"""Voice Activity Projection (VAP) rebuilt in PyTorch, so it can run continuously and be re-trained here.

Same network and weights as models/vap/vap.onnx (otoSpeech checkpoint, CC BY 4.0 weights from
viks66/VAP_checkpoints; MIT architecture by Erik Ekstedt). The weights are read from the ONNX file's
tensors (a plain data format, no code is loaded), checked against the ONNX model by tools/vap_port_check.py.

Two ways to run it:
  VAP.forward(audio)    a whole window of both voices, (batch, 2, samples) at 16 kHz -> every 20 ms frame
  VAPStream.step(a, b)  20 ms of each voice at a time, keeping its state: a fresh answer every 20 ms at a
                        small fraction of the cost of re-reading a window (phone-class)

Channel 0 is the person, channel 1 is Alba. Outputs per 20 ms frame:
  p_now[..., k]     share of the talking in the next 0-0.6 s that is speaker k's
  p_future[..., k]  the same for 0.6-2.0 s
  probs             the 256 joint futures: which of 4 time bins (0.2, 0.4, 0.6, 0.8 s) each speaker talks in
  vad               is each speaker talking now
"""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

RATE = 16000
FRAME = 320                     # 20 ms: one output frame
BINS = (10, 20, 30, 40)         # frames per future bin (0.2, 0.4, 0.6, 0.8 s)
DIM, HEADS = 256, 4


class ChannelNorm(nn.Module):
    """CPC's norm: over the channels of each time step, unbiased variance."""

    def __init__(self, n):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, n, 1))
        self.bias = nn.Parameter(torch.zeros(1, n, 1))

    def forward(self, x):                      # (B, C, T)
        mean = x.mean(1, keepdim=True)
        var = x.var(1, keepdim=True, unbiased=True)
        return (x - mean) / torch.sqrt(var + 1e-5) * self.weight + self.bias


class Encoder(nn.Module):
    """CPC encoder (frozen in VAP's training): 5 convolutions to 100 Hz, an LSTM, then a causal
    convolution down to 50 Hz."""
    CONVS = ((10, 5, 3), (8, 4, 2), (4, 2, 1), (4, 2, 1), (4, 2, 1))   # kernel, stride, padding

    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv1d(1 if i == 0 else DIM, DIM, k, s, padding=p)
                                    for i, (k, s, p) in enumerate(self.CONVS)])
        self.norms = nn.ModuleList([ChannelNorm(DIM) for _ in self.CONVS])
        self.lstm = nn.LSTM(DIM, DIM, batch_first=True)
        self.down = nn.Conv1d(DIM, DIM, 5, 2)
        self.down_ln = nn.LayerNorm(DIM)

    def frames100(self, wav):                  # (B, N) -> (B, D, T100)
        x = wav[:, None]
        for conv, norm in zip(self.convs, self.norms):
            x = F.relu(norm(conv(x)))
        return x

    def downsample(self, z):                   # (B, T100, D) -> (B, T50, D), causal
        y = self.down(F.pad(z.transpose(1, 2), (4, 0)))
        return F.gelu(self.down_ln(y.transpose(1, 2)))

    def forward(self, wav):
        z, _ = self.lstm(self.frames100(wav).transpose(1, 2))
        return self.downsample(z)


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.key = nn.Linear(DIM, DIM, bias=False)
        self.query = nn.Linear(DIM, DIM, bias=False)
        self.value = nn.Linear(DIM, DIM, bias=False)
        self.proj = nn.Linear(DIM, DIM, bias=False)

    @staticmethod
    def heads(x):                              # (B, T, D) -> (B, H, T, D/H)
        b, t, _ = x.shape
        return x.view(b, t, HEADS, DIM // HEADS).transpose(1, 2)

    def forward(self, q_in, kv_in, bias):
        q, k, v = self.heads(self.query(q_in)), self.heads(self.key(kv_in)), self.heads(self.value(kv_in))
        att = torch.softmax(q @ k.transpose(-1, -2) * (1.0 / math.sqrt(DIM)) + bias, dim=-1)
        y = (att @ v).transpose(1, 2).reshape(q_in.shape[0], q_in.shape[1], DIM)
        return self.proj(y)


def ffn():
    return nn.Sequential(nn.Linear(DIM, 3 * DIM, bias=False), nn.GELU(), nn.Linear(3 * DIM, DIM, bias=False))


SLOPES = torch.tensor([2.0 ** -(2 * (h + 1)) for h in range(HEADS)])   # ALiBi: 1/4, 1/16, 1/64, 1/256


def alibi(n_q, n_k, offset=0, device=None):
    """Causal ALiBi bias, (1, H, n_q, n_k): query i (at absolute position offset+i) sees keys j <= offset+i,
    farther ones a little less (as in the ONNX table)."""
    qi = torch.arange(n_q, device=device)[:, None] + offset
    kj = torch.arange(n_k, device=device)[None, :] + (offset + n_q - n_k)
    dist = (qi - kj).float()
    b = -SLOPES.to(device)[:, None, None] * dist[None]
    return torch.where(dist[None] >= 0, b, torch.full_like(b, float('-inf')))[None]


class SelfLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln_self_attn, self.ln_ffnetwork = nn.LayerNorm(DIM), nn.LayerNorm(DIM)
        self.mha, self.ffnetwork = Attention(), ffn()

    def forward(self, x, bias, kv=None):
        h = self.ln_self_attn(x)
        x = x + self.mha(h, h if kv is None else kv, bias)
        return x + self.ffnetwork(self.ln_ffnetwork(x))


class StereoLayer(nn.Module):
    """Each voice attends to itself, then to the other voice (keys and values from the other voice's
    layer input, not normalised), then a feed-forward; weights shared by both voices."""

    def __init__(self):
        super().__init__()
        self.ln_self_attn, self.ln_src_attn, self.ln_ffnetwork = nn.LayerNorm(DIM), nn.LayerNorm(DIM), nn.LayerNorm(DIM)
        self.mha, self.mha_cross, self.ffnetwork = Attention(), Attention(), ffn()

    def one(self, x, other, bias, self_kv=None, other_kv=None):
        h = self.ln_self_attn(x)
        z = x + self.mha(h, h if self_kv is None else self_kv, bias)
        z = z + self.mha_cross(self.ln_src_attn(z), other if other_kv is None else other_kv, bias)
        return z + self.ffnetwork(self.ln_ffnetwork(z))

    def forward(self, x1, x2, bias):
        return self.one(x1, x2, bias), self.one(x2, x1, bias)


class VAP(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.ar_channel = SelfLayer()
        self.ar = nn.ModuleList([StereoLayer() for _ in range(3)])
        self.comb_a = nn.Linear(DIM, DIM, bias=False)
        self.comb_b = nn.Linear(DIM, DIM, bias=False)
        self.comb_ln = nn.LayerNorm(DIM)
        self.va_classifier = nn.Linear(DIM, 1)
        self.vap_head = nn.Linear(DIM, 256)
        self.register_buffer('near_activity', torch.zeros(256, 2))
        self.register_buffer('later_activity', torch.zeros(256, 2))

    def encode(self, audio):                   # (B, 2, N) -> two (B, T, D)
        return self.encoder(audio[:, 0]), self.encoder(audio[:, 1])

    def heads(self, y1, y2):
        x = F.gelu(self.comb_ln(self.comb_a(y1))) + F.gelu(self.comb_ln(self.comb_b(y2)))
        logits = self.vap_head(x)
        vad_logits = torch.cat([self.va_classifier(y1), self.va_classifier(y2)], -1)
        return logits, vad_logits

    def readout(self, logits, vad_logits):
        probs = torch.softmax(logits, -1)
        near, later = probs @ self.near_activity, probs @ self.later_activity
        return {'logits': logits, 'vad_logits': vad_logits, 'probs': probs, 'vad': torch.sigmoid(vad_logits),
                'p_now': near / (near.sum(-1, keepdim=True) + 1e-5),
                'p_future': later / (later.sum(-1, keepdim=True) + 1e-5)}

    def forward(self, audio):
        e1, e2 = self.encode(audio)
        bias = alibi(e1.shape[1], e1.shape[1], device=audio.device)
        x1, x2 = self.ar_channel(e1, bias), self.ar_channel(e2, bias)
        for layer in self.ar:
            x1, x2 = layer(x1, x2, bias)
        return self.readout(*self.heads(x1, x2))


# ----------------------------------------------------------------------------------------------------------
# Weights from the ONNX file

def load_from_onnx(path):
    """Reads the tensors of models/vap/vap.onnx by the role of the node that uses them."""
    import onnx
    from onnx import numpy_helper
    m = onnx.load(str(path))
    g = m.graph
    init = {t.name: numpy_helper.to_array(t) for t in g.initializer}
    alias = {n.output[0]: n.input[0] for n in g.node if n.op_type == 'Identity'}

    def W(name):
        while name in alias:
            name = alias[name]
        return init[name]

    by_node = {}
    for n in g.node:
        if n.op_type in ('MatMul', 'Add', 'LSTM', 'Conv', 'LayerNormalization'):
            by_node[n.name] = n
    model = VAP()
    sd = {}

    def lin(dst, node_name, transpose=True):
        w = W(by_node[node_name].input[1])
        sd[dst] = torch.from_numpy(w.T.copy() if transpose else w.copy())

    # encoder
    for i in range(5):
        sd[f'encoder.convs.{i}.weight'] = torch.from_numpy(init[f'model.encoder.encoder.gEncoder.conv{i}.weight'])
        sd[f'encoder.convs.{i}.bias'] = torch.from_numpy(init[f'model.encoder.encoder.gEncoder.conv{i}.bias'])
        sd[f'encoder.norms.{i}.weight'] = torch.from_numpy(init[f'model.encoder.encoder.gEncoder.batchNorm{i}.weight'])
        sd[f'encoder.norms.{i}.bias'] = torch.from_numpy(init[f'model.encoder.encoder.gEncoder.batchNorm{i}.bias'])
    lstm = by_node['/model/encoder/gAR/baseNet/LSTM']
    w_ih, w_hh, b = W(lstm.input[1])[0], W(lstm.input[2])[0], W(lstm.input[3])[0]
    # ONNX gate order i, o, f, c -> PyTorch i, f, g(c), o
    order = lambda a: np.concatenate([a[0:256], a[512:768], a[768:1024], a[256:512]], 0)
    sd['encoder.lstm.weight_ih_l0'] = torch.from_numpy(order(w_ih))
    sd['encoder.lstm.weight_hh_l0'] = torch.from_numpy(order(w_hh))
    sd['encoder.lstm.bias_ih_l0'] = torch.from_numpy(order(b[:1024]))
    sd['encoder.lstm.bias_hh_l0'] = torch.from_numpy(order(b[1024:]))
    sd['encoder.down.weight'] = torch.from_numpy(init['model.encoder.downsample.1.weight'])
    sd['encoder.down.bias'] = torch.from_numpy(init['model.encoder.downsample.1.bias'])
    sd['encoder.down_ln.weight'] = torch.from_numpy(init['model.encoder.downsample.2.ln.weight'])
    sd['encoder.down_ln.bias'] = torch.from_numpy(init['model.encoder.downsample.2.ln.bias'])
    # transformers
    for ln in ('ln_self_attn', 'ln_ffnetwork'):
        for p in ('weight', 'bias'):
            sd[f'ar_channel.{ln}.{p}'] = torch.from_numpy(init[f'model.ar_channel.layers.0.{ln}.{p}'])
    for r in ('key', 'query', 'value', 'proj'):
        lin(f'ar_channel.mha.{r}.weight', f'/model/ar_channel/layers.0/mha/{r}/MatMul')
    lin('ar_channel.ffnetwork.0.weight', '/model/ar_channel/layers.0/ffnetwork/ffnetwork.0/MatMul')
    lin('ar_channel.ffnetwork.2.weight', '/model/ar_channel/layers.0/ffnetwork/ffnetwork.3/MatMul')
    for i in range(3):
        for ln in ('ln_self_attn', 'ln_src_attn', 'ln_ffnetwork'):
            for p in ('weight', 'bias'):
                sd[f'ar.{i}.{ln}.{p}'] = torch.from_numpy(init[f'model.ar.layers.{i}.{ln}.{p}'])
        for att in ('mha', 'mha_cross'):
            for r in ('key', 'query', 'value', 'proj'):
                lin(f'ar.{i}.{att}.{r}.weight', f'/model/ar/layers.{i}/{att}/{r}/MatMul')
        lin(f'ar.{i}.ffnetwork.0.weight', f'/model/ar/layers.{i}/ffnetwork/ffnetwork.0/MatMul')
        lin(f'ar.{i}.ffnetwork.2.weight', f'/model/ar/layers.{i}/ffnetwork/ffnetwork.3/MatMul')
    lin('comb_a.weight', '/model/ar/combinator/h0_a/MatMul')
    lin('comb_b.weight', '/model/ar/combinator/h0_b/MatMul')
    sd['comb_ln.weight'] = torch.from_numpy(init['model.ar.combinator.ln.weight'])
    sd['comb_ln.bias'] = torch.from_numpy(init['model.ar.combinator.ln.bias'])
    lin('va_classifier.weight', '/model/va_classifier/MatMul')
    va_add = by_node['/model/va_classifier/Add']
    sd['va_classifier.bias'] = torch.from_numpy(np.asarray(W(va_add.input[0]), np.float32).reshape(1))
    lin('vap_head.weight', '/model/vap_head/MatMul')
    sd['vap_head.bias'] = torch.from_numpy(init['model.vap_head.bias'])
    sd['near_activity'] = torch.from_numpy(init['near_activity'])
    sd['later_activity'] = torch.from_numpy(init['later_activity'])
    missing, unexpected = model.load_state_dict({k: v.float() for k, v in sd.items()}, strict=True), None
    return model.eval()


# ----------------------------------------------------------------------------------------------------------
# Streaming: 20 ms in, one fresh frame out

class VAPStream:
    """Runs VAP one 20 ms frame at a time, keeping its state, so every answer is fresh and costs a small
    fraction of re-reading a window.

    Exact by construction: 100 Hz frame k of the convolutions depends on samples 160k-153 .. 160k+311, so
    once the 20 ms hop t has arrived (samples up to 320t+319), frames 2t-1 and 2t are final. They go through
    the LSTM (state carried), the causal downsampling makes 50 Hz frame t from frames 2t-4..2t, and the
    transformer layers attend to cached keys/values of the last `context_s` seconds. A window run over the
    same stretch differs only in starting its LSTM and attention at the window's first sample."""

    KEEP = 2400                                            # samples of audio kept per voice

    def __init__(self, model, context_s=6.0):
        self.m = model.eval()
        self.ctx = int(round(context_s * 50))
        self.reset()

    def reset(self):
        self.buf = [np.zeros(0, np.float32), np.zeros(0, np.float32)]
        self.buf_start = 0                                 # global sample index of buf[0] (same for both voices)
        self.lstm = [None, None]
        self.z = [torch.zeros(1, 4, DIM), torch.zeros(1, 4, DIM)]   # last LSTM outputs (zeros = causal padding)
        self.cache = None
        self.t = 0

    def _frames(self, ch, k0, k1):
        """LSTM outputs for 100 Hz frames k0..k1 of voice ch."""
        s0 = max(0, 160 * (k0 - 2))
        s1 = min(self.buf_start + len(self.buf[ch]), 160 * k1 + 312)
        seg = self.buf[ch][s0 - self.buf_start:s1 - self.buf_start]
        x = torch.from_numpy(np.ascontiguousarray(seg)).float()[None]
        f = self.m.encoder.frames100(x)                    # (1, D, n)
        j0 = k0 - s0 // 160
        f = f[:, :, j0:j0 + (k1 - k0 + 1)]
        assert f.shape[2] == k1 - k0 + 1, (f.shape, k0, k1, s0, s1)
        z, self.lstm[ch] = self.m.encoder.lstm(f.transpose(1, 2), self.lstm[ch])
        return z

    @torch.no_grad()
    def step(self, person, agent):
        """320 samples (20 ms at 16 kHz) of each voice -> this frame's p_now, p_future, vad, probs."""
        t = self.t
        for ch, x in enumerate((person, agent)):
            x = np.asarray(x, np.float32).reshape(-1)
            assert len(x) == FRAME
            self.buf[ch] = np.concatenate([self.buf[ch], x])
        if len(self.buf[0]) > self.KEEP:
            drop = (len(self.buf[0]) - self.KEEP) // 160 * 160
            self.buf = [b[drop:] for b in self.buf]
            self.buf_start += drop
        e = []
        for ch in (0, 1):
            z = self._frames(ch, max(0, 2 * t - 1), 2 * t)
            self.z[ch] = torch.cat([self.z[ch], z], 1)[:, -5:]
            y = self.m.encoder.down(self.z[ch].transpose(1, 2))            # (1, D, 1): frames 2t-4 .. 2t
            e.append(F.gelu(self.m.encoder.down_ln(y.transpose(1, 2))))
        out = self._transformer(e[0], e[1])
        self.t += 1
        return out

    # keys and values are kept already projected, so a step only projects the new frame
    def _push(self, slot, att, x):
        k, v = Attention.heads(att.key(x)), Attention.heads(att.value(x))
        K, V = self.cache.get(slot, (None, None))
        K = k if K is None else torch.cat([K, k], 2)[:, :, -self.ctx:]
        V = v if V is None else torch.cat([V, v], 2)[:, :, -self.ctx:]
        self.cache[slot] = (K, V)
        return K, V

    def _attend(self, att, q_in, K, V):
        q = Attention.heads(att.query(q_in))
        n = K.shape[2]
        s = torch.softmax(q @ K.transpose(-1, -2) * (1.0 / math.sqrt(DIM)) + self.bias[..., -n:], dim=-1)
        return att.proj((s @ V).transpose(1, 2).reshape(1, 1, DIM))

    def _transformer(self, e1, e2):
        m = self.m
        if self.cache is None:
            self.cache = {}
            dist = torch.arange(self.ctx - 1, -1, -1).float()
            self.bias = (-SLOPES[:, None, None] * dist[None, None, :])[None]     # (1, H, 1, ctx)
        xs = []
        for k, e in enumerate((e1, e2)):
            sa = m.ar_channel
            h = sa.ln_self_attn(e)
            x = e + self._attend(sa.mha, h, *self._push(('ch', k), sa.mha, h))
            xs.append(x + sa.ffnetwork(sa.ln_ffnetwork(x)))
        x1, x2 = xs
        for i, layer in enumerate(m.ar):
            h = [layer.ln_self_attn(x1), layer.ln_self_attn(x2)]
            kv_self = [self._push(('self', i, k), layer.mha, h[k]) for k in (0, 1)]
            # each voice's cross-attention reads the other voice's raw layer input
            kv_cross = [self._push(('cross', i, 0), layer.mha_cross, x2), self._push(('cross', i, 1), layer.mha_cross, x1)]
            ys = []
            for k, x in enumerate((x1, x2)):
                z = x + self._attend(layer.mha, h[k], *kv_self[k])
                z = z + self._attend(layer.mha_cross, layer.ln_src_attn(z), *kv_cross[k])
                ys.append(z + layer.ffnetwork(layer.ln_ffnetwork(z)))
            x1, x2 = ys
        out = m.readout(*m.heads(x1, x2))
        return {k: v[0, -1] for k, v in out.items()}
