"""Fast streaming hearing for Alba Duplex: the LFM3-Audio speech encoder, one 80 ms token per step, replayed as a
captured CUDA graph.

Why: the encoder is tiny (17 layers of width 768, ~0.1 GFLOP per step), but run through the library it launches
hundreds of small GPU tasks per step and spends 50-110 ms on bookkeeping (masks, growing caches, Python). Here
every stateful part has a fixed shape and lives in preallocated buffers, so the whole step can be captured once
and replayed: the GPU then runs it without the processor in the way.

What it computes is the model's own streaming encoder (Lfm3AudioEncoder.stream + the projector), unchanged:
  * front end: the downsampler over the 16-frame window [previous token's 8 frames, this token's 8], keeping its
    second output (the first token of a call uses its own 8 frames only, as the library does);
  * 17 layers, each the model's own sub-modules for everything stateless (norms, projections, gates, feed-forward);
    only the memory is ours: the short convolutions' last 2 inputs, and for the attention layers a 32-slot ring
    buffer of keys/values (the encoder's sliding window: a token sees itself and the 31 before it), with the
    positions of what each slot holds, and rotary positions computed on the GPU;
  * the final norm and the projector to the language model's width.
Checked against the library's own streaming path by tools/test_fast_audio.py.
"""
import torch
from torch.nn import functional as F


def _rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


class FastEncoderStream:
    def __init__(self, model, device='cuda', use_graph=True):
        m = model.model if hasattr(model, 'model') and hasattr(model.model, 'audio_encoder') else model
        self.enc, self.proj = m.audio_encoder, m.audio_projector
        lfm = self.enc.lfm
        self.cfg = cfg = lfm.config
        self.layers = list(lfm.layers[: cfg.num_hidden_layers])
        self.types = list(cfg.layer_types[: cfg.num_hidden_layers])
        self.norm = lfm.embedding_norm
        self.device = device
        p = next(self.enc.parameters())
        self.dtype = p.dtype
        self.W = int(self.enc.config.sliding_window)
        self.H = cfg.num_attention_heads
        self.KV = cfg.num_key_value_heads
        self.hd = getattr(cfg, 'head_dim', None) or cfg.hidden_size // cfg.num_attention_heads
        D = cfg.hidden_size
        # rotary frequencies (the model's own buffers): LFM2 layers (this encoder) have one table and rotate in the
        # model's number format; LFM3 layers have one per attention type and rotate in float32
        rot = lfm.rotary_emb
        self.lfm2 = hasattr(rot, 'inv_freq')
        if self.lfm2:
            self.inv_freq = {t: rot.inv_freq.to(device, torch.float32) for t in set(self.types) if t != 'conv'}
            self.rot_scale = {t: float(rot.attention_scaling) for t in self.inv_freq}
        else:
            self.inv_freq = {t: getattr(rot, f'{t}_inv_freq').to(device, torch.float32) for t in set(self.types) if t != 'conv'}
            self.rot_scale = {t: float(getattr(rot, f'{t}_attention_scaling')) for t in self.inv_freq}
        # state (fixed shapes; reset() refills them in place so a captured graph keeps pointing at them)
        K = cfg.conv_L_cache
        self.conv_state = {i: torch.zeros(1, D, K - 1, device=device, dtype=self.dtype)
                           for i, t in enumerate(self.types) if t == 'conv'}
        self.k_buf = {i: torch.zeros(self.KV, self.W, self.hd, device=device, dtype=self.dtype)
                      for i, t in enumerate(self.types) if t != 'conv'}
        self.v_buf = {i: torch.zeros(self.KV, self.W, self.hd, device=device, dtype=self.dtype)
                      for i, t in enumerate(self.types) if t != 'conv'}
        self.slot_pos = torch.full((self.W,), -1, device=device, dtype=torch.long)
        self.pos = torch.zeros((), device=device, dtype=torch.long)
        self.window = torch.zeros(1, self.enc.config.num_mel_bins, 16, device=device, dtype=self.dtype)
        self.out = None
        self.graph = None
        self.use_graph = use_graph and str(device).startswith('cuda')
        self.started = False

    # ---- one token, everything on the GPU, fixed shapes ---------------------------------------------------------
    def _layers(self, x):
        """x: (1, 1, D) the downsampled token -> (2048,) the projected token. Advances the state in place."""
        pos = self.pos
        slot = torch.remainder(pos, self.W).view(1)
        self.slot_pos.index_copy_(0, slot, pos.view(1))
        valid = (self.slot_pos >= 0) & (pos - self.slot_pos < self.W)
        bias = torch.where(valid, 0.0, float('-inf')).to(torch.float32)            # (W,)
        cs = {}
        for t, inv in self.inv_freq.items():
            f = pos.to(torch.float32) * inv
            e = torch.cat((f, f))
            # rounded to the model's number format first, as the library hands them over
            cs[t] = ((e.cos() * self.rot_scale[t]).to(self.dtype).float(), (e.sin() * self.rot_scale[t]).to(self.dtype).float())
        h = x
        for i, (layer, t) in enumerate(zip(self.layers, self.types)):
            res = h
            a = layer.operator_norm(h)
            if t == 'conv':
                c = layer.conv
                BCx = c.in_proj(a)[0, 0]                                             # (3D,)
                B, C, u = BCx.chunk(3)
                u = (B * u).view(1, -1, 1)
                st = self.conv_state[i]
                win = torch.cat([st, u], dim=-1)                                     # (1, D, K)
                w = c.conv.weight.squeeze(1)                                         # (D, K)
                y = (win[0] * w).sum(-1)
                if c.conv.bias is not None:
                    y = y + c.conv.bias
                st.copy_(win[:, :, 1:])
                o = c.out_proj((C * y).view(1, 1, -1))
            else:
                at = layer.self_attn
                qn = getattr(at, 'q_layernorm', None) or at.q_norm
                kn = getattr(at, 'k_layernorm', None) or at.k_norm
                q = qn(at.q_proj(a).view(self.H, self.hd))                           # (H, hd)
                k = kn(at.k_proj(a).view(self.KV, self.hd))
                v = at.v_proj(a).view(self.KV, self.hd)
                cos, sin = cs[t]
                if self.lfm2:                                    # in the model's number format, as Lfm2Attention
                    c_, s_ = cos.to(self.dtype), sin.to(self.dtype)
                    q = q * c_ + _rotate_half(q) * s_
                    k = k * c_ + _rotate_half(k) * s_
                else:                                            # in float32, as Lfm3Attention
                    q = (q.float() * cos + _rotate_half(q.float()) * sin).to(self.dtype)
                    k = (k.float() * cos + _rotate_half(k.float()) * sin).to(self.dtype)
                kb, vb = self.k_buf[i], self.v_buf[i]
                kb.index_copy_(1, slot, k.view(self.KV, 1, self.hd))
                vb.index_copy_(1, slot, v.view(self.KV, 1, self.hd))
                rep = self.H // self.KV
                kk = kb.repeat_interleave(rep, dim=0)                                 # (H, W, hd)
                vv = vb.repeat_interleave(rep, dim=0)
                s = torch.einsum('hd,hwd->hw', q.float(), kk.float()) * at.scaling + bias
                p_ = torch.softmax(s, dim=-1).to(self.dtype)
                o = torch.einsum('hw,hwd->hd', p_, vv)                               # (H, hd)
                if getattr(at, 'gate_proj', None) is not None:   # LFM3's gated attention
                    o = o * torch.sigmoid(at.gate_proj(a)).view(self.H, 1)
                o = (getattr(at, 'out_proj', None) or at.o_proj)(o.reshape(1, 1, -1))
            if getattr(layer, 'use_sandwich_norm', False):
                o = layer.post_operator_norm(o)
            h = o + res
            f = layer.feed_forward(layer.ffn_norm(h))
            if getattr(layer, 'use_sandwich_norm', False):
                f = layer.post_ffn_norm(f)
            h = h + f
        h = self.norm(h)
        self.pos += 1
        return self.proj(h)[0, 0]

    def _steady(self):
        tok = self.enc.downsample(self.window).transpose(1, 2)[:, 1:2]               # (1, 1, D)
        return self._layers(tok)

    # ---- public ---------------------------------------------------------------------------------------------------
    def reset(self):
        for s in self.conv_state.values():
            s.zero_()
        for b in list(self.k_buf.values()) + list(self.v_buf.values()):
            b.zero_()
        self.slot_pos.fill_(-1)
        self.pos.zero_()
        self.window.zero_()
        self.started = False

    @torch.no_grad()
    def push(self, feats):
        """feats: (1, 128, 8) mel frames of the step just completed -> its token (2048,), on the GPU."""
        feats = feats.to(self.device, self.dtype)
        if not self.started:
            # first token of a call: its own 8 frames only (as the library), eagerly
            self.window[:, :, 8:].copy_(feats)
            tok = self.enc.downsample(feats).transpose(1, 2)[:, 0:1]
            self.started = True
            return self._layers(tok).clone()
        self.window[:, :, :8].copy_(self.window[:, :, 8:])
        self.window[:, :, 8:].copy_(feats)
        if not self.use_graph:
            return self._steady().clone()
        if self.graph is None:
            self._capture()
        self.graph.replay()
        return self.out.clone()

    def _capture(self):
        """Captures the steady step once. Warm-up and capture run on copies of the state, which is then restored."""
        saved = self._save()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                self._steady()
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.out = self._steady()
        self._restore(saved)

    def _save(self):
        return ({i: v.clone() for i, v in self.conv_state.items()}, {i: v.clone() for i, v in self.k_buf.items()},
                {i: v.clone() for i, v in self.v_buf.items()}, self.slot_pos.clone(), self.pos.clone(),
                self.window.clone())

    def _restore(self, saved):
        cs, kb, vb, sp, pos, win = saved
        for i, v in cs.items():
            self.conv_state[i].copy_(v)
        for i, v in kb.items():
            self.k_buf[i].copy_(v)
        for i, v in vb.items():
            self.v_buf[i].copy_(v)
        self.slot_pos.copy_(sp)
        self.pos.copy_(pos)
        self.window.copy_(win)


class FastFrameSampler:
    """The model's own MaskGIT frame sampler (Lfm3AudioForConditionalGeneration.sample_audio_frame), replayed as
    one captured CUDA graph per frame: codebooks revealed in a random order over the passes of the model's schedule;
    each pass scores the hidden codebooks with the model's depth decoder; repetition penalty over the call's last
    `hist_len` frames (as Hugging Face's RepetitionPenaltyLogitsProcessor), temperature, top-k, then a draw. Two
    changes make it capturable, neither changes what is sampled: the random order is the argsort of uniform noise
    (any uniformly random permutation), and the draw is the Gumbel-max trick (exactly the softmax distribution)."""

    def __init__(self, model, device='cuda', temperature=1.0, top_k=128, repetition_penalty=1.1, hist_len=64,
                 use_graph=True, eos_first=False):
        cfg = model.config
        self.dd = model.model.depth_decoder
        self.n, self.V = cfg.num_codebooks, cfg.codebook_size
        self.schedule = list(cfg.maskgit_schedule)
        if eos_first and self.schedule[0] > 1:
            # one codebook alone in the first pass ([1, 3, 4, 4] for the release's [4, 4, 4]): an end-of-speech
            # frame is all end-of-speech codes, so the stop is decided once, by one draw from the model's chance
            # of it, as exact sampling would; four codebooks drawn together each get a chance to stop the
            # utterance (the generator stops on any), about four times the model's chance mid-sentence
            self.schedule = [1, self.schedule[0] - 1] + self.schedule[1:]
        self.T, self.k, self.rp = float(temperature), int(top_k), float(repetition_penalty)
        p = next(self.dd.parameters())
        self.dtype, self.device = p.dtype, device
        self.h = torch.zeros(1, cfg.text_config.hidden_size, device=device, dtype=self.dtype)
        self.hist = torch.zeros(hist_len, self.n, device=device, dtype=torch.long)
        self.hist_valid = torch.zeros(hist_len, device=device, dtype=torch.float32)
        self.hist_len, self.ptr = hist_len, 0
        self.order_override = None            # tests only: a fixed reveal order
        self.use_graph = use_graph and str(device).startswith('cuda')
        self.graph, self.out = None, None

    def reset(self):
        self.hist.zero_()
        self.hist_valid.zero_()
        self.ptr = 0

    def _sample(self):
        n = self.n
        codes = torch.zeros(1, n, device=self.device, dtype=torch.long)
        revealed = torch.zeros(1, n, device=self.device, dtype=torch.bool)
        order = (self.order_override if self.order_override is not None
                 else torch.rand(n, device=self.device).argsort())
        # which codes each codebook has said in the last frames (for the repetition penalty)
        pen = torch.zeros(n, self.V, device=self.device, dtype=torch.float32)
        pen.scatter_reduce_(1, self.hist.T, self.hist_valid.expand(n, -1), reduce='amax')
        start = 0
        for size in self.schedule:
            pick = order[start:start + size]
            start += size
            logits = self.dd(self.h, codes, ~revealed)[0].index_select(0, pick).float()      # (size, V)
            if self.rp != 1.0:
                pl = pen.index_select(0, pick) > 0
                logits = torch.where(pl, torch.where(logits < 0, logits * self.rp, logits / self.rp), logits)
            if self.T:
                logits = logits / self.T
                kth = logits.topk(min(self.k, self.V), dim=-1).values[:, -1:]
                logits = logits.masked_fill(logits < kth, float('-inf'))
                g = -torch.log(-torch.log(torch.rand_like(logits).clamp_(1e-20, 1.0)))
                sel = (logits + g).argmax(-1)
            else:
                sel = logits.argmax(-1)
            codes[0].index_copy_(0, pick, sel)
            revealed[0].index_fill_(0, pick, True)
        return codes[0]

    @torch.no_grad()
    def sample(self, hidden):
        """hidden: (hidden_size,) the language model's state before the frame -> (12,) codes on the GPU."""
        self.h.copy_(torch.as_tensor(hidden).view(1, -1).to(self.device, self.dtype))
        if not self.use_graph or self.order_override is not None:
            c = self._sample()
        else:
            if self.graph is None:
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(2):
                        self._sample()
                torch.cuda.current_stream().wait_stream(s)
                self.graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph):
                    self.out = self._sample()
            self.graph.replay()
            c = self.out
        return c.clone()

    def remember(self, codes):
        """A frame that was spoken (not the end-of-speech frame) joins the penalty history."""
        self.hist[self.ptr].copy_(codes)
        self.hist_valid[self.ptr] = 1.0
        self.ptr = (self.ptr + 1) % self.hist_len


class FastCodecDecoder:
    """The Liquid codec's streaming decoder (LiquidCodecModel.decode with a streaming cache): one speech frame
    (12 codes) -> its sound, replayed as a captured CUDA graph.

    Per frame, as the library does it: the codes become latents (the quantizer's own decode), are projected
    (from_quantizer) and stretched to 6 positions; 8 LFM3 layers (the model's own sub-modules; ours only the memory:
    the 7-wide short convolutions' last 6 inputs, and a ring buffer for the sliding-window attention, 30 positions)
    and the final norm; then the spectrum head and the inverse STFT; the overlap with the previous frame is added
    and divided by the window envelope, the last 960 samples held back for the next frame. The first frame of an
    utterance drops its first 480 samples and flush() returns the held-back 480, exactly as decode() does."""

    def __init__(self, codec, device='cuda', use_graph=True):
        self.codec = codec
        cfg = codec.config
        dec = codec.decoder
        self.dec, self.lfm = dec, dec.lfm
        lc = self.lfm.config
        self.layers = list(self.lfm.layers[: lc.num_hidden_layers])
        self.types = list(lc.layer_types[: lc.num_hidden_layers])
        self.S = dec.frame_stack
        self.W = int(cfg.decoder_window)
        self.R = self.W + self.S
        self.H, self.KV = lc.num_attention_heads, lc.num_key_value_heads
        self.hd = getattr(lc, 'head_dim', None) or lc.hidden_size // lc.num_attention_heads
        p = next(dec.parameters())
        self.dtype, self.device = p.dtype, device
        rot = self.lfm.rotary_emb
        self.inv_freq = {t: getattr(rot, f'{t}_inv_freq').to(device, torch.float32) for t in set(self.types) if t != 'conv'}
        self.rot_scale = {t: float(getattr(rot, f'{t}_attention_scaling')) for t in self.inv_freq}
        D, K = lc.hidden_size, lc.conv_L_cache
        self.conv_state = {i: torch.zeros(1, D, K - 1, device=device, dtype=self.dtype)
                           for i, t in enumerate(self.types) if t == 'conv'}
        self.k_buf = {i: torch.zeros(self.KV, self.R, self.hd, device=device, dtype=self.dtype)
                      for i, t in enumerate(self.types) if t != 'conv'}
        self.v_buf = {i: torch.zeros(self.KV, self.R, self.hd, device=device, dtype=self.dtype)
                      for i, t in enumerate(self.types) if t != 'conv'}
        self.slot_pos = torch.full((self.R,), -1, device=device, dtype=torch.long)
        self.pos = torch.zeros((), device=device, dtype=torch.long)
        self.trim = (cfg.stft_win_length - cfg.stft_hop_length) // 2
        self.tail = 2 * self.trim
        self.ola_audio = torch.zeros(1, self.tail, device=device, dtype=torch.float32)
        self.ola_env = torch.zeros(self.tail, device=device, dtype=torch.float32)
        self.codes_in = torch.zeros(1, cfg.n_q, 1, device=device, dtype=torch.long)
        self.ar = torch.arange(self.S, device=device)
        self.started = False
        self.use_graph = use_graph and str(device).startswith('cuda')
        self.graph, self.out = None, None

    def _step(self):
        S = self.S
        latents = self.codec.quantizer.decode(self.codes_in).mT.to(self.codec.from_quantizer.weight.dtype)
        emb = self.codec.from_quantizer(latents)                                            # (1, 1, D)
        h = F.interpolate(emb.mT, S, mode='nearest-exact').mT                               # (1, S, D)
        qpos = self.pos + self.ar                                                           # (S,)
        slots = torch.remainder(qpos, self.R)
        self.slot_pos.index_copy_(0, slots, qpos)
        sp = self.slot_pos.view(1, -1)
        q_ = qpos.view(-1, 1)
        valid = (sp >= 0) & (sp <= q_) & (q_ - sp < self.W)                                  # (S, R)
        bias = torch.where(valid, 0.0, float('-inf')).to(torch.float32)
        cs = {}
        for t, inv in self.inv_freq.items():
            f = qpos.to(torch.float32).view(-1, 1) * inv.view(1, -1)                         # (S, hd/2)
            e = torch.cat((f, f), dim=-1)
            cs[t] = (e.cos() * self.rot_scale[t], e.sin() * self.rot_scale[t])               # float32 (the codec's hook)
        for i, (layer, t) in enumerate(zip(self.layers, self.types)):
            res = h
            a = layer.operator_norm(h)                                                      # (1, S, D)
            if t == 'conv':
                c = layer.conv
                B, C, u = c.in_proj(a).transpose(-1, -2).chunk(3, dim=-2)                   # each (1, D, S)
                u = B * u
                st = self.conv_state[i]
                win = torch.cat([st, u], dim=-1)                                            # (1, D, K-1+S)
                y = F.conv1d(win.to(c.conv.weight.dtype), c.conv.weight, c.conv.bias, groups=win.shape[1])
                st.copy_(win[:, :, -st.shape[-1]:])
                o = c.out_proj((C * y.to(C.dtype)).transpose(-1, -2).contiguous())
            else:
                at = layer.self_attn
                q = at.q_norm(at.q_proj(a).view(S, self.H, self.hd)).transpose(0, 1)         # (H, S, hd)
                k = at.k_norm(at.k_proj(a).view(S, self.KV, self.hd)).transpose(0, 1)       # (KV, S, hd)
                v = at.v_proj(a).view(S, self.KV, self.hd).transpose(0, 1)
                cos, sin = cs[t]
                q = (q.float() * cos + _rotate_half(q.float()) * sin).to(self.dtype)
                k = (k.float() * cos + _rotate_half(k.float()) * sin).to(self.dtype)
                kb, vb = self.k_buf[i], self.v_buf[i]
                kb.index_copy_(1, slots, k)
                vb.index_copy_(1, slots, v)
                rep = self.H // self.KV
                kk = kb.repeat_interleave(rep, dim=0)                                        # (H, R, hd)
                vv = vb.repeat_interleave(rep, dim=0)
                s = torch.einsum('hsd,hrd->hsr', q.float(), kk.float()) * at.scaling + bias
                p_ = torch.softmax(s, dim=-1).to(self.dtype)
                o = torch.einsum('hsr,hrd->hsd', p_, vv).transpose(0, 1)                     # (S, H, hd)
                if getattr(at, 'gate_proj', None) is not None:
                    o = o * torch.sigmoid(at.gate_proj(a)).view(S, self.H, 1)
                o = at.o_proj(o.reshape(1, S, -1))
            if getattr(layer, 'use_sandwich_norm', False):
                o = layer.post_operator_norm(o)
            h = o + res
            f = layer.feed_forward(layer.ffn_norm(h))
            if getattr(layer, 'use_sandwich_norm', False):
                f = layer.post_ffn_norm(f)
            h = h + f
        h = self.lfm.embedding_norm(h)
        log_mag, angle = torch.chunk(self.dec.lin(h).mT.contiguous(), 2, 1)
        audio, env = self.dec.istft(torch.polar(log_mag.float().exp(), angle.float()))   # (1, L), (L,)
        tail = self.tail
        audio = torch.cat([audio[:, :tail] + self.ola_audio, audio[:, tail:]], dim=1)
        env = torch.cat([env[:tail] + self.ola_env, env[tail:]])
        self.ola_audio.copy_(audio[:, -tail:])
        self.ola_env.copy_(env[-tail:])
        self.pos += S
        return (audio[:, :-tail] / env[:-tail])[0]

    def reset(self):
        for s in self.conv_state.values():
            s.zero_()
        for b in list(self.k_buf.values()) + list(self.v_buf.values()):
            b.zero_()
        self.slot_pos.fill_(-1)
        self.pos.zero_()
        self.ola_audio.zero_()
        self.ola_env.zero_()
        self.started = False

    @torch.no_grad()
    def frame(self, codes):
        """codes: (12,) one frame -> its sound (float32 on the GPU): 1440 samples for an utterance's first frame,
        1920 after (the codec's overlap is held back until the next frame or flush())."""
        self.codes_in.copy_(torch.as_tensor(codes).view(1, -1, 1).to(self.device))
        if not self.use_graph:
            out = self._step()
        else:
            if self.graph is None:
                saved = self._save()
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(2):
                        self._step()
                torch.cuda.current_stream().wait_stream(s)
                self.graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph):
                    self.out = self._step()
                self._restore(saved)
            self.graph.replay()
            out = self.out
        if not self.started:
            self.started = True
            return out[self.trim:].clone()
        return out.clone()

    @torch.no_grad()
    def flush(self):
        """End of an utterance: the held-back overlap (480 samples); the state starts afresh."""
        if not self.started:
            self.reset()
            return torch.zeros(0, device=self.device)
        keep = self.tail - self.trim
        out = (self.ola_audio[0, :keep] / self.ola_env[:keep]).clone()
        self.reset()
        return out

    def _save(self):
        return ({i: v.clone() for i, v in self.conv_state.items()}, {i: v.clone() for i, v in self.k_buf.items()},
                {i: v.clone() for i, v in self.v_buf.items()}, self.slot_pos.clone(), self.pos.clone(),
                self.ola_audio.clone(), self.ola_env.clone())

    def _restore(self, saved):
        cs, kb, vb, sp, pos, oa, oe = saved
        for i, v in cs.items():
            self.conv_state[i].copy_(v)
        for i, v in kb.items():
            self.k_buf[i].copy_(v)
        for i, v in vb.items():
            self.v_buf[i].copy_(v)
        self.slot_pos.copy_(sp)
        self.pos.copy_(pos)
        self.ola_audio.copy_(oa)
        self.ola_env.copy_(oe)
