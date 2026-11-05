"""Milestone 3: re-train the learned turn-taking model (VAP, app/vap_torch.py) on real two-person conversations.

Data: Seamless Interaction train conversations (tools/fetch_seamless.py --split train; both people are real,
so it learns human timing, not the old Alba's). Each conversation is used both ways round.
Objective (VAP's own): at every 20 ms frame, predict which of 4 future time bins (0.2/0.4/0.6/0.8 s) each person
will be talking in: one of 256 joint futures (class bit 4*speaker+bin), plus "is each person talking now".
The sound encoder (CPC convolutions + LSTM + downsampling) stays as it is; its output is computed once per
conversation and cached, so training touches only the transformer layers and heads (cheap, runs anywhere).

Checked on the held-out timing set (data/tests/timing: Seamless test conversations and your 3 newest calls,
never trained on): how well "the other person talks next" separates turn ends from pauses (AUC) at 0.1-0.5 s
of silence, before and after.

  python -I tools/train_vap.py --data data/seamless/train --epochs 4 [--device cuda]
Writes models/vap/vap_ft.pt (state dict) and results/vap_ft.json.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.vap_torch import load_from_onnx  # noqa: E402

FPS = 50
BINS = (10, 20, 30, 40)          # frames: 0.2, 0.4, 0.6, 0.8 s
HORIZON = sum(BINS)


def vad_frames(segs, n):
    v = np.zeros(n, np.float32)
    for s, e in segs:
        v[int(s * FPS):int(e * FPS)] = 1
    return v


def targets(va, vb):
    """Class index per frame from both people's future activity (VAP's bit layout)."""
    n = len(va)
    cls = np.zeros(n - HORIZON, np.int64)
    edges = np.cumsum((0,) + BINS)
    for spk, v in enumerate((va, vb)):
        c = np.concatenate([[0], np.cumsum(v)])
        for b in range(4):
            lo, hi = edges[b] + 1, edges[b + 1] + 1         # frames t+1 .. t+bin_end
            t = np.arange(n - HORIZON)
            frac = (c[t + hi] - c[t + lo]) / (hi - lo)
            cls += (frac >= 0.5).astype(np.int64) << (4 * spk + b)
    return cls


@torch.no_grad()
def encode(model, wav, device, chunk_s=60):
    """(T50, 256) sound features. Long recordings go through the convolutions a minute at a time (with 0.1 s
    of overlap that is thrown away; the result is the same, and memory stays small), then the LSTM and the
    last step over the whole thing."""
    enc = model.encoder
    if len(wav) <= 2 * chunk_s * 16000:
        return enc(torch.from_numpy(wav).float()[None].to(device))[0].cpu()
    n = len(wav) - len(wav) % 160
    step, pad = chunk_s * 16000, 1600
    parts = []
    for a in range(0, n, step):
        b = min(n, a + step)
        lo, hi = max(0, a - pad), min(n, b + pad)
        f = enc.frames100(torch.from_numpy(wav[lo:hi]).float()[None].to(device))
        off = (a - lo) // 160
        parts.append(f[:, :, off:off + (b - a) // 160])
    z, _ = enc.lstm(torch.cat(parts, -1).transpose(1, 2))
    return enc.downsample(z)[0].cpu()


def load_conv(src, pa, pb):
    def segs(fid):
        return [(json.loads(l)['start'], json.loads(l)['end']) for l in open(src / 'meta' / f'{fid}.vad.jsonl', encoding='utf8') if l.strip()]
    xa, _ = sf.read(src / 'audio' / f'{pa}.flac', dtype='float32')
    xb, _ = sf.read(src / 'audio' / f'{pb}.flac', dtype='float32')
    n = min(len(xa), len(xb))
    return xa[:n], xb[:n], segs(pa), segs(pb)


def forward_feats(model, e1, e2):
    from app.vap_torch import alibi
    bias = alibi(e1.shape[1], e1.shape[1], device=e1.device)
    x1, x2 = model.ar_channel(e1, bias), model.ar_channel(e2, bias)
    for layer in model.ar:
        x1, x2 = layer(x1, x2, bias)
    return model.heads(x1, x2)


_EVAL_FEATS = {}


def evaluate(model, device, context_s=6.0):
    """AUC of p(other talks next) at shifts vs holds, from the frozen timing test set. Each moment is judged
    from a window of the last context_s seconds (as the app sees it); the frozen encoder's output is computed
    once per track and kept."""
    items = json.loads((ROOT / 'data' / 'tests' / 'timing' / 'items.json').read_text(encoding='utf8'))['items']
    rows = {0.1: ([], []), 0.2: ([], []), 0.3: ([], []), 0.5: ([], [])}
    ctx = int(context_s * FPS)
    model.eval()
    for it in items:
        if it['id'] not in _EVAL_FEATS:
            xa, _ = sf.read(ROOT / 'data' / 'tests' / 'timing' / it['caller'], dtype='float32')
            if it.get('partner'):
                xb, _ = sf.read(ROOT / 'data' / 'tests' / 'timing' / it['partner'], dtype='float32')
            else:
                xb, _ = sf.read(ROOT / 'sessions' / it['id'].replace('call-', '') / 'voice16k.flac', dtype='float32')
            n = min(len(xa), len(xb))
            _EVAL_FEATS[it['id']] = (encode(model, xa[:n], device), encode(model, xb[:n], device))
        ea, eb = _EVAL_FEATS[it['id']]
        moments = []
        for off in rows:
            for sh in it['shifts']:
                t = sh['a_end'] + off
                if t < sh['partner_start'] - 0.02:
                    moments.append((off, 0, int(t * FPS)))
            for h in it['holds']:
                t = h['a_end'] + off
                if t < h['a_resume'] - 0.02:
                    moments.append((off, 1, int(t * FPS)))
        moments = [m for m in moments if ctx <= m[2] <= len(ea)]
        for k in range(0, len(moments), 64):
            chunk = moments[k:k + 64]
            e1 = torch.stack([ea[i - ctx:i] for _, _, i in chunk]).to(device)
            e2 = torch.stack([eb[i - ctx:i] for _, _, i in chunk]).to(device)
            with torch.no_grad():
                pf = model.readout(*forward_feats(model, e1, e2))['p_future'][:, -1, 1].cpu().numpy()
            for (off, kind, _), v in zip(chunk, pf):
                rows[off][kind].append(float(v))

    def auc(p, q):
        p, q = np.asarray(p), np.asarray(q)
        return float((p[:, None] > q[None, :]).mean() + 0.5 * (p[:, None] == q[None, :]).mean()) if len(p) and len(q) else float('nan')
    return {str(o): {'auc': round(auc(*rows[o]), 3), 'shifts': len(rows[o][0]), 'holds': len(rows[o][1])} for o in rows}


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=str(ROOT / 'data' / 'seamless' / 'train'))
    ap.add_argument('--epochs', type=int, default=4)
    ap.add_argument('--lr', type=float, default=3e-5)
    ap.add_argument('--window-s', type=float, default=20.0)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    torch.manual_seed(0)
    dev = args.device
    model = load_from_onnx(ROOT / 'models' / 'vap' / 'vap.onnx').to(dev)
    before = evaluate(model, dev)
    print('held-out AUC before:', before, flush=True)
    src = Path(args.data)
    pairs = json.loads((src / 'pairs.json').read_text(encoding='utf8'))
    cache = src / '_vap_feats'
    cache.mkdir(exist_ok=True)
    data = []
    t0 = time.time()
    for pa, pb in pairs:
        p = cache / f'{pa}.pt'
        if p.exists():
            d = torch.load(p, weights_only=True)
        else:
            xa, xb, sa, sb = load_conv(src, pa, pb)
            d = {'ea': encode(model, xa, dev), 'eb': encode(model, xb, dev)}
            n = min(len(d['ea']), len(d['eb']))
            d['va'] = torch.from_numpy(vad_frames(sa, n)); d['vb'] = torch.from_numpy(vad_frames(sb, n))
            torch.save(d, p)
        data.append(d)
    print(f'{len(data)} conversations encoded in {time.time() - t0:.0f}s', flush=True)
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    win = int(args.window_s * FPS)
    rng = random.Random(0)
    model.train()
    for ep in range(args.epochs):
        losses = []
        order = [(i, swap) for i in range(len(data)) for swap in (0, 1)]
        rng.shuffle(order)
        for i, swap in order:
            d = data[i]
            ea, eb, va, vb = (d['eb'], d['ea'], d['vb'], d['va']) if swap else (d['ea'], d['eb'], d['va'], d['vb'])
            n = min(len(ea), len(va))
            for s in range(0, max(1, n - win - HORIZON), win):
                e1 = ea[s:s + win + HORIZON][None].to(dev); e2 = eb[s:s + win + HORIZON][None].to(dev)
                a = va[s:s + win + HORIZON].numpy(); b = vb[s:s + win + HORIZON].numpy()
                if len(a) <= HORIZON + 10:
                    continue
                cls = torch.from_numpy(targets(a, b)).to(dev)
                logits, vad_logits = forward_feats(model, e1, e2)
                m = len(cls)
                loss = F.cross_entropy(logits[0, :m], cls) + F.binary_cross_entropy_with_logits(
                    vad_logits[0, :len(a)], torch.from_numpy(np.stack([a, b], -1)).to(dev))
                opt.zero_grad(); loss.backward(); opt.step()
                losses.append(float(loss))
        res = evaluate(model, dev)
        model.train()
        print(f'epoch {ep + 1}: loss {np.mean(losses):.3f} | held-out AUC {res}', flush=True)
    after = evaluate(model, dev)
    torch.save(model.state_dict(), ROOT / 'models' / 'vap' / 'vap_ft.pt')
    (ROOT / 'results' / 'vap_ft.json').write_text(json.dumps({'before': before, 'after': after, 'epochs': args.epochs,
                                                              'conversations': len(data)}, indent=1), encoding='utf8')
    print('before', before, '\nafter ', after, flush=True)


if __name__ == '__main__':
    main()
