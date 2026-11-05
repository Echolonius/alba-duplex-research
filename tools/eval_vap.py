"""Learned turn-taking vs today's turn model, on the same moments of Dessa's captioned demos.

For each caption end in Dessa's demos (the same events calibrate_turn.py uses: 'end' = the other side took
over, 'mid' = the same speaker carried on), after a given stretch of silence it asks:
  Smart Turn  (today): "has this speaker finished?" from that speaker's audio alone
  VAP otoSpeech      : "who will be talking in 0.2-0.8 s?" from BOTH voices (models/vap, CC BY 4.0)
and reports how well each tells turn ends from pauses (AUC: 0.5 = coin flip, 1.0 = perfect), plus the
share of turn ends each catches while letting through only 5% / 10% of pauses.

Only moments before anyone speaks again are scored, so neither model sees the answer. Dessa's audio is
used here only to measure, as everywhere in this project.

  python -I tools/eval_vap.py [--offsets 0.1,0.2,0.3,0.5] [--threads 4]
Writes results/vap_vs_smartturn.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
from scipy import signal

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.config import Config  # noqa: E402
from app.listen import SmartTurn  # noqa: E402

SR = 16000
WIN = 320000          # VAP's fixed 20 s window


def auc(pos, neg):
    """Probability that a random turn end scores above a random pause (ties count half)."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    if not len(pos) or not len(neg):
        return float('nan')
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return float(gt + 0.5 * eq)


def catch_at(pos, neg, let_through):
    th = np.quantile(neg, 1 - let_through) if len(neg) else 1.0
    return float(np.mean(np.asarray(pos) > th)) if len(pos) else float('nan')


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser()
    ap.add_argument('--offsets', default='0.1,0.2,0.3,0.5')
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--window', type=float, default=20.0, help='seconds of both voices VAP sees (20 = its full window)')
    args = ap.parse_args()
    offsets = [float(x) for x in args.offsets.split(',')]
    so = ort.SessionOptions()
    so.intra_op_num_threads = args.threads
    name = 'vap.onnx' if args.window >= 20 else 'vap_dyn.onnx'      # vap_dyn: same weights, any window length
    vap = ort.InferenceSession(str(ROOT / 'models' / 'vap' / name), so, providers=['CPUExecutionProvider'])
    win = int(args.window * SR)
    st = SmartTurn(Config().paths.smart_turn)
    recs = json.loads((ROOT / 'reference' / 'dessa-page' / 'recordings.json').read_text(encoding='utf8'))['recordings']
    audio_dir = ROOT / 'reference' / 'dessa-audio'
    rows, t_vap = [], []
    for r in recs:
        if not r['captions'] or not all((audio_dir / f).exists() for f in r['files']):
            continue
        caps = sorted(r['captions'], key=lambda c: c['start'])
        tracks = []
        for f in r['files'][:2]:
            x, fs = sf.read(audio_dir / f, dtype='float64')
            if x.ndim > 1:
                x = x.mean(axis=1)
            g = np.gcd(SR, fs)
            tracks.append(signal.resample_poly(x, SR // g, fs // g).astype(np.float32))
        n = min(len(t) for t in tracks)
        tracks = [t[:n] for t in tracks]
        for spk, track in [(1, 0), (2, 1)]:
            me, other = tracks[track], tracks[1 - track]
            mine = [c for c in caps if c['speaker'] == spk]
            for k, c in enumerate(mine):
                nxt_mine = mine[k + 1] if k + 1 < len(mine) else None
                other_next = next((d for d in caps if d['speaker'] != spk and d['start'] >= c['end'] - 0.3), None)
                gap_mine = (nxt_mine['start'] - c['end']) if nxt_mine else 99.0
                if gap_mine > 1.5 and other_next is not None and other_next['start'] - c['end'] < 1.5:
                    kind, next_t = 'end', other_next['start']
                elif gap_mine < 1.2 and (other_next is None or other_next['start'] > c['end'] + gap_mine):
                    kind, next_t = 'mid', nxt_mine['start']
                else:
                    continue
                for off in offsets:
                    t = c['end'] + off
                    if t >= next_t - 0.02:
                        continue                      # someone is already talking again: no peeking
                    i = int(t * SR)
                    if i > n:
                        continue
                    # Smart Turn: this speaker's own audio up to t (its usual 8 s view)
                    p_st = float(st(me[max(0, i - 8 * SR):i]))
                    # VAP: both voices, the last 20 s, "user" = this speaker, "agent" = the other
                    w = np.zeros((1, 2, win), np.float32)
                    a, b = max(0, i - win), i
                    w[0, 0, win - (b - a):] = me[a:b]
                    w[0, 1, win - (b - a):] = other[a:b]
                    t0 = time.perf_counter()
                    p_now, p_fut, vad = vap.run(None, {'audio': w})
                    t_vap.append(time.perf_counter() - t0)
                    rows.append({'rec': r['id'], 'speaker': spk, 'kind': kind, 'offset': off,
                                 'smart_turn': p_st, 'vap_now': float(p_now[0, -1, 1]),
                                 'vap_future': float(p_fut[0, -1, 1])})
        print(f"{r['id']}: {len(rows)} moments so far", flush=True)
    out = {'vap_seconds_per_window_median': float(np.median(t_vap)), 'by_offset': {}}
    print(f'\nVAP: {np.median(t_vap) * 1000:.0f} ms per {args.window:g} s window on this processor ({args.threads} threads)')
    print(f"{'silence':>8} {'ends':>5} {'pauses':>6} | {'AUC':^23} | turn ends caught, 10% of pauses let through")
    print(f"{'':>8} {'':>5} {'':>6} | SmartTurn VAPnow VAPfut | SmartTurn  VAPnow  VAPfut")
    for off in offsets:
        e = [w for w in rows if w['offset'] == off and w['kind'] == 'end']
        m = [w for w in rows if w['offset'] == off and w['kind'] == 'mid']
        res = {}
        for key in ('smart_turn', 'vap_now', 'vap_future'):
            pos, neg = [w[key] for w in e], [w[key] for w in m]
            res[key] = {'auc': auc(pos, neg), 'catch_5': catch_at(pos, neg, 0.05), 'catch_10': catch_at(pos, neg, 0.10)}
        out['by_offset'][str(off)] = {'ends': len(e), 'pauses': len(m), **res}
        print(f"{off:7.1f}s {len(e):5d} {len(m):6d} | {res['smart_turn']['auc']:9.2f} {res['vap_now']['auc']:6.2f} "
              f"{res['vap_future']['auc']:6.2f} | {res['smart_turn']['catch_10']:8.0%} {res['vap_now']['catch_10']:7.0%} "
              f"{res['vap_future']['catch_10']:7.0%}")
    (ROOT / 'results' / f'vap_vs_smartturn_{args.window:g}s.json').write_text(json.dumps({**out, 'rows': rows}, indent=1), encoding='utf8')


if __name__ == '__main__':
    main()
