"""Turn models on the user's own recorded calls: the real voice, through the laptop microphone, talking to Alba.

The simulated caller in tools/benchmark.py is a synthetic voice, and the learned model (VAP) cannot read
turn ends in synthetic speech at all (AUC about 0.5 on those recordings, against 0.87-0.97 on the people in
Dessa's demos). These calls are the real thing, so the timing is tuned and checked here.

For every moment the person stops talking (Silero VAD on person_clean16k.flac):
  'end'   Alba replied before they spoke again (and they did not talk over the reply's start),
  'mid'   they went on within 1 s, before Alba replied.
At each pause length both models are asked "are they done?" with only the audio up to that moment:
  VAP        both voices, the last 6 s  (models/vap/vap_dyn.onnx)
  SmartTurn  the person's voice alone, the last 8 s
Then the conductor's own rule (answer at the first check whose bar the score clears) is replayed with the
bars in app/config.py, and with bars fitted to these calls, to show how fast it would answer and how often
it would cut in at a pause.

Only the user's calls on this computer are read; nothing is uploaded.

  python -I tools/eval_turns_calls.py [--sessions DIR] [--threads 4]
Writes results/turns_on_calls.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.config import Config  # noqa: E402
from app.listen import SileroVAD, SmartTurn, VAPTurn  # noqa: E402

SR = 16000
OFFSETS = (0.06, 0.10, 0.15, 0.20, 0.30, 0.45, 0.60, 0.80)


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return float('nan')
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def load16(path):
    x, fs = sf.read(path, dtype='float32')
    if x.ndim > 1:
        x = x.mean(axis=1)
    if fs != SR:
        from scipy import signal
        g = np.gcd(SR, fs)
        x = signal.resample_poly(x, SR // g, fs // g).astype(np.float32)
    return x


def person_segments(x, vad, on=0.5, off=0.35, min_gap=0.06, min_len=0.12):
    """Speech stretches from Silero VAD (32 ms frames, with hysteresis)."""
    vad.reset()
    f = SileroVAD.FRAME
    probs = np.array([vad(x[i:i + f]) for i in range(0, len(x) - f + 1, f)])
    speech, segs, start = False, [], 0
    for k, p in enumerate(probs):
        if not speech and p >= on:
            speech, start = True, k
        elif speech and p < off:
            speech = False
            segs.append([start * f / SR, k * f / SR])
    if speech:
        segs.append([start * f / SR, len(probs) * f / SR])
    merged = []
    for s in segs:
        if merged and s[0] - merged[-1][1] < min_gap:
            merged[-1][1] = s[1]
        else:
            merged.append(s)
    return [s for s in merged if s[1] - s[0] >= min_len]


def agent_segments(y, floor_db=-45, min_gap=0.25):
    """Alba's own output is clean, so plain energy finds it (10 ms frames)."""
    h = SR // 100
    n = len(y) // h
    e = 10 * np.log10(np.mean(y[:n * h].reshape(n, h) ** 2, axis=1) + 1e-12)
    on = e > floor_db
    segs, start = [], None
    for k, v in enumerate(on):
        if v and start is None:
            start = k
        elif not v and start is not None:
            segs.append([start / 100, k / 100]); start = None
    if start is not None:
        segs.append([start / 100, n / 100])
    merged = []
    for s in segs:
        if merged and s[0] - merged[-1][1] < min_gap:
            merged[-1][1] = s[1]
        else:
            merged.append(s)
    return merged


def moments(psegs, asegs):
    """(kind, stop, limit): limit = when someone spoke next (no peeking past it)."""
    replies = [s for s in asegs if s[1] - s[0] >= 0.6]           # not "mhm" listening sounds
    out = []
    for k, (a, b) in enumerate(psegs):
        resume = psegs[k + 1][0] if k + 1 < len(psegs) else 1e9
        rep = next((s for s in replies if s[0] >= b - 0.05), None)
        rep_t = rep[0] if rep else 1e9
        # Alba already talking when they stopped (they were talking over her): not a clean moment
        if any(s[0] < b - 0.05 < s[1] for s in asegs):
            continue
        if rep_t < resume and rep_t - b < 4.0:
            if resume - rep_t < 0.5:
                continue                     # they talked over her start: may have been cut off mid-turn
            out.append(('end', b, rep_t))
        elif resume < rep_t and resume - b <= 1.0:
            out.append(('mid', b, resume))
    return out


def fit_bars(rows, offsets, key, target_false=0.10):
    """Lowest bar per check (in schedule order) such that at most target_false of the pauses get cut into.
    Greedy: set each step's bar so it adds at most an even share of the false cut-ins."""
    mids = [r for r in rows if r['kind'] == 'mid']
    share = target_false / len(offsets)
    bars, cut = [], set()
    for k, off in enumerate(offsets):
        vals = sorted([r[key][k] for i, r in enumerate(mids) if r[key][k] is not None and i not in cut], reverse=True)
        allowed = int(share * len(mids) * (k + 1)) - len(cut)
        bar = vals[allowed] + 1e-6 if 0 <= allowed < len(vals) else (0.0 if allowed >= len(vals) else 1.0)
        bar = float(min(0.95, max(0.05, bar)))
        bars.append(round(bar, 3))
        for i, r in enumerate(mids):
            if r[key][k] is not None and r[key][k] >= bar:
                cut.add(i)
    return bars


def replay(rows, offsets, key, bars, give_up=1.0, lat=0.0):
    """The conductor's rule: answer at the first check that clears its bar (plus the check's own compute time),
    else at give_up seconds."""
    delays, cut, n_mid = [], 0, 0
    for r in rows:
        hit = next((off for off, bar, v in zip(offsets, bars, r[key]) if v is not None and v >= bar), None)
        if r['kind'] == 'end':
            delays.append(min(give_up, hit + lat) if hit is not None else give_up)
        else:
            n_mid += 1
            cut += hit is not None
    return {'answer_delay_median_s': float(np.median(delays)) if delays else float('nan'),
            'answer_delay_p90_s': float(np.percentile(delays, 90)) if delays else float('nan'),
            'answered_by_model': float(np.mean([d < give_up - 1e-9 for d in delays])) if delays else float('nan'),
            'pauses_cut_into': cut / n_mid if n_mid else float('nan')}


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser()
    ap.add_argument('--sessions', default=str(ROOT / 'sessions'))
    ap.add_argument('--threads', type=int, default=4)
    args = ap.parse_args()
    cfg = Config()
    vad = SileroVAD(cfg.paths.vad)
    vap = VAPTurn(cfg.paths.vap, threads=args.threads)
    st = SmartTurn(cfg.paths.smart_turn, threads=args.threads)
    rows = []
    t_vap, t_st = [], []
    for d in sorted(Path(args.sessions).iterdir()):
        if not (d / 'person_clean16k.flac').exists() or not (d / 'voice16k.flac').exists():
            continue
        p, a = load16(d / 'person_clean16k.flac'), load16(d / 'voice16k.flac')
        n = min(len(p), len(a)); p, a = p[:n], a[:n]
        ps, as_ = person_segments(p, vad), agent_segments(a)
        ms = moments(ps, as_)
        for kind, stop, limit in ms:
            r = {'session': d.name, 'kind': kind, 'stop': round(stop, 3), 'next': round(limit, 3), 'vap': [], 'smart_turn': []}
            for off in OFFSETS:
                t = stop + off
                if t >= limit - 0.02:
                    r['vap'].append(None); r['smart_turn'].append(None)
                    continue
                i = int(t * SR)
                t0 = time.perf_counter()
                r['vap'].append(round(vap(p[max(0, i - vap.WINDOW):i], a[max(0, i - vap.WINDOW):i])[0], 4))
                t_vap.append(time.perf_counter() - t0)
                t0 = time.perf_counter()
                r['smart_turn'].append(round(st(p[max(0, i - st.WINDOW):i]), 4))
                t_st.append(time.perf_counter() - t0)
            rows.append(r)
        print(f"{d.name}: {sum(k == 'end' for k, _, _ in ms)} turn ends, {sum(k == 'mid' for k, _, _ in ms)} pauses", flush=True)
    out = {'offsets': OFFSETS, 'vap_ms_median': 1000 * float(np.median(t_vap)), 'smart_turn_ms_median': 1000 * float(np.median(t_st)),
           'by_offset': {}, 'policies': {}}
    lv, ls = out['vap_ms_median'] / 1000, out['smart_turn_ms_median'] / 1000
    print(f"\n{sum(r['kind'] == 'end' for r in rows)} turn ends, {sum(r['kind'] == 'mid' for r in rows)} pauses; "
          f"VAP {out['vap_ms_median']:.0f} ms per check, SmartTurn {out['smart_turn_ms_median']:.0f} ms")
    print(f"{'silence':>8} {'ends':>5} {'pauses':>6} | AUC SmartTurn  VAP | VAP median: end  pause")
    for k, off in enumerate(OFFSETS):
        e = [r for r in rows if r['kind'] == 'end' and r['vap'][k] is not None]
        m = [r for r in rows if r['kind'] == 'mid' and r['vap'][k] is not None]
        res = {'ends': len(e), 'pauses': len(m),
               'auc_smart_turn': auc([r['smart_turn'][k] for r in e], [r['smart_turn'][k] for r in m]),
               'auc_vap': auc([r['vap'][k] for r in e], [r['vap'][k] for r in m]),
               'vap_median_end': float(np.median([r['vap'][k] for r in e])) if e else float('nan'),
               'vap_median_pause': float(np.median([r['vap'][k] for r in m])) if m else float('nan')}
        out['by_offset'][str(off)] = res
        print(f"{off:7.2f}s {len(e):5d} {len(m):6d} | {res['auc_smart_turn']:13.2f} {res['auc_vap']:4.2f} | "
              f"{res['vap_median_end']:15.2f} {res['vap_median_pause']:6.2f}")
    T = cfg.timing
    sched_off = [o for o, _ in T.vap_schedule]
    idx = [OFFSETS.index(o) if o in OFFSETS else int(np.argmin(np.abs(np.array(OFFSETS) - o))) for o in sched_off]

    def sub(key):
        return [{**r, key: [r[key][i] for i in idx]} for r in rows]
    vap_rows, st_rows = sub('vap'), sub('smart_turn')
    pol = {
        'VAP, bars in config': replay(vap_rows, sched_off, 'vap', [b for _, b in T.vap_schedule], lat=lv),
        'SmartTurn, its bars in config': replay([{**r, 'smart_turn': [r['smart_turn'][OFFSETS.index(min(OFFSETS, key=lambda x: abs(x - o)))] for o, _ in T.turn_schedule]} for r in rows],
                                                [o for o, _ in T.turn_schedule], 'smart_turn', [b for _, b in T.turn_schedule], lat=ls),
    }
    for target in (0.05, 0.10, 0.15):
        bars = fit_bars(rows, OFFSETS, 'vap', target)
        pol[f'VAP, bars fitted for {target:.0%} cut-ins'] = {**replay(rows, OFFSETS, 'vap', bars, lat=lv), 'bars': bars}
        bars_st = fit_bars(rows, OFFSETS, 'smart_turn', target)
        pol[f'SmartTurn, bars fitted for {target:.0%} cut-ins'] = {**replay(rows, OFFSETS, 'smart_turn', bars_st, lat=ls), 'bars': bars_st}
    out['policies'] = pol
    print(f"\n{'rule':42s} | answer median  p90 | by model | pauses cut into")
    for name, r in pol.items():
        print(f"{name:42s} | {r['answer_delay_median_s']:12.2f}s {r['answer_delay_p90_s']:4.2f}s | {r['answered_by_model']:8.0%} | "
              f"{r['pauses_cut_into']:6.0%}  {r.get('bars', '')}")
    (ROOT / 'results' / 'turns_on_calls.json').write_text(json.dumps({**out, 'rows': rows}, indent=1), encoding='utf8')


if __name__ == '__main__':
    main()
