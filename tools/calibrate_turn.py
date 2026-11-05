"""Calibrates the "is the person done?" rule on real conversation audio.

For each caption boundary in Dessa's captioned demos, the Smart Turn score is computed
after 0.1-1.0 s of silence. Turn ends (the other side took over) are compared with
mid-sentence pauses (the same person carried on). Prints, for each silence length,
the score threshold that still lets through at most a given share of mid-sentence
pauses, and how many real turn ends it catches.
  python -I calibrate_turn.py <recordings.json> <audio_dir> <out.json>
"""
import json, sys
from pathlib import Path
import numpy as np, soundfile as sf
from scipy import signal
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.listen import SmartTurn
from app.config import Config

PAUSES = [0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0]
st = SmartTurn(Config().paths.smart_turn)
recs = json.loads(Path(sys.argv[1]).read_text(encoding='utf8'))['recordings']
rows = []
for r in recs:
    if not r['captions'] or not all((Path(sys.argv[2]) / f).exists() for f in r['files']):
        continue
    caps = sorted(r['captions'], key=lambda c: c['start'])
    for spk, track in [(1, 0), (2, 1)]:
        x, fs = sf.read(Path(sys.argv[2]) / r['files'][track], dtype='float64')
        x = signal.resample_poly(x, 160, 441).astype(np.float32)
        mine = [c for c in caps if c['speaker'] == spk]
        for k, c in enumerate(mine):
            nxt_mine = mine[k + 1] if k + 1 < len(mine) else None
            other_next = next((d for d in caps if d['speaker'] != spk and d['start'] >= c['end'] - 0.3), None)
            gap_mine = (nxt_mine['start'] - c['end']) if nxt_mine else 99.0
            if gap_mine > 1.5 and other_next is not None and other_next['start'] - c['end'] < 1.5:
                kind = 'end'
            elif gap_mine < 1.2 and (other_next is None or other_next['start'] > c['end'] + gap_mine):
                kind = 'mid'
            else:
                continue
            ps = {}
            for pz in PAUSES:
                if kind == 'mid' and pz >= gap_mine:
                    break
                b = int((c['end'] + pz) * 16000)
                ps[pz] = st(x[max(0, b - 8 * 16000):b])
            rows.append({'rec': r['id'], 'speaker': spk, 'kind': kind, 'gap': gap_mine, 'p': ps})
ends = [w for w in rows if w['kind'] == 'end']; mids = [w for w in rows if w['kind'] == 'mid']
print(f"{len(ends)} turn ends, {len(mids)} mid-turn pauses")
out = {}
for pz in PAUSES:
    pe = np.array([w['p'][pz] for w in ends if pz in w['p']]); pm = np.array([w['p'][pz] for w in mids if pz in w['p']])
    line = f"after {pz:.2f}s silence: ends median {np.median(pe):.3f} | mids still silent n={len(pm):3d} median {np.median(pm) if len(pm) else float('nan'):.3f}"
    for fp in (0.05, 0.10):
        thr = float(np.quantile(pm, 1 - fp)) if len(pm) >= 10 else None
        if thr is not None:
            line += f" | thr@{int(fp*100)}%FP={thr:.3f} catches {np.mean(pe >= thr)*100:.0f}% of ends"
    print(line)
    out[str(pz)] = {'ends': pe.tolist(), 'mids': pm.tolist()}
Path(sys.argv[3]).write_text(json.dumps(out), encoding='utf8')

# ---- policy search: respond at the first check where score >= threshold, else at 1.0 s ----
Path(sys.argv[3]).with_name('turn_calibration_rows.json').write_text(json.dumps(rows), encoding='utf8')
import itertools
checks = [0.1, 0.2, 0.3, 0.45, 0.6, 0.8]
grid = [0.9, 0.7, 0.5, 0.35, 0.25, 0.15, 0.1, 0.06, 0.04, 0.0]
def evaluate(th):
    delays, cut = [], 0
    for w in ends:
        d = next((s for s, t in zip(checks, th) if s in w['p'] and w['p'][s] >= t), 1.0)
        delays.append(d)
    for w in mids:
        if any(s in w['p'] and w['p'][s] >= t for s, t in zip(checks, th)):
            cut += 1
    return float(np.median(delays)), float(np.mean(delays)), cut / len(mids)
best = {}
for th in itertools.product(grid, repeat=len(checks)):
    if any(th[i] < th[i + 1] for i in range(len(th) - 1)):
        continue   # thresholds only fall as the silence grows
    med, mean, fp = evaluate(th)
    for cap in (0.02, 0.04, 0.06, 0.10):
        if fp <= cap and (cap not in best or mean < best[cap][1]):
            best[cap] = (med, mean, fp, th)
for cap, (med, mean, fp, th) in sorted(best.items()):
    print(f"cut-ins <= {cap*100:.0f}% of mid pauses: median decision {med:.2f}s, mean {mean:.2f}s, actual {fp*100:.1f}% | thresholds at {checks}: {th}")
