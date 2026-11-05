"""Can the turn model tell a turn is ending *before* the speaker goes quiet?

Uses the measured talkspurts of Dessa's demos (both tracks, both speakers). A spurt
end is a turn end if the other side takes the floor next and this speaker stays
quiet >1.5 s; it is a mid-turn pause if the same speaker carries on within 1.0 s.
Scores Smart Turn on audio cut at offsets around each spurt end.

  python -I predictive_turn.py <dessa_metrics.json> <audio_dir> <out.json>
"""
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import Config      # noqa: E402
from app.listen import SmartTurn   # noqa: E402

OFFSETS = [-0.3, -0.2, -0.1, 0.0, 0.032, 0.064, 0.1]
st = SmartTurn(Config().paths.smart_turn)
metrics = json.loads(Path(sys.argv[1]).read_text(encoding='utf8'))
rows = []
for m in metrics:
    if m['speaker1'] != 'Human':
        continue
    sp = [m['tracks'][0]['spurts'], m['tracks'][1]['spurts']]
    for k in (0, 1):
        x, fs = sf.read(Path(sys.argv[2]) / m['tracks'][k]['file'], dtype='float64')
        x = signal.resample_poly(x, 160, 441).astype(np.float32)
        mine, other = sp[k], sp[1 - k]
        for i, (a, b) in enumerate(mine):
            if b - a < 0.4:
                continue
            nxt = mine[i + 1][0] - b if i + 1 < len(mine) else 99.0
            other_next = next((oa for oa, ob in other if oa >= b - 0.5), None)
            if nxt > 1.5 and other_next is not None and other_next - b < 1.5:
                kind = 'end'
            elif nxt < 1.0 and (other_next is None or other_next > b + nxt):
                kind = 'mid'
            else:
                continue
            ps = {}
            for off in OFFSETS:
                if kind == 'mid' and off > 0 and off >= nxt:
                    continue
                end = int((b + off) * 16000)
                ps[str(off)] = st(x[max(0, end - 8 * 16000):end])
            rows.append({'kind': kind, 'gap': nxt, 'p': ps, 'speaker': 'Dessa' if k == 1 else 'person'})
Path(sys.argv[3]).write_text(json.dumps(rows), encoding='utf8')
ends = [r for r in rows if r['kind'] == 'end']
mids = [r for r in rows if r['kind'] == 'mid']
print(f'{len(ends)} turn ends, {len(mids)} mid-turn pauses')
for off in OFFSETS:
    pe = np.array([r['p'][str(off)] for r in ends if str(off) in r['p']])
    pm = np.array([r['p'][str(off)] for r in mids if str(off) in r['p']])
    line = f'cut {off:+.3f}s from stop: ends median {np.median(pe):.3f} share>0.5 {np.mean(pe > 0.5):.2f} | mids median {np.median(pm):.3f} share>0.5 {np.mean(pm > 0.5):.2f}'
    for thr in (0.7, 0.9):
        line += f' | >{thr}: ends {np.mean(pe > thr):.2f} mids {np.mean(pm > thr):.2f}'
    print(line)
