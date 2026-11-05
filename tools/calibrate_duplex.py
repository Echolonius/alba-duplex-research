"""Calibrates Alba Duplex's one runtime knob, the LISTEN bias (how strongly "keep listening" is favoured over
"start talking"), on a VALIDATION set: the Seamless test-split conversations that are NOT in the frozen timing
test and were never trained on. The frozen test is never used here.

The rule, fixed before any result was seen (5 Nov 2026, 05:30): among the candidate biases, pick the one that
answers the most hand-offs while cutting into fewer than 8% of thinking pauses and starting replies within
0.35 s median; if none meets both, the one with the fewest cut-ins. The theory value is log(1/LISTEN_KEEP) =
1.61 (it undoes the training's down-weighting of LISTEN exactly); the grid brackets it.

  .venv-train/Scripts/python.exe -I tools/calibrate_duplex.py --model models/alba-duplex [--n 8]
Writes results/calibration_<model>.json; then use --listen-bias with run_timing_duplex / duplex.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import signal

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
from build_testsets import classify, events_for, load16, merge  # noqa: E402
from run_timing_duplex import model_settings, run_track  # noqa: E402
from score_timing import energy_segments, score_item, summarize  # noqa: E402
from app.duplex import DELAY, AlbaDuplex  # noqa: E402


def validation_items(n):
    src = ROOT / 'data' / 'seamless' / 'test'
    pairs = json.loads((src / 'pairs.json').read_text(encoding='utf8'))
    manifest = json.loads((src / 'manifest.json').read_text(encoding='utf8'))['files']
    frozen = json.loads((ROOT / 'data' / 'tests' / 'timing' / 'items.json').read_text(encoding='utf8'))['items']
    used = {it['id'].replace('seamless-', '') for it in frozen}
    # (your own test calls have no partner file; a pair is skipped if either side is in the frozen test)
    used |= {Path(it['partner']).name.replace('-partner.flac', '').replace('seamless-', '') for it in frozen
             if it.get('partner')}

    def seg(fid):
        vad = [json.loads(l) for l in open(src / 'meta' / f'{fid}.vad.jsonl', encoding='utf8') if l.strip()]
        tr = src / 'meta' / f'{fid}.transcript.jsonl'
        words = [w for l in open(tr, encoding='utf8') if l.strip() for w in json.loads(l).get('words', [])
                 if isinstance(w.get('start'), (int, float)) and isinstance(w.get('end'), (int, float))] if tr.exists() else []
        return classify(merge([(v['start'], v['end']) for v in vad]), words)

    cands = []
    for pa, pb in pairs:
        if pa in used or pb in used:
            continue
        secs = min(manifest[pa]['seconds'], manifest[pb]['seconds'])
        if secs < 100:
            continue
        sa, sb = seg(pa), seg(pb)
        for ca, cb, a, b in ((pa, pb, sa, sb), (pb, pa, sb, sa)):
            ev = events_for(a, b, secs)
            if len(ev['shifts']) >= 4 and len(ev['holds']) >= 4:
                cands.append((len(ev['shifts']) + 0.5 * len(ev['holds']), ca, cb, a, b))
    chosen, seen = [], set()
    for c in sorted(cands, key=lambda c: -c[0]):
        key = tuple(sorted(c[1:3]))
        if key not in seen:
            chosen.append(c)
            seen.add(key)
        if len(chosen) == n:
            break
    items = []
    for _, pa, pb, a, b in chosen:
        xa, xb = load16(src / 'audio' / f'{pa}.flac'), load16(src / 'audio' / f'{pb}.flac')
        m = min(len(xa), len(xb))
        items.append(({'id': f'val-{pa}', 'seconds': m / 16000, 'caller_segments': a, 'partner_segments': b,
                       **events_for(a, b, m / 16000)}, xa[:m]))
    return items


def choose(table):
    ok = [r for r in table if r['cut_in_share'] < 0.08 and (r['reply_start_median_s'] or 9) <= 0.35]
    if ok:
        return max(ok, key=lambda r: r['answered_share'])
    return min(table, key=lambda r: r['cut_in_share'])


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--n', type=int, default=8)
    ap.add_argument('--grid', default='0,0.8,1.61,2.4')
    args = ap.parse_args()
    md = ROOT / args.model
    st = model_settings(md)
    delay = int(st.get('delay_steps', DELAY))
    items = validation_items(args.n)
    print(f'{len(items)} validation conversations ({sum(it["seconds"] for it, _ in items) / 60:.0f} min), none in the '
          f'frozen test', flush=True)
    alba = AlbaDuplex(md / 'model-Q4_K_M.gguf', md / 'hf')
    table = []
    for b in [float(v) for v in args.grid.split(',')]:
        rows = []
        for it, x in items:
            voice24, _, _ = run_track(alba, x, delay, b)
            v16 = signal.resample_poly(voice24, 2, 3).astype(np.float32)
            rows.append(score_item(it, energy_segments(v16, 16000)))
        s = summarize(rows)
        r = {'listen_bias': b, **{k: s.get(k) for k in ('reply_start_median_s', 'answered_share', 'cut_in_share',
                                                       'stop_median_s', 'talk_through_share', 'mmhm_per_min')}}
        table.append(r)
        print(json.dumps(r), flush=True)
    pick = choose(table)
    print('chosen by the fixed rule:', pick, flush=True)
    out = ROOT / 'results' / f'calibration_{Path(args.model).name}.json'
    out.write_text(json.dumps({'rule': 'max answered with cut-ins < 8% and reply median <= 0.35 s; else fewest cut-ins',
                               'validation': [it['id'] for it, _ in items], 'table': table, 'chosen': pick},
                              indent=1), encoding='utf8')
    alba.close()


if __name__ == '__main__':
    main()
