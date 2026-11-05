"""Per-utterance breakdown of Alba Duplex on ONE validation conversation (the calibration set: Seamless test-split
pairs that are not in the frozen test and were never trained on). For every utterance: when she is heard to start,
how long it ran, the words she planned, whether her plan had ended when she stopped, and the stop chance at the
stop frame. Then how the frozen scorer's segmentation (score_timing.energy_segments) sees her voice: turns
(>= 0.8 s), short sounds, and which short sounds are whole short utterances vs pieces of longer ones split at
pauses of 0.25 s or more.

  .venv-train/Scripts/python.exe -I tools/diag_duplex_track.py --model models/alba-duplex2 [--k 0]
      [--listen-bias 1.61] [--eos-first] [--minutes 3]
Writes results/diag_track_<model>[-eosfirst]-<k>.json.
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
from app.duplex import DELAY  # noqa: E402
from calibrate_duplex import validation_items  # noqa: E402
from diag_duplex_decode import Diag  # noqa: E402
from run_timing_duplex import model_settings, run_track  # noqa: E402
from score_timing import BC_MAX, energy_segments, score_item, summarize  # noqa: E402


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--k', type=int, default=0, help='which validation conversation (calibration order)')
    ap.add_argument('--listen-bias', type=float, default=None)
    ap.add_argument('--eos-first', action='store_true')
    ap.add_argument('--minutes', type=float, default=3.0)
    ap.add_argument('--ctx', type=int, default=32768, help='memory size in positions (small = rebuilt often)')
    ap.add_argument('--keep', type=int, default=2400, help='positions kept when the memory is rebuilt')
    args = ap.parse_args()
    md = ROOT / args.model
    st = model_settings(md)
    delay = int(st.get('delay_steps', DELAY))
    bias = args.listen_bias if args.listen_bias is not None else float(st.get('listen_bias', 0.0))
    it, x = validation_items(args.k + 1)[args.k]
    if args.minutes:
        cut = args.minutes * 60
        x = x[:int(cut * 16000)]
        it = {**it, 'seconds': min(it['seconds'], cut),
              'shifts': [s for s in it['shifts'] if s['a_next'] <= cut],
              'holds': [h for h in it['holds'] if h['a_resume'] <= cut],
              'a_onsets': [o for o in it['a_onsets'] if o['start'] <= cut - 1],
              'a_backchannels': [b for b in it['a_backchannels'] if b['end'] <= cut - 1],
              'caller_segments': [c for c in it['caller_segments'] if c[1] <= cut]}
    a = Diag(md / 'model-Q4_K_M.gguf', md / 'hf', eos_first=args.eos_first, n_ctx=args.ctx, keep_tokens=args.keep)
    a.plan_log = []
    voice24, _, _ = run_track(a, x, delay, bias)
    utts, cur = [], None
    for e in a.events:
        if e[0] == 'start':
            cur = {'start_step': e[1]}
        elif e[0] == 'stop' and cur is not None:
            cur.update(stop_step=e[1], frames=e[3], plan_done=bool(e[4]))
            utts.append(cur)
            cur = None
    if cur is not None:
        utts.append({**cur, 'stop_step': None, 'frames': None, 'plan_done': None})
    for u in utts:
        lo, hi = u['start_step'], (u['stop_step'] if u['stop_step'] is not None else 10 ** 9)
        toks = [e[3] for e in a.plan_log if e[0] == 'piece' and lo <= e[1] <= hi]
        u['plan'] = a.tok.convert_tokens_to_string(toks).replace('<|text_end|>', '|').strip()
        fr = [e for e in a.plan_log if e[0] == 'frame' and lo <= e[1] <= hi]
        u['p_eos_stop'] = [fr[-1][4], fr[-1][5]] if fr else None
        u['heard_s'] = round((lo + delay) * 0.08, 2)
        u['secs'] = round(u['frames'] * 0.08, 2) if u['frames'] is not None else None
    v16 = signal.resample_poly(voice24, 2, 3).astype(np.float32)
    segs = energy_segments(v16, 16000)
    split_pieces = whole_short = 0
    for s, e in segs:
        if e - s >= BC_MAX:
            continue
        owner = next((u for u in utts if u['heard_s'] - 0.1 <= s <= u['heard_s'] + (u['secs'] or 0) + 0.3), None)
        if owner is not None and (owner['secs'] or 0) >= BC_MAX:
            split_pieces += 1
        else:
            whole_short += 1
    score = summarize([score_item(it, segs)])
    secs = [u['secs'] for u in utts if u['secs'] is not None]
    summary = {'conversation': it['id'], 'minutes': round(len(x) / 16000 / 60, 2), 'listen_bias': bias,
               'ctx': args.ctx, 'keep': args.keep, 'rebuilds': sum(1 for e in a.events if e[0] == 'rebuilt'),
               'eos_first': args.eos_first, 'utterances': len(utts),
               'utterance_s_median': float(np.median(secs)) if secs else None,
               'utterances_under_0.8s': sum(1 for s in secs if s < BC_MAX),
               'stopped_before_plan_end': sum(1 for u in utts if u['plan_done'] is False),
               'energy_segments': len(segs), 'turn_segments': sum(1 for s, e in segs if e - s >= BC_MAX),
               'short_segments_whole_utterances': whole_short, 'short_segments_split_from_longer': split_pieces,
               **{k: score.get(k) for k in ('answered_share', 'cut_in_share', 'reply_start_median_s', 'mmhm_per_min',
                                            'stop_median_s')}}
    for u in utts:
        print(f"{u['heard_s']:7.2f}s {str(u['secs']):>5}s done={str(u['plan_done'])[0]} "
              f"eos={u['p_eos_stop']} | {u['plan'][:90]!r}", flush=True)
    print(json.dumps(summary), flush=True)
    tag = Path(args.model).name + ('-eosfirst' if args.eos_first else '') + (f'-ctx{args.ctx}' if args.ctx != 32768 else '')
    (ROOT / 'results' / f'diag_track_{tag}-{args.k}.json').write_text(
        json.dumps({'summary': summary, 'utterances': utts, 'segments': segs}, indent=1, ensure_ascii=False),
        encoding='utf8')
    a.close()


if __name__ == '__main__':
    main()
