"""Scores the timing test (check 2) from a system's recorded output against the real callers' moments.

For every caller track in data/tests/timing/items.json and the system's aligned output
(results/tests/<tag>/timing/<id>-voice.flac):
  reply start   at each "shift" (the caller finished; the real partner took over): first system speech
                (not a short "mm-hm") from 1 s before the caller's end until the caller speaks again;
                start - end in seconds (below 0 = it started while the caller was finishing)
  answered      share of shifts with a reply before the caller spoke again
  cut-ins       share of "holds" (the caller paused and went on) where the system started talking
  stops         when the caller starts a real turn (>= 0.8 s) while the system is talking: seconds until it stops
  talks through when the caller says "mm-hm" while the system is talking: share where it keeps going 0.5 s on
  mm-hms        the system's own short listening sounds per minute of caller speech
The same is measured for the real human partners of the Seamless tracks: the human reference.

Targets (goal, check 2): reply start <= 0.3 s median; stops <= 0.4 s median; cut-ins < 10%.

  python -I tools/score_timing.py <tag> [<tag> ...]      (or 'human' for the reference)
Writes results/tests/<tag>/timing_scores.json.

--public scores the public timing set instead (data/tests-public/timing: the same 12 Seamless tracks without the
owner's calls, which anyone can rebuild): the system's output is read from results/tests-public/<tag>/timing/ and
the scores go to results/tests-public/<tag>/timing_scores.json, apart from the frozen set's.
"""
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
TESTS = ROOT / 'data' / 'tests' / 'timing'
PUBLIC = ROOT / 'data' / 'tests-public' / 'timing'
BC_MAX = 0.8         # a system segment shorter than this is a listening sound, not a turn


def energy_segments(x, sr=16000, floor_db=-45, min_gap=0.25):
    h = sr // 100
    n = len(x) // h
    on = 10 * np.log10(np.mean(x[:n * h].reshape(n, h) ** 2, axis=1) + 1e-12) > floor_db
    segs, st = [], None
    for k, v in enumerate(on):
        if v and st is None:
            st = k
        elif not v and st is not None:
            segs.append([st / 100, k / 100]); st = None
    if st is not None:
        segs.append([st / 100, n / 100])
    out = []
    for s in segs:
        if out and s[0] - out[-1][1] < min_gap:
            out[-1][1] = s[1]
        else:
            out.append(s)
    return [(round(s, 3), round(e, 3)) for s, e in out if e - s >= 0.08]


def score_item(it, sys_segs):
    """sys_segs: [(start, end)] of the system (or partner) speech."""
    turns = [(s, e) for s, e in sys_segs if e - s >= BC_MAX]
    bcs = [(s, e) for s, e in sys_segs if e - s < BC_MAX]
    lat, answered = [], 0
    for sh in it['shifts']:
        on = next((s for s, e in turns if sh['a_end'] - 1.0 <= s < sh['a_next']), None)
        if on is not None:
            answered += 1
            lat.append(on - sh['a_end'])
    cut = 0
    for h in it['holds']:
        if any(h['a_end'] - 0.3 <= s < h['a_resume'] for s, e in turns):
            cut += 1
    stops = []
    for o in it['a_onsets']:
        seg = next(((s, e) for s, e in turns if s <= o['start'] - 0.3 and e > o['start']), None)
        if seg:
            stops.append(min(5.0, seg[1] - o['start']))
    through, through_n = 0, 0
    for b in it['a_backchannels']:
        seg = next(((s, e) for s, e in turns if s <= b['start'] - 0.3 and e > b['start']), None)
        if seg:
            through_n += 1
            through += seg[1] > b['end'] + 0.5
    a_speech = sum(e - s for s, e, bc in it['caller_segments'] if not bc) / 60
    return {'shifts': len(it['shifts']), 'answered': answered, 'latencies': [round(x, 3) for x in lat],
            'holds': len(it['holds']), 'cut_ins': cut, 'stops': [round(x, 3) for x in stops],
            'talk_through': through, 'talk_through_n': through_n, 'mmhm': len(bcs), 'caller_speech_min': round(a_speech, 3)}


def summarize(rows):
    lat = [x for r in rows for x in r['latencies']]
    stops = [x for r in rows for x in r['stops']]
    sh, ans = sum(r['shifts'] for r in rows), sum(r['answered'] for r in rows)
    holds, cut = sum(r['holds'] for r in rows), sum(r['cut_ins'] for r in rows)
    tt, ttn = sum(r['talk_through'] for r in rows), sum(r['talk_through_n'] for r in rows)
    mins = sum(r['caller_speech_min'] for r in rows)
    f = lambda v: round(float(v), 3)
    return {'reply_start_median_s': f(np.median(lat)) if lat else None,
            'reply_start_p90_s': f(np.percentile(lat, 90)) if lat else None,
            'replies_overlapping_share': f(np.mean(np.array(lat) < 0)) if lat else None,
            'replies_slower_than_0.7s_share': f(np.mean(np.array(lat) > 0.7)) if lat else None,
            'answered_share': f(ans / sh) if sh else None, 'shifts': sh,
            'cut_in_share': f(cut / holds) if holds else None, 'holds': holds,
            'stop_median_s': f(np.median(stops)) if stops else None, 'stops_n': len(stops),
            'stop_within_0.4s_share': f(np.mean(np.array(stops) <= 0.4)) if stops else None,
            'talk_through_share': f(tt / ttn) if ttn else None, 'talk_through_n': ttn,
            'mmhm_per_min': f(sum(r['mmhm'] for r in rows) / mins) if mins else None}


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    tags = [a for a in sys.argv[1:] if a != '--public']
    tests, res = (PUBLIC, ROOT / 'results' / 'tests-public') if '--public' in sys.argv[1:] else \
        (TESTS, ROOT / 'results' / 'tests')
    items = json.loads((tests / 'items.json').read_text(encoding='utf8'))['items']
    for tag in tags:
        rows, per = [], {}
        for it in items:
            if tag == 'human':
                if not it.get('partner_segments'):
                    continue
                segs = [(s, e) for s, e, bc in it['partner_segments']]
            else:
                p = res / tag / 'timing' / f"{it['id']}-voice.flac"
                if not p.exists():
                    continue
                x, sr = sf.read(p, dtype='float32')
                segs = energy_segments(x, sr)
            r = score_item(it, segs)
            rows.append(r)
            per[it['id']] = r
        groups = {'all': rows}
        groups['seamless (strangers)'] = [per[k] for k in per if k.startswith('seamless')]
        groups['your calls'] = [per[k] for k in per if k.startswith('call-')]
        out = {g: summarize(v) for g, v in groups.items() if v}
        out['targets'] = {'reply_start_median_s': '<= 0.3', 'stop_median_s': '<= 0.4', 'cut_in_share': '< 0.10'}
        out['per_item'] = per
        dest = res / tag
        dest.mkdir(parents=True, exist_ok=True)
        (dest / 'timing_scores.json').write_text(json.dumps(out, indent=1), encoding='utf8')
        print(f'== {tag}')
        for g in groups:
            if groups[g]:
                print(f'  {g:22s}', json.dumps(out[g]))


if __name__ == '__main__':
    main()
