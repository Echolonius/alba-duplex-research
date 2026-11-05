"""Measures benchmark conversations with the same tool used on Dessa and prints both.

  python -I compare.py <bench_dir> <tag>
Writes results/<tag>_metrics.json and results/<tag>_summary.json.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
bench, tag = Path(sys.argv[1]), sys.argv[2]
index = bench / f'{tag}-recordings.json'
metrics = ROOT / 'results' / f'{tag}_metrics.json'
summary = ROOT / 'results' / f'{tag}_summary.json'
tools = ROOT / 'tools'
subprocess.run([PY, '-I', str(tools / 'conversation_metrics.py'), str(index), str(bench), str(metrics)], check=True,
               stderr=subprocess.DEVNULL)
subprocess.run([PY, '-I', str(tools / 'summarize_metrics.py'), str(metrics), str(index), str(summary), 'LocalVoice'],
               check=True, stdout=subprocess.DEVNULL)

d = json.loads((ROOT / 'results' / 'dessa_summary.json').read_text(encoding='utf8'))
o = json.loads(summary.read_text(encoding='utf8'))
D, H, L, P = d['test_with_people'], d['people_with_test'], o['test_with_people'], o['people_with_test']
rows = [
    ('Starts its turn after the other stops (median s)', lambda s: s['response_offset_s']['median']),
    ('...share of turn starts that overlap', lambda s: s['response_offset_s']['share_overlapping_start']),
    ('...share slower than 0.7 s', lambda s: s['response_offset_s']['share_over_700ms']),
    ('Listening sounds per min of other speech', lambda s: s['backchannels_per_min_of_other_speech']),
    ('Interrupts per minute', lambda s: s['interruptions_per_min']),
    ('Keeps talking when talked over (median s)', lambda s: s['kept_talking_when_talked_over_s']['median']),
    ('Share of talk time', lambda s: s['talk_share']),
    ('Both talking at once (share of time)', lambda s: s['overlap_share_of_time']),
    ('Audio bandwidth (Hz)', lambda s: s['bandwidth_hz_median']),
    ('Pitch range (semitones, 5-95%)', lambda s: s['f0_range_semitones_median']),
    ('Voice steadiness: jitter %', lambda s: s['jitter_pct_median']),
    ('Harmonics-to-noise (dB)', lambda s: s['hnr_db_median']),
]
print(f"{'':48s} {'Dessa':>10s} {'People w/ Dessa':>16s} {'LocalVoice':>11s} {'Sim person':>11s}")
for name, f in rows:
    vals = []
    for s in (D, H, L, P):
        try:
            v = f(s)
            vals.append(f'{v:.2f}' if isinstance(v, float) and abs(v) < 100 else f'{v:.0f}')
        except (TypeError, KeyError):
            vals.append('-')
    print(f'{name:48s} {vals[0]:>10s} {vals[1]:>16s} {vals[2]:>11s} {vals[3]:>11s}')
print(f"conversations: Dessa {D['conversations']} ({D['minutes']:.0f} min), LocalVoice {L['conversations']} ({L['minutes']:.1f} min); "
      f"response offsets counted: {D['response_offset_s']['n']} vs {L['response_offset_s']['n']}")
