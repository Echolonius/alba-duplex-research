"""How the person said things: pitch, loudness, pace and how each line ends, line by line.

Reads a live call's person track and the line timings from analysis.json (tools/analyze_call.py),
and prints each line with, relative to that person's own usual voice in the call:
  pitch   median pitch, in semitones above (+) or below (-) their usual
  range   how far the pitch moved within the line (semitones, 5th to 95th percentile)
  loud    loudness in dB above or below their usual
  pace    words per second
  end     the pitch over the line's last half second: rising, falling or level
Lines whose pitch or loudness jumps from the line before are marked "shift" (a change of tone).
It is the start of giving the brain how something was said, not only the words.

  python -I tools/prosody.py sessions/<call> [--from MM:SS] [--to MM:SS]
Writes prosody.json in the call's folder.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import parselmouth
import soundfile as sf


def clock(s):
    m, sec = s.split(':')
    return int(m) * 60 + float(sec)


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser()
    ap.add_argument('session')
    ap.add_argument('--from', dest='t0', default='0:00')
    ap.add_argument('--to', dest='t1', default='999:00')
    args = ap.parse_args()
    S = Path(args.session)
    lines = json.loads((S / 'analysis.json').read_text(encoding='utf8'))['said']
    audio = S / 'person_clean16k.flac' if (S / 'person_clean16k.flac').exists() else S / 'person.flac'
    x, sr = sf.read(audio, dtype='float32')
    if x.ndim > 1:
        x = x.mean(axis=1)
    snd = parselmouth.Sound(x.astype(np.float64), sampling_frequency=sr)
    pitch = snd.to_pitch(time_step=0.01, pitch_floor=60, pitch_ceiling=400)
    f0 = pitch.selected_array['frequency']
    f0_t = pitch.xs()
    inten = snd.to_intensity(time_step=0.01, minimum_pitch=60)
    db, db_t = inten.values[0], inten.xs()
    voiced = f0[f0 > 0]
    base_f0 = float(np.median(voiced))
    rows = []
    for ln in lines:
        a, b = ln['t'], ln['end']
        k = (f0_t >= a) & (f0_t <= b) & (f0 > 0)
        if k.sum() < 10:
            continue
        st = 12 * np.log2(f0[k] / base_f0)
        kd = (db_t >= a) & (db_t <= b)
        loud = float(np.percentile(db[kd], 90)) if kd.any() else np.nan
        tail = k & (f0_t >= b - 0.5)
        end = 'level'
        if tail.sum() >= 5:
            slope = np.polyfit(f0_t[tail], 12 * np.log2(f0[tail] / base_f0), 1)[0]   # semitones per second
            end = 'rising' if slope > 4 else ('falling' if slope < -4 else 'level')
        words = len(ln['text'].split())
        rows.append({'t': a, 'end_t': b, 'text': ln['text'], 'pitch_st': float(np.median(st)),
                     'range_st': float(np.percentile(st, 95) - np.percentile(st, 5)), 'loud_db': loud,
                     'pace_wps': words / max(0.5, b - a), 'ending': end})
    base_loud = float(np.nanmedian([r['loud_db'] for r in rows]))
    for r in rows:
        r['loud_db'] -= base_loud
    t0, t1 = clock(args.t0), clock(args.t1)
    print(f'usual pitch {base_f0:.0f} Hz; {len(rows)} lines with enough voice to measure\n')
    print(' time   pitch  range  loud  pace  end      what was said')
    prev = None
    for r in rows:
        shift = prev is not None and (abs(r['pitch_st'] - prev['pitch_st']) >= 2.0 or abs(r['loud_db'] - prev['loud_db']) >= 4)
        r['shift'] = bool(shift)
        if t0 <= r['t'] <= t1:
            print(f"{int(r['t'] // 60):02d}:{r['t'] % 60:04.1f} {r['pitch_st']:+5.1f} {r['range_st']:5.1f} {r['loud_db']:+5.1f} "
                  f"{r['pace_wps']:4.1f}  {r['ending']:8s} {'shift ' if shift else '      '}{r['text'][:90]}")
        prev = r
    (S / 'prosody.json').write_text(json.dumps({'usual_pitch_hz': base_f0, 'lines': rows}, indent=1), encoding='utf8')


if __name__ == '__main__':
    main()
