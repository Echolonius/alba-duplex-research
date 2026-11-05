"""Pitch range and pace of voice clips (results/voices/...), to compare candidate voices.

  python -I tools/voice_measures.py results/voices/pocket results/voices/lfm3
Writes voice_measures.json in the first folder: per voice, the median over its clips of
  range_st   pitch range, 5th to 95th percentile, in semitones (flat ~3-4, lively ~8+)
  wps        words per second while speaking (silence at the ends trimmed)
  first_s    silence before the first word, in seconds
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import parselmouth
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.try_pocket_tts import LINES  # noqa: E402


def measure(path, words):
    x, sr = sf.read(path, dtype='float32')
    snd = parselmouth.Sound(x.astype(np.float64), sampling_frequency=sr)
    f0 = snd.to_pitch(time_step=0.01, pitch_floor=70, pitch_ceiling=450).selected_array['frequency']
    f0 = f0[f0 > 0]
    st = 12 * np.log2(f0 / np.median(f0)) if len(f0) else np.array([0.0])
    frame = int(0.02 * sr)
    rms = np.sqrt(np.mean(x[:len(x) // frame * frame].reshape(-1, frame) ** 2, axis=1))
    loud = np.where(rms > 0.1 * rms.max())[0]
    start, end = (loud[0], loud[-1] + 1) if len(loud) else (0, len(rms))
    return {'range_st': float(np.percentile(st, 95) - np.percentile(st, 5)),
            'wps': words / max(0.3, (end - start) * 0.02), 'first_s': start * 0.02}


def main():
    folders = [Path(p) for p in sys.argv[1:]]
    by_voice = defaultdict(list)
    for folder in folders:
        for f in sorted(folder.glob('*.wav')):
            m = re.match(r'(.+)-(\d+)\.wav$', f.name)
            if m is None:
                continue
            voice, k = m.group(1), int(m.group(2))
            by_voice[voice].append(measure(f, len(LINES[k - 1].split())))
    rows = {v: {k: round(float(np.median([r[k] for r in rs])), 2) for k in rs[0]} for v, rs in by_voice.items()}
    for v, r in sorted(rows.items(), key=lambda kv: -kv[1]['range_st']):
        print(f"{v:22s} pitch range {r['range_st']:5.1f} semitones | {r['wps']:.1f} words/s | "
              f"starts after {r['first_s']:.2f}s")
    (folders[0] / 'voice_measures.json').write_text(json.dumps(rows, indent=1), encoding='utf8')


if __name__ == '__main__':
    main()
