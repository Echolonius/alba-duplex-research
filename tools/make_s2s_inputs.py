"""Test inputs for speech-to-speech models: a few spoken lines in a synthetic voice (Pocket TTS 'michael'),
plus a few real turns cut from the user's own recorded calls (local only; used as input, never to copy the
voice). Writes results/s2s-inputs/*.wav (16 kHz mono) and inputs.json with what was said.

  python -I tools/make_s2s_inputs.py [--real 6]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SR = 16000
LINES = ["Hey! How's it going?",
         "Ugh, I've had kind of a rough week. My manager keeps adding deadlines.",
         "Do you know that one superhero with an S on his shirt? He wears a cape and flies.",
         "What would you do with a free weekend?"]


def real_turns(limit):
    """Whole turns the person said in recorded calls: from their first sound after Alba's reply to the
    moment Alba answered, with what the transcriber heard."""
    out = []
    for d in sorted((ROOT / 'sessions').iterdir(), reverse=True):
        ev_path, wav = d / 'events.jsonl', d / 'person_clean16k.flac'
        if not ev_path.exists() or not wav.exists():
            continue
        x, fs = sf.read(wav, dtype='float32')
        ev = [json.loads(line) for line in open(ev_path, encoding='utf8') if line.strip()]
        start, stop = None, None
        for e in ev:
            if e['type'] == 'user_start' and start is None:
                start = e['at']
            elif e['type'] == 'user_stop':
                stop = e['at']
            elif e['type'] == 'heard' and start is not None and stop is not None:
                text = e['text'].strip()
                dur = stop - start
                if (1.5 <= dur <= 8.0 and len(text.split()) >= 4 and 'claude' not in text.lower()
                        and all(text[:40] != o[2][:40] for o in out)):
                    a, b = max(0, int((start - 0.15) * fs)), min(len(x), int((stop + 0.25) * fs))
                    out.append((d.name, round(start, 2), text, x[a:b]))
                start, stop = None, None
            elif e['type'] in ('respond', 'first_audio'):
                start, stop = None, None
        if len(out) >= limit * 3:
            break
    rng = np.random.default_rng(7)
    pick = rng.choice(len(out), size=min(limit, len(out)), replace=False) if out else []
    return [out[i] for i in sorted(pick)]


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser()
    ap.add_argument('--real', type=int, default=6)
    args = ap.parse_args()
    from app.speech import PocketVoice
    out = ROOT / 'results' / 's2s-inputs'
    out.mkdir(parents=True, exist_ok=True)
    items = []
    person = PocketVoice('michael', threads=4)
    for k, line in enumerate(LINES):
        wav, rate = person.say(line, seed=k)
        g = np.gcd(SR, rate)
        x = signal.resample_poly(np.asarray(wav, np.float64), SR // g, rate // g).astype(np.float32)
        name = f'synthetic-{k + 1}.wav'
        sf.write(out / name, x, SR)
        items.append({'file': name, 'kind': 'synthetic voice', 'text': line})
    for k, (sess, t, text, x) in enumerate(real_turns(args.real)):
        name = f'real-{k + 1}.wav'
        sf.write(out / name, x, SR)
        items.append({'file': name, 'kind': 'real call (local only)', 'session': sess, 'at_s': t, 'text': text})
    (out / 'inputs.json').write_text(json.dumps(items, indent=1, ensure_ascii=False), encoding='utf8')
    for it in items:
        print(f"{it['file']:16s} {it['kind']:22s} {it['text'][:90]}")


if __name__ == '__main__':
    main()
