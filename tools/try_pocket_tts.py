"""Phone-class voice candidate: Kyutai's Pocket TTS (100M, runs on the processor).

Speaks the same conversational lines in several of its built-in voices, measures how soon the
first sound is ready and how much faster than real time it runs, and saves the clips for
listening. Weights: kyutai/pocket-tts-without-voice-cloning (CC-BY-4.0, safetensors, pinned in
the package's config files); downloads stay in models/hf (tools/env.sh).

  python -I tools/try_pocket_tts.py results/voices/pocket [--voices alba,cosette] [--threads 2]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

LINES = [
    "Oh no, again? That toaster is clearly out to get you.",
    "Honestly? I'd pick the mountains. Big views, cold air, terrible snacks.",
    "Wait, they cancelled it after a whole month of work? That's rough.",
    "Ha, okay, that's fair. I walked right into that one.",
    "Mm, I hear you. Sometimes you just need to say it out loud.",
]
VOICES = 'alba,cosette,jean,marius,javert,anna,eve,caro_davy,vera,michael'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--voices', default=VOICES)
    ap.add_argument('--language', default='english')
    ap.add_argument('--threads', type=int, default=2, help='processor threads for the model (phone-like: 2)')
    ap.add_argument('--quantize', action='store_true', help='8-bit weights')
    args = ap.parse_args()
    import torch
    from pocket_tts import TTSModel
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t = time.perf_counter()
    model = TTSModel.load_model(language=args.language, quantize=args.quantize)
    torch.set_num_threads(args.threads)
    print(f'model {args.language} loaded in {time.perf_counter() - t:.1f}s, voice cloning: {model.has_voice_cloning}, '
          f'rate {model.sample_rate} Hz', flush=True)
    rows = []
    for voice in args.voices.split(','):
        t = time.perf_counter()
        state = model.get_state_for_audio_prompt(voice)
        load = time.perf_counter() - t
        for k, line in enumerate(LINES):
            t0 = time.perf_counter()
            first, chunks = None, []
            for c in model.generate_audio_stream(state, line):
                if first is None:
                    first = time.perf_counter() - t0
                chunks.append(c.numpy())
            total = time.perf_counter() - t0
            audio = np.concatenate(chunks)
            dur = len(audio) / model.sample_rate
            sf.write(out / f'{args.language}-{voice}-{k + 1}.wav', audio, model.sample_rate)
            rows.append({'voice': voice, 'line': k + 1, 'first_audio_s': round(first, 3), 'seconds': round(dur, 2),
                         'x_real_time': round(dur / total, 2)})
        mine = [r for r in rows if r['voice'] == voice]
        print(f"{voice:10s} voice ready in {load:.2f}s | first sound median "
              f"{np.median([r['first_audio_s'] for r in mine]):.2f}s | "
              f"{np.median([r['x_real_time'] for r in mine]):.1f}x real time", flush=True)
    (out / f'{args.language}-timings.json').write_text(json.dumps(rows, indent=1), encoding='utf8')


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    main()
