"""Tries system prompts for a casual, short-turn speaking style.

  python -I probe_persona.py <question.wav> <out_dir>
Reports reply length, words, speaking rate and pitch for each prompt.
"""
import asyncio
import sys
import time
from pathlib import Path

import numpy as np
import parselmouth
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.dsp import Resampler, from_pcm16, to_pcm16  # noqa: E402
from app.realtime import Realtime, decode_audio      # noqa: E402

BASE = 'Respond with interleaved text and audio.'
PROMPTS = {
    'card': BASE,
    'friend': BASE + ' You are chatting with a friend on a voice call. Keep every reply short: one or two '
                     'casual sentences, then let them talk. React like a real person: "oh no", "wait, really?", '
                     'a laugh. Ask about them instead of giving advice.',
    'friend_first': 'You are chatting with a friend on a voice call. Keep every reply short: one or two casual '
                    'sentences, then let them talk. React like a real person and ask about them instead of giving '
                    'advice. ' + BASE,
}


def pitch(x, rate):
    snd = parselmouth.Sound(x.astype(np.float64), sampling_frequency=rate)
    f0 = snd.to_pitch_ac(time_step=0.01, pitch_floor=60, pitch_ceiling=600).selected_array['frequency']
    f0 = f0[f0 > 0]
    if len(f0) < 20:
        return float('nan'), float('nan')
    p5, p50, p95 = np.percentile(f0, [5, 50, 95])
    return p50, 12 * np.log2(p95 / p5)


async def run(prompt, q16, out, tag):
    rt = Realtime('127.0.0.1', 9931, prompt)
    await rt.connect()
    step = int(0.04 * rt.in_rate)
    for i in range(0, len(q16), step):
        await rt.append(to_pcm16(q16[i:i + step]))
    t0 = time.perf_counter()
    await rt.respond()
    audio, text, first = [], [], None
    async for ev in rt.events():
        if ev['type'] == 'response.output_audio.delta':
            first = first or time.perf_counter()
            audio.append(from_pcm16(decode_audio(ev['delta'])))
        elif ev['type'] == 'response.output_text.delta':
            text.append(ev['delta'])
        elif ev['type'] in ('response.done', 'error'):
            break
    await rt.close()
    x = np.concatenate(audio) if audio else np.zeros(10)
    sf.write(out / f'persona_{tag}.wav', x, rt.out_rate)
    words = len(''.join(text).split())
    dur = len(x) / rt.out_rate
    f0, rng = pitch(x, rt.out_rate)
    print(f"[{tag}] {dur:5.1f}s, {words} words ({words/dur*60 if dur else 0:.0f} wpm), first audio {1000*(first-t0):.0f} ms, "
          f"pitch {f0:.0f} Hz, range {rng:.1f} st\n    {''.join(text)!r}")


def main():
    q, rate = sf.read(sys.argv[1], dtype='float32')
    q16 = Resampler(rate, 16000)(q).astype(np.float32) if rate != 16000 else q
    out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
    for tag, p in PROMPTS.items():
        for k in range(2):
            asyncio.run(run(p, q16, out, f'{tag}{k}'))


if __name__ == '__main__':
    main()
