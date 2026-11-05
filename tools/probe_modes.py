"""Reply-speed experiments on the running model server.

1. Interleaved text+speech vs speech-only replies: time to the first real sound
   (not the silent lead-in) after the person stops.
2. What a cancelled reply leaves behind in the conversation.

  python -I probe_modes.py <out_dir>
"""
import asyncio
import base64
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app import sounds                                  # noqa: E402
from app.dsp import Resampler, from_pcm16, to_pcm16     # noqa: E402
from app.main import PERSONA                            # noqa: E402
from app.realtime import Realtime                       # noqa: E402

LINES = ["Ugh, I've had kind of a rough week.",
         "Okay, random question. Coffee or tea?",
         "I watched an old sci-fi movie last night and the effects were so bad, but I loved it.",
         "I just got back from a hike and my friend forgot all the snacks."]
INSTR = PERSONA + 'Respond with interleaved text and audio.'


def first_loud(x, rate, floor=0.008):
    w = int(0.01 * rate)
    n = len(x) // w
    if not n:
        return None
    rms = np.sqrt(np.mean(x[:n * w].reshape(n, w) ** 2, axis=1))
    k = np.nonzero(rms > floor)[0]
    return k[0] * 0.01 if len(k) else None


async def ask(q16, modalities, cancel_after=None):
    rt = Realtime('127.0.0.1', 9931, INSTR, modalities=modalities)
    await rt.connect()
    for i in range(0, len(q16), 640):           # speak the question in real time
        await rt.append(to_pcm16(q16[i:i + 640]))
        await asyncio.sleep(0.04)
    t0 = time.perf_counter()
    await rt.respond()
    audio, text, t_first_delta, gen_s = [], [], None, 0.0
    async for ev in rt.events():
        k = ev['type']
        if k == 'response.output_audio.delta':
            t_first_delta = t_first_delta or time.perf_counter()
            audio.append(from_pcm16(base64.b64decode(ev['delta'])))
        elif k == 'response.output_text.delta':
            text.append(ev['delta'])
        elif k in ('response.done', 'error'):
            gen_s = time.perf_counter() - t0
            break
    x = np.concatenate(audio) if audio else np.zeros(1, np.float32)
    lead = first_loud(x, rt.out_rate)
    await rt.close()
    first_sound = (t_first_delta - t0) + (lead or 0) / max(1e-6, (len(x) / rt.out_rate) / max(1e-6, gen_s)) if t_first_delta else None
    return {'text': ''.join(text), 'delta_s': (t_first_delta - t0) if t_first_delta else None, 'lead_s': lead,
            'audio_s': len(x) / rt.out_rate, 'gen_s': gen_s, 'first_sound_s': first_sound, 'audio': x, 'rate': rt.out_rate}


async def cancel_probe(q16):
    """Ask, cancel 0.15 s in, then ask in text what it had started to say."""
    rt = Realtime('127.0.0.1', 9931, INSTR)
    await rt.connect()
    for i in range(0, len(q16), 640):
        await rt.append(to_pcm16(q16[i:i + 640]))
    await rt.respond()
    t0 = time.perf_counter()
    got_text, cancelled = [], False
    async for ev in rt.events():
        if ev['type'] == 'response.output_text.delta':
            got_text.append(ev['delta'])
        if not cancelled and time.perf_counter() - t0 > 0.15:
            await rt.cancel()
            cancelled = True
        if ev['type'] == 'response.done':
            break
    await rt.send({'type': 'conversation.item.delete', 'item_id': 'probe'})
    delete_reply = None
    await rt.say_text('Quick check: repeat, word for word, the last thing you said to me. If you said nothing, say "nothing".')
    await rt.send({'type': 'response.create', 'response': {'output_modalities': ['text']}})
    answer = []
    async for ev in rt.events():
        if ev['type'] == 'error' and delete_reply is None:
            delete_reply = ev['error'].get('message')
            continue
        if ev['type'] == 'response.output_text.delta':
            answer.append(ev['delta'])
        elif ev['type'] == 'response.done':
            break
    await rt.close()
    return ''.join(got_text), ''.join(answer), delete_reply


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    base = 'http://127.0.0.1:9931'
    qs = []
    for k, line in enumerate(LINES):
        x, rate = sounds.tts(base, line, 'male', 300 + k)
        qs.append(Resampler(rate, 16000)(sounds.trim(x, rate, floor_db=-45)).astype(np.float32))
    for mode in (['text', 'audio'], ['audio']):
        res = [asyncio.run(ask(q, mode)) for q in qs for _ in range(2)]
        fs = [r['first_sound_s'] for r in res if r['first_sound_s'] is not None]
        leads = [r['lead_s'] for r in res if r['lead_s'] is not None]
        speed = np.median([r['audio_s'] / r['gen_s'] for r in res if r['gen_s']])
        print(f"{'+'.join(mode):10s} first audio packet {np.median([r['delta_s'] for r in res if r['delta_s']]):.3f}s | "
              f"silent lead-in {np.median(leads):.2f}s | first real sound ~{np.median(fs):.3f}s | x{speed:.1f} real time | "
              f"reply {np.median([r['audio_s'] for r in res]):.1f}s long")
        for k, r in enumerate(res[::2]):
            sf.write(out / f"mode_{'_'.join(mode)}_{k}.wav", r['audio'], r['rate'])
            if r['text']:
                print(f"     {r['text'][:100]!r}")
    said, answer, delete_reply = asyncio.run(cancel_probe(qs[0]))
    print(f"cancelled reply had produced text: {said!r}\n  asked afterwards what it last said: {answer!r}\n  item.delete: {delete_reply!r}")


if __name__ == '__main__':
    main()
