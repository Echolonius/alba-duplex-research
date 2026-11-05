"""Speed and latency probe for the local LFM3-Audio server.

  python -I probe_server.py <out_dir>

1. /tts: seconds of speech produced per second of compute, and time to first audio when streaming.
2. /v1/realtime: a spoken question goes in at real-time pace; measures how long after
   the end of the question the first reply audio arrives, and how fast the reply streams.
"""
import asyncio
import io
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.dsp import Resampler, from_pcm16, to_pcm16  # noqa: E402
from app.realtime import Realtime, decode_audio      # noqa: E402

BASE = 'http://127.0.0.1:9931'


def tts(text, voice, stream=False, seed=7):
    body = json.dumps({'input': text, 'voice': voice, 'seed': seed, 'stream': stream, 'response_format': 'pcm'}).encode()
    req = urllib.request.Request(BASE + '/tts', data=body, headers={'Content-Type': 'application/json'})
    t0 = time.perf_counter(); first = None; chunks = []
    with urllib.request.urlopen(req, timeout=300) as r:
        rate = int(r.headers.get('Content-Type', 'rate=24000').split('rate=')[1].split(';')[0])
        while True:
            b = r.read(4096)
            if not b:
                break
            first = first or time.perf_counter()
            chunks.append(b)
    t1 = time.perf_counter()
    x = np.frombuffer(b''.join(chunks), dtype='<f4')
    return x, rate, (first - t0) if first else None, t1 - t0


async def converse(question16, out_dir, tag):
    rt = Realtime('127.0.0.1', 9931, 'Respond with interleaved text and audio.')
    await rt.connect()
    step = int(0.04 * rt.in_rate)
    t_end = None
    async def send():
        nonlocal t_end
        for i in range(0, len(question16), step):
            await rt.append(to_pcm16(question16[i:i + step]))
            await asyncio.sleep(step / rt.in_rate)
        t_end = time.perf_counter()
        await rt.respond()
    sender = asyncio.create_task(send())
    first_audio = first_text = None; audio = []; text = []
    async for ev in rt.events():
        t = ev.get('type')
        if t == 'response.output_audio.delta':
            first_audio = first_audio or time.perf_counter()
            audio.append(from_pcm16(decode_audio(ev['delta'])))
        elif t == 'response.output_text.delta':
            first_text = first_text or time.perf_counter()
            text.append(ev['delta'])
        elif t == 'response.done':
            break
        elif t == 'error':
            print('error', ev); break
    t_done = time.perf_counter()
    await sender
    await rt.close()
    reply = np.concatenate(audio) if audio else np.zeros(1)
    sf.write(out_dir / f'reply_{tag}.wav', reply, rt.out_rate)
    dur = len(reply) / rt.out_rate
    print(f"[{tag}] first audio {1000*(first_audio - t_end):.0f} ms after question ends | first text {1000*(first_text - t_end):.0f} ms | "
          f"reply {dur:.2f}s of audio generated in {t_done - t_end:.2f}s (x{dur/(t_done - t_end):.2f} real time)")
    print(f"[{tag}] text: {''.join(text)!r}")


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    line = "Oh wow, honestly? That's kind of amazing. So, what happened next?"
    for voice in ('female', 'male'):
        x, rate, ttfa, total = tts(line, voice)
        print(f"/tts {voice}: {len(x)/rate:.2f}s audio in {total:.2f}s (x{len(x)/rate/total:.2f} real time), rate {rate}")
    x, rate, ttfa, total = tts(line, 'female', stream=True)
    print(f"/tts streaming: first audio after {1000*ttfa:.0f} ms, done {total:.2f}s")
    q = "Hey! So I just got back from a really long hike and I'm totally exhausted. Have you ever done something like that?"
    qx, qrate, _, _ = tts(q, 'male', seed=3)
    q16 = Resampler(qrate, 16000)(qx).astype(np.float32)
    sf.write(out / 'question.wav', q16, 16000)
    for k in range(2):
        asyncio.run(converse(q16, out, f'try{k}'))


if __name__ == '__main__':
    main()
