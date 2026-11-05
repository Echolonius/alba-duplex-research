"""How fast could a split design be? (the voice model transcribes and speaks, a text model writes)

  python -I probe_cascade_speed.py
Needs the voice server on 9931 and a text server on 9941.
Measures: transcript ready after the end of the person's speech, text model's first
sentence, and first real sound from the speech generator in the chat's built-in voice.
"""
import asyncio
import base64
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import websockets

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app import sounds                          # noqa: E402
from app.dsp import Resampler, to_pcm16         # noqa: E402
from app.main import PERSONA                    # noqa: E402

LINES = ["Okay so honestly, I've had kind of a weird day.", "My car broke down on the way to work, and my boss was annoyed."]


async def transcribe_live(clip):
    async with websockets.connect('ws://127.0.0.1:9931/v1/realtime?intent=transcription', max_size=None) as ws:
        json.loads(await ws.recv())
        await ws.send(json.dumps({'type': 'session.update', 'session': {
            'type': 'transcription', 'audio': {'input': {'transcription': {'language': 'en'}}}}}))
        while json.loads(await ws.recv()).get('type') != 'session.updated':
            pass
        for i in range(0, len(clip), 640):          # in real time, like a microphone
            await ws.send(json.dumps({'type': 'input_audio_buffer.append',
                                      'audio': base64.b64encode(to_pcm16(clip[i:i + 640])).decode()}))
            await asyncio.sleep(0.04)
        t0 = time.perf_counter()
        await ws.send(json.dumps({'type': 'input_audio_buffer.commit'}))
        while True:
            ev = json.loads(await ws.recv())
            if ev.get('type') == 'conversation.item.input_audio_transcription.completed':
                return ev.get('transcript', ''), time.perf_counter() - t0


def first_sentence(text):
    body = json.dumps({'messages': [{'role': 'system', 'content': PERSONA}, {'role': 'user', 'content': text}],
                       'max_tokens': 60, 'stream': True}).encode()
    req = urllib.request.Request('http://127.0.0.1:9941/v1/chat/completions', data=body, headers={'Content-Type': 'application/json'})
    t0 = time.perf_counter()
    out = ''
    with urllib.request.urlopen(req, timeout=60) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith('data:') or line.endswith('[DONE]'):
                continue
            delta = json.loads(line[5:])['choices'][0]['delta'].get('content') or ''
            out += delta
            if any(p in delta for p in '.!?'):
                return out.strip(), time.perf_counter() - t0
    return out.strip(), time.perf_counter() - t0


def first_sound(text):
    body = json.dumps({'input': text, 'voice': 'female', 'stream': True, 'response_format': 'pcm'}).encode()
    req = urllib.request.Request('http://127.0.0.1:9931/tts', data=body, headers={'Content-Type': 'application/json'})
    t0 = time.perf_counter()
    buf = b''
    with urllib.request.urlopen(req, timeout=60) as r:
        rate = int(r.headers.get('Content-Type').split('rate=')[1].split(';')[0])
        while True:
            b = r.read(4800)
            if not b:
                return None
            buf += b
            x = np.frombuffer(buf[:len(buf) // 4 * 4], dtype='<f4')
            w = int(0.01 * rate)
            n = len(x) // w
            if n and (np.sqrt(np.mean(x[:n * w].reshape(n, w) ** 2, axis=1)) > 0.008).any():
                return time.perf_counter() - t0


def main():
    base = 'http://127.0.0.1:9931'
    for k, line in enumerate(LINES):
        x, rate = sounds.tts(base, line, 'male', 600 + k)
        clip = Resampler(rate, 16000)(sounds.trim(x, rate, floor_db=-45)).astype(np.float32)
        heard, t_asr = asyncio.run(transcribe_live(clip))
        reply, t_llm = first_sentence(heard)
        t_tts = first_sound(reply)
        print(f'transcript after {t_asr:.2f}s ("{heard}") | first sentence after {t_llm:.2f}s ("{reply}") | '
              f'first sound after {t_tts:.2f}s | total {t_asr + t_llm + t_tts:.2f}s')


if __name__ == '__main__':
    main()
