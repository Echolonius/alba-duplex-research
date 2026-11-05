"""Client for llama-server's /v1/realtime WebSocket (OpenAI Realtime-style events).

The server runs LFM3-Audio: audio goes in as PCM16 at 16 kHz while the person
talks (it is prefilled as it arrives), `respond()` ends their turn, and the
reply streams back as text deltas plus PCM16 audio deltas at 24 kHz.
"""
import asyncio
import base64
import json

import websockets


class Realtime:
    def __init__(self, host, port, instructions, modalities=('text', 'audio')):
        self.url = f'ws://{host}:{port}/v1/realtime'
        self.instructions = instructions
        self.modalities = list(modalities)
        self.in_rate = 16000
        self.out_rate = 24000
        self.ws = None

    async def connect(self):
        self.ws = await websockets.connect(self.url, max_size=None, ping_interval=None, open_timeout=30)
        ev = json.loads(await self.ws.recv())
        if ev.get('type') == 'error':
            raise RuntimeError(ev['error'].get('message'))
        audio = ev.get('session', {}).get('audio', {})
        self.in_rate = audio.get('input', {}).get('format', {}).get('rate', 16000)
        self.out_rate = (audio.get('output') or {}).get('format', {}).get('rate', 24000)
        await self.send({'type': 'session.update', 'session': {
            'type': 'realtime', 'instructions': self.instructions, 'output_modalities': self.modalities}})
        while True:
            ev = json.loads(await self.ws.recv())
            if ev.get('type') == 'session.updated':
                return ev
            if ev.get('type') == 'error':
                raise RuntimeError(ev['error'].get('message'))

    async def send(self, obj):
        await self.ws.send(json.dumps(obj))

    async def append(self, pcm16: bytes):
        await self.send({'type': 'input_audio_buffer.append', 'audio': base64.b64encode(pcm16).decode()})

    async def say_text(self, text):
        await self.send({'type': 'conversation.item.create', 'item': {
            'type': 'message', 'role': 'user', 'content': [{'type': 'input_text', 'text': text}]}})

    async def respond(self, commit=True, initiative=False, guess=None):
        await self.send({'type': 'response.create'})

    # The split design (app/brain.py) needs these; here the server keeps the conversation itself.
    checks_text = False

    async def commit(self):
        pass

    def reply_started(self):
        pass

    def reply_cut(self, played_s):
        pass

    async def prefetch(self):
        pass

    def discard_prefetch(self):
        pass

    def turn_cue(self):
        return None

    def still_thinking(self):
        return False

    def early_reaction(self):
        return None

    def use_reaction(self, word):
        pass

    async def cancel(self):
        await self.send({'type': 'response.cancel'})

    async def events(self):
        async for msg in self.ws:
            yield json.loads(msg)

    async def close(self):
        if self.ws is not None:
            await self.ws.close()


def decode_audio(delta_b64):
    return base64.b64decode(delta_b64)
