"""Phone-class hearing: Moonshine Voice's streaming speech recognition on the processor (MIT).

The voice model's own transcriber (LFM3-Audio, 2B) shares the graphics card with the brain; in the
seventh live call the transcript of the person's last words sometimes came 1.3 s after they
stopped. Moonshine's streaming models are built for phones: they work on the audio as it arrives,
so when the person stops only the last fraction of a second is left to do, and they run on the
processor, leaving the graphics card to the brain.

The person's audio goes to a worker thread (a transcription pass blocks for ~0.1-0.3 s). At each of
their pauses the brain asks for what they said since the last pause: that is the new text of
Moonshine's lines since then. Words that keep coming are passed on as a live transcript.
Model files: fetched by the package from Moonshine AI on first use (checksums verified by it) into
models/moonshine.
"""
import os
import queue
import threading
import time

import numpy as np

from .config import ROOT

os.environ.setdefault('MOONSHINE_VOICE_CACHE', str(ROOT / 'models' / 'moonshine'))

IN_RATE = 16000


class MoonshineHearing:
    def __init__(self, arch='small-streaming', keyterms=(), update_s=0.3, note=None, stream_update_s=0.3):
        from moonshine_voice import ModelArch, Transcriber, get_model_for_language
        from moonshine_voice.moonshine_api import string_to_model_arch
        self.note = note or (lambda *a, **k: None)
        want = string_to_model_arch(arch) if isinstance(arch, str) else arch
        self.model_path, self.arch = get_model_for_language('en', want)
        self.tr = Transcriber(model_path=self.model_path, model_arch=self.arch, update_interval=update_s)
        if keyterms:
            try:
                self.tr.set_keyterms(list(keyterms))    # names it should hear as names ("Claude", not "clock")
            except Exception as e:                      # older models may not take key terms
                self.note('hearing_keyterms_failed', error=repr(e))
        # The stream also transcribes on its own as audio comes in (every stream_update_s of audio):
        # small regular passes keep the backlog short, so the pass asked for at a pause has little
        # left to do. With none of its own (stream_update_s=None) a pause waited 0.54 s, not 0.3 s.
        self.update_s = update_s
        self.stream = self.tr.create_stream(update_interval=stream_update_s or 1e9)
        self.stream.start()
        self.ModelArch = ModelArch
        self.q = queue.Queue()
        self.on_live = None          # callback(text): what they are saying now, not yet handed out
        self.loop = None
        self.handed = {}             # line id -> how much of its text was already handed out
        self.lines = {}              # line id -> latest text
        self.pending = []
        self.thread = threading.Thread(target=self._work, daemon=True)
        self.thread.start()

    # ---- called from the event loop -------------------------------------------
    def append(self, pcm16: bytes):
        self.q.put(('audio', np.frombuffer(pcm16, dtype='<i2').astype(np.float32) / 32768.0))

    def commit(self, fut):
        """Resolves fut (in the event loop) with what was said since the last commit."""
        self.q.put(('commit', fut))

    def close(self):
        self.q.put(('stop', None))
        self.thread.join(timeout=3)

    # ---- worker thread ----------------------------------------------------------
    def _feed(self, force=False):
        if not self.pending:
            return
        x = np.concatenate(self.pending)
        if not force and len(x) < 0.1 * IN_RATE:
            return
        self.pending = []
        self.stream.add_audio(x.tolist(), IN_RATE)

    def _take(self, transcript):
        """New text since the last hand-out, line by line (a line can run across several pauses)."""
        parts = []
        for line in transcript.lines:
            text = ' '.join((line.text or '').split())
            self.lines[line.line_id] = text
            done = self.handed.get(line.line_id, 0)
            if len(text) > done:
                parts.append(text[done:].strip())
                self.handed[line.line_id] = len(text)
        return ' '.join(p for p in parts if p)

    def _live(self, transcript):
        if self.on_live is None or self.loop is None:
            return
        parts = []
        for line in transcript.lines:
            text = ' '.join((line.text or '').split())
            done = self.handed.get(line.line_id, 0)
            if len(text) > done:
                parts.append(text[done:].strip())
        self.loop.call_soon_threadsafe(self.on_live, ' '.join(p for p in parts if p))

    def _work(self):
        force = getattr(self.tr, 'MOONSHINE_FLAG_FORCE_UPDATE', 0)
        last_live = 0.0
        while True:
            kind, x = self.q.get()
            if kind == 'stop':
                try:
                    self.stream.stop()
                    self.tr.close()
                except Exception:
                    pass
                return
            if kind == 'audio':
                self.pending.append(x)
                self._feed()
                now = time.monotonic()
                if now - last_live >= self.update_s and self.q.empty():
                    last_live = now
                    try:
                        self._live(self.stream.update_transcription(0))
                    except Exception as e:
                        self.note('hearing_error', error=repr(e))
            elif kind == 'commit':
                t0 = time.perf_counter()
                text = ''
                try:
                    self._feed(force=True)
                    text = self._take(self.stream.update_transcription(force))
                except Exception as e:
                    self.note('hearing_error', error=repr(e))
                ms = round(1000 * (time.perf_counter() - t0))
                if self.loop is not None:
                    self.loop.call_soon_threadsafe(_resolve, x, text, ms, self.note)


def _resolve(fut, text, ms, note):
    note('hearing_commit', ms=ms, words=len(text.split()))
    if not fut.done():
        fut.set_result(text)
