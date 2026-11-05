"""Writes both sides of a call to disk while it happens.

The audio callback only hands blocks to a queue; a writer thread appends them to
two FLAC files and flushes every second, so even a force-closed window leaves
everything up to the last second on disk.
"""
import queue
import threading
import time
from pathlib import Path

import soundfile as sf


class DiskRecorder:
    def __init__(self, folder, rate=48000):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.rate = rate
        self.q = queue.Queue()
        self.files = {name: sf.SoundFile(self.folder / f'{name}.flac', 'w', rate, 1, 'PCM_16', format='FLAC')
                      for name in ('person', 'voice')}
        self.samples = 0
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def add(self, mic, out):
        """Called from the audio callback: never blocks."""
        self.q.put_nowait((mic.copy(), out.copy()))

    def _run(self):
        last_flush = time.time()
        while self.running or not self.q.empty():
            try:
                mic, out = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            self.files['person'].write(mic)
            self.files['voice'].write(out)
            self.samples += len(mic)
            if time.time() - last_flush > 1.0:
                for f in self.files.values():
                    f.flush()
                last_flush = time.time()

    @property
    def seconds(self):
        return self.samples / self.rate

    def close(self):
        self.running = False
        self.thread.join(timeout=10)
        for f in self.files.values():
            f.close()
