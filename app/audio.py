"""Sound in and out.

One duplex callback every 10 ms hands the conductor a block of microphone
audio together with the block the speakers played at the same moment, so the
echo canceller and the session recording share one clock.

LiveAudio:  the real microphone and speakers (sounddevice / WASAPI, 48 kHz).
SimAudio:   a scripted, simulated person for testing without a microphone.
"""
import collections
import threading
import time

import numpy as np

RATE = 48000
BLOCK = 480  # 10 ms


class Playback:
    """What the speakers will play: the voice's reply plus short listening sounds."""

    def __init__(self):
        self.lock = threading.Lock()
        self.main = collections.deque()
        self.sfx = collections.deque()
        self.fade_left = 0
        self.fade_total = 0
        self.gain = 1.0          # current level of the reply...
        self.target = 1.0        # ...and where it is heading (ducking while talked over)
        self.ramp = 1.0 / (0.06 * RATE)

    def push(self, x):
        with self.lock:
            self.main.append(np.asarray(x, np.float32))

    def push_sfx(self, x):
        with self.lock:
            self.sfx.append(np.asarray(x, np.float32))

    def fade_out(self, seconds):
        with self.lock:
            self.fade_total = self.fade_left = max(1, int(seconds * RATE))

    def cut_last_sentence(self, expected_s, floor=0.006, min_pause_s=0.12, fade_s=0.05):
        """Drops the queued reply's last sentence: everything after its last pause.

        Only cuts if what would be removed is about as long as the sentence is expected to
        be (so a pause between two words of it is never mistaken for the sentence break).
        Returns the seconds removed (0 if nothing was cut).
        """
        with self.lock:
            if not self.main:
                return 0.0
            x = np.concatenate(list(self.main))
            w = int(0.01 * RATE)
            n = len(x) // w
            if n < 20:
                return 0.0
            loud = np.sqrt(np.mean(x[:n * w].reshape(n, w) ** 2, axis=1)) > floor
            last_loud = int(np.nonzero(loud)[0][-1]) if loud.any() else 0
            need = int(min_pause_s / 0.01)
            quiet, cut = 0, None
            for k in range(last_loud, -1, -1):      # walk back to the pause before the last sentence
                if not loud[k]:
                    quiet += 1
                    if quiet >= need:
                        cut = k + quiet // 2
                        break
                else:
                    quiet = 0
            if cut is None or cut < 5:
                return 0.0
            end = cut * w
            removed = (len(x) - end) / RATE
            if not 0.5 * expected_s <= removed <= 2.5 * expected_s:
                return 0.0
            y = x[:end].copy()
            f = min(len(y), int(fade_s * RATE))
            y[-f:] *= np.linspace(1, 0, f, dtype=np.float32)
            self.main.clear()
            self.main.append(y)
            return (len(x) - end) / RATE

    def trim_tail(self, floor=0.004, keep_s=0.15):
        """Drops trailing near-silence from the queued reply (keeps keep_s of it)."""
        with self.lock:
            if not self.main:
                return 0
            x = np.concatenate(list(self.main))
            w = int(0.01 * RATE)
            n = len(x) // w
            if n == 0:
                return 0
            loud = np.sqrt(np.mean(x[:n * w].reshape(n, w) ** 2, axis=1)) > floor
            last = int(np.nonzero(loud)[0][-1]) + 1 if loud.any() else 0
            end = min(len(x), last * w + int(keep_s * RATE))
            self.main.clear()
            if end > 0:
                self.main.append(x[:end])
            return len(x) - end

    def duck(self, gain):
        """Quieter while being talked over; back to full with duck(1.0)."""
        with self.lock:
            self.target = gain

    def busy(self):
        with self.lock:
            return bool(self.main)

    def sfx_busy(self):
        with self.lock:
            return bool(self.sfx)

    @staticmethod
    def _take(q, n):
        out = np.zeros(n, np.float32)
        i = 0
        while i < n and q:
            c = q[0]
            k = min(n - i, len(c))
            out[i:i + k] = c[:k]
            if k == len(c):
                q.popleft()
            else:
                q[0] = c[k:]
            i += k
        return out

    def pull(self, n):
        with self.lock:
            main = self._take(self.main, n)
            if self.gain != self.target or self.gain != 1.0:
                step = np.clip(self.target - self.gain, -self.ramp * n, self.ramp * n)
                main *= np.linspace(self.gain, self.gain + step, n, dtype=np.float32)
                self.gain += step
            if self.fade_left:
                k = min(n, self.fade_left)
                start = self.fade_left / self.fade_total
                ramp = np.linspace(start, max(0.0, start - k / self.fade_total), k, dtype=np.float32)
                main[:k] *= ramp
                main[k:] = 0
                self.fade_left -= k
                if not self.fade_left:
                    self.main.clear()
                    self.gain = self.target = 1.0
            return main + self._take(self.sfx, n)


class Recorder:
    """Keeps both sides of the conversation, sample-aligned, for later measurement."""

    def __init__(self):
        self.mic, self.out = [], []

    def add(self, mic, out):
        self.mic.append(mic.copy())
        self.out.append(out.copy())

    def tracks(self):
        return (np.concatenate(self.mic) if self.mic else np.zeros(0, np.float32),
                np.concatenate(self.out) if self.out else np.zeros(0, np.float32))


class LiveAudio:
    """Microphone + speakers through one duplex stream."""

    def __init__(self, playback, on_block, recorder=None, input_device=None, output_device=None):
        import sounddevice as sd
        self.sd = sd
        self.playback, self.on_block, self.recorder = playback, on_block, recorder
        self.input_device, self.output_device = input_device, output_device
        self.stream = None

    def _pick_devices(self):
        sd = self.sd
        if self.input_device is not None and self.output_device is not None:
            return self.input_device, self.output_device, None
        apis = sd.query_hostapis()
        wasapi = next((i for i, a in enumerate(apis) if 'WASAPI' in a['name']), None)
        if wasapi is not None:
            a = apis[wasapi]
            return a['default_input_device'], a['default_output_device'], wasapi
        return sd.default.device[0], sd.default.device[1], None

    def start(self):
        sd = self.sd
        din, dout, wasapi = self._pick_devices()
        extra = None
        if wasapi is not None:
            extra = (sd.WasapiSettings(auto_convert=True), sd.WasapiSettings(auto_convert=True))
        in_ch = 1

        def cb(indata, outdata, frames, time_info, status):
            out = self.playback.pull(frames)
            outdata[:] = out[:, None]
            mic = indata[:, 0].astype(np.float32)
            if self.recorder is not None:
                self.recorder.add(mic, out)
            self.on_block(mic, out)

        self.stream = sd.Stream(samplerate=RATE, blocksize=BLOCK, device=(din, dout), channels=(in_ch, 2),
                                dtype='float32', latency='low', callback=cb, extra_settings=extra)
        self.stream.start()
        return sd.query_devices(din)['name'], sd.query_devices(dout)['name']

    def stop(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()


class SimAudio:
    """Runs the same 10 ms callback in real time, with a scripted person as the microphone.

    `person` is an object with `next_block(agent_block, now_s) -> mic_block` that hears
    what the voice plays and decides when to talk (see tools/benchmark.py).
    """

    def __init__(self, playback, on_block, recorder, person, seconds_limit=900):
        self.playback, self.on_block, self.recorder, self.person = playback, on_block, recorder, person
        self.limit = seconds_limit
        self.thread = None
        self.running = False
        self.done = threading.Event()

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return 'simulated person', 'simulated speaker'

    def _run(self):
        t0 = time.perf_counter()
        n = 0
        while self.running:
            out = self.playback.pull(BLOCK)
            now = n * BLOCK / RATE
            mic = self.person.next_block(out, now)
            if mic is None or now > self.limit:
                break
            self.recorder.add(mic, out)
            self.on_block(mic, out)
            n += 1
            delay = t0 + n * BLOCK / RATE - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
        self.running = False
        self.done.set()

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2)
