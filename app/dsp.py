"""Streaming signal processing: sample-rate conversion and echo cancellation."""
from math import gcd

import numpy as np
from scipy import signal


class Resampler:
    """Polyphase FIR resampler that keeps its state between blocks."""

    def __init__(self, rate_in, rate_out, taps_per_phase=24):
        g = gcd(rate_in, rate_out)
        self.up, self.down = rate_out // g, rate_in // g
        span = max(self.up, self.down)
        self.h = signal.firwin(taps_per_phase * span + 1, 0.95 / span, window=('kaiser', 8.0)) * self.up
        self.zi = np.zeros(len(self.h) - 1)
        self.phase = 0  # global sample index (mod down) of the next upsampled sample

    def __call__(self, x):
        x = np.asarray(x, np.float64)
        if self.up > 1:
            y = np.zeros(len(x) * self.up)
            y[::self.up] = x
        else:
            y = x
        y, self.zi = signal.lfilter(self.h, 1.0, y, zi=self.zi)
        start = (-self.phase) % self.down
        self.phase = (self.phase + len(y)) % self.down
        return y[start::self.down]


class EchoCanceller:
    """Partitioned-block frequency-domain NLMS echo canceller (overlap-save).

    `ref` is what the speakers played during the same block as `mic`; the
    filter learns the speaker-to-microphone path (delay included, up to
    `tail_s`) and subtracts the predicted echo from the microphone signal.
    """

    def __init__(self, block=160, tail_s=0.32, rate=16000, mu=0.6):
        self.B, self.N = block, 2 * block
        self.P = int(np.ceil(tail_s * rate / block))
        k = self.N // 2 + 1
        self.W = np.zeros((self.P, k), complex)
        self.X = np.zeros((self.P, k), complex)
        self.xprev = np.zeros(block)
        self.norm = np.full(k, 1e-3)
        self.mu = mu

    def __call__(self, mic, ref, adapt=True):
        ref = np.asarray(ref, np.float64)
        Xk = np.fft.rfft(np.concatenate([self.xprev, ref]))
        self.xprev = ref.copy()
        self.X[1:] = self.X[:-1]
        self.X[0] = Xk
        y = np.fft.irfft((self.W * self.X).sum(axis=0), self.N)[self.B:]
        e = mic - y
        if adapt and float(np.dot(ref, ref)) > 1e-6 * self.B:
            self.norm = 0.8 * self.norm + 0.2 * (np.abs(self.X) ** 2).sum(axis=0)
            E = np.fft.rfft(np.concatenate([np.zeros(self.B), e]))
            G = self.mu * np.conj(self.X) * E / (self.norm + 1e-6)
            g = np.fft.irfft(G, self.N, axis=1)
            g[:, self.B:] = 0                      # keep the filter causal (gradient constraint)
            self.W += np.fft.rfft(g, axis=1)
            if np.dot(e, e) > 4 * np.dot(mic, mic) + 1e-9:   # diverging: start over
                self.W[:] = 0
        return e, y

    # The speaker-to-microphone path of a laptop barely changes, so what the
    # filter learned in one conversation is a good start for the next.
    def save(self, path):
        np.save(path, self.W)

    def load(self, path):
        try:
            W = np.load(path)
            if W.shape == self.W.shape and np.all(np.isfinite(W)):
                self.W = W
        except (OSError, ValueError):
            pass


class EchoGate:
    """With laptop speakers: is sound at the microphone the person, or the voice's own echo?

    Works from the first second, before the echo canceller has learned the room. It
    measures how strongly the speakers leak into the microphone (coupling) and how much
    the canceller already removes, and only counts sound clearly louder than the echo
    that should be left. Errs toward "echo": the voice missing an interruption is
    better than the voice interrupting itself.
    """

    def __init__(self, margin_db=10.0, delay_frames=10, window=320, start_db=-15.0):
        import collections
        self.recent_ref = collections.deque(maxlen=delay_frames)   # ~0.3 s of playback energy
        self.leak_db = collections.deque(maxlen=window)            # leftover echo vs playback, dB
        self.margin_db = margin_db
        self.start_db = start_db

    def observe(self, e_mic, e_ref, e_out):
        self.recent_ref.append(e_ref)
        R = max(self.recent_ref)
        # Only while the voice is actually sounding: in the gaps between its sentences the
        # microphone goes quiet while R still holds the last word, and those very low readings
        # pulled the estimate down until its own echo counted as the person (split design,
        # which pauses between sentences more).
        if R > 1e-8 and max(list(self.recent_ref)[-3:]) > 0.1 * R:
            self.leak_db.append(10 * np.log10((e_out + 1e-14) / R))

    def playing(self):
        return bool(self.recent_ref) and max(self.recent_ref) > 1e-8

    def leak(self):
        """How loud the leftover echo is relative to what the speakers play, in dB.

        Taken from the quiet end of what the microphone hears while the voice plays:
        the person talking over it only adds loud frames, so a low percentile stays the
        echo's level. (A high percentile was used first and was wrong: in the first live
        call, the person's own talking-over raised the estimate until they were blocked.)
        """
        if len(self.leak_db) < 15:
            return self.start_db
        return float(np.percentile(self.leak_db, 35))

    def is_person(self, e_out):
        R = max(self.recent_ref)
        return 10 * np.log10((e_out + 1e-14) / R) > self.leak() + self.margin_db


def to_pcm16(x):
    return (np.clip(x, -1.0, 1.0) * 32767).astype('<i2').tobytes()


def from_pcm16(b):
    return np.frombuffer(b, dtype='<i2').astype(np.float32) / 32768.0
