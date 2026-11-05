"""Listening models, both small enough to run on the CPU every few milliseconds.

SileroVAD: is someone speaking in this 32 ms frame?            (Silero VAD v6, ONNX)
SmartTurn: given the person's turn so far, have they finished? (Pipecat Smart Turn v3.2, ONNX)
VAPTurn:   hearing both voices, who talks next?               (Voice Activity Projection, ONNX)
"""
import numpy as np
import onnxruntime as ort

from .whisper_features import compute_whisper_log_mel_features

RATE = 16000


def _session(path, threads):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), sess_options=opts, providers=['CPUExecutionProvider'])


class SileroVAD:
    """Streaming speech probability for consecutive 512-sample (32 ms) frames."""
    FRAME = 512
    CONTEXT = 64

    def __init__(self, path):
        self.sess = _session(path, 1)
        self.reset()

    def reset(self):
        self.state = np.zeros((2, 1, 128), np.float32)
        self.context = np.zeros((1, self.CONTEXT), np.float32)

    def __call__(self, frame):
        x = np.concatenate([self.context, np.asarray(frame, np.float32)[None, :]], axis=1)
        out, self.state = self.sess.run(None, {'input': x, 'state': self.state,
                                               'sr': np.array(RATE, dtype=np.int64)})
        self.context = x[:, -self.CONTEXT:]
        return float(out.reshape(-1)[0])


class VAPTurn:
    """Who will be talking in 0.2-0.8 s, from BOTH voices: Voice Activity Projection, otoSpeech
    checkpoint (5.8M parameters; weights CC BY 4.0 viks66/VAP_checkpoints, architecture MIT Erik
    Ekstedt; ONNX export aufklarer/VAP-OTO-ONNX, made window-length free in models/vap/vap_dyn.onnx).

    On Dessa's 21 captioned demos (tools/eval_vap.py) it told the end of a turn from a pause far
    better than Smart Turn, which hears only the speaker: AUC 0.80-0.94 against 0.70-0.74 after
    0.1-0.5 s of silence, at 0.15 s per 6 s window on this laptop's processor (4 threads).
    """
    WINDOW = 6 * RATE

    def __init__(self, path, threads=2):
        self.sess = _session(path, threads)

    def __call__(self, person, agent):
        """(chance the agent is the one talking 0.4-0.8 s from now, ... 0-0.4 s from now)."""
        w = np.zeros((1, 2, self.WINDOW), np.float32)
        for k, x in enumerate((person, agent)):
            x = np.asarray(x, np.float32)[-self.WINDOW:]
            if len(x):
                w[0, k, self.WINDOW - len(x):] = x
        p_now, p_future, _ = self.sess.run(None, {'audio': w})
        return float(p_future[0, -1, 1]), float(p_now[0, -1, 1])


class VAPLive:
    """The same learned turn-taking model, running all the time instead of on request: every 20 ms of both
    voices goes through the streaming PyTorch version (app/vap_torch.py, 11-13 ms per step on the processor) in a
    background thread, so the newest answer is always ready and checking costs nothing.
    weights: optional re-trained state dict (tools/train_vap.py -> models/vap/vap_ft.pt)."""

    def __init__(self, onnx_path, weights=None, context_s=6.0, threads=2):
        import queue
        import threading

        import torch

        from .vap_torch import VAPStream, load_from_onnx
        torch.set_num_threads(threads)
        model = load_from_onnx(onnx_path)
        if weights:
            model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True))
        self.stream = VAPStream(model.eval(), context_s=context_s)
        self.q = queue.Queue()
        self.buf_p, self.buf_a = np.zeros(0, np.float32), np.zeros(0, np.float32)
        self.p_future, self.p_now, self.updated = 0.0, 0.0, 0.0
        self.step_ms = 0.0
        self.alive = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def push(self, person, agent):
        """Any amount of 16 kHz audio of both voices, in step with each other."""
        self.buf_p = np.concatenate([self.buf_p, np.asarray(person, np.float32)])
        self.buf_a = np.concatenate([self.buf_a, np.asarray(agent, np.float32)])
        while len(self.buf_p) >= 320 and len(self.buf_a) >= 320:
            self.q.put((self.buf_p[:320], self.buf_a[:320]))
            self.buf_p, self.buf_a = self.buf_p[320:], self.buf_a[320:]

    def _run(self):
        import time
        while self.alive:
            p, a = self.q.get()
            if p is None:
                break
            t0 = time.perf_counter()
            out = self.stream.step(p, a)
            self.step_ms = 0.9 * self.step_ms + 0.1 * 1000 * (time.perf_counter() - t0)
            self.p_future, self.p_now = float(out['p_future'][1]), float(out['p_now'][1])
            self.updated = time.monotonic()

    def latest(self):
        """(chance Alba should talk 0.6-2 s from now, ... 0-0.6 s from now, frames still waiting)."""
        return self.p_future, self.p_now, self.q.qsize()

    def close(self):
        self.alive = False
        self.q.put((None, None))


class SmartTurn:
    """Probability (0-1) that the speaker has finished their turn.

    Mirrors Pipecat's reference inference: last 8 s of audio, zero-padded at
    the start, Whisper log-mel features, one ONNX forward pass.
    """
    WINDOW = 8 * RATE

    def __init__(self, path, threads=2):
        self.sess = _session(path, threads)

    def __call__(self, audio):
        a = np.asarray(audio, np.float32)[-self.WINDOW:]
        if len(a) < self.WINDOW:
            a = np.pad(a, (self.WINDOW - len(a), 0))
        feats = compute_whisper_log_mel_features(a, do_normalize=True)[None]
        out = self.sess.run(None, {'input_features': feats})
        return float(np.asarray(out[0]).reshape(-1)[0])
