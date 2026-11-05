"""Replays a recorded speaker-mode call through the same echo handling and talk-over
rule as the app, and prints when the voice would soften and stop.

  python -I replay_bargein.py <session_dir>
"""
import collections
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import Config                            # noqa: E402
from app.dsp import EchoCanceller, EchoGate, Resampler   # noqa: E402
from app.listen import SileroVAD                         # noqa: E402

S = Path(sys.argv[1])
cfg = Config()
T = cfg.timing
mic, fs = sf.read(S / 'person.flac', dtype='float32')
out, _ = sf.read(S / 'voice.flac', dtype='float32')
rm, rr = Resampler(fs, 16000), Resampler(fs, 16000)
aec, gate, vad = EchoCanceller(), EchoGate(margin_db=T.echo_margin_db), SileroVAD(cfg.paths.vad)
removed = collections.deque(maxlen=500)
hist = collections.deque(maxlen=28)
voice_recent = collections.deque(maxlen=10)
buf = {k: np.zeros(0, np.float32) for k in 'mre'}
active, ducked, n_frames, log = False, False, 0, []
fsec = 512 / 16000
for i in range(0, min(len(mic), len(out)) - 480, 480):
    m, r = rm(mic[i:i + 480]), rr(out[i:i + 480])
    a, _ = aec(m, r, adapt=not active)
    if float(np.dot(r, r)) > 1e-6 * len(r) and not active:
        removed.append(10 * np.log10((np.dot(m, m) + 1e-12) / (np.dot(a, a) + 1e-12)))
    e = a if (len(removed) >= 100 and np.median(removed) >= 3.0) else m
    for k, x in zip('mre', (m, r, e)):
        buf[k] = np.concatenate([buf[k], np.asarray(x, np.float32)])
    while len(buf['e']) >= 512:
        mm, rf, ee = (buf[k][:512] for k in 'mre')
        for k in 'mre':
            buf[k] = buf[k][512:]
        n_frames += 1
        t = n_frames * fsec
        p = vad(ee)
        gate.observe(float(np.mean(mm ** 2)), float(np.mean(rf ** 2)), float(np.mean(ee ** 2)))
        sp = p >= (T.vad_off if active else T.vad_on)
        if sp and gate.playing():
            sp = p >= 0.7 and gate.is_person(float(np.mean(ee ** 2)))
        active = sp
        voice_recent.append(float(np.sqrt(np.mean(rf ** 2))) > 0.003)
        if any(voice_recent):
            hist.append(sp)
            ev = sum(hist) * fsec
            if ev >= T.yield_after_s:
                log.append((round(t, 2), 'would stop'))
                hist.clear()
                ducked = False
            elif ev >= T.duck_after_s and not ducked:
                ducked = True
                log.append((round(t, 2), 'would soften'))
            elif ducked and not any(list(hist)[-10:]):
                ducked = False
                log.append((round(t, 2), 'back to full'))
        else:
            hist.clear()
            ducked = False
print('echo canceller used:', bool(len(removed) >= 100 and np.median(removed) >= 3.0), f'(it removes {np.median(removed):.1f} dB)')
print(log)
