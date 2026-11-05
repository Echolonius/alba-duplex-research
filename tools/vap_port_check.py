"""Checks app/vap_torch.py against models/vap/vap_dyn.onnx: the same window must give the same answers,
and the 20 ms streaming version must track a long window run.

  python -I tools/vap_port_check.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.vap_torch import VAPStream, load_from_onnx  # noqa: E402


def main():
    torch.set_num_threads(2)
    model = load_from_onnx(ROOT / 'models' / 'vap' / 'vap.onnx')
    sess = ort.InferenceSession(str(ROOT / 'models' / 'vap' / 'vap_dyn.onnx'), providers=['CPUExecutionProvider'])
    # a real call: the person and Alba, 16 kHz
    d = sorted(p for p in (ROOT / 'sessions').iterdir() if (p / 'person_clean16k.flac').exists())[-1]
    p, _ = sf.read(d / 'person_clean16k.flac', dtype='float32')
    a, _ = sf.read(d / 'voice16k.flac', dtype='float32')
    start = 16000 * 20
    win = np.stack([p[start:start + 6 * 16000], a[start:start + 6 * 16000]])[None].astype(np.float32)
    pn, pf, vad = sess.run(None, {'audio': win})
    with torch.no_grad():
        out = model(torch.from_numpy(win))
    for name, ref in (('p_now', pn), ('p_future', pf), ('vad', vad)):
        got = out[name].numpy()
        print(f'window: {name:8s} frames {got.shape[1]} vs {ref.shape[1]} | max diff {np.abs(got - ref).max():.2e}')
    # streaming over 30 s vs a window run over the same 30 s (identical start: both begin at `start`)
    secs = 30
    pp, aa = p[start:start + secs * 16000], a[start:start + secs * 16000]
    with torch.no_grad():
        full = model(torch.from_numpy(np.stack([pp, aa])[None].astype(np.float32)))
    st = VAPStream(model, context_s=60)
    t0 = time.perf_counter()
    outs = [st.step(pp[i:i + 320], aa[i:i + 320]) for i in range(0, len(pp) - 319, 320)]
    dt = (time.perf_counter() - t0) / len(outs)
    got = torch.stack([o['p_future'] for o in outs]).numpy()
    ref = full['p_future'][0].numpy()[:len(got)]
    print(f'stream vs 30 s window (same start, 60 s context): frames {len(got)} vs {full["p_future"].shape[1]}, '
          f'max diff {np.abs(got - ref).max():.2e}, {1000 * dt:.1f} ms per 20 ms step')
    # the app's case: 6 s of context, against 6 s ONNX windows ending at the same moments
    st = VAPStream(model, context_s=6)
    outs = [st.step(pp[i:i + 320], aa[i:i + 320]) for i in range(0, len(pp) - 319, 320)]
    diffs, pairs = [], []
    for t in range(400, len(outs), 37):
        e = (t + 1) * 320
        w = np.stack([pp[e - 96000:e], aa[e - 96000:e]])[None].astype(np.float32)
        r = sess.run(None, {'audio': w})[1][0, -1]
        diffs.append(np.abs(outs[t]['p_future'].numpy() - r).max())
        pairs.append((float(outs[t]['p_future'][1]), float(r[1])))
    pr = np.array(pairs)
    print(f'stream (6 s context) vs 6 s ONNX windows ending at the same frame: median diff {np.median(diffs):.3f}, '
          f'max {np.max(diffs):.3f}, correlation {np.corrcoef(pr[:, 0], pr[:, 1])[0, 1]:.3f} over {len(diffs)} moments')


if __name__ == '__main__':
    main()
