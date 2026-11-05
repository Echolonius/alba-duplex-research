"""Where the time goes in one Alba Duplex step: for the streaming speech encoder, the frame decoder and the codec,
the wall time per call vs the graphics card's own busy time (CUDA kernel time), and how many GPU calls each
makes. Wall >> kernel time means the processor-side overhead (Python, launching many tiny GPU calls) dominates,
which captured CUDA graphs or a leaner hand-written step remove.

  .venv-train/Scripts/python.exe -I tools/profile_duplex.py [--model DIR]     (writes results/profile_duplex.json)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.duplex import FIRST_STEP, STEP, AlbaDuplex  # noqa: E402


def measure(name, fn, n=30):
    from torch.profiler import ProfilerActivity, profile
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    wall = (time.perf_counter() - t) / n * 1000
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
    ev = prof.key_averages()
    kernel = sum(e.self_device_time_total for e in ev) / n / 1000
    launches = sum(e.count for e in ev if e.key.startswith(('cudaLaunchKernel', 'cuLaunchKernel'))) / n
    print(f'{name:14s} wall {wall:6.1f} ms | GPU kernels {kernel:6.1f} ms | {launches:5.0f} GPU launches per call', flush=True)
    return {'wall_ms': round(wall, 2), 'gpu_kernel_ms': round(kernel, 2), 'launches': round(launches)}


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='')
    args = ap.parse_args()
    if args.model:
        md = ROOT / args.model
        a = AlbaDuplex(md / 'model-Q4_K_M.gguf', md / 'hf')
    else:
        a = AlbaDuplex(ROOT / 'models' / 'lfm3-audio' / 'LFM3-Audio-2B-Q4_K_M.gguf', ROOT / 'models' / 'lfm3-audio-hf')
    a.start()
    x, sr = sf.read(ROOT / 'results' / 's2s-inputs' / 'synthetic-2.wav', dtype='float32')
    a.hear(x[:FIRST_STEP])
    pos = [FIRST_STEP]

    def hear():
        if pos[0] + STEP > len(x):
            a.fe_cache = a.enc_cache = None
            a.hear(x[:FIRST_STEP])
            pos[0] = FIRST_STEP
        a.hear(x[pos[0]:pos[0] + STEP])
        pos[0] += STEP

    h = np.random.default_rng(0).standard_normal(a.lm.n_embd).astype(np.float32)
    res = {'encoder_step': measure('encoder step', hear)}
    res['frame_sampler'] = measure('frame sampler', lambda: a.frame(h))
    codes = a.frame(h).clamp(max=2015)

    def sound():
        if a.codec_cache is not None and np.random.rand() < 0.02:
            a.flush()
        a.sound(codes)
    res['codec_frame'] = measure('codec frame', sound)
    vec = np.stack([a.vec(a.ids['listen']), h])
    t = time.perf_counter()
    for _ in range(30):
        a.lm.feed_vectors(vec)
    res['lm_two_vectors_ms'] = round((time.perf_counter() - t) / 30 * 1000, 2)
    print(f"language model (2 vectors, llama.cpp) {res['lm_two_vectors_ms']} ms", flush=True)
    out = ROOT / 'results' / 'profile_duplex.json'
    out.write_text(json.dumps(res, indent=1), encoding='utf8')
    a.close()


if __name__ == '__main__':
    main()
