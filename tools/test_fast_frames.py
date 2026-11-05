"""Checks app/fast_audio.FastFrameSampler against the model's own sample_audio_frame: with the same reveal order and
greedy picks (top-k 1), both must choose the same 12 codes, with and without a repetition-penalty history; then
times both with the real settings (temperature 1, top-k 128). Writes results/fast_frames_check.json.

  .venv-train/Scripts/python.exe -I tools/test_fast_frames.py
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('NO_TORCH_COMPILE', '1')
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.fast_audio import FastFrameSampler  # noqa: E402


@torch.no_grad()
def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    from transformers import AutoModelForMultimodalLM
    hf = ROOT / 'models' / 'lfm3-audio-hf'
    model = AutoModelForMultimodalLM.from_pretrained(hf, dtype=torch.bfloat16, device_map='cpu',
                                                     trust_remote_code=True, local_files_only=True).eval()
    model.requires_grad_(False)
    model.model.language_model = torch.nn.Identity()
    model.model.depth_decoder.to('cuda')
    emb = model.model.audio_projector  # noqa: F841  (unused; keeps the structure explicit)
    n, V = model.config.num_codebooks, model.config.codebook_size
    fast = FastFrameSampler(model, temperature=1.0, top_k=1, repetition_penalty=1.1)
    g = torch.Generator(device='cuda').manual_seed(0)
    same, trials = 0, 30
    real_randperm = torch.randperm
    for t in range(trials):
        h = torch.randn(model.config.text_config.hidden_size, device='cuda', generator=g).to(torch.bfloat16) * 2
        order = torch.randperm(n, device='cuda', generator=g)
        hist = None if t % 3 == 0 else torch.randint(0, V - 1, (16, n), device='cuda', generator=g)
        torch.randperm = lambda *a, **k: order.clone()          # the model's sampler draws this order
        try:
            ref = model.sample_audio_frame(h, temperature=1.0, top_k=1, repetition_penalty=1.1, history=hist)
        finally:
            torch.randperm = real_randperm
        fast.reset()
        if hist is not None:
            for row in hist:
                fast.remember(row)
        fast.order_override = order
        got = fast.sample(h)
        same += int(torch.equal(ref.cpu(), got.cpu()))
    fast.order_override = None
    res = {'greedy_same_codes': f'{same}/{trials}'}
    real = FastFrameSampler(model, temperature=1.0, top_k=128, repetition_penalty=1.1)
    h = torch.randn(model.config.text_config.hidden_size, device='cuda').to(torch.bfloat16)
    hist = torch.randint(0, V - 1, (64, n), device='cuda')
    for row in hist:
        real.remember(row)
    for name, fn in (('model', lambda: model.sample_audio_frame(h, temperature=1.0, top_k=128, repetition_penalty=1.1,
                                                                history=hist)),
                     ('fast-graph', lambda: real.sample(h))):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(30):
            fn()
        torch.cuda.synchronize()
        res[f'{name}_ms'] = round((time.perf_counter() - t0) / 30 * 1000, 2)
    print(json.dumps(res))
    (ROOT / 'results' / 'fast_frames_check.json').write_text(json.dumps(res, indent=1), encoding='utf8')


if __name__ == '__main__':
    main()
