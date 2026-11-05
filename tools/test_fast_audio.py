"""Checks app/fast_audio.FastEncoderStream against the library's own streaming encoder (the path the model was
checked with), token by token, on a real clip, and times both. Writes results/fast_audio_check.json.

  .venv-train/Scripts/python.exe -I tools/test_fast_audio.py
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('NO_TORCH_COMPILE', '1')
import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.fast_audio import FastEncoderStream  # noqa: E402

FIRST, STEP = 1520, 1280


@torch.no_grad()
def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    from transformers import AutoModelForMultimodalLM, AutoProcessor
    hf = ROOT / 'models' / 'lfm3-audio-hf'
    proc = AutoProcessor.from_pretrained(hf, trust_remote_code=True, local_files_only=True)
    model = AutoModelForMultimodalLM.from_pretrained(hf, dtype=torch.bfloat16, device_map='cpu',
                                                     trust_remote_code=True, local_files_only=True).eval()
    model.requires_grad_(False)
    m = model.model
    m.language_model = torch.nn.Identity()
    m.audio_encoder.to('cuda')
    m.audio_projector.to('cuda')
    x, sr = sf.read(ROOT / 'results' / 's2s-inputs' / 'synthetic-2.wav', dtype='float32')
    x = np.concatenate([x, x])                                  # long enough to fill the 32-token window twice
    n = (len(x) - FIRST) // STEP + 1
    chunks = [x[0:FIRST]] + [x[FIRST + (k - 1) * STEP:FIRST + k * STEP] for k in range(1, n)]
    res = {}
    for name in ('library', 'fast-eager', 'fast-graph'):
        fe = None
        enc = None
        fast = None if name == 'library' else FastEncoderStream(model, use_graph=(name == 'fast-graph'))
        toks, times = [], []
        for c in chunks:
            feats, fe = proc.feature_extractor.stream(c, fe)
            xf = feats['input_features']
            torch.cuda.synchronize()
            t = time.perf_counter()
            if fast is None:
                tok, enc = m.get_audio_features_stream(xf.to('cuda', torch.bfloat16), enc)
                tok = tok[0, -1]
            else:
                tok = fast.push(xf)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t)
            toks.append(tok.float().cpu().numpy())
        res[name] = (np.stack(toks), np.array(times))
    ref = res['library'][0]
    out = {'steps': n}
    for name in ('fast-eager', 'fast-graph'):
        L = res[name][0]
        cos = (L * ref).sum(1) / (np.linalg.norm(L, axis=1) * np.linalg.norm(ref, axis=1) + 1e-9)
        out[name] = {'cosine_min': round(float(cos.min()), 5), 'cosine_mean': round(float(cos.mean()), 5),
                     'max_abs_diff': round(float(np.abs(L - ref).max()), 4)}
    for name in res:
        t = res[name][1][3:]
        out.setdefault(name, {})['ms_per_step'] = round(float(t.mean() * 1000), 2)
    print(json.dumps(out, indent=1))
    (ROOT / 'results' / 'fast_audio_check.json').write_text(json.dumps(out, indent=1), encoding='utf8')


if __name__ == '__main__':
    main()
