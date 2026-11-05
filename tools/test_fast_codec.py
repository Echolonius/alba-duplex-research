"""Checks app/fast_audio.FastCodecDecoder against the codec's own streaming decode (LiquidCodecModel.decode with a
LiquidCodecStreamingCache, as app/duplex.py used it), frame by frame on a real clip of Alba's voice, including the
first frame and the final flush; then times both. Writes results/fast_codec_check.json.

  .venv-train/Scripts/python.exe -I tools/test_fast_codec.py
"""
import json
import os
import sys
import time
from importlib import import_module
from pathlib import Path

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('NO_TORCH_COMPILE', '1')
import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.fast_audio import FastCodecDecoder  # noqa: E402


@torch.no_grad()
def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(ROOT / 'models' / 'lfm3-audio-hf', trust_remote_code=True, local_files_only=True)
    codec = proc.audio_tokenizer.to('cuda').eval()
    codec.requires_grad_(False)
    cache_cls = getattr(import_module(type(codec).__module__), 'LiquidCodecStreamingCache')
    clip = sorted((ROOT / 'data' / 'convo' / 'audio').glob('*-1.wav'))[3]
    y, sr = sf.read(clip, dtype='float32')
    codes = proc.encode_audio(y).to('cuda')                            # (frames, 12)
    res, outs = {'frames': len(codes), 'clip_rate': sr, 'codec_dtype': str(next(codec.parameters()).dtype)}, {}
    for name in ('library', 'fast-eager', 'fast-graph'):
        parts, times = [], []
        if name == 'library':
            cache = cache_cls(codec.config)
            for c in codes:
                torch.cuda.synchronize(); t = time.perf_counter()
                o = codec.decode(c.view(1, -1, 1), streaming_cache=cache, finalize=False, return_dict=True).audio_values[0]
                torch.cuda.synchronize(); times.append(time.perf_counter() - t)
                parts.append(o.float().cpu().numpy())
            empty = torch.zeros(1, codec.config.n_q, 0, dtype=torch.long, device='cuda')
            parts.append(codec.decode(empty, streaming_cache=cache, finalize=True, return_dict=True).audio_values[0].float().cpu().numpy())
        else:
            fast = FastCodecDecoder(codec, use_graph=(name == 'fast-graph'))
            for c in codes:
                torch.cuda.synchronize(); t = time.perf_counter()
                o = fast.frame(c)
                torch.cuda.synchronize(); times.append(time.perf_counter() - t)
                parts.append(o.float().cpu().numpy())
            parts.append(fast.flush().float().cpu().numpy())
        outs[name] = np.concatenate(parts)
        res[name] = {'samples': int(len(outs[name])), 'first_frame_samples': int(len(parts[0])),
                     'ms_per_frame': round(float(np.mean(times[3:]) * 1000), 2)}
    ref = outs['library']
    for name in ('fast-eager', 'fast-graph'):
        a = outs[name]
        m = min(len(a), len(ref))
        res[name]['max_abs_diff'] = round(float(np.abs(a[:m] - ref[:m]).max()), 5)
        res[name]['snr_db'] = round(float(10 * np.log10(np.sum(ref[:m] ** 2) / (np.sum((a[:m] - ref[:m]) ** 2) + 1e-12))), 1)
    print(json.dumps(res, indent=1))
    (ROOT / 'results' / 'fast_codec_check.json').write_text(json.dumps(res, indent=1), encoding='utf8')


if __name__ == '__main__':
    main()
