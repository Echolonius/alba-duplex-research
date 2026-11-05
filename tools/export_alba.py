"""Turns a trained Alba LoRA into a model the app can run fast: merge into LFM3-Audio-2B (on the processor,
bf16), save it as a plain checkpoint, then convert with llama.cpp's official converter (tools/llama-src, the
same build as the app's server, b12094) to GGUF (language model + the audio side "mmproj"), and quantize the
language model to Q4_K_M like the official release.

  .venv-train/Scripts/python.exe -I tools/export_alba.py --lora models/alba-lora --out models/alba-m2
  .venv-train/Scripts/python.exe -I tools/export_alba.py --base-only --out models/lfm3-convert-check
      (converts the untouched base model, to check our pipeline reproduces the official files)
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('NO_TORCH_COMPILE', '1')
import torch

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / 'models' / 'lfm3-audio-hf'
CONVERT = ROOT / 'tools' / 'llama-src' / 'llama.cpp-b12094' / 'convert_hf_to_gguf.py'
QUANTIZE = ROOT / 'bin' / 'llama-b12094' / 'llama-quantize.exe'


def run(cmd):
    print('>', ' '.join(str(c) for c in cmd), flush=True)
    env = {**os.environ, 'PYTHONUTF8': '1'}
    subprocess.run([str(c) for c in cmd], check=True, env=env)


def to_release_format(hf):
    """Rewrites the saved weights in the release's own layout, which llama.cpp's converter reads: the same tensor
    names (this transformers version leaves a stray '$' on 'mask_embedding'), the 12 tied depth-decoder output
    tables (not saved, never trained) taken from the release, the release's number formats; checked name by name
    and shape by shape against the release, then one model.safetensors."""
    import json
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file
    saved = {}
    for f in sorted(hf.glob('*.safetensors')):
        saved.update(load_file(str(f)))
    saved = {k.rstrip('$'): v for k, v in saved.items()}
    out, changed = {}, 0
    with safe_open(str(BASE / 'model.safetensors'), 'pt') as rel:   # read one release tensor at a time
        names = list(rel.keys())
        missing = sorted(set(names) - set(saved))
        extra = sorted(set(saved) - set(names))
        if extra or any('to_logits' not in k for k in missing):
            raise SystemExit(f'cannot match the release layout: extra {extra[:5]}, missing {missing[:5]}')
        for k in names:
            v = rel.get_tensor(k)
            w = saved.pop(k, None)
            if w is None:
                out[k] = v
                continue
            if w.shape != v.shape:
                raise SystemExit(f'{k}: shape {tuple(w.shape)} vs release {tuple(v.shape)}')
            out[k] = w.to(v.dtype).contiguous()
            changed += int(not torch.equal(out[k], v))
    for f in hf.glob('*.safetensors'):
        f.unlink()
    (hf / 'model.safetensors.index.json').unlink(missing_ok=True)
    save_file(out, str(hf / 'model.safetensors'), metadata={'format': 'pt'})
    (hf / 'EXPORT.json').write_text(json.dumps({'tensors': len(out), 'from_release_untouched': missing,
                                                'changed_by_training': changed}, indent=1), encoding='utf8')
    print(f'release layout: {len(out)} tensors, {changed} changed by training, {len(missing)} tied ones from the '
          f'release', flush=True)


# release names -> the model's own names (the model code's register_checkpoint_conversion_mapping, in order)
RENAMES = [(r'^lfm\.model\.', 'model.language_model.'), (r'^lfm\.lm_head\.', 'lm_head.'),
           (r'^encoder\.downsample\.', 'model.audio_encoder.downsample.'),
           (r'^encoder\.lfm\.lfm\.', 'model.audio_encoder.lfm.'),
           (r'^mlp\.model\.0\.', 'model.audio_projector.layer_norm.'),
           (r'^mlp\.model\.1\.', 'model.audio_projector.linear_1.'),
           (r'^mlp\.model\.3\.', 'model.audio_projector.linear_2.'),
           (r'^codebook_embedding\.', 'model.codebook_embedding.'),
           (r'^depth_linear\.', 'model.depth_decoder.depth_linear.'),
           (r'^mask_embedding$', 'model.depth_decoder.mask_embedding'),
           (r'^depthformer\.model\.', 'model.depth_decoder.model.'),
           (r'^depth_embeddings\.', 'model.depth_decoder.depth_embeddings.')]
EMBED = 'lfm.model.embed_tokens.weight'


def internal_name(k):
    import re
    for pat, rep in RENAMES:
        if re.search(pat, k):
            return re.sub(pat, rep, k)
    return k


def slim(out, hf):
    """After conversion the language model lives in the GGUF. What else training changed (the depth decoder,
    and the rows of the word table it touched, e.g. LISTEN) goes into one small trained.safetensors in the
    model's own names; the duplex runtime loads the untouched release for the rest. Then hf/ is removed
    (it was a 4 GB full copy, made only for the converter)."""
    import json
    from safetensors import safe_open
    from safetensors.torch import save_file
    keep, rows = {}, None
    with safe_open(str(hf / 'model.safetensors'), 'pt') as new, safe_open(str(BASE / 'model.safetensors'), 'pt') as rel:
        for k in new.keys():
            a, b = new.get_tensor(k), rel.get_tensor(k)
            if torch.equal(a, b):
                continue
            if k == EMBED:
                idx = torch.nonzero((a != b).any(dim=1)).flatten()
                rows = (idx, a[idx].clone())
            elif not k.startswith('lfm.'):
                keep[internal_name(k)] = a.contiguous()
    if rows is not None:
        keep['embed_rows.idx'], keep['embed_rows.val'] = rows[0].to(torch.int64), rows[1].contiguous()
    save_file(keep, str(out / 'trained.safetensors'), metadata={'format': 'pt', 'base': 'models/lfm3-audio-hf'})
    (out / 'trained.json').write_text(json.dumps({'tensors': sorted(k for k in keep if not k.startswith('embed_rows')),
                                                  'embed_rows': [] if rows is None else rows[0].tolist()},
                                                 indent=1), encoding='utf8')
    shutil.rmtree(hf)
    print(f'slim: {len(keep)} trained tensors (+ {0 if rows is None else len(rows[0])} word rows) in '
          f'trained.safetensors; the full copy removed', flush=True)


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser()
    ap.add_argument('--lora', default='')
    ap.add_argument('--out', required=True)
    ap.add_argument('--base-only', action='store_true')
    ap.add_argument('--keep-f16', action='store_true')
    ap.add_argument('--keep-hf', action='store_true', help='keep the full 4 GB merged copy (default: slim export)')
    args = ap.parse_args()
    out = Path(args.out)
    hf = out / 'hf'
    out.mkdir(parents=True, exist_ok=True)
    if args.base_only:
        src = BASE
    else:
        from transformers import AutoModelForMultimodalLM
        from peft import PeftModel
        model = AutoModelForMultimodalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map='cpu',
                                                         trust_remote_code=True, local_files_only=True)
        model = PeftModel.from_pretrained(model, args.lora).merge_and_unload()
        hf.mkdir(parents=True, exist_ok=True)
        # this transformers version refuses to save the release's own generation settings (temperature 0 without
        # sampling): save without them, then the original file is copied over unchanged below
        from transformers import GenerationConfig
        model.generation_config = GenerationConfig()
        model.save_pretrained(hf, safe_serialization=True, max_shard_size='5GB')
        (hf / 'generation_config.json').unlink(missing_ok=True)
        del model                       # one copy of the model in memory at a time
        import gc
        gc.collect()
        to_release_format(hf)
        # the processor, tokenizer, codec, config and the reviewed model code travel with the weights, unchanged
        for p in BASE.iterdir():
            if p.name in ('model.safetensors', 'MANIFEST.json') or p.suffix == '.bad':
                continue
            dst = hf / p.name
            if p.is_dir():
                if not dst.exists():
                    shutil.copytree(p, dst)
            elif not dst.exists() or p.name == 'config.json':
                shutil.copy2(p, dst)
        src = hf
        if (Path(args.lora) / 'duplex.json').exists():       # the duplex timing settings travel with the model
            shutil.copy2(Path(args.lora) / 'duplex.json', out / 'duplex.json')
    f16 = out / 'model-f16.gguf'
    mm = out / 'mmproj-f16.gguf'
    # -E -s (not -I): the converter imports its own 'conversion' package from its folder (official source)
    run([sys.executable, '-E', '-s', CONVERT, src, '--outfile', f16, '--outtype', 'f16'])
    run([sys.executable, '-E', '-s', CONVERT, src, '--outfile', mm, '--outtype', 'f16', '--mmproj'])
    q4 = out / 'model-Q4_K_M.gguf'
    run([QUANTIZE, f16, q4, 'Q4_K_M'])
    # the tools can report success on a file a full disk cut short (seen 5 Nov): check every output
    for p, least in ((f16, 2.5e9), (mm, 4e8), (q4, 9e8)):
        with open(p, 'rb') as fh:
            magic = fh.read(4)
        if magic != b'GGUF' or p.stat().st_size < least:
            raise SystemExit(f'{p.name} is broken ({p.stat().st_size} bytes, header {magic!r}): check free disk space')
    if not args.keep_f16:
        f16.unlink()
    if not args.base_only and not args.keep_hf:
        slim(out, hf)
    print('done:', q4, mm, flush=True)


if __name__ == '__main__':
    main()
