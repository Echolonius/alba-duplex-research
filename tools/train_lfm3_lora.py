"""Step 3: teach LFM3-Audio-2B to be Alba, end to end in audio (LoRA, on this laptop's 4 GB graphics card).

The model hears the person's voice (no transcript in between) and answers in interleaved text and speech:
six text tokens, then twelve 80 ms speech frames, and so on, then the rest of the speech, the same layout it
generates in. Both are trained at once: the words (text loss) and Alba's voice (audio loss on the Liquid
codec's 12 codes per frame), as in Liquid's official LFM3-Audio speech fine-tuning recipe (LoRA on the
language model and the speech decoder; the hearing encoder stays frozen).

To fit 4 GB, the language model's weights are held in 4 bits (bitsandbytes NF4, "QLoRA"); the small parts
stay in bf16; batches of one conversation with gradient accumulation; gradient checkpointing.

Weights: models/lfm3-audio-hf (LiquidAI/LFM3-Audio-2B, pinned revision, checked by tools/hf_fetch.py, LFM Open
License v1.0). Its model code ships with the weights; it was read before use (standard model code: no network,
no files written, no other programs) and is loaded only from that local copy, offline.

  .venv-train/Scripts/python.exe -I tools/train_lfm3_lora.py --steps 2000 [--accum 8] [--lr 2e-4] [--out models/alba-lora]
"""
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('NO_TORCH_COMPILE', '1')
import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / 'models' / 'lfm3-audio-hf'
DATA = ROOT / 'data' / 'convo'
N_TEXT, N_AUDIO = 6, 12


def load(quantize=True, device='cuda'):
    from transformers import AutoModelForMultimodalLM, AutoProcessor, BitsAndBytesConfig
    proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True, local_files_only=True)
    kw = dict(dtype=torch.bfloat16, device_map=device, trust_remote_code=True, local_files_only=True)
    if quantize:
        kw['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=['audio_encoder', 'audio_projector', 'depth_decoder', 'lm_head', 'codebook_embedding'])
    model = AutoModelForMultimodalLM.from_pretrained(MODEL, **kw)
    return proc, model


class Tokens:
    def __init__(self, proc, model):
        c, tok = model.config, proc.tokenizer
        self.tok = tok
        self.mixed_start = c.mixed_start_token_id
        self.text_end = c.text_end_token_id
        self.audio_out = c.audio_out_token_id
        self.im_end = tok.convert_tokens_to_ids('<|im_end|>')
        self.newline = tok('\n', add_special_tokens=False)['input_ids']
        self.eos_code = proc.audio_eos_code


def interleave(T, text, frames):
    """Assistant reply -> (ids, labels, codes): <|mixed_start|> [6 text] [12 frames] ... <|text_end|> [rest of
    the frames] [end-of-speech frame] <|im_end|>. None if the speech runs out before the text (rare)."""
    text_ids = T.tok(text, add_special_tokens=False)['input_ids'] + [T.text_end]
    ids, labels, codes = [T.mixed_start], [T.mixed_start], []
    ti = fi = 0
    while True:
        chunk = text_ids[ti:ti + N_TEXT]
        ids += chunk; labels += chunk; ti += len(chunk)
        if T.text_end in chunk:
            break
        if fi + N_AUDIO > len(frames):
            return None
        ids += [T.audio_out] * N_AUDIO; labels += [-100] * N_AUDIO
        codes.append(frames[fi:fi + N_AUDIO]); fi += N_AUDIO
    tail = np.concatenate([frames[fi:], np.full((1, frames.shape[1]), T.eos_code, frames.dtype)])
    ids += [T.audio_out] * len(tail); labels += [-100] * len(tail); codes.append(tail)
    end = [T.im_end] + T.newline
    ids += end; labels += end
    return ids, labels, np.concatenate(codes)


def codes_for(proc, wav_path, cache_dir):
    """Liquid codec codes of one of Alba's clips (24 kHz), cached as .npy."""
    out = cache_dir / (wav_path.stem + '.npy')
    if out.exists():
        return np.load(out)
    x, sr = sf.read(wav_path, dtype='float32')
    if sr != 24000:
        import librosa
        x = librosa.resample(x, orig_sr=sr, target_sr=24000)
    c = proc.encode_audio(np.clip(np.nan_to_num(x), -1, 1)).numpy().astype(np.int64)
    np.save(out, c)
    return c


def samples_from(conv, audio_dir):
    """One training sample per Alba turn: the earlier exchanges as text, the person's last line as audio."""
    out, hist = [], []
    turns = conv['turns']
    for k in range(1, len(turns)):
        if turns[k]['role'] == 'alba' and turns[k - 1]['role'] == 'person':
            out.append({'id': f"{conv['id']}-{k}", 'system': conv['system'] + (f"\n{conv['clock']}" if conv.get('clock') else ''),
                        'history': list(hist), 'user_audio': audio_dir / turns[k - 1]['audio'],
                        'reply': turns[k]['text'], 'reply_audio': audio_dir / turns[k]['audio']})
            hist += [{'role': 'user', 'content': turns[k - 1]['text']}, {'role': 'assistant', 'content': turns[k]['text']}]
    return out


def build(proc, T, s, cache_dir, device):
    user, sr = sf.read(s['user_audio'], dtype='float32')
    msgs = [{'role': 'system', 'content': s['system']}, *s['history'],
            {'role': 'user', 'content': [{'type': 'audio', 'audio': user}]}]
    prompt = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    enc = proc(text=prompt, audio=[user], return_tensors='pt', add_special_tokens=False, sampling_rate=16000)
    frames = codes_for(proc, s['reply_audio'], cache_dir)
    lay = interleave(T, s['reply'], frames)
    if lay is None:
        return None
    ids, labels, codes = lay
    p = enc['input_ids'][0].tolist()
    batch = {'input_ids': torch.tensor([p + ids]), 'attention_mask': torch.ones(1, len(p) + len(ids), dtype=torch.long),
             'labels': torch.tensor([[-100] * len(p) + labels]),
             'audio_out_codes': torch.from_numpy(codes), 'audio_labels': torch.from_numpy(codes.copy())}
    for k in ('input_features', 'feature_attention_mask'):
        if k in enc:
            batch[k] = enc[k]
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=2000, help='optimizer steps')
    ap.add_argument('--accum', type=int, default=8)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--rank', type=int, default=16)
    ap.add_argument('--out', default=str(ROOT / 'models' / 'alba-lora'))
    ap.add_argument('--data', default=str(DATA / 'audio.jsonl'))
    ap.add_argument('--max-tokens', type=int, default=900)
    ap.add_argument('--check', action='store_true', help='build a few samples and run one step, then stop')
    args = ap.parse_args()
    torch.manual_seed(0)
    from peft import LoraConfig, get_peft_model
    proc, model = load()
    print({n: type(m).__name__ for n, m in model.named_children()}, flush=True)
    T = Tokens(proc, model)
    convs = [json.loads(line) for line in open(args.data, encoding='utf8') if line.strip()]
    samples = [s for c in convs for s in samples_from(c, DATA / 'audio')]
    random.Random(0).shuffle(samples)
    n_held = min(40, max(1, len(samples) // 10))
    held = samples[:n_held]
    train = samples[n_held:]
    (Path(args.out)).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / 'held_out.json').write_text(json.dumps([str(s['id']) for s in held]), encoding='utf8')
    cache = DATA / 'codes'
    cache.mkdir(exist_ok=True)
    print(f'{len(train)} training samples, {len(held)} held out', flush=True)
    # (peft's prepare_model_for_kbit_training would copy every bf16 part to fp32: too big for 4 GB)
    for p in model.parameters():
        p.requires_grad_(False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads()
    cfg = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, bias='none',
                     target_modules=r"model\.(language_model|depth_decoder)\..*\.(q_proj|k_proj|v_proj|o_proj|in_proj|out_proj|w1|w2|w3)")
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))
    warm = max(10, args.steps // 30)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps)))))
    model.train()
    step, micro, t0 = 0, 0, time.time()
    log = {'text': [], 'audio': []}
    order = iter(())
    while step < args.steps:
        try:
            s = next(order)
        except StopIteration:
            random.shuffle(train)
            order = iter(train)
            s = next(order)
        b = build(proc, T, s, cache, 'cuda')
        if b is None or b['input_ids'].shape[1] > args.max_tokens:
            continue
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = model(**b)
        (out.loss / args.accum).backward()
        log['text'].append(float(out.text_loss)); log['audio'].append(float(out.audio_loss))
        micro += 1
        if micro % args.accum == 0:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % 10 == 0 or args.check:
                el = time.time() - t0
                print(f"step {step}/{args.steps} | text loss {np.mean(log['text'][-80:]):.3f} | audio loss "
                      f"{np.mean(log['audio'][-80:]):.3f} | {el / step:.1f}s/step | gpu peak "
                      f"{torch.cuda.max_memory_allocated() / 1e9:.2f} GB", flush=True)
            if step % 250 == 0 or step == args.steps:
                model.save_pretrained(args.out)
            if args.check:
                break
    model.save_pretrained(args.out)
    print('saved', args.out, flush=True)


if __name__ == '__main__':
    main()
