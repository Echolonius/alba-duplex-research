"""Audio-native baseline: Liquid's LFM2.5-Audio-1.5B hears a spoken turn and answers in speech, in one model
(official liquid-audio package, weights in models/lfm25-audio, checked by tools/hf_fetch.py; runs offline).

Inputs come from tools/make_s2s_inputs.py (synthetic lines plus real turns from the user's calls, local only).
For each it measures how soon the first audio of the reply is ready, how fast speech is generated compared
with real time, and saves the reply (text and audio) to results/lfm25-audio/.

  .venv-train/Scripts/python.exe -I tools/try_lfm25_audio.py [--device cuda|cpu] [--persona] [--tag NAME]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('NO_TORCH_COMPILE', '1')     # torch.compile needs Triton, which has no official Windows build
import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parent.parent
PERSONA = ("You are Alba, a witty, warm friend on a voice call. Talk like a real person: short spoken replies, "
           "your own opinions, no assistant talk. Respond with interleaved text and audio.")


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--persona', action='store_true', help="Alba's instructions instead of the model card's own")
    ap.add_argument('--threads', type=int, default=6)
    ap.add_argument('--tag', default='')
    ap.add_argument('--max-new', type=int, default=600)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    from liquid_audio import ChatState, LFM2AudioModel, LFM2AudioProcessor
    inputs = ROOT / 'results' / 's2s-inputs'
    items = json.loads((inputs / 'inputs.json').read_text(encoding='utf8'))
    out = ROOT / 'results' / 'lfm25-audio' / (args.tag or f"{args.device}{'-persona' if args.persona else ''}")
    out.mkdir(parents=True, exist_ok=True)
    model_dir = ROOT / 'models' / 'lfm25-audio'
    dtype = torch.bfloat16 if args.device == 'cuda' else torch.float32
    t = time.perf_counter()
    proc = LFM2AudioProcessor.from_pretrained(model_dir, device=args.device).eval()
    model = LFM2AudioModel.from_pretrained(model_dir, device=args.device, dtype=dtype).eval()
    mimi = proc.mimi.eval()
    with mimi.streaming(1), torch.no_grad():               # warm up the speech decoder, as the official demo does
        for _ in range(5):
            mimi.decode(torch.randint(2048, (1, 8, 1), device=args.device))
    print(f'loaded on {args.device} in {time.perf_counter() - t:.0f}s', flush=True)
    if args.device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    results = []
    for k, it in enumerate(items):
        wav, rate = sf.read(inputs / it['file'], dtype='float32')
        chat = ChatState(proc, dtype=dtype)
        chat.new_turn('system')
        chat.add_text(PERSONA if args.persona else 'Respond with interleaved text and audio.')
        chat.end_turn()
        chat.new_turn('user')
        chat.add_audio(torch.tensor(wav, dtype=torch.float32)[None], rate)
        chat.end_turn()
        chat.new_turn('assistant')
        text_tok, audio, first, first_text = [], [], None, None
        if args.device == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad(), mimi.streaming(1):
            for tok in model.generate_interleaved(**chat, max_new_tokens=args.max_new, audio_temperature=1.0, audio_top_k=4):
                if tok.numel() == 1:
                    text_tok.append(tok)
                    if first_text is None:
                        first_text = time.perf_counter() - t0
                elif tok.numel() == 8:
                    if (tok == 2048).any():
                        continue
                    chunk = mimi.decode(tok[None, :, None])[0]
                    if first is None:
                        if args.device == 'cuda':
                            torch.cuda.synchronize()
                        first = time.perf_counter() - t0
                    audio.append(chunk.float().cpu().numpy().reshape(-1))
        total = time.perf_counter() - t0
        reply = proc.text.decode(torch.cat(text_tok)) if text_tok else ''
        reply = reply.replace('<|text_end|>', '').strip()
        x = np.concatenate(audio) if audio else np.zeros(1, np.float32)
        secs = len(x) / 24000
        sf.write(out / f"reply-{Path(it['file']).stem}.wav", x, 24000)
        r = {'input': it['file'], 'kind': it['kind'], 'heard_text': it['text'], 'reply': reply,
             'first_audio_s': first, 'speech_s': secs, 'gen_s': total, 'x_real_time': secs / total if total else 0}
        results.append(r)
        print(f"[{it['file']}] first audio {first if first is not None else float('nan'):.2f}s | {secs:.1f}s of speech in "
              f"{total:.1f}s ({r['x_real_time']:.2f}x real time) | {reply[:150]}", flush=True)
    summary = {'device': args.device, 'dtype': str(dtype), 'persona': args.persona,
               'first_audio_median_s': float(np.median([r['first_audio_s'] for r in results if r['first_audio_s'] is not None])),
               'x_real_time_median': float(np.median([r['x_real_time'] for r in results]))}
    if args.device == 'cuda':
        summary['gpu_peak_gb'] = torch.cuda.max_memory_allocated() / 1e9
    print(json.dumps(summary), flush=True)
    (out / 'results.json').write_text(json.dumps({'summary': summary, 'replies': results}, indent=1, ensure_ascii=False),
                                      encoding='utf8')


if __name__ == '__main__':
    main()
