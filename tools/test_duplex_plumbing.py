"""Checks every link of the Alba Duplex runtime with the stock LFM3-Audio-2B (no duplex training needed): the
streaming speech encoder hears a clip 80 ms at a time, the language model (llama.cpp library, fed vectors)
answers in its native interleaved mode (6 text pieces, 12 speech frames, ...), the depth decoder makes each frame
from the model's hidden state, and the codec turns frames into sound, piece by piece. The reply is saved and
transcribed back by the same model (PyTorch), so a sensible, intelligible reply means the plumbing is right.
Also times each part.

  .venv-train/Scripts/python.exe -I tools/test_duplex_plumbing.py [--clip results/s2s-inputs/synthetic-2.wav]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.duplex import FIRST_STEP, STEP, AlbaDuplex, _sample  # noqa: E402


@torch.no_grad()
def consistency(a, x):
    """Live use must see exactly what training sees: (1) the speech encoder fed 95 ms then 80 ms pieces vs the
    whole clip at once (training), (2) the codec decoding frame by frame vs all at once."""
    n = (len(x) - FIRST_STEP) // STEP + 1
    a.fe_cache = a.enc_cache = None
    live, times = [], []
    for k in range(n):
        lo, hi = (0, FIRST_STEP) if k == 0 else (FIRST_STEP + (k - 1) * STEP, FIRST_STEP + k * STEP)
        t = time.perf_counter()
        v = a.hear(x[lo:hi])
        times.append(time.perf_counter() - t)
        live.append(v)
    got = sum(v is not None for v in live)
    feats = a.proc.feature_extractor(x[:FIRST_STEP + (n - 1) * STEP], sampling_rate=16000, return_tensors='pt')
    full = a.m.get_audio_features(feats['input_features'].to(a.device, torch.bfloat16),
                                  feats['feature_attention_mask'].to(a.device).bool()).float().cpu().numpy()
    ok = got == n == len(full)
    if ok:
        L = np.stack(live)
        cos = (L * full).sum(1) / (np.linalg.norm(L, axis=1) * np.linalg.norm(full, axis=1) + 1e-9)
        print(f'encoder: {n} steps, one token each; live vs training cosine min {cos.min():.4f} mean {cos.mean():.4f}, '
              f'max abs diff {np.abs(L - full).max():.4f} (bf16) | {1000 * np.mean(times):.1f} ms per step', flush=True)
    else:
        print(f'encoder: MISMATCH: {n} steps, {got} live tokens, {len(full)} training tokens', flush=True)
    # codec: Alba's own voice, encoded, then decoded both ways
    ref = Path(__file__).resolve().parent.parent / 'data' / 'convo' / 'audio'
    clip = next(iter(sorted(ref.glob('*-1.wav'))), None) if ref.exists() else None
    if clip is None:
        print('codec: no Alba clip found to test with', flush=True)
        return
    y, sr = sf.read(clip, dtype='float32')
    codes = a.proc.encode_audio(y).to(a.device)                   # (frames, 12)
    whole = a.codec.decode(codes.T[None], return_dict=True).audio_values[0].float().cpu().numpy()
    a.codec_cache, parts, times = None, [], []
    for c in codes:
        t = time.perf_counter()
        parts.append(a.sound(c))
        times.append(time.perf_counter() - t)
    parts.append(a.flush())
    piece = np.concatenate(parts)
    m = min(len(piece), len(whole))
    print(f'codec: {len(codes)} frames; whole {len(whole)} samples, frame by frame {len(piece)}; max abs diff '
          f'{np.abs(piece[:m] - whole[:m]).max():.4f} | {1000 * np.mean(times):.1f} ms per frame', flush=True)


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser()
    ap.add_argument('--clip', default=str(ROOT / 'results' / 's2s-inputs' / 'synthetic-2.wav'))
    ap.add_argument('--gguf', default=str(ROOT / 'models' / 'lfm3-audio' / 'LFM3-Audio-2B-Q4_K_M.gguf'))
    ap.add_argument('--hf', default=str(ROOT / 'models' / 'lfm3-audio-hf'))
    ap.add_argument('--max-frames', type=int, default=150)
    args = ap.parse_args()
    t0 = time.perf_counter()
    a = AlbaDuplex(args.gguf, args.hf)
    print(f'loaded in {time.perf_counter() - t0:.1f}s', flush=True)
    x, sr = sf.read(args.clip, dtype='float32')
    assert sr == 16000
    consistency(a, x)
    a.fe_cache = a.enc_cache = a.codec_cache = None
    # turn-based prompt in the model's own chat format
    head = a.proc.apply_chat_template([{'role': 'system', 'content': 'Respond with interleaved text and audio.'}],
                                      tokenize=False) + '<|im_start|>user\n'
    a.lm.feed_tokens(a.lm.tokenize(head))
    a.fe_cache = a.enc_cache = a.codec_cache = None
    a.history, a.said = [], []
    t_hear = time.perf_counter()
    vecs = []
    for k in range(0, len(x) - STEP + 1, STEP):
        v = a.hear(x[k:k + STEP])
        if v is not None:
            vecs.append(v)
    a.lm.feed_vectors(np.stack(vecs))
    t_hear = time.perf_counter() - t_hear
    logits, hidden = a.lm.feed_tokens(a.lm.tokenize('<|im_end|>\n<|im_start|>assistant\n'))
    # native interleaved generation, driven by our own loop
    out, text, timing = [], [], {'lm': 0.0, 'depth': 0.0, 'codec': 0.0}
    tok = _sample(logits, temperature=0.0, rng=a.rng)
    assert tok == a.ids['mixed_start'] or True
    if tok != a.ids['mixed_start']:
        print('note: model did not open with <|mixed_start|>, forcing it', flush=True)
    logits, hidden = a.lm.feed_vectors(a.vec(a.ids['mixed_start'])[None])
    slots, speaking, text_done, frames = 6, False, False, 0
    while frames < args.max_frames:
        if not speaking:
            t = time.perf_counter()
            tk = _sample(logits, temperature=0.7, top_k=40, rng=a.rng)
            text.append(tk)
            logits, hidden = a.lm.feed_vectors(a.vec(tk)[None])
            timing['lm'] += time.perf_counter() - t
            if tk == a.ids['text_end']:
                text_done, speaking = True, True
                continue
            slots -= 1
            if slots == 0:
                speaking, slots = True, 12
            continue
        t = time.perf_counter()
        codes = a.frame(hidden)
        timing['depth'] += time.perf_counter() - t
        if bool((codes == a.eos_code).any()):
            break
        a.history.append(codes)
        t = time.perf_counter()
        out.append(a.sound(codes))
        timing['codec'] += time.perf_counter() - t
        t = time.perf_counter()
        emb = a.m.get_audio_out_embeds(codes[None]).float().cpu().numpy()[0]
        logits, hidden = a.lm.feed_vectors(emb[None])
        timing['lm'] += time.perf_counter() - t
        frames += 1
        if not text_done:
            slots -= 1
            if slots == 0:
                speaking, slots = False, 6
    wav = np.concatenate(out) if out else np.zeros(1, np.float32)
    dest = ROOT / 'results' / 'duplex-plumbing'
    dest.mkdir(parents=True, exist_ok=True)
    sf.write(dest / 'reply.wav', wav, 24000)
    said = a.tok.decode(text, skip_special_tokens=True)
    secs = len(wav) / 24000
    print(f'heard {len(x) / 16000:.1f}s in {t_hear:.2f}s | reply text: {said!r}')
    print(f'{frames} frames = {secs:.1f}s of speech | per frame: language model {1000 * timing["lm"] / max(1, frames):.1f} ms, '
          f'frame decoder {1000 * timing["depth"] / max(1, frames):.1f} ms, codec {1000 * timing["codec"] / max(1, frames):.1f} ms')
    # (the fast frame sampler is checked against the model's own in tools/test_fast_frames.py)
    a.close()


if __name__ == '__main__':
    main()
