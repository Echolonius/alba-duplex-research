"""Where do Alba Duplex's speech frames lose quality? Teacher-forced check on one training-layout window: the
same tokens and frames go through (a) the exported llama.cpp model (as live) and (b) the stock llama.cpp model
(before any duplex training), and at every speech frame we score how likely the model's depth decoder finds the
TRUE frame codes given each hidden state (mean negative log-likelihood per code, greedy-revealed in the model's
own order). Lower is better; the stock model shows the starting point.

  .venv-train/Scripts/python.exe -I tools/check_frame_path.py --model models/alba-duplex2
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('NO_TORCH_COMPILE', '1')
import numpy as np
import soundfile as sf
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
import train_duplex as T  # noqa: E402
from app.duplex import AlbaDuplex  # noqa: E402


@torch.no_grad()
def frame_nll(a, hidden, codes):
    """Mean NLL of the true codes for one frame, revealing codebooks one pass at a time in a fixed order."""
    h = torch.from_numpy(hidden).to(a.device, a.dtype)[None]
    n = codes.shape[0]
    true = torch.as_tensor(codes, device=a.device).view(1, -1)
    revealed = torch.zeros(1, n, dtype=torch.bool, device=a.device)
    cur = torch.zeros(1, n, dtype=torch.long, device=a.device)
    nll, start = 0.0, 0
    order = torch.arange(n, device=a.device)
    for size in a.model.config.maskgit_schedule:
        pick = order[start:start + size]
        start += size
        logits = a.m.depth_decoder(h, cur, ~revealed)[0].float()
        nll += float(F.cross_entropy(logits[pick], true[0, pick], reduction='sum'))
        cur[0, pick] = true[0, pick]
        revealed[0, pick] = True
    return nll / n


def run(a, ids, frames, user_vecs, G, n_head):
    """Feeds the layout through the language model as live: tokens by their embedding rows, <|audio_in|> as the
    user's encoder vectors, <|audio_out|> as the frame embeddings; returns the NLL at every frame."""
    a.lm.rewind(0)
    a.lm.feed_tokens(ids[:n_head])
    ui = fi = 0
    out = []
    hidden = None
    for t in ids[n_head:]:
        if t == G.audio_in:
            v = user_vecs[ui]; ui += 1
        elif t == G.audio_out:
            out.append(frame_nll(a, hidden, frames[fi]))
            v = a.m.get_audio_out_embeds(torch.as_tensor(frames[fi], device=a.device)[None]).float().cpu().numpy()[0]
            fi += 1
        else:
            v = a.vec(t)
        _, hidden = a.lm.feed_vectors(v[None])
    return out


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    args = ap.parse_args()
    md = ROOT / args.model
    res = {}
    for name, gguf, hf in (('trained', md / 'model-Q4_K_M.gguf', md / 'hf'),
                           ('stock', ROOT / 'models' / 'lfm3-audio' / 'LFM3-Audio-2B-Q4_K_M.gguf', ROOT / 'models' / 'lfm3-audio-hf')):
        a = AlbaDuplex(gguf, hf, n_ctx=8192)
        G = T.Grammar(a.proc, a.model)
        cdir = sorted((ROOT / 'data' / 'duplex' / 'synthetic').glob('syn-c0*'))[5]
        meta = json.loads((cdir / 'meta.json').read_text(encoding='utf8'))
        utts = T.timeline(meta, T.person_segments(meta), np.random.default_rng(0))
        spans = T.spans_of(utts, 3)
        codes = np.load(ROOT / 'data' / 'duplex' / '_codes_utt2' / (cdir.name + '.npy'))
        n_steps = min(len(codes) - 3, (sf.info(str(cdir / 'user.flac')).frames - T.LOOK) // 1280)
        chunks = (json.loads((md / 'duplex.json').read_text(encoding='utf8')).get('plan_chunks', 'predicted')
                  if (md / 'duplex.json').exists() else 'predicted')
        ids, labels, frames = T.build(G, spans, codes, 0, n_steps, random.Random(0), clock=meta.get('clock'),
                                      plan_chunks=chunks)
        user, _ = sf.read(cdir / 'user.flac', dtype='float32')
        x = user[:n_steps * 1280 + T.LOOK]
        feats = a.proc.feature_extractor(x, sampling_rate=16000, return_tensors='pt')
        uv = a.m.get_audio_features(feats['input_features'].to(a.device, a.dtype),
                                    feats['feature_attention_mask'].to(a.device).bool()).float().cpu().numpy()
        n_head = len(G.head(meta.get('clock')))
        nll = run(a, ids, frames, uv, G, n_head)
        # frames right after plan text vs frames right after "what I just heard"
        after_text, after_in = [], []
        k = 0
        prev = None
        for t in ids[n_head:]:
            if t == G.audio_out:
                (after_in if prev == G.audio_in else after_text).append(nll[k]); k += 1
            prev = t
        res[name] = {'frames': len(nll), 'nll_mean': round(float(np.mean(nll)), 3),
                     'nll_after_plan_text': round(float(np.mean(after_text)), 3) if after_text else None,
                     'nll_after_heard': round(float(np.mean(after_in)), 3) if after_in else None}
        print(name, json.dumps(res[name]), flush=True)
        a.close()
        del a
        torch.cuda.empty_cache()
    (ROOT / 'results' / f'frame_path_{Path(args.model).name}.json').write_text(json.dumps(res, indent=1), encoding='utf8')


if __name__ == '__main__':
    main()
