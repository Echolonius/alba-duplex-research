"""Decoding diagnostics on VALIDATION prompts only (data/convo-pilot first turns: made before training, never
trained on, never a frozen test). Every plan decision the runtime makes (app/duplex.AlbaDuplex.plan_log: each
word-piece taken with the top choice's chance, each chunk end with the token the model pointed at), and for every
speech frame the depth decoder's chance of the end-of-speech code in each of the 12 codebooks before any is
revealed (what the frame sampler's first pass draws from).

  .venv-train/Scripts/python.exe -I tools/diag_duplex_decode.py --model models/alba-duplex2 [--n 4]
Writes results/diag_decode_<model>.json.
"""
import argparse
import json
import sys
from pathlib import Path

import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
from app.duplex import DELAY, AlbaDuplex  # noqa: E402
from run_items_duplex import run_item, to16  # noqa: E402
from run_timing_duplex import model_settings  # noqa: E402


class Diag(AlbaDuplex):
    @torch.no_grad()
    def frame(self, hidden, **kw):
        h = torch.from_numpy(hidden).to(self.device, self.dtype)[None]
        n = self.model.config.num_codebooks
        cur = torch.zeros(1, n, dtype=torch.long, device=self.device)
        lg = self.m.depth_decoder(h, cur, torch.ones(1, n, dtype=torch.bool, device=self.device))[0].float()
        p = torch.softmax(lg, -1)[:, self.eos_code].cpu().numpy()
        codes = super().frame(hidden, **kw)
        self.plan_log.append(('frame', self.n_steps - 1, self.utt_steps, self.text_done, round(float(p.mean()), 4),
                              round(float(p.max()), 4), int((codes == self.eos_code).sum())))
        return codes


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--eos-first', action='store_true', help='stop decided by one codebook (app/fast_audio)')
    ap.add_argument('--n', type=int, default=4)
    ap.add_argument('--listen-bias', type=float, default=None)
    args = ap.parse_args()
    md = ROOT / args.model
    st = model_settings(md)
    bias = args.listen_bias if args.listen_bias is not None else float(st.get('listen_bias', 0.0))
    a = Diag(md / 'model-Q4_K_M.gguf', md / 'hf', eos_first=args.eos_first)
    clips = sorted((ROOT / 'data' / 'convo-pilot' / 'audio').glob('*-0.wav'))[:args.n]
    out = {}
    for p in clips:
        a.plan_log = []
        x = to16(*sf.read(p, dtype='float32'))
        _, reply, _ = run_item(a, [x], listen_bias=bias, delay=int(st.get('delay_steps', DELAY)))
        out[p.stem] = {'reply': reply, 'log': a.plan_log}
        print(f'== {p.stem}: {reply!r}', flush=True)
        for e in a.plan_log:
            print('   ', *e, flush=True)
    (ROOT / 'results' / f'diag_decode_{Path(args.model).name}.json').write_text(json.dumps(out, indent=1),
                                                                               encoding='utf8')
    a.close()


if __name__ == '__main__':
    main()
