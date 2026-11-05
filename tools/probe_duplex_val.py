"""Reply-quality probe on VALIDATION prompts only (never the frozen tests): the first spoken turn of each pilot
conversation (data/convo-pilot, made before training and never trained on). Prints what Alba says and counts
stray word-pieces (non-ASCII fragments in an English reply). For checking decoding and runtime settings.

  .venv-train/Scripts/python.exe -I tools/probe_duplex_val.py --model models/alba-duplex2 [--listen-bias 1.61]
      [--save results/tests/duplex2-probe-val]
--save writes each prompt's voice track (<id>-voice.wav, 24 kHz) and plans.json (everything she planned to say),
for tools/speech_match.py (does her speech say what she planned?).
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
from app.duplex import DELAY, AlbaDuplex  # noqa: E402
from run_items_duplex import run_item, to16  # noqa: E402
from run_timing_duplex import model_settings  # noqa: E402


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--eos-first', action='store_true', help='stop decided by one codebook (app/fast_audio)')
    ap.add_argument('--listen-bias', type=float, default=None)
    ap.add_argument('--save', default='')
    ap.add_argument('--mic', action='store_true', help="the prompts through training's laptop-mic sound (fixed seed)")
    args = ap.parse_args()
    md = ROOT / args.model
    st = model_settings(md)
    bias = args.listen_bias if args.listen_bias is not None else float(st.get('listen_bias', 0.0))
    alba = AlbaDuplex(md / 'model-Q4_K_M.gguf', md / 'hf', eos_first=args.eos_first)
    clips = sorted((ROOT / 'data' / 'convo-pilot' / 'audio').glob('*-0.wav'))
    stray = answered = 0
    utt_frames, early = [], 0
    plans = {}
    save = ROOT / args.save if args.save else None
    if save:
        save.mkdir(parents=True, exist_ok=True)
    for p in clips:
        x = to16(*sf.read(p, dtype='float32'))
        if args.mic:
            from train_duplex import room_mic
            x = room_mic(x, np.random.default_rng(int(p.stem[1:6]))).astype(np.float32)
        voice, reply, _ = run_item(alba, [x], listen_bias=bias, delay=int(st.get('delay_steps', DELAY)))
        if save:
            sf.write(save / f'{p.stem}-voice.wav', voice, 24000, subtype='PCM_16')
            plans[p.stem] = {'reply': reply, 'all_planned': alba.tok.decode(alba.said, skip_special_tokens=True).strip()}
        for e in alba.events:
            if e[0] == 'stop' and len(e) >= 5:
                utt_frames.append(e[3])
                early += not e[4]
        bad = bool(re.search(r'[^\x00-\x7f’‘“”—–]', reply))
        stray += bad
        answered += len(reply.split()) >= 4
        print(f'{p.stem}: {reply[:160]!r}{"  <- stray pieces" if bad else ""}', flush=True)
    print(json.dumps({'prompts': len(clips), 'answered_4plus_words': answered, 'with_stray_pieces': stray,
                      'listen_bias': bias, 'utterances': len(utt_frames),
                      'utterance_frames_median': float(np.median(utt_frames)) if utt_frames else None,
                      'stopped_before_plan_end': early}), flush=True)
    if save:
        (save / 'plans.json').write_text(json.dumps(plans, indent=1, ensure_ascii=False), encoding='utf8')
    alba.close()


if __name__ == '__main__':
    main()
