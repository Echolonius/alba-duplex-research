"""Does the voice fall into repeating itself? A conversation built like the live calls
(the person complains, asks why it keeps saying that) is played to the running server.

  python -I probe_loops.py <label> [trials]

Prints how many replies repeat an earlier sentence and how often "What would you like
to talk about?" comes back, plus how clearly the replies are spoken (word error rate of
the model's own speech recognition on the reply audio against the reply text).
"""
import asyncio
import base64
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
from app import sounds                               # noqa: E402
from app.dsp import Resampler, from_pcm16, to_pcm16  # noqa: E402
from app.main import PERSONA                         # noqa: E402
from app.realtime import Realtime                    # noqa: E402
from intelligibility import transcribe, wer          # noqa: E402

LINES = ["Hello?",
         "What do you mean? I said hello.",
         "Your voice sounds kind of weird.",
         "It just sounds a little abrupt.",
         "Why do you keep saying that?",
         "Can you look things up online?",
         "That's a problem.",
         "You keep saying things the exact same way.",
         "Okay, tell me about your favorite movie.",
         "Why do you like it?"]
ASSIST = re.compile(r"digital assistant|here to help|i'?m just an? (ai|assistant)|don'?t have (feelings|preferences|opinions|personal|a personality)|as an ai", re.I)
STOCK = re.compile(r"what would you like to talk about|what can i help|what'?s on your mind|hiccup|i'?m an ai", re.I)


def sentences(t):
    return [s.strip().lower() for s in re.split(r'[.!?]+', t) if len(s.strip()) > 8]


async def converse(clips, persona, tail, anchor=None):
    rt = Realtime('127.0.0.1', 9931, persona + tail)
    await rt.connect()
    if anchor:
        await rt.say_text(anchor)      # the friendship set up inside the conversation itself
    out = []
    for clip in clips:
        for i in range(0, len(clip), 640):
            await rt.append(to_pcm16(clip[i:i + 640]))
            await asyncio.sleep(0.005)
        await rt.respond()
        text, audio = [], []
        async for ev in rt.events():
            if ev['type'] == 'response.output_text.delta':
                text.append(ev['delta'])
            elif ev['type'] == 'response.output_audio.delta':
                audio.append(from_pcm16(base64.b64decode(ev['delta'])))
            elif ev['type'] in ('response.done', 'error'):
                break
        a = np.concatenate(audio) if audio else np.zeros(1, np.float32)
        out.append((''.join(text).strip(), Resampler(rt.out_rate, 16000)(a).astype(np.float32)))
    await rt.close()
    return out


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    label = sys.argv[1]
    trials = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    persona = PERSONA + (sys.argv[3] if len(sys.argv) > 3 else '')
    tail = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] else 'Respond with interleaved text and audio.'
    anchor = sys.argv[5] if len(sys.argv) > 5 else None
    base = 'http://127.0.0.1:9931'
    clips = []
    for k, line in enumerate(LINES):
        x, rate = sounds.tts(base, line, 'male', 500 + k)
        clips.append(Resampler(rate, 16000)(sounds.trim(x, rate, floor_db=-45)).astype(np.float32))
    rows = []
    for trial in range(trials):
        res = asyncio.run(converse(clips, persona, tail, anchor))
        heard = asyncio.run(transcribe([a for _, a in res]))
        seen, repeats = set(), 0
        for t, _ in res:
            ss = sentences(t)
            repeats += any(s in seen for s in ss)
            seen.update(ss)
        stock = sum(bool(STOCK.search(t)) for t, _ in res)
        assist = sum(bool(ASSIST.search(t)) for t, _ in res)
        errs = [wer(t, h) for (t, _), h in zip(res, heard) if t]
        w = sum(e for e, _ in errs) / max(1, sum(n for _, n in errs))
        rows.append((repeats, stock, w, assist))
        print(f"[{label} #{trial}] replies repeating an earlier sentence {repeats}/{len(res)} | stock phrases {stock} | "
              f"assistant-mode {assist} | speech clarity: word error {w * 100:.1f}%")
        for line, (t, _) in zip(LINES, res):
            print(f"   > {line}\n     {t}")
    r = np.array(rows)
    print(f"== {label}: repeats {r[:, 0].mean():.1f}, stock {r[:, 1].mean():.1f}, assistant-mode {r[:, 3].mean():.1f}, word error {r[:, 2].mean() * 100:.1f}%")


if __name__ == '__main__':
    main()
