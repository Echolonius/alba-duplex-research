"""Multi-turn personality check: one spoken conversation per instruction set.

  python -I probe_multiturn.py <out.json> [style,style,...]

Counts, across the whole conversation, replies that repeat an earlier sentence,
assistant-style phrases, and "I'm an AI" mentions.
"""
import asyncio
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app import sounds                      # noqa: E402
from app.dsp import Resampler, to_pcm16     # noqa: E402
from app.realtime import Realtime           # noqa: E402

TAIL = 'Respond with interleaved text and audio.'
STYLES = {
    'current': 'You are chatting with a friend on a voice call. Keep every reply short: one or two casual sentences, '
               'then let them talk. React like a real person and ask about them instead of giving advice. ',
    'named_friend': 'You are Juno, a warm, playful AI friend on a casual voice call. Talk like a real person: short, '
                    'relaxed sentences, your own opinions, a little humor. One or two sentences, then let them talk. ',
    'two_friends': 'This is a casual voice call between two friends. You speak briefly and naturally, with your own '
                   'opinions and feelings, and you pick up on how they feel. One or two short sentences per turn. ',
}
TURNS = ["Hey, can you hear me?",
         "Cool. So, are you an AI?",
         "Okay. Honestly I've had a weird day. My car broke down on the way to work.",
         "Yeah, and then my boss was annoyed because I was late.",
         "What would you do if you were me?",
         "Ha, maybe. Anyway, what's two plus two?",
         "Right. Okay, tell me something fun about you."]
ASSISTANT = re.compile(r"\b(let me know|i'?m here to help|how can i help|what can i help|is there anything|"
                       r"what would you like to talk about|have you (thought|tried|considered)|it'?s important to|"
                       r"i recommend|i don'?t have (feelings|emotions|a personality|personality|memory))\b", re.I)
AI = re.compile(r"\bi'?m an ai\b|\bas an ai\b", re.I)


def sentences(t):
    return [s.strip().lower() for s in re.split(r'[.!?]+', t) if len(s.strip()) > 8]


async def converse(style, clips):
    rt = Realtime('127.0.0.1', 9931, style + TAIL)
    await rt.connect()
    replies = []
    for clip in clips:
        for i in range(0, len(clip), 640):
            await rt.append(to_pcm16(clip[i:i + 640]))
            await asyncio.sleep(0.005)
        await rt.respond()
        text = []
        async for ev in rt.events():
            if ev['type'] == 'response.output_text.delta':
                text.append(ev['delta'])
            elif ev['type'] in ('response.done', 'error'):
                break
        replies.append(''.join(text).strip())
    await rt.close()
    return replies


def score(replies):
    seen, repeats = set(), 0
    for r in replies:
        ss = sentences(r)
        if any(s in seen for s in ss):
            repeats += 1
        seen.update(ss)
    return {'repeats': repeats, 'assistant': sum(bool(ASSISTANT.search(r)) for r in replies),
            'ai_mentions': sum(bool(AI.search(r)) for r in replies),
            'mean_words': float(np.mean([len(r.split()) for r in replies]))}


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')   # replies can contain emoji
    base = 'http://127.0.0.1:9931'
    clips = []
    for k, line in enumerate(TURNS):
        x, rate = sounds.tts(base, line, 'male', 400 + k)
        clips.append(Resampler(rate, 16000)(sounds.trim(x, rate, floor_db=-45)).astype(np.float32))
    names = sys.argv[2].split(',') if len(sys.argv) > 2 else list(STYLES)
    report = {}
    for name in names:
        for trial in range(2):
            replies = asyncio.run(converse(STYLES[name], clips))
            s = score(replies)
            report[f'{name}#{trial}'] = {'score': s, 'replies': replies}
            print(f"\n[{name} #{trial}] repeats {s['repeats']}/{len(replies)}, assistant-style {s['assistant']}, "
                  f"'I'm an AI' {s['ai_mentions']}, {s['mean_words']:.1f} words")
            for t, r in zip(TURNS, replies):
                print(f"   > {t}\n     {r}")
    Path(sys.argv[1]).write_text(json.dumps(report, indent=1), encoding='utf8')


if __name__ == '__main__':
    main()
