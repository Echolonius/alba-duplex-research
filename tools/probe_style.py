"""Compares personality instructions on the same spoken lines.

  python -I probe_style.py <out.json>
Each instruction set answers the same six spoken lines (fresh conversation each time).
Counts assistant-style phrases and reports reply length.
"""
import asyncio
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app import sounds                              # noqa: E402
from app.dsp import Resampler, to_pcm16             # noqa: E402
from app.realtime import Realtime                   # noqa: E402

TAIL = 'Respond with interleaved text and audio.'
STYLES = {
    'current': 'You are chatting with a friend on a voice call. Keep every reply short: one or two casual sentences, '
               'then let them talk. React like a real person and ask about them instead of giving advice. ',
    'friend_opinions': "You're on a voice call with a close friend. Talk like a real person, not an assistant: casual "
                       'words, your own opinions and reactions, a little humor. One or two short sentences, then let '
                       "them talk. Don't offer help or advice unless they ask. ",
    'react_first': 'This is a relaxed phone call between two friends. Speak the way people really talk: react first '
                   '(oh no, wait, seriously?, ha), say what you think, and sometimes ask something back. One or two '
                   'short sentences. No assistant talk, and no advice unless asked. ',
    'person_not_assistant': 'You are a warm, curious person on a casual voice call with a friend, not an assistant. '
                            'Keep every reply short: one or two casual sentences, then let them talk. Share your own '
                            'thoughts and feelings, and ask about theirs instead of giving advice. ',
}
if len(sys.argv) > 2:   # only the named styles, e.g. "current,person_not_assistant"
    STYLES = {k: v for k, v in STYLES.items() if k in sys.argv[2].split(',')}
LINES = ["Ugh, I've had kind of a rough week.",
         "My manager keeps adding deadlines and I'm just so tired.",
         "Okay, random question. Coffee or tea?",
         "I watched an old sci-fi movie last night and the effects were so bad, but I loved it.",
         "I just got back from a hike and my friend forgot all the snacks.",
         "Part of me wants to quit and just move to the beach."]
ASSISTANT = re.compile(r"\b(let me know|i'?m here to help|how can i help|is there anything|have you (thought|tried|considered)|"
                       r"it'?s okay to|it'?s important to|you (could|might) (try|consider)|i recommend|as an ai)\b", re.I)


async def ask(style, pcm16):
    rt = Realtime('127.0.0.1', 9931, style + TAIL)
    await rt.connect()
    for i in range(0, len(pcm16), 640):
        await rt.append(to_pcm16(pcm16[i:i + 640]))
    await rt.respond()
    text = []
    async for ev in rt.events():
        if ev['type'] == 'response.output_text.delta':
            text.append(ev['delta'])
        elif ev['type'] in ('response.done', 'error'):
            break
    await rt.close()
    return ''.join(text).strip()


def main():
    base = 'http://127.0.0.1:9931'
    audio = []
    for k, line in enumerate(LINES):
        x, rate = sounds.tts(base, line, 'male', 200 + k)
        audio.append(Resampler(rate, 16000)(sounds.trim(x, rate, floor_db=-45)).astype(np.float32))
    report = {}
    for name, style in STYLES.items():
        replies = [asyncio.run(ask(style, a)) for a in audio]
        words = [len(r.split()) for r in replies]
        flags = sum(bool(ASSISTANT.search(r)) for r in replies)
        report[name] = {'replies': replies, 'mean_words': float(np.mean(words)), 'assistant_phrases': flags}
        print(f"\n[{name}] mean {np.mean(words):.1f} words, assistant-style replies {flags}/{len(replies)}")
        for line, r in zip(LINES, replies):
            print(f"   > {line}\n     {r}")
    Path(sys.argv[1]).write_text(json.dumps(report, indent=1), encoding='utf8')


if __name__ == '__main__':
    main()
