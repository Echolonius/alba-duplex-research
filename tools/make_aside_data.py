"""Training conversations for the standing test "ignores my side remarks to Claude" (goal check 4).

In your calls you sometimes talk to Claude (the developer) instead of to Alba. A full-duplex Alba hears
everything, so she has to learn that a remark addressed to Claude is not for her: she stays quiet, and answers
again when you talk to her. Here, some of the spoken synthetic conversations (tools/make_convo_audio.py) get
one such remark inserted after one of Alba's replies, in the same person's voice, followed by a short pause;
the conversation then goes on as written.

The remarks are written by the local open model (Gemma 4 E4B, never Claude) and checked against the frozen
test items by the same leak filter as all training text (the test's own remark is held out).

  python -I tools/make_aside_data.py --phase text  [--n 160]     (graphics card: the open model writes remarks)
  python -I tools/make_aside_data.py --phase audio [--convs 300] (processor: speaks them into conversations)
then: python -I tools/make_duplex_data.py --source synthetic --convo data/asides --out data/duplex/asides --source-name asides
"""
import argparse
import json
import random
import re
import sys
import zlib
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
OUT = ROOT / 'data' / 'asides'
KINDS = ['a note about how the call is going (timing, voice, a reply that was off) for Claude to write down',
         'asking Claude to check or fix something in the program later',
         'telling Claude what to test next',
         'a quick question to Claude about the project, not to Alba',
         'thinking out loud to Claude about whether the voice sounded natural just now']


def phase_text(args):
    from app import server
    from app.config import Config
    from make_convo_text import chat, clean, leaks
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'lines.jsonl'
    have = [json.loads(l)['text'] for l in open(path, encoding='utf8')] if path.exists() else []
    cfg = Config()
    proc = None
    if not server.healthy(cfg, port=args.port):
        proc = server._launch(cfg, ['-m', args.model, '-c', '4096', '-np', '1'], args.port,
                              ROOT / 'logs' / 'server-asides.log', 180)
    rng = random.Random(len(have))
    try:
        with open(path, 'a', encoding='utf8') as f:
            tries = 0
            while len(have) < args.n and tries < args.n * 4:
                tries += 1
                kind = rng.choice(KINDS)
                prompt = ("Someone is testing a voice assistant named Alba on a call. While the call is on, they "
                          "briefly turn to Claude, the developer who is building Alba, and say something to Claude, "
                          f"not to Alba: {kind}. Write only what they say out loud, one or two short casual "
                          "sentences, addressed to Claude by name (start with 'Claude,' or put 'Claude' early). "
                          "No quotes, no stage directions.")
                line = clean(chat(args.port, [{'role': 'user', 'content': prompt}], temperature=1.0, max_tokens=80))
                low = line.lower()
                if ('claude' not in low or len(line.split()) > 30 or leaks(line) or 'alba,' in low[:8]
                        or any(_close(line, h) for h in have)):
                    continue
                have.append(line)
                f.write(json.dumps({'text': line, 'kind': kind}, ensure_ascii=False) + '\n')
                f.flush()
        print(f'{len(have)} remarks', flush=True)
    finally:
        if proc is not None:
            server.stop(proc)


def _close(a, b):
    import difflib
    return difflib.SequenceMatcher(None, re.sub(r'\W', '', a.lower()), re.sub(r'\W', '', b.lower())).ratio() > 0.8


def phase_audio(args):
    from app.speech import PocketVoice
    from make_convo_audio import laptop_mic, to16k
    lines = [json.loads(l)['text'] for l in open(OUT / 'lines.jsonl', encoding='utf8') if l.strip()]
    convs = [json.loads(l) for l in open(ROOT / 'data' / 'convo' / 'audio.jsonl', encoding='utf8') if l.strip()]
    convs = [c for c in convs if len(c['turns']) >= 4 and not c['id'].startswith('call')]
    rng = random.Random(1)
    rng.shuffle(convs)
    adir = OUT / 'audio'
    adir.mkdir(parents=True, exist_ok=True)
    src = ROOT / 'data' / 'convo' / 'audio'
    voices, rows = {}, []
    for c in convs[:args.convs]:
        crng = np.random.default_rng(zlib.crc32(('aside-' + c['id']).encode()))
        pv = c['turns'][0]['voice']
        if pv not in voices:
            voices[pv] = PocketVoice(pv, threads=args.threads)
        line = lines[int(crng.integers(len(lines)))]
        aid = f"a{c['id']}"
        path = adir / f'{aid}-aside.wav'
        if not path.exists():
            wav, rate = voices[pv].say(line, seed=int(crng.integers(1 << 30)))
            sf.write(path, laptop_mic(to16k(wav, rate), 16000, crng), 16000)
        # after one of Alba's replies that is followed by another person turn
        spots = [k for k in range(1, len(c['turns']) - 1) if c['turns'][k]['role'] == 'alba'
                 and c['turns'][k + 1]['role'] == 'person']
        if not spots:
            continue
        k = spots[int(crng.integers(len(spots)))]
        turns = [{**t, 'audio': '../../convo/audio/' + t['audio']} for t in c['turns']]
        turns.insert(k + 1, {'role': 'person', 'text': line, 'aside': True, 'audio': path.name, 'voice': pv})
        rows.append({**c, 'id': aid, 'turns': turns})
    with open(OUT / 'audio.jsonl', 'w', encoding='utf8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'{len(rows)} conversations with a remark to Claude', flush=True)


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', choices=('text', 'audio'), required=True)
    ap.add_argument('--n', type=int, default=160)
    ap.add_argument('--convs', type=int, default=300)
    ap.add_argument('--port', type=int, default=9946)
    ap.add_argument('--model', default=str(ROOT / 'models' / 'text' / 'gemma-4-E4B-it-Q4_0.gguf'))
    ap.add_argument('--threads', type=int, default=2)
    args = ap.parse_args()
    phase_text(args) if args.phase == 'text' else phase_audio(args)


if __name__ == '__main__':
    main()
