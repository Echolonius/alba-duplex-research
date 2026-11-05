"""Opt-in learning from a call (goal check 7): proposals you review, approve, undo or delete.

Runs only for a call you allowed it for (session.json "learning_opt_in": true: the box in the call window).
1. Your side of the call is transcribed on this PC (Moonshine, the app's own recogniser, on the processor);
   Alba's side is her own plan text (events.jsonl).
2. The local open model (Gemma 4 E4B, on this PC) lists up to 5 things you said about yourself or what you
   like that you might want her to remember next time: only things you said yourself, nothing about health,
   money, passwords, addresses or other people's private details, nothing said to Claude.
3. Each goes into the review queue of Alba's local API (data/api/store.db) with the words it came from.
   Nothing is used until you approve it at http://127.0.0.1:8765/review; approve, undo or delete any time.
Nothing leaves this PC.

  .venv/Scripts/python.exe -I tools/learn_from_call.py sessions/<date-time> [--force-opt-in]   (the flag: tests only)
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
os.environ.setdefault('MOONSHINE_VOICE_CACHE', str(ROOT / 'models' / 'moonshine'))
NOT_ALLOWED = re.compile(r'password|passcode|pin\b|bank|credit|card number|social security|ssn|address|diagnos|'
                         r'medication|prescription|salary|debt|claude', re.I)


def transcribe(path):
    from moonshine_voice import Transcriber, get_model_for_language
    from moonshine_voice.moonshine_api import string_to_model_arch
    from scipy import signal
    x, sr = sf.read(path, dtype='float32')
    if x.ndim > 1:
        x = x.mean(1)
    if sr != 16000:
        g = np.gcd(16000, sr)
        x = signal.resample_poly(x, 16000 // g, sr // g).astype(np.float32)
    mp, arch = get_model_for_language('en', string_to_model_arch('small-streaming'))
    tr = Transcriber(model_path=mp, model_arch=arch)
    try:
        out = []
        for k in range(0, len(x), 16000 * 60):          # a minute at a time keeps memory small
            r = tr.transcribe_without_streaming(x[k:k + 16000 * 60].tolist(), 16000)
            out += [(k / 16000 + l.start_time, ' '.join((l.text or '').split())) for l in r.lines if (l.text or '').strip()]
        return out
    finally:
        tr.close()


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser()
    ap.add_argument('session')
    ap.add_argument('--force-opt-in', action='store_true', help='tests only: treat the call as opted in')
    ap.add_argument('--store', default='', help='tests only: a throwaway store folder')
    ap.add_argument('--port', type=int, default=9946)
    ap.add_argument('--model', default=str(ROOT / 'models' / 'text' / 'gemma-4-E4B-it-Q4_0.gguf'))
    args = ap.parse_args()
    sess = Path(args.session)
    info = json.loads((sess / 'session.json').read_text(encoding='utf8')) if (sess / 'session.json').exists() else {}
    if not (info.get('learning_opt_in') or args.force_opt_in):
        print('This call was not opted in to learning: nothing done.')
        return
    lines = [(t, 'Person', s) for t, s in transcribe(sess / 'person.flac')]
    if (sess / 'events.jsonl').exists():
        for l in open(sess / 'events.jsonl', encoding='utf8'):
            e = json.loads(l)
            if e.get('type') == 'stop' and e.get('planned'):
                lines.append((e['t'], 'Alba', e['planned']))
    lines.sort()
    transcript = '\n'.join(f'{who}: {s}' for _, who, s in lines)[-12000:]
    if not any(w == 'Person' for _, w, _ in lines):
        print('Nothing of yours was heard: nothing to propose.')
        return
    from app import server
    from app.config import Config
    from make_convo_text import chat
    cfg, proc = Config(), None
    if not server.healthy(cfg, port=args.port):
        proc = server._launch(cfg, ['-m', args.model, '-c', '8192', '-np', '1'], args.port,
                              ROOT / 'logs' / 'server-learn.log', 180)
    try:
        prompt = ("Here is the transcript of a voice call between a person and their AI friend Alba (the person's "
                  "side was transcribed automatically and may have small errors):\n\n" + transcript + "\n\n"
                  "List up to 5 things the PERSON said about themselves, their plans or what they like, that they "
                  "would probably want Alba to remember in a later call. Only what the person clearly said about "
                  "themselves; never anything about health, money, passwords, addresses or other people's private "
                  "details; nothing they said to someone called Claude. If there is nothing, return []. Answer with "
                  'JSON only: [{"remember": "short fact, in the third person", "quote": "their own words"}]')
        raw = chat(args.port, [{'role': 'user', 'content': prompt}], temperature=0.2, max_tokens=500)
    finally:
        if proc is not None:
            server.stop(proc)
    m = re.search(r'\[.*\]', raw, re.S)
    try:
        items = json.loads(m.group(0)) if m else []
    except json.JSONDecodeError:
        items = []
    from app.api import Store
    store, n = (Store(args.store) if args.store else Store()), 0
    for it in items[:5]:
        fact, quote = str(it.get('remember', '')).strip(), str(it.get('quote', '')).strip()
        if not fact or NOT_ALLOWED.search(fact + ' ' + quote):
            continue
        store.propose(sess.name, fact[:300], quote[:300])
        n += 1
    print(f'{n} proposals waiting for your review at http://127.0.0.1:8765/review (start: python -m app.api)')


if __name__ == '__main__':
    main()
