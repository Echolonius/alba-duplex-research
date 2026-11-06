"""Training conversations from your own calls (they stay on this PC): your real voice through your laptop
microphone is the *listening* side; Alba's reply to each of your turns is newly written by the local open model
(Gemma 4 E4B, never Claude) under her persona, and spoken by Pocket TTS 'alba'. Your voice is never a speaking
target and is never cloned.

Left out: the three newest calls (held out for testing, listed in data/tests/private_calls.json), anything you
said to Claude (the aside rule), anything near a test item (the leak filter), turns under 1.2 s or over 10 s.
Without that list (it stays on your PC, git-ignored) this stops, so the test calls never end up in training.

  python -I tools/make_convo_from_calls.py [--port 9946]
Writes data/convo/calls.jsonl and the clips in data/convo/audio/ (same layout as make_convo_audio.py's output).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
from app import server  # noqa: E402
from app.brain import ASIDE  # noqa: E402
from app.config import Config  # noqa: E402
from build_testsets import held_out_calls  # noqa: E402
from make_convo_text import TRAIN_SYSTEM, alba_reply, leaks  # noqa: E402


def turns_of(sess):
    d = ROOT / 'sessions' / sess
    ev = [json.loads(l) for l in open(d / 'events.jsonl', encoding='utf8') if l.strip()]
    x, fs = sf.read(d / 'person_clean16k.flac', dtype='float32')
    out, start, stop = [], None, None
    for e in ev:
        if e['type'] == 'user_start' and start is None:
            start = e['at']
        elif e['type'] == 'user_stop':
            stop = e['at']
        elif e['type'] == 'heard' and start is not None and stop is not None:
            text = e['text'].strip()
            if 1.2 <= stop - start <= 10 and len(text.split()) >= 3 and not ASIDE.search(text) and not leaks(text) \
                    and all(text[:40] != o[1][:40] for o in out):
                a, b = max(0, int((start - 0.15) * fs)), min(len(x), int((stop + 0.3) * fs))
                out.append((round(start, 2), text, x[a:b]))
            start = stop = None
        elif e['type'] in ('respond', 'first_audio', 'aside'):
            start = stop = None
    return out


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=9946)
    ap.add_argument('--model', default=str(ROOT / 'models' / 'text' / 'gemma-4-E4B-it-Q4_0.gguf'))
    args = ap.parse_args()
    held_out = held_out_calls()
    if not held_out:
        sys.exit('needs data/tests/private_calls.json, the list of your held-out test calls (it stays on your PC), '
                 'so that they are left out of training')
    from app.speech import PocketVoice
    adir = ROOT / 'data' / 'convo' / 'audio'
    adir.mkdir(parents=True, exist_ok=True)
    cfg = Config()
    proc = None
    if not server.healthy(cfg, port=args.port):
        proc = server._launch(cfg, ['-m', args.model, '-c', '4096', '-np', '1'], args.port,
                              ROOT / 'logs' / 'server-convo-text.log', 180)
    alba = PocketVoice('alba', threads=4)
    rows = []
    try:
        for d in sorted((ROOT / 'sessions').iterdir()):
            if d.name in held_out or not (d / 'person_clean16k.flac').exists() or not (d / 'events.jsonl').exists():
                continue
            for k, (t, text, clip) in enumerate(turns_of(d.name)):
                cid = f'call-{d.name}-{k:03d}'
                reply, _ = alba_reply(args.port, [{'role': 'person', 'text': text}], None, False)
                if not reply:
                    continue
                sf.write(adir / f'{cid}-0.wav', clip, 16000)
                wav, rate = alba.say(reply, seed=k)
                sf.write(adir / f'{cid}-1.wav', np.asarray(wav, np.float32), rate)
                rows.append({'id': cid, 'situation': 'your call', 'clock': None, 'system': TRAIN_SYSTEM,
                             'turns': [{'role': 'person', 'text': text, 'audio': f'{cid}-0.wav', 'voice': 'you (listening side only)'},
                                       {'role': 'alba', 'text': reply, 'audio': f'{cid}-1.wav', 'voice': 'alba'}]})
                print(f'{cid}: {text[:60]!r} -> {reply[:70]!r}', flush=True)
    finally:
        server.stop(proc)
    with open(ROOT / 'data' / 'convo' / 'calls.jsonl', 'w', encoding='utf8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'{len(rows)} turns from your calls', flush=True)


if __name__ == '__main__':
    main()
