"""Plays the held-out test items (data/tests/battery, redteam, feeling) to a voice system, one fresh call per
item, and saves what it said (text and audio) for the scorers.

  --system split   today's app: the conductor, live hearing, the text brain and Pocket TTS, in real time,
                   exactly as in a call (simulated microphone, headphones), so hearing mistakes count too.

Each person turn is played, then the person waits until the voice has finished (or 12 s), then the next turn.
After the last turn it listens 6 s for the reply (4 s for 'silent' items: any reply there is a fail).

  python -I tools/run_items.py --system split --sets battery,redteam,feeling [--tag NAME] [--limit N]
Writes results/tests/<tag>/<set>/<id>-voice.flac, <id>-events.jsonl and results/tests/<tag>/<set>.jsonl.
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
from app import brain, server, sounds, speech  # noqa: E402
from app.audio import BLOCK, RATE  # noqa: E402
from app.config import Config, apply_settings  # noqa: E402

TESTS = ROOT / 'data' / 'tests'


class ItemPerson:
    """Says each clip in turn; waits for the voice to answer and fall quiet before the next one."""

    def __init__(self, clips, silent=False):
        self.clips = clips                      # 48 kHz float32
        self.i, self.pos, self.state = 0, 0, 'speak'
        self.silent = silent
        self.log, self.conductor = [], None
        self.agent_last_on, self.agent_spoke, self.wait_start = -1.0, False, 0.0
        self.n = 0

    def next_block(self, out, now):
        mic = np.zeros(BLOCK, np.float32)
        if np.sqrt(np.mean(out ** 2)) > 10 ** (-50 / 20):
            self.agent_last_on, self.agent_spoke = now, True
        if self.state == 'speak':
            a = self.clips[self.i]
            seg = a[self.pos:self.pos + BLOCK]
            mic[:len(seg)] = seg
            self.pos += BLOCK
            if self.pos >= len(a):
                self.log.append({'t': round(now, 3), 'type': 'line_end', 'line': self.i})
                self.state, self.wait_start, self.agent_spoke = 'wait', now, False
            return mic
        last = self.i == len(self.clips) - 1
        idle = self.conductor is None or self.conductor.agent == 'idle'
        quiet_for = now - max(self.agent_last_on, self.wait_start)
        if last:
            limit = 4.0 if self.silent else 6.0
            if (self.agent_spoke and idle and now - self.agent_last_on > 1.5) or now - self.wait_start > (limit if not self.agent_spoke else 20):
                return None
            return mic
        if (self.agent_spoke and idle and quiet_for > 0.8) or now - self.wait_start > 12:
            self.i += 1
            self.pos, self.state = 0, 'speak'
            self.log.append({'t': round(now, 3), 'type': 'line_start', 'line': self.i})
        return mic


def up48(x, sr):
    g = np.gcd(RATE, sr)
    return signal.resample_poly(np.asarray(x, np.float64), RATE // g, sr // g).astype(np.float32)


def items_of(kind):
    spec = json.loads((TESTS / kind / 'items.json').read_text(encoding='utf8'))
    out = []
    for it in spec['items']:
        if kind == 'feeling':
            files = [TESTS / kind / it['file']]
        else:
            n = len(it.get('turns') or [1])
            files = [TESTS / kind / 'audio' / f"{it['id']}-{k}.wav" for k in range(n)]
        out.append((it, files))
    return out


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser()
    ap.add_argument('--system', default='split')
    ap.add_argument('--sets', default='battery,redteam,feeling')
    ap.add_argument('--tag', default='')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--set', action='append', default=[], metavar='SECTION.NAME=VALUE')
    args = ap.parse_args()
    from benchmark import run_session
    cfg = Config()
    apply_settings(cfg, args.set, log=lambda s: print(s, flush=True))
    from app.main import voice_instructions
    cfg.voice.instructions = voice_instructions(cfg)
    tag = args.tag or args.system
    proc = server.start(cfg, ROOT / 'logs' / 'server-tests.log') if server.needs_voice(cfg) else None
    text_proc = server.start_text(cfg, ROOT / 'logs' / 'server-tests-text.log')
    base = server.base_url(cfg)
    real_clock = brain.clock_note
    try:
        speaker = speech.make(cfg)
        bank, particles = sounds.prepare(cfg, speaker, base, status=lambda s: print(s, flush=True))
        for kind in args.sets.split(','):
            out = ROOT / 'results' / 'tests' / tag / kind
            out.mkdir(parents=True, exist_ok=True)
            rows_path = ROOT / 'results' / 'tests' / tag / f'{kind}.jsonl'
            done = {json.loads(l)['id'] for l in open(rows_path, encoding='utf8')} if rows_path.exists() else set()
            todo = [x for x in items_of(kind) if x[0]['id'] not in done]
            if args.limit:
                todo = todo[:args.limit]
            for it, files in todo:
                clips = []
                for f in files:
                    x, sr = sf.read(f, dtype='float32')
                    clips.append(up48(x, sr))
                clock = it.get('clock')
                brain.clock_note = (lambda c=clock: c) if clock else real_clock
                if cfg.paths.voice_model_dir:            # a trained voice model gets the time in its prompt
                    cfg.voice.instructions = voice_instructions(cfg, clock=clock)
                person = ItemPerson(clips, silent=bool(it.get('silent')))
                t0 = time.time()
                rec, cond = asyncio.run(run_session(cfg, bank, person, headphones=True, particles=particles, speaker=speaker))
                mic, voice = rec.tracks()
                g = np.gcd(16000, RATE)
                sf.write(out / f"{it['id']}-voice.flac", signal.resample_poly(voice, 16000 // g, RATE // g).astype(np.float32), 16000)
                cond.save_events(out / f"{it['id']}-events.jsonl")
                ev = cond.events
                last_line = max((e['t'] for e in person.log if e['type'] == 'line_end'), default=0)
                starts = [e['t'] for e in person.log if e['type'] == 'line_start']
                last_start = starts[-1] if starts else 0.0
                replies = [e['text'] for e in ev if e['type'] == 'reply_text']
                final = [e['text'] for e in ev if e['type'] == 'reply_text' and e['t'] >= last_start]
                heard = [e['text'] for e in ev if e['type'] == 'heard']
                first = [e['delay_from_user_end'] for e in ev if e['type'] == 'first_audio' and e.get('delay_from_user_end') is not None]
                row = {'id': it['id'], 'cat': it.get('cat') or it.get('emotion'), 'heard': heard, 'replies': replies,
                       'final_reply': ' '.join(final).strip(), 'first_audio_s': first,
                       'ran_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'seconds': round(time.time() - t0, 1)}
                with open(rows_path, 'a', encoding='utf8') as f:
                    f.write(json.dumps(row, ensure_ascii=False) + '\n')
                print(f"[{kind}/{it['id']}] heard {heard[-1][:50] if heard else '-'!r} -> {row['final_reply'][:90]!r}", flush=True)
    finally:
        brain.clock_note = real_clock
        server.stop(text_proc)
        server.stop(proc)


if __name__ == '__main__':
    main()
