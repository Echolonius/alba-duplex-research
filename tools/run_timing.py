"""Timing test (check 2): plays each held-out caller track (real people: data/tests/timing) to a voice system
in real time, open loop (the caller does not wait for the system), and records what the system says, aligned.

  --system split   today's app (conductor, hearing, text brain, Pocket TTS)
  --speakers       simulate laptop speakers: the system's own voice leaks back into the microphone (check 1:
                   it must never answer its own echo)

  python -I tools/run_timing.py --system split [--tag NAME] [--only ID,ID] [--speakers]
Writes results/tests/<tag>/timing/<id>-voice.flac (16 kHz, same clock as the caller) and <id>-events.jsonl.
Score with tools/score_timing.py.
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
from app import server, sounds, speech  # noqa: E402
from app.audio import BLOCK, RATE  # noqa: E402
from app.config import Config, apply_settings  # noqa: E402

TESTS = ROOT / 'data' / 'tests' / 'timing'


class ReplayCaller:
    """The caller's real track, block by block, whatever the system does. Optional speaker echo."""

    def __init__(self, x48, echo=False, seed=0):
        self.x, self.pos, self.log, self.conductor = x48, 0, [], None
        self.echo = echo
        if echo:
            rng = np.random.default_rng(seed)
            h = np.zeros(int(0.2 * RATE)); d0 = int(0.04 * RATE)
            h[d0:] = rng.standard_normal(len(h) - d0) * np.exp(-np.arange(len(h) - d0) / (0.03 * RATE))
            h = signal.lfilter(*signal.butter(2, [250 / 24000, 7000 / 24000], 'band'), h)
            self.h = h * (10 ** (-10 / 20)) / np.sqrt(np.sum(h ** 2))
            self.zi = np.zeros(len(self.h) - 1)

    def next_block(self, out, now):
        if self.pos >= len(self.x) + 3 * RATE:
            return None
        seg = self.x[self.pos:self.pos + BLOCK]
        mic = np.zeros(BLOCK, np.float32)
        mic[:len(seg)] = seg
        self.pos += BLOCK
        if self.echo:
            e, self.zi = signal.lfilter(self.h, 1.0, np.tanh(2 * out) / 2, zi=self.zi)
            mic += e.astype(np.float32)
        return mic


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser()
    ap.add_argument('--system', default='split')
    ap.add_argument('--tag', default='')
    ap.add_argument('--only', default='')
    ap.add_argument('--speakers', action='store_true')
    ap.add_argument('--set', action='append', default=[], metavar='SECTION.NAME=VALUE')
    args = ap.parse_args()
    from benchmark import run_session
    cfg = Config()
    apply_settings(cfg, args.set, log=lambda s: print(s, flush=True))
    from app.main import voice_instructions
    cfg.voice.instructions = voice_instructions(cfg)
    tag = args.tag or (args.system + ('-speakers' if args.speakers else ''))
    out = ROOT / 'results' / 'tests' / tag / 'timing'
    out.mkdir(parents=True, exist_ok=True)
    items = json.loads((TESTS / 'items.json').read_text(encoding='utf8'))['items']
    if args.only:
        items = [i for i in items if i['id'] in args.only.split(',')]
    proc = server.start(cfg, ROOT / 'logs' / 'server-tests.log') if server.needs_voice(cfg) else None
    text_proc = server.start_text(cfg, ROOT / 'logs' / 'server-tests-text.log')
    base = server.base_url(cfg)
    try:
        speaker = speech.make(cfg)
        bank, particles = sounds.prepare(cfg, speaker, base, status=lambda s: print(s, flush=True))
        for it in items:
            if (out / f"{it['id']}-voice.flac").exists():
                continue
            x, sr = sf.read(TESTS / it['caller'], dtype='float32')
            g = np.gcd(RATE, sr)
            x48 = signal.resample_poly(x, RATE // g, sr // g).astype(np.float32)
            caller = ReplayCaller(x48, echo=args.speakers)
            t0 = time.time()
            rec, cond = asyncio.run(run_session(cfg, bank, caller, headphones=not args.speakers, particles=particles,
                                                speaker=speaker))
            mic, voice = rec.tracks()
            g = np.gcd(16000, RATE)
            v16 = signal.resample_poly(voice, 16000 // g, RATE // g).astype(np.float32)
            sf.write(out / f"{it['id']}-voice.flac", v16, 16000)
            cond.save_events(out / f"{it['id']}-events.jsonl")
            ev = cond.events
            print(f"[{it['id']}] {len(x) / sr:.0f}s in {time.time() - t0:.0f}s | replies {sum(e['type'] == 'respond' for e in ev)} "
                  f"| yields {sum(e['type'] == 'yield' for e in ev)} | backchannels {sum(e['type'] == 'backchannel' for e in ev)}",
                  flush=True)
    finally:
        server.stop(text_proc)
        server.stop(proc)


if __name__ == '__main__':
    main()
