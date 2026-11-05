"""Benchmark: a simulated person has real-time conversations with the local voice.

The person's lines are spoken by the model's male voice. The person behaves like the
people in Dessa's demos: they wait a human-like moment after the voice finishes
(sampled from the measured human response times), sometimes pause mid-thought,
sometimes talk over the voice, and sometimes say "mm-hm" while it talks.
The two recorded tracks are then measured with the same tool used on Dessa.

  python -I benchmark.py <out_dir> [--speakers] [--sessions A,B,C] [--tag name]
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
from app import sounds, server, speech                             # noqa: E402
from app.audio import BLOCK, RATE, Playback, Recorder, SimAudio    # noqa: E402
from app.conductor import Conductor                                # noqa: E402
from app.config import Config, apply_settings                      # noqa: E402
from app.dsp import Resampler                                      # noqa: E402
from app.main import PERSONA                                       # noqa: E402

SCRIPTS = {
    'A': ["Hey! How's it going?",
          "Pretty good, actually. I just got back from a hiking trip this weekend.",
          "Yeah, we went up to this lake in the mountains. And it was | honestly way harder than I expected.",
          "The last two miles were basically straight uphill.",
          "And my friend forgot the snacks, so we were starving the whole way.",
          "We had like one granola bar between three people.",
          "But the view at the top was totally worth it.",
          "Have you ever been somewhere where the view just stops you?",
          "Okay, so what would you do with a free weekend?",
          "That sounds nice. I think I need a slower weekend next time | maybe just coffee and a book.",
          "Alright, I should go unpack. Thanks for chatting!"],
    'B': ["Hi. Ugh, I've had kind of a rough week.",
          "Work has just been nonstop. My manager keeps adding deadlines.",
          "And on Thursday I found out the project I spent a month on | might get cancelled.",
          "Yeah. I'm trying not to take it personally.",
          "I don't know, part of me wants to just look for a new job.",
          "But I actually like the people I work with, so it's complicated.",
          "What would you do if you were me?",
          "Hmm. That's fair. Maybe I should talk to my manager first.",
          "I think I just needed to say it out loud, you know?",
          "Okay, I feel a little better. Thanks for listening."],
    'C': ["Okay, random question. What's the best movie you've seen lately?",
          "Ha! I watched an old sci-fi movie last night, and the effects were so bad | but I loved it anyway.",
          "There's this scene where the robot is clearly a guy in a cardboard costume.",
          "Do you like movies that are so bad they're good?",
          "Fair enough. Okay, different question. Coffee or tea?",
          "I'm a coffee person. Like, three cups a day, which is probably too much.",
          "My doctor says I should cut back, but | I mean, come on.",
          "What's something you could never give up?",
          "Ha, okay. That makes sense.",
          "Alright, I'm going to go make coffee. Talk later!"],
}
SCRIPTS['L'] = ["Hello?",                                   # built like the live calls that made it loop
                "What do you mean? I said hello.",
                "Your voice sounds kind of weird.",
                "It just sounds a little abrupt.",
                "Why do you keep saying that?",
                "Can you look things up online?",
                "That's a problem.",
                "You keep saying things the exact same way.",
                "Okay, tell me about your favorite movie.",
                "Why do you like it?"]
SCRIPTS['U'] = ["Hello?",                                   # what the person said in the fourth live call
                "Okay, let's test this.",
                "That sounds off, that's a bad response.",
                "This sounds very stagnant to me. Not really what I'm looking for at all.",
                "I don't know why you would laugh there, it just sounds very inaccurate.",
                "The responses are very generic.",
                "Who's Sonic the Hedgehog?",
                "Who is Sonic?",
                "That's not true."]
SCRIPTS['K'] = ["What's the capital of Australia?",           # plain questions a friend would just answer
                "Who wrote the Harry Potter books?",
                "How many legs does a spider have?",
                "Do you know what Minecraft is?",
                "What's the tallest mountain in the world?",
                "What's a good movie from the nineties?",
                "What's your favorite food?",
                "Can you recommend a podcast?"]
SCRIPTS['G'] = ["Hey! Okay, I have a quick one for you.",     # describing things without naming them (live calls 7-8)
                "So, do you know that one superhero | that has an S on his shirt, | wears blue and a red cape, | "
                "and he flies?",
                "Yes! Exactly. Okay, what's his name when he's at work, | like as a normal guy?",
                "Right. Okay, another one. What's that show called | with the doctor who travels | in a blue police box?",
                "Yeah! I love that show.",
                "Okay, that's all I had. Thanks!"]
PERSON_BACKCHANNELS = ['Mm-hm.', 'Yeah.', 'Uh-huh.']


# The simulated person's voice: the voice server's male voice, or a Pocket TTS voice (--person-voice),
# so a run with hearing and speaking on the processor needs no voice server at all.
PERSON = {'speaker': None}


def speak(base, text, seed, voice='male'):
    if PERSON['speaker'] is not None:
        x, rate = PERSON['speaker'].say(text, seed)
    else:
        x, rate = sounds.tts(base, text, voice, seed)
    return Resampler(rate, RATE)(sounds.trim(x, rate, floor_db=-45)).astype(np.float32)


def build_lines(base, script, rng):
    lines = []
    for k, text in enumerate(script):
        parts = [p.strip() for p in text.split('|')]
        audio, pauses, t = [], [], 0.0
        for j, part in enumerate(parts):
            a = speak(base, part, seed=100 + 7 * k + j)
            audio.append(a)
            t += len(a) / RATE
            if j < len(parts) - 1:
                gap = rng.uniform(0.7, 1.0)
                audio.append(np.zeros(int(gap * RATE), np.float32))
                pauses.append((t, t + gap))
                t += gap
        lines.append({'text': text, 'audio': np.concatenate(audio), 'pauses': pauses})
    return lines


class ScriptedPerson:
    def __init__(self, lines, bc_clips, gaps, rng, echo=False, p_interrupt=0.2, p_backchannel=0.35):
        self.lines, self.bc_clips, self.gaps, self.rng = lines, bc_clips, gaps, rng
        self.p_interrupt, self.p_backchannel = p_interrupt, p_backchannel
        self.i, self.pos, self.state = 0, 0, 'speak'
        self.line_start = 0.0
        self.agent_last_on = -9.0
        self.agent_spoke = False
        self.agent_start = None
        self.wait_start = 0.0
        self.plan = {}
        self.bc = None
        self.next_start = None
        self.log = []
        self.conductor = None
        self.noise = rng.standard_normal(RATE * 1200).astype(np.float32) * 10 ** (-66 / 20)
        self.n = 0
        self.echo = echo
        if echo:  # laptop-speaker echo: 40 ms delay, short room tail, -10 dB, mild distortion
            h = np.zeros(int(0.2 * RATE)); d0 = int(0.04 * RATE)
            h[d0:] = rng.standard_normal(len(h) - d0) * np.exp(-np.arange(len(h) - d0) / (0.03 * RATE))
            h = signal.lfilter(*signal.butter(2, [250 / 24000, 7000 / 24000], 'band'), h)
            self.h = h * (10 ** (-10 / 20)) / np.sqrt(np.sum(h ** 2))
            self.zi = np.zeros(len(self.h) - 1)

    def _start_line(self, now, how):
        self.state, self.pos, self.line_start = 'speak', 0, now
        line = self.lines[self.i]
        self.log.append({'t': round(now, 3), 'type': 'line_start', 'line': self.i, 'how': how,
                         'pauses': [(round(now + a, 3), round(now + b, 3)) for a, b in line['pauses']]})

    def next_block(self, out, now):
        mic = self.noise[self.n:self.n + BLOCK].copy()
        self.n += BLOCK
        if self.echo:
            e, self.zi = signal.lfilter(self.h, 1.0, np.tanh(2 * out) / 2, zi=self.zi)
            mic += e.astype(np.float32)
        if np.sqrt(np.mean(out ** 2)) > 10 ** (-50 / 20):
            if not self.agent_spoke and self.state == 'wait':
                self.agent_spoke, self.agent_start = True, now
            self.agent_last_on = now
        agent_talking = now - self.agent_last_on < 0.25
        if self.state == 'speak':
            a = self.lines[self.i]['audio']
            mic += a[self.pos:self.pos + BLOCK] if self.pos + BLOCK <= len(a) else np.pad(a[self.pos:], (0, self.pos + BLOCK - len(a)))
            self.pos += BLOCK
            if self.pos >= len(a):
                self.log.append({'t': round(now, 3), 'type': 'line_end', 'line': self.i})
                self.i += 1
                self.state, self.wait_start, self.agent_spoke, self.next_start = 'wait', now, False, None
                self.plan = {'interrupt_after': self.rng.uniform(1.2, 2.5) if self.rng.random() < self.p_interrupt else None,
                             'bc_after': self.rng.uniform(1.5, 3.0) if self.rng.random() < self.p_backchannel else None}
            return mic
        # waiting for the voice
        if self.bc is not None:
            clip, k = self.bc
            mic += clip[k:k + BLOCK] if k + BLOCK <= len(clip) else np.pad(clip[k:], (0, k + BLOCK - len(clip)))
            self.bc = (clip, k + BLOCK) if k + BLOCK < len(clip) else None
        if self.i >= len(self.lines):
            if (self.agent_spoke and not agent_talking and now - self.agent_last_on > 1.5) or now - self.wait_start > 12:
                return None
            return mic
        if self.agent_spoke and agent_talking:
            since = now - self.agent_start
            if self.plan.get('interrupt_after') is not None and since >= self.plan['interrupt_after']:
                self.plan['interrupt_after'] = None
                self.log.append({'t': round(now, 3), 'type': 'interrupt'})
                self._start_line(now, 'interrupt')
            elif self.plan.get('bc_after') is not None and since >= self.plan['bc_after'] and self.bc is None:
                self.plan['bc_after'] = None
                self.bc = (self.bc_clips[int(self.rng.integers(len(self.bc_clips)))], 0)
                self.log.append({'t': round(now, 3), 'type': 'person_backchannel'})
        elif self.agent_spoke and not agent_talking:
            # People hear from words and tone when a turn is over; the simulation is told
            # instead (the voice has finished its reply), then waits a human-like gap.
            finished = self.conductor is None or self.conductor.agent == 'idle' or now - self.agent_last_on > 1.5
            if not finished:
                self.next_start = None
            else:
                if self.next_start is None:
                    gap = float(np.clip(self.rng.choice(self.gaps), 0.1, 1.2))
                    self.next_start = self.agent_last_on + gap
                if now >= self.next_start:
                    self._start_line(now, 'after_voice')
        elif not self.agent_spoke and now - self.wait_start > 6.0:
            self.log.append({'t': round(now, 3), 'type': 'no_answer'})
            self._start_line(now, 'no_answer')
        return mic


async def run_session(cfg, bank, person, headphones, particles=None, speaker=None):
    playback, rec = Playback(), Recorder()
    conductor = Conductor(cfg, playback, bank, headphones=headphones, log=lambda s: None, particles=particles,
                          speaker=speaker)
    person.conductor = conductor
    sim = SimAudio(playback, conductor.on_block, rec, person)
    stop = asyncio.Event()
    task = asyncio.create_task(conductor.run(stop))
    while conductor.reply_rs is None:
        await asyncio.sleep(0.05)
    sim.start()
    while not sim.done.is_set():
        await asyncio.sleep(0.2)
    await asyncio.sleep(0.3)
    stop.set()
    await task
    return rec, conductor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--speakers', action='store_true', help='simulate laptop-speaker echo (echo canceller on)')
    ap.add_argument('--sessions', default='A,B,C')
    ap.add_argument('--tag', default='run')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--no-openers', action='store_true', help='turn off the quick "oh,"/"yeah," turn openers')
    ap.add_argument('--brain', choices=('split', 'audio'), default=None, help='default: as in app/config.py')
    ap.add_argument('--set', action='append', default=[], metavar='SECTION.NAME=VALUE',
                    help='override a setting from app/config.py, e.g. brain.early_reaction=1 or timing.sure_hold_s=0.25')
    ap.add_argument('--person-voice', default='',
                    help="a Pocket TTS voice for the simulated person (e.g. 'michael'); default: the voice server's male voice")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cfg = Config()
    if args.brain:
        cfg.brain.mode = args.brain
    apply_settings(cfg, args.set, log=lambda s: print(s, flush=True))
    cfg.voice.instructions = PERSONA + cfg.voice.instructions
    if args.person_voice:
        PERSON['speaker'] = speech.PocketVoice(args.person_voice, threads=4)
    need = server.needs_voice(cfg) or not args.person_voice
    proc = server.start(cfg, ROOT / 'logs' / 'server-bench.log') if need else None
    text_proc = server.start_text(cfg, ROOT / 'logs' / 'server-bench-text.log')
    mind_proc = server.start_mind(cfg, ROOT / 'logs' / 'server-bench-mind.log')
    base = server.base_url(cfg)
    try:
        speaker = speech.make(cfg)                     # Pocket TTS when brain.speech=pocket, else None
        bank, all_particles = sounds.prepare(cfg, speaker, base, status=lambda s: print(s, flush=True))
        dessa = json.loads((ROOT / 'results' / 'dessa_metrics.json').read_text(encoding='utf8'))
        gaps = [g for m in dessa if m['speaker1'] == 'Human' for g in m['turn_taking']['floor_transfers']['1'] if g > 0]
        bc_clips = [speak(base, w, seed=50 + k) for k, w in enumerate(PERSON_BACKCHANNELS)]
        recs = []
        for name in args.sessions.split(','):
            rng = np.random.default_rng(args.seed * 100 + ord(name))
            lines = build_lines(base, SCRIPTS[name], rng)
            person = ScriptedPerson(lines, bc_clips, gaps, rng, echo=args.speakers)
            t0 = time.time()
            particles = {} if args.no_openers else all_particles
            rec, cond = asyncio.run(run_session(cfg, bank, person, headphones=not args.speakers, particles=particles,
                                                speaker=speaker))
            mic, voice = rec.tracks()
            sid = f'{args.tag}-{name}'
            sf.write(out / f'{sid}-person.flac', mic, RATE)
            sf.write(out / f'{sid}-voice.flac', voice, RATE)
            cond.save_events(out / f'{sid}-events.jsonl')
            (out / f'{sid}-person.json').write_text(json.dumps(person.log, indent=1), encoding='utf8')
            ev = cond.events
            first = [e['delay_from_user_end'] for e in ev if e['type'] == 'first_audio' and e['delay_from_user_end'] is not None]
            print(f"[{sid}] {len(mic)/RATE:.0f}s talk in {time.time()-t0:.0f}s | replies {sum(e['type']=='respond' for e in ev)} "
                  f"| first-audio delay median {np.median(first) if first else float('nan'):.2f}s "
                  f"| yields {sum(e['type']=='yield' for e in ev)} | backchannels {sum(e['type']=='backchannel' for e in ev)} "
                  f"| person interrupts {sum(e['type']=='interrupt' for e in person.log)}", flush=True)
            recs.append({'id': sid, 'source': 'benchmark', 'duration': len(mic) / RATE,
                         'files': [f'{sid}-person.flac', f'{sid}-voice.flac'],
                         'speaker1': 'Human', 'speaker2': 'LocalVoice', 'markers': [], 'captions': []})
        index = out / f'{args.tag}-recordings.json'
        index.write_text(json.dumps({'recordings': recs}, indent=1), encoding='utf8')
    finally:
        server.stop(mind_proc)
        server.stop(text_proc)
        server.stop(proc)


if __name__ == '__main__':
    main()
