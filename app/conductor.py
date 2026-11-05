"""The conductor: decides, every 32 ms, whether the voice listens, answers,
makes a listening sound, or stops because it is being talked over.

The language model (LFM3-Audio, served by llama-server) does the hearing and
the speaking; this module gives it human conversational timing, tuned to the
numbers measured from Dessa's demos (see config.Timing).
"""
import asyncio
import collections
import json
import random
import re
import time

import numpy as np

from .audio import RATE as OUT_RATE
from .brain import SplitBrain
from .dsp import EchoCanceller, EchoGate, Resampler, from_pcm16, to_pcm16
from .listen import SileroVAD, SmartTurn, VAPLive, VAPTurn
from .realtime import Realtime, decode_audio
from .sounds import PARTICLES

IN = 16000
FRAME = 512          # one listening decision per 32 ms
SEND_CHUNK = 640     # stream the person's audio to the model in 40 ms pieces

# First word of a plain reply -> what kind of opener fits in front of it.
OPENER_KIND = {w: 'react' for w in ('that', "that's", 'thats', 'those', 'it', "it's", 'its', 'sounds', 'this', 'what')}
OPENER_KIND.update({w: 'question' for w in ('how', 'did', 'do', 'does', 'have', 'are', 'is', 'where', 'when', 'why', 'which')})
OPENER_KIND.update({w: 'sorry' for w in ("i'm", 'im', 'sorry')})
ADDED_OPENERS = {'react': ('Oh,', 'Mm,', 'Ooh,'), 'question': ('So,', 'Oh,', 'Mm,'), 'sorry': ('Oh,', 'Mm,')}


ASSISTANT_MODE = re.compile(r"digital assistant|here to help|how can i (help|assist)|what can i help|i'?m (just )?an? (ai|assistant)\b"
                            r"|don'?t have (feelings|emotions|preferences|opinions|personal|a personality|personality|memory)"
                            r"|as an ai|language model", re.I)
# Openings that nearly always lead into assistant-speak, caught from the first few words.
ASSISTANT_OPENING = re.compile(r"\s*(i'?m (just |only )?(an? )?(ai|digital|virtual|assistant|chatbot)\b|as an ai\b"
                               r"|i don'?t have (feelings|emotions|preferences|opinions|personal|a personality|personality)"
                               r"|i'?m here to (help|assist)|i (can'?t|cannot) (browse|search|look things up|access the internet))", re.I)


def _norm(sentence):
    return ' '.join(re.findall(r"[a-z0-9']+", sentence.lower()))


class Tape:
    """Growing 16 kHz recording of what the microphone heard (after echo removal)."""

    def __init__(self):
        self.buf = np.zeros(IN * 60, np.float32)
        self.keep = np.ones(IN * 60, np.float32)    # 0 where it was only the voice's own echo
        self.n = 0

    def add(self, x):
        if self.n + len(x) > len(self.buf):
            grow = max(len(self.buf), len(x))
            self.buf = np.concatenate([self.buf, np.zeros(grow, np.float32)])
            self.keep = np.concatenate([self.keep, np.ones(grow, np.float32)])
        self.buf[self.n:self.n + len(x)] = x
        self.n += len(x)

    def get(self, a, b):
        return self.buf[max(0, a):max(0, min(b, self.n))]

    def mute(self, a, b):
        self.keep[max(0, a):max(0, b)] = 0.0

    def unmute(self, a, b):
        self.keep[max(0, a):max(0, b)] = 1.0

    def get_person(self, a, b):
        """Like get(), with the stretches that were only the voice's echo silenced: for the transcriber,
        which otherwise wrote the voice's own words into what the person said (speakers only)."""
        a, b = max(0, a), max(0, min(b, self.n))
        return self.buf[a:b] * self.keep[a:b]


class Conductor:
    def __init__(self, cfg, playback, sounds=(), headphones=True, log=print, echo_state=None, events_path=None,
                 particles=None, speaker=None):
        self.cfg, self.T = cfg, cfg.timing
        self.particles = particles or {}     # quick turn-openers ("Oh,", "Yeah,") in the voice's own voice
        self.opener = None
        self.last_opener = None
        # Repeat guard: the small model falls into saying the same lines (third live call:
        # "It's just a hiccup... What would you like to talk about?" four times).
        self.recent_sentences = collections.deque(maxlen=24)
        self.first_checked = False
        self.rerolls = 0
        self.skip_word = False
        self.word_skipped = False
        self.opener_end_t = 0.0
        # Events are also appended to disk as they happen (survives a force-closed window).
        self.events_file = open(events_path, 'a', encoding='utf8') if events_path else None
        self.mic_level_db = -90.0       # for a level meter
        self.playback = playback
        self.sounds = list(sounds)
        self.headphones = headphones
        self.say = log
        self.vad = SileroVAD(cfg.paths.vad)
        self.turn_model = SmartTurn(cfg.paths.smart_turn)
        # Learned turn-taking from both voices (timing.turn_model = 'vap'), else Smart Turn alone.
        self.vap = VAPTurn(cfg.paths.vap) if cfg.timing.turn_model == 'vap' else None
        # 'vap_live': the same model running continuously (app/listen.VAPLive), always fresh, no check delay
        self.vap_live = None
        if cfg.timing.turn_model == 'vap_live':
            w = cfg.paths.vap_weights if cfg.paths.vap_weights and cfg.paths.vap_weights.exists() else None
            self.vap_live = VAPLive(cfg.paths.vap.with_name('vap.onnx'), weights=w)
            self.vap = self.vap_live
        self.ref_tape = Tape()               # what the voice itself played, in step with self.tape (for VAP)
        self.mic_rs, self.ref_rs = Resampler(OUT_RATE, IN), Resampler(OUT_RATE, IN)
        self.aec = EchoCanceller()
        self.aec_removed = collections.deque(maxlen=500)   # dB the canceller removes while the voice plays
        self.aec_on = False
        self.gate = EchoGate(margin_db=cfg.timing.echo_margin_db)
        self.echo_state = echo_state
        if echo_state is not None and not headphones:
            self.aec.load(echo_state)
        if cfg.brain.mode == 'split':
            self.rt = SplitBrain(cfg, note=self.note)
            self.rt.speech = speaker
            self.rt.on_opener = self.say_opener
            self.rt.clips = set(self.particles)
        else:
            self.rt = Realtime(cfg.server.host, cfg.server.port, cfg.voice.instructions)
        self.uncommitted_speech = False   # split design: speech not yet sent off to be transcribed
        self.agent_end_t, self.spoke_up = 0.0, False   # speaking up after a long quiet (timing.speak_up_after_s)
        self.speak_up_wait = 6.0
        self.reply_rs = None
        self.tape = Tape()
        self.events = []
        self.q = None
        self.loop = None
        # listening state
        self.frame_e, self.frame_m, self.frame_r = [], [], []
        self.pending = np.zeros(0, np.float32)
        self.user_active = False
        self.speech_run = self.silence_run = 0
        self.onset_t = self.last_voice_t = 0.0
        # turn state
        self.turn_open = False
        self.turn_id = 0
        self.turn_start = 0
        self.sent_upto = 0
        self.next_check = None
        self.checking = False
        self.last_done_p = None
        self.bc_this_pause = False
        self.last_bc_t = -99.0
        self.user_speech_since_bc = 0.0
        self.last_bc_clip = -1
        # the voice's state: idle | waiting (asked, no audio yet) | speaking
        self.agent = 'idle'
        self.requests = 0          # replies asked for; only one is ever in progress
        self.current_req = 0
        self.dropped = set()       # request numbers we cancelled
        self.idle_server = None    # set when no reply is in progress on the server
        self.waiting_since = 0.0
        self.generation_done = True
        self.overlap_start = None
        self.overlap_first = None          # first moment the person spoke over the current reply
        self.over_hist = collections.deque(maxlen=28)   # ~0.9 s of "person talking over the voice"
        self.last_duck_t = -99.0           # when the voice last softened for them
        self.burst_start = None            # when the current stretch of talking over it began
        # It was talked over and stopped: until it next gets a second of speech in, it waits for them
        # to clearly finish. Otherwise it jumped back in at their next short pause and was cut off
        # again (third live call, and back-to-back cut-offs in the benchmark), which sounded like
        # "clicking in and out".
        self.collided = False
        self.masked_run = 0
        self.ducked = False
        self.reply_text = []
        self.t_first_audio = None
        self.hold_until = 0.0
        self.held = []
        self.onset_found = False
        self.closed = False
        # Adapts to the person: each time the voice jumps in while they were only pausing,
        # it waits a little longer next time; each clean turn relaxes that again.
        self.bar_scale = 1.0

    # ---- timeline -------------------------------------------------------
    @property
    def t(self):
        return self.tape.n / IN

    def note(self, kind, **kw):
        ev = {'t': round(self.t, 3), 'type': kind, **kw}
        self.events.append(ev)
        if self.events_file is not None and not self.events_file.closed:
            self.events_file.write(json.dumps(ev) + '\n')
            self.events_file.flush()
        if kind == 'heard' and kw.get('text'):
            self.say(f"You: {kw['text']}")          # the call window shows both sides as they happen
        return ev

    # ---- audio thread -> asyncio ------------------------------------------
    def on_block(self, mic48, out48):
        if self.loop is not None and not self.closed:
            self.loop.call_soon_threadsafe(self.q.put_nowait, (mic48, out48))

    async def run(self, stop_event):
        self.loop = asyncio.get_running_loop()
        self.q = asyncio.Queue()
        self.idle_server = asyncio.Event()
        self.idle_server.set()
        info = await self.rt.connect()
        self.reply_rs = Resampler(self.rt.out_rate, OUT_RATE)
        self.note('session', in_rate=self.rt.in_rate, out_rate=self.rt.out_rate)
        listener = asyncio.create_task(self.server_events())
        try:
            while not stop_event.is_set():
                try:
                    mic, out = await asyncio.wait_for(self.q.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue
                await self.process_block(mic, out)
        finally:
            self.closed = True
            listener.cancel()
            await self.rt.close()
            if self.echo_state is not None and not self.headphones:
                self.aec.save(self.echo_state)
            if self.events_file is not None:
                self.events_file.close()

    # ---- every 10 ms ----------------------------------------------------
    async def process_block(self, mic48, out48):
        m = self.mic_rs(mic48)
        r = self.ref_rs(out48)
        if self.headphones:
            e = m
        else:
            a, _ = self.aec(m, r, adapt=not self.user_active)
            # Many laptop microphones already cancel the speakers. There this canceller has
            # nothing to remove and only leaks a faint copy of the voice back in (first live
            # call), so its output is used only once it really removes echo (>= 3 dB).
            if float(np.dot(r, r)) > 1e-6 * len(r) and not self.user_active:
                self.aec_removed.append(10 * np.log10((np.dot(m, m) + 1e-12) / (np.dot(a, a) + 1e-12)))
            removed = float(np.median(self.aec_removed)) if len(self.aec_removed) >= 100 else 0.0
            helps = removed >= (2.0 if self.aec_on else 4.0)   # switch on at 4 dB, off below 2 dB
            if helps != self.aec_on:
                self.aec_on = helps
                self.note('echo_canceller', on=helps, removes_db=round(float(np.median(self.aec_removed)), 1))
            e = a if helps else m
        e = np.asarray(e, np.float32)
        self.tape.add(e)
        self.ref_tape.add(np.asarray(r, np.float32)[:len(e)])
        if self.vap_live is not None:
            self.vap_live.push(e, np.asarray(r, np.float32)[:len(e)])
        self.frame_e.append(e)
        self.frame_m.append(m)
        self.frame_r.append(r)
        self.pending = np.concatenate([self.pending, e])
        while len(self.pending) >= FRAME:
            start = self.tape.n - len(self.pending)          # where this frame sits on the tape
            frame, self.pending = self.pending[:FRAME], self.pending[FRAME:]
            energies = [float(np.mean(np.concatenate(f)[-FRAME:] ** 2)) for f in (self.frame_m, self.frame_r, self.frame_e)]
            self.frame_e, self.frame_m, self.frame_r = [], [], []
            await self.on_frame(frame, *energies, start=start)
        if self.turn_open and self.tape.n - self.sent_upto >= SEND_CHUNK:
            await self.rt.append(to_pcm16(self.tape.get_person(self.sent_upto, self.tape.n)))
            self.sent_upto = self.tape.n
        if self.agent == 'waiting' and self.held and self.ready_to_speak():
            self.start_speaking()
        if self.collided and self.agent == 'speaking' and self.t - (self.t_first_audio or self.t) >= 1.0:
            self.collided = False        # it got a full second in: the bump is sorted out
        if self.agent == 'speaking' and self.generation_done and not self.playback.busy():
            self.agent = 'idle'
            self.collided = False
            self.bar_scale = max(0.8, self.bar_scale * 0.9)
            self.note('agent_end', bar_scale=round(self.bar_scale, 2))
            self.agent_end_t = self.t            # (speaks up once per silence: reset when they speak)
            self.speak_up_wait = random.uniform(*self.T.speak_up_after_s) if self.T.speak_up_after_s else 0
            if self.user_active:
                # They started answering before the voice finished, as people do.
                self.open_turn_from_first_overlap()
        elif (self.agent == 'idle' and self.T.speak_up_after_s and not self.spoke_up and not self.user_active
              and not self.turn_open and getattr(self.rt, 'checks_text', False) and not getattr(self.rt, 'aside_mode', False)
              and self.t - max(self.agent_end_t, self.last_voice_t) >= self.speak_up_wait and self.agent_end_t > 0):
            self.spoke_up = True               # once per silence
            await self.end_turn('quiet', initiative=True)
        elif self.agent == 'waiting' and self.t - self.waiting_since > 6.0:
            self.note('no_reply')
            self.dropped.add(self.current_req)
            self.agent = 'idle'
            self.generation_done = True

    # ---- every 32 ms ----------------------------------------------------
    async def on_frame(self, frame, e_mic, e_ref, e_out, start=None):
        T = self.T
        frame_s = FRAME / IN
        self.mic_level_db = 10 * np.log10(e_mic + 1e-12)
        p = self.vad(frame)
        speech = p >= (T.vad_off if self.user_active else T.vad_on)
        masked = False
        if not self.headphones:
            self.gate.observe(e_mic, e_ref, e_out)
            if speech and self.gate.playing():
                # With speakers, the voice's own echo must not count as the person talking...
                speech = p >= 0.7 and self.gate.is_person(e_out)
                # ...but sound the gate held back is not silence either: it may be the person.
                masked = not speech
            if not speech and start is not None and self.gate.playing():
                self.tape.mute(start, start + FRAME)   # only its own echo: keep it from the transcriber
        if speech:
            self.speech_run += 1
            self.silence_run = 0
            self.masked_run = 0
        elif masked and self.user_active and self.masked_run < int(1.0 / frame_s):
            self.masked_run += 1          # hold the person's turn open through the echo
        else:
            self.silence_run += 1
            self.speech_run = 0
            self.masked_run = 0
        if not self.user_active and self.speech_run * frame_s >= T.min_speech_s:
            self.user_active = True
            self.onset_t = self.t - self.speech_run * frame_s
            await self.speech_started()
        elif self.user_active and self.silence_run >= 2:
            self.user_active = False
            self.last_voice_t = self.t - self.silence_run * frame_s
            await self.speech_ended()
        if self.user_active:
            self.user_speech_since_bc += frame_s
        # Being talked over: count how much of the last ~0.9 s was the person, so speech
        # that the echo gate lets through only in pieces still adds up to an interruption.
        talking_over = speech and self.agent == 'speaking'
        if talking_over and not any(list(self.over_hist)[-10:]):
            self.burst_start = self.t - frame_s       # a new stretch of them talking over it
        self.over_hist.append(talking_over)
        if talking_over and self.overlap_first is None:
            self.overlap_first = self.t - frame_s
        if self.agent == 'speaking':
            evidence = sum(self.over_hist) * frame_s
            # Both started at once: the voice came in during a pause and they went on. People
            # sort that out within a few hundred ms, and a "mm-hm" this early is rare. With
            # speakers it takes a little more of them, so a few frames of the voice's own echo
            # as it starts cannot stop it.
            collision = (self.burst_start is not None and self.t_first_audio is not None
                         and self.burst_start - self.t_first_audio < T.collision_window_s)
            need = T.yield_after_s
            if collision:
                need = T.collision_yield_s if self.headphones else T.collision_yield_speakers_s
            if evidence >= need:
                await self.yield_turn()
            elif evidence >= T.duck_after_s and not self.ducked:
                # A "mm-hm" is one short burst. A second one soon after means they are trying to
                # get a word in ("Fair enough." ... "Okay." ...), each piece too short to count alone.
                if self.t - self.last_duck_t < T.second_burst_s:
                    self.note('second_burst')
                    await self.yield_turn()
                    return
                self.ducked = True
                self.last_duck_t = self.t
                self.playback.duck(T.duck_gain)
                self.note('duck')
            elif self.ducked and not any(list(self.over_hist)[-10:]):
                self.ducked = False          # it was only a "mm-hm": carry on at full voice
                self.playback.duck(1.0)
                self.over_hist.clear()       # start counting afresh (no instant re-duck)
                self.note('unduck')
        elif self.turn_open and self.agent == 'idle' and not self.user_active:
            await self.during_pause()

    async def speech_started(self):
        self.note('user_start', at=round(self.onset_t, 3))
        self.spoke_up = False
        self.uncommitted_speech = True
        self.rt.discard_prefetch()
        if not self.headphones:
            # Speech is only detected a moment after it starts; with the voice playing, that moment
            # was silenced as echo and the first word lost (fourth live call: "Who's Sonic the
            # Hedgehog?" was heard as "Sonic the Hedgehog"). Keep it for the transcriber.
            self.tape.unmute(int((self.onset_t - self.T.preroll_s) * IN), self.tape.n)
        if self.agent == 'waiting':
            # They kept going right after we decided they were done: take it back.
            await self.yield_turn(immediate=True)
        elif self.agent == 'speaking':
            self.overlap_start = self.onset_t
        else:
            self.open_turn()

    async def speech_ended(self):
        self.note('user_stop', at=round(self.last_voice_t, 3))
        self.overlap_start = None
        if self.turn_open:
            self.next_check = 0          # index into Timing.turn_schedule
            self.bc_this_pause = False
            # Split design: have what they said so far transcribed now, so the words are
            # ready by the time the turn model decides they are done.
            if self.tape.n > self.sent_upto:
                await self.rt.append(to_pcm16(self.tape.get_person(self.sent_upto, self.tape.n)))
                self.sent_upto = self.tape.n
            await self.rt.commit()
            self.uncommitted_speech = False

    def open_turn(self):
        if self.turn_open:
            return
        self.turn_open = True
        self.turn_id += 1
        # Start a little before the speech, but never re-send audio the model already has.
        self.turn_start = max(self.sent_upto, int((self.onset_t - self.T.preroll_s) * IN))
        self.sent_upto = self.turn_start
        self.next_check = None
        self.last_done_p = None

    async def during_pause(self):
        T = self.T
        pause = self.t - self.last_voice_t
        cue = self.rt.turn_cue()
        give_up = T.give_up_pause_s
        if cue == 'open':          # mid-sentence, or thinking out loud: give them time to go on
            give_up = T.thinking_give_up_s if self.rt.still_thinking() else T.open_give_up_s
        if pause >= give_up or (cue == 'done' and pause >= T.done_give_up_s):
            await self.end_turn('long pause')
            return
        sched = T.vap_schedule if self.vap is not None else T.turn_schedule
        if (self.next_check is not None and self.next_check < len(sched) and not self.checking
                and pause >= sched[self.next_check][0]):
            while self.next_check + 1 < len(sched) and pause >= sched[self.next_check + 1][0]:
                self.next_check += 1           # a slow check made us skip a step: use the latest bar
            bar = sched[self.next_check][1] * self.bar_scale
            self.next_check += 1
            self.checking = True
            tid = self.turn_id
            t0 = time.perf_counter()
            if self.vap_live is not None:
                p_done, _, _ = self.vap_live.latest()          # always fresh: no waiting
            elif self.vap is not None:
                # Learned turn-taking: both voices, the last 6 s ("who talks next?")
                n = self.tape.n
                person = self.tape.get_person(n - self.vap.WINDOW, n).copy()
                agent = self.ref_tape.get(n - self.vap.WINDOW, n).copy()
                p_done, _ = await self.loop.run_in_executor(None, self.vap, person, agent)
            else:
                audio = self.tape.get(self.turn_start, self.tape.n).copy()
                p_done = await self.loop.run_in_executor(None, self.turn_model, audio)
            self.checking = False
            self.last_done_p = p_done
            cue = self.rt.turn_cue()             # split design: how their words end, once transcribed
            if cue == 'open':
                bar *= T.open_bar_scale
            elif cue == 'question':
                bar *= T.question_bar_scale
            bar = round(min(0.95, bar), 3)
            self.note('turn_check', pause=round(pause, 3), p_done=round(p_done, 3), bar=bar, cue=cue,
                      ms=round(1000 * (time.perf_counter() - t0), 1))
            if tid == self.turn_id and self.turn_open and not self.user_active and self.agent == 'idle':
                if p_done >= bar:
                    await self.end_turn('turn model')
                    return
                if p_done >= (T.vap_prefetch_bar if self.vap is not None else T.prefetch_bar) or cue == 'question':
                    await self.rt.prefetch()     # split design: start the reply in case they are done
        # They are describing something without naming it and the brain is sure what it is: say the
        # guess at this short pause ("Superman?"), the way people cut in (seventh and eighth live calls).
        guess = getattr(self.rt, 'pending_guess', lambda: None)()
        if (guess and pause >= T.guess_pause_s and self.turn_open and not self.user_active and self.agent == 'idle'
                and not self.checking):
            self.note('guess', guess=guess, pause=round(pause, 3))
            await self.end_turn('guess', guess=guess)
            return
        lo, hi = T.backchannel_pause_s
        cue = self.rt.turn_cue()
        # Their words stop mid-sentence: they are clearly going on, so a "mm-hm" fits even when
        # the turn model is less sure of that.
        if self.vap is not None:
            most = T.vap_backchannel_max_open if cue == 'open' else T.vap_backchannel_max
        else:
            most = T.backchannel_max_done_prob_open if cue == 'open' else T.backchannel_max_done_prob
        if (self.sounds and not self.bc_this_pause and lo <= pause <= hi and self.last_done_p is not None
                and cue != 'question' and self.last_done_p <= most and not getattr(self.rt, 'aside_mode', False)
                and self.t - self.last_bc_t >= T.backchannel_min_gap_s):
            self.bc_this_pause = True
            chance = min(1.0, T.backchannel_per_min * self.user_speech_since_bc / 60)
            if random.random() < chance:
                k = random.choice([i for i in range(len(self.sounds)) if i != self.last_bc_clip] or [0])
                self.last_bc_clip = k
                self.playback.push_sfx(self.sounds[k] * T.backchannel_gain)
                self.last_bc_t = self.t
                self.user_speech_since_bc = 0.0
                self.note('backchannel', clip=k, pause=round(pause, 3))

    async def end_turn(self, why, initiative=False, guess=None):
        if self.tape.n > self.sent_upto:
            await self.rt.append(to_pcm16(self.tape.get_person(self.sent_upto, self.tape.n)))
            self.sent_upto = self.tape.n
        self.turn_open = False
        self.agent = 'waiting'
        self.waiting_since = self.t
        self.ducked = False
        self.playback.duck(1.0)
        self.generation_done = False
        self.reply_text = []
        self.t_first_audio = None
        self.overlap_start = None
        if not self.idle_server.is_set():     # a cancelled reply is still closing
            try:
                await asyncio.wait_for(self.idle_server.wait(), 1.0)
            except asyncio.TimeoutError:
                pass
        self.idle_server.clear()
        self.requests += 1
        self.current_req = self.requests
        sure = why == 'long pause' or (self.last_done_p or 0) >= (self.T.vap_sure_bar if self.vap is not None
                                                                  else self.T.sure_bar)
        self.hold_until = self.last_voice_t + (self.T.sure_hold_s if sure else self.T.play_hold_s) * max(1.0, self.bar_scale)
        self.held = []
        self.held_s = 0.0
        self.onset_found = False
        self.opener, self.skip_word, self.word_skipped, self.opener_end_t = None, False, False, 0.0
        self.reply_rs = Resampler(self.rt.out_rate, OUT_RATE)   # no tail from a cancelled reply
        self.first_checked = False
        if why != 'reroll':
            self.rerolls = 0
            # Split design: a reaction already picked from what they said ("Oh no,") opens the turn
            # now, the moment they are done, and the reply is written on from it.
            word = None if (initiative or guess) else self.rt.early_reaction()
            if word and self.say_opener(word):
                self.rt.use_reaction(word)
        if initiative:
            await self.rt.respond(commit=False, initiative=True)
        elif guess:
            await self.rt.respond(commit=self.uncommitted_speech, guess=guess)
        else:
            await self.rt.respond(commit=self.uncommitted_speech)
        self.uncommitted_speech = False
        self.note('respond', why=why, user_end=round(self.last_voice_t, 3), p_done=self.last_done_p,
                  hold_until=round(self.hold_until, 3))

    async def reroll(self):
        """Ask again for a reply that would have opened with something it just said."""
        try:
            await asyncio.wait_for(self.idle_server.wait(), 1.0)
        except asyncio.TimeoutError:
            pass
        if self.agent == 'waiting' and not self.user_active:
            self.rerolls += 1
            self.turn_open = False
            await self.end_turn('reroll')

    def check_first_sentence(self):
        """Called as reply text arrives, before any of its sound is heard.

        The text leads the sound, but a long first sentence finishes only after the sound
        has started, so the opening words are checked as soon as a few have arrived.
        """
        text = ''.join(self.reply_text)
        m = re.match(r'\s*(.+?[.!?])(\s|$)', text)
        words = len(text.split())
        if not m and words < 4:
            return False
        sentence = m.group(1) if m else text
        if m or words >= 8:
            self.first_checked = True
        first = _norm(sentence)
        repeat = bool(m) and len(first) > 12 and first in self.recent_sentences
        # The small model slips into assistant mode about half the time when the talk turns
        # to itself (tools/probe_loops.py); asking once more usually gets a casual reply.
        assistant = bool(ASSISTANT_MODE.search(sentence) or ASSISTANT_OPENING.match(sentence))
        if (repeat or assistant) and self.rerolls < 1:
            self.first_checked = True
            self.note('reroll', sentence=sentence, reason='repeat' if repeat else 'assistant')
            self.dropped.add(self.current_req)
            asyncio.ensure_future(self.rt.cancel())
            asyncio.ensure_future(self.reroll())
            return True
        return False

    def hold_reply_audio(self, x, floor=0.008, lead_s=0.08, give_up_s=1.5):
        """Keeps reply audio until it is time to speak, skipping the model's silent lead-in.

        LFM3-Audio often opens a reply with 0.4-1.2 s of near-silence before the first
        word. Only the last lead_s of that silence is kept, so the voice starts right
        away but its first sound still has a natural attack.
        """
        self.held.append(x)
        self.held_s += len(x) / OUT_RATE
        if self.onset_found:
            return
        a = np.concatenate(self.held)
        w = int(0.01 * OUT_RATE)
        n = len(a) // w
        loud = np.sqrt(np.mean(a[:n * w].reshape(n, w) ** 2, axis=1)) > floor if n else np.zeros(0, bool)
        if loud.any():
            start = max(0, int(np.argmax(loud)) * w - int(lead_s * OUT_RATE))
            self.held = [a[start:]]
            self.onset_found = True
            self.note('lead_in_skipped', seconds=round(start / OUT_RATE, 2))
        elif self.held_s > give_up_s:
            self.onset_found = True          # nothing loud yet: speak anyway
        else:
            self.held = [a[-int(lead_s * OUT_RATE):]]

    def maybe_open_with_particle(self):
        """If the reply starts with "Oh,", "Yeah,"..., say our ready-made one right away.

        The model's text arrives ~0.1 s after it is asked; its sound ~0.3 s later. Saying
        the opening word at once is what people (and Dessa, 44% of its turns) do, and if
        the person was not finished after all, a lone "oh" works as a listening sound.
        """
        text = ''.join(self.reply_text)
        m = re.match(r"\s*([A-Za-z']+)([^A-Za-z']?)", text)
        if not m or (not m.group(2) and len(text) < 12):
            return                           # the first word is not complete yet
        first = m.group(1).lower()
        if self.t - self.last_bc_t < 1.0 or self.playback.sfx_busy():
            self.opener = ''                 # a listening sound just played: it already opened the turn
            return
        if self.rt.turn_cue() == 'open':
            self.opener = ''                 # their words stop mid-sentence: no early "oh" into the gap
            return
        spoken = PARTICLES.get(first)
        clip = self.particles.get(spoken) if spoken else None
        skip = clip is not None              # the model says this word itself: drop its copy
        if clip is None and random.random() < self.T.added_opener_rate:
            # The reply starts plainly ("That sounds...", "What..."): put a fitting opener in
            # front, the way people do, never the same one twice in a row.
            kind = OPENER_KIND.get(first)
            choices = [c for c in ADDED_OPENERS.get(kind, ()) if c in self.particles and c != self.last_opener]
            if choices:
                spoken = random.choice(choices)
                clip = self.particles[spoken]
        self.opener = spoken or ''
        if clip is not None:
            self.playback.push_sfx(clip)
            self.opener_end_t = self.t + len(clip) / OUT_RATE
            self.skip_word = skip
            self.last_opener = spoken
            last = next((e for e in reversed(self.events) if e['type'] == 'respond'), None)
            self.note('opener', word=spoken, added=not skip,
                      delay_from_user_end=round(self.t - last['user_end'], 3) if last else None)

    def say_opener(self, word):
        """Split design: the reply's first word ("Oh,") the moment it is written. Says it at once
        from the ready-made clips; the brain then leaves it out of the spoken sentence, so nothing
        has to be cut from the speech. Returns whether it was said."""
        if self.agent != 'waiting' or self.opener is not None:
            return False
        spoken = PARTICLES.get(word.lower(), word)      # "Oh," ... or a whole quick reaction, "Totally!"
        clip = self.particles.get(spoken)
        # Not the same clip twice in a row: an identical "Oh," every turn sounds canned, so then the
        # sentence is simply voiced with its first word.
        if (clip is None or spoken == self.last_opener or self.t - self.last_bc_t < 1.0
                or self.playback.sfx_busy() or self.rt.turn_cue() == 'open'):
            self.opener = ''
            self.last_opener = None
            return False
        self.opener = spoken
        self.playback.push_sfx(clip)
        self.opener_end_t = self.t + len(clip) / OUT_RATE
        self.last_opener = spoken
        self.note('opener', word=spoken, added=False, delay_from_user_end=round(self.t - self.last_voice_t, 3))
        return True

    def skip_model_first_word(self, floor=0.008, give_up_s=0.9):
        """Drops the model's own first word (we already said it), up to the pause after it."""
        a = np.concatenate(self.held)
        w = int(0.01 * OUT_RATE)
        n = len(a) // w
        if n < 10:
            return
        loud = np.sqrt(np.mean(a[:n * w].reshape(n, w) ** 2, axis=1)) > floor
        k0 = int(np.argmax(loud)) if loud.any() else 0
        quiet = 0
        for k in range(k0 + 8, n):
            if not loud[k]:
                quiet += 1
            else:
                if quiet >= 6:               # a 60 ms pause after the first word
                    self.held = [a[max(0, (k - 2) * w):]]
                    self.word_skipped = True
                    self.note('opener_word_skipped', seconds=round((k - 2) * w / OUT_RATE, 2))
                    return
                quiet = 0
        if (n - k0) * 0.01 > give_up_s:
            self.word_skipped = True         # no pause found: play the model's audio as it is
            self.note('opener_no_pause')

    def ready_to_speak(self):
        hold = self.hold_until
        if self.opener:              # "Oh," already took the turn: go straight on after it
            hold = min(hold, self.opener_end_t)
        cue = self.rt.turn_cue()
        if cue == 'question':        # a question wants a quick answer
            hold = min(hold, self.last_voice_t + self.T.question_hold_s * max(1.0, self.bar_scale))
        elif cue == 'open':          # their words stop mid-sentence: give them time to go on
            hold = max(hold, self.last_voice_t + self.T.open_hold_s * max(1.0, self.bar_scale))
        if self.collided and cue != 'question':
            hold = max(hold, self.last_voice_t + self.T.after_collision_hold_s)
        # Never on top of a short sound that is still playing (second live call: "two things at once").
        return (self.onset_found and self.t >= hold and not self.playback.sfx_busy()
                and (not self.skip_word or self.word_skipped) and self.t >= self.opener_end_t - 0.03)

    def start_speaking(self):
        for x in self.held:
            self.playback.push(x)
        self.held = []
        self.over_hist.clear()
        self.overlap_first = None
        self.burst_start = None
        self.last_duck_t = -99.0
        self.agent = 'speaking'
        self.t_first_audio = self.t
        self.rt.reply_started()
        last = next((e for e in reversed(self.events) if e['type'] == 'respond'), None)
        self.note('first_audio', delay_from_user_end=round(self.t - last['user_end'], 3) if last else None)

    async def yield_turn(self, immediate=False):
        """Being talked over: stop speaking and listen."""
        self.dropped.add(self.current_req)
        await self.rt.cancel()
        heard = round(self.t - (self.t_first_audio or self.t), 3)
        self.held = []
        if immediate:                     # they were still mid-thought (nothing was heard)
            self.bar_scale = min(1.6, self.bar_scale * 1.05)
        elif heard < 1.2:                 # ...and this time they heard the voice jump in
            self.bar_scale = min(1.8, self.bar_scale * 1.35)
        if not immediate:
            self.collided = True          # they took the floor: let them finish this time
            self.playback.fade_out(self.T.fade_s)
            self.rt.reply_cut(heard)
        self.note('yield', immediate=immediate, overlap_s=round(self.t - (self.overlap_start or self.t), 3),
                  played_s=heard)
        self.agent = 'idle'
        self.generation_done = True
        self.overlap_start = None
        self.ducked = False
        self.open_turn_from_first_overlap()

    def open_turn_from_first_overlap(self):
        """The model hears everything the person said while talking over the voice."""
        if self.overlap_first is not None and self.t - self.overlap_first < 4.0:
            self.onset_t = min(self.onset_t, self.overlap_first)
        self.overlap_first = None
        self.over_hist.clear()
        self.open_turn()

    # ---- what the model sends back ----------------------------------------
    async def server_events(self):
        async for ev in self.rt.events():
            kind = ev.get('type')
            if kind == 'response.done':
                self.idle_server.set()
            if self.current_req in self.dropped:
                continue
            if kind == 'response.output_audio.delta':
                if self.agent == 'idle':
                    continue
                x = self.reply_rs(from_pcm16(decode_audio(ev['delta']))).astype(np.float32)
                if self.agent == 'waiting':
                    self.hold_reply_audio(x)
                    if self.onset_found and self.skip_word and not self.word_skipped:
                        self.skip_model_first_word()
                    if self.ready_to_speak():
                        self.start_speaking()
                else:
                    self.playback.push(x)
            elif kind == 'response.output_text.delta':
                self.reply_text.append(ev.get('delta', ''))
                if (not self.rt.checks_text and not self.first_checked and self.agent == 'waiting'
                        and self.check_first_sentence()):
                    continue
                if self.opener is None and self.agent == 'waiting' and self.particles and not self.rt.checks_text:
                    self.maybe_open_with_particle()
            elif kind == 'response.done':
                self.generation_done = True
                if self.agent == 'waiting' and self.held and not self.onset_found:
                    self.onset_found = True      # a very quiet reply: play what there is
                if self.agent == 'waiting' and self.skip_word and not self.word_skipped:
                    self.word_skipped = True     # short reply: nothing more to wait for
                if self.agent == 'waiting' and not self.held:
                    self.agent = 'idle'          # the reply had no sound: listen again
                    self.note('empty_reply')
                trimmed = self.playback.trim_tail()
                if trimmed:
                    self.note('trimmed_tail', seconds=round(trimmed / OUT_RATE, 2))
                text = ''.join(self.reply_text).strip()
                said = [s for s in re.findall(r'[^.!?]+[.!?]', text) if s.strip()]
                if (not self.rt.checks_text and len(said) >= 2 and _norm(said[-1]) in self.recent_sentences
                        and self.agent == 'speaking'):
                    # It would end on a line it already used: stop at the pause before it.
                    cut = self.playback.cut_last_sentence(expected_s=0.3 * len(said[-1].split()) + 0.2)
                    if cut:
                        self.note('repeat_cut', sentence=said[-1].strip(), seconds=round(cut, 2))
                        said = said[:-1]
                        text = ' '.join(s.strip() for s in said)
                for s in said:
                    self.recent_sentences.append(_norm(s))
                if text:
                    self.note('reply_text', text=text)
                    self.say(f"{getattr(self.rt, 'name', 'Voice')}: {text}")
            elif kind == 'error':
                self.note('server_error', message=(ev.get('error') or {}).get('message'))

    def save_events(self, path):
        with open(path, 'w', encoding='utf8') as f:
            for ev in self.events:
                f.write(json.dumps(ev) + '\n')
