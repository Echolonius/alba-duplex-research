"""All tunable numbers in one place.

The timing targets come from measuring Block's 21 public Dessa conversations
(results/dessa_summary.json): Dessa starts its turn a median 0.16 s after the
other person stops (41% of its turn starts overlap their last words), makes
about 11 short listening sounds ("mm-hm", "yeah") per minute of the other
person's speech, and keeps talking a median 0.38 s when talked over.
"""
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Paths:
    llama_server: Path = ROOT / 'bin' / 'llama-b12094' / 'llama-server.exe'
    # 'Q4_K_M' (default): ~30% faster replies on the 4 GB card. 'Q8_0': a little closer to the original.
    quant: str = 'Q4_K_M'
    vad: Path = ROOT / 'models' / 'vad' / 'silero_vad.onnx'
    smart_turn: Path = ROOT / 'models' / 'vad' / 'smart-turn-v3.2-cpu.onnx'
    vap: Path = ROOT / 'models' / 'vap' / 'vap_dyn.onnx'      # learned turn-taking (app/listen.VAPTurn)
    vap_weights: Path = ROOT / 'models' / 'vap' / 'vap_ft.pt'  # re-trained weights for 'vap_live', if present
    sounds: Path = ROOT / 'models' / 'sounds'          # cached listening sounds per voice
    sessions: Path = ROOT / 'sessions'                 # recordings of each conversation
    # A trained voice model instead of the stock one, e.g. 'models/alba-m2' (holds model-Q4_K_M.gguf and
    # mmproj-f16.gguf, from tools/export_alba.py). Empty: the stock LFM3-Audio-2B.
    voice_model_dir: str = ''

    @property
    def model(self):
        if self.voice_model_dir:
            return ROOT / self.voice_model_dir / 'model-Q4_K_M.gguf'
        return ROOT / 'models' / 'lfm3-audio' / f'LFM3-Audio-2B-{self.quant}.gguf'

    @property
    def mmproj(self):
        if self.voice_model_dir:
            return ROOT / self.voice_model_dir / 'mmproj-f16.gguf'
        return ROOT / 'models' / 'lfm3-audio' / f'mmproj-LFM3-Audio-2B-{self.quant}.gguf'


@dataclass
class Server:
    host: str = '127.0.0.1'                            # never reachable from other computers
    port: int = 9931
    context: int = 8192
    gpu_layers: int = 99


@dataclass
class Timing:
    # Listening
    vad_on: float = 0.5               # speech probability that starts speech
    vad_off: float = 0.35             # ...and that ends it
    min_speech_s: float = 0.064       # ignore blips shorter than this
    preroll_s: float = 0.30           # audio kept from before a detected speech start
    # Taking the turn. After each silence length, answer if the Smart Turn score
    # reaches that step's bar. Calibrated on the captioned Dessa demos (118 turn
    # ends, 342 mid-sentence pauses; tools/calibrate_turn.py): median decision
    # after 0.2 s of silence, about 6% of mid-sentence pauses trigger an early
    # reply, most of which is cancelled before any sound plays.
    # The 0.2/0.3 s bars were raised from 0.15 after the first live call, where the voice
    # answered "Alright, uh..." (score 0.19) before the question came.
    turn_schedule: tuple = ((0.064, 0.80), (0.10, 0.35), (0.20, 0.22), (0.30, 0.20), (0.45, 0.10), (0.60, 0.10), (0.80, 0.10))
    # Which model decides that a turn is over: 'vap' (learned from two-voice conversations, hears both
    # sides) or 'smart_turn' (hears only the person). Its own bars, from Dessa's demos
    # (tools/eval_vap.py --window 6): each lets through about 5-10% of mid-sentence pauses and catches
    # 46-76% of real turn ends, where Smart Turn caught 29-33% at the same rate.
    turn_model: str = 'vap'
    vap_schedule: tuple = ((0.10, 0.54), (0.20, 0.52), (0.30, 0.50), (0.45, 0.47), (0.60, 0.42), (0.80, 0.40))   # fitted on the user's calls (tools/eval_turns_calls.py, ~10% of pauses cut into)
    vap_prefetch_bar: float = 0.40      # start writing a reply in case they are done
    vap_sure_bar: float = 0.75
    vap_backchannel_max: float = 0.35   # an "mm-hm" only while VAP is sure they go on
    vap_backchannel_max_open: float = 0.50
    give_up_pause_s: float = 1.0         # answer anyway after this much silence
    open_give_up_s: float = 1.6          # ...after words that stop mid-sentence ("...and the")
    thinking_give_up_s: float = 2.6      # ...after thinking out loud ("...you know, uhm")
    done_give_up_s: float = 0.7          # ...or this much, after a finished sentence (split design)
    # Split design: from this turn-model score on (the lowest bar in the schedule), the reply
    # is written and voiced ahead, unheard, so it is ready the moment the turn is decided.
    prefetch_bar: float = 0.10
    # The reply is asked for at once but only heard after this much silence, unless the
    # turn model is sure. The model needs ~0.26 s to start talking anyway, so this costs
    # little, and a person who was only pausing never hears the voice jump in.
    play_hold_s: float = 0.40
    sure_bar: float = 0.8
    # Even when the turn model is sure, wait this much silence before sound. The reply takes
    # ~0.3 s to arrive anyway; in the third live call, two "sure" starts landed in pauses
    # where the person was only thinking.
    sure_hold_s: float = 0.30
    # Split design: the transcript says how their words end (app/brain.py, turn_cue). Ending
    # mid-sentence ("...and", no full stop) raises the bar and waits longer before sound;
    # a question lowers the bar and is answered sooner.
    open_bar_scale: float = 2.5
    open_hold_s: float = 0.70
    question_bar_scale: float = 0.5
    question_hold_s: float = 0.15
    # Turn openers: when a reply starts plainly ("That sounds...", "What..."), put a quick
    # "Oh,"/"Mm,"/"So," in front this often. Dessa opens 44% of its turns with one, but in
    # the second live call these fixed-tone clips felt out of step with the mood, so the
    # voice now only says the opener early when its own reply starts with it.
    added_opener_rate: float = 0.0
    # Listening sounds (backchannels)
    # Per minute of the person's speech (Dessa ~11, people 8.5). Set above the target because the
    # voice usually takes its turn at a pause, so there are few chances: at 8 it played only 7 of 12.
    backchannel_per_min: float = 12.0
    backchannel_min_gap_s: float = 4.0   # never two closer than this (eighth live call: 118 in 18 min, "it keeps saying uh huh")
    backchannel_pause_s: tuple = (0.15, 0.50)  # play inside a pause of this length...
    backchannel_max_done_prob: float = 0.15    # ...only when the person is clearly not finished
    backchannel_max_done_prob_open: float = 0.35   # ...or their words stop mid-sentence (split design)
    backchannel_gain: float = 0.4
    # Only the soft, neutral sounds: "Yeah."/"Okay."/"Right." carry a tone of their own,
    # which felt off in the second live call.
    backchannel_words: tuple = ('Mhm.', 'Hmm.', 'Mm-hm.')
    # After the voice has spoken and they stay quiet this long, it picks the conversation back up
    # (brain.SPEAK_UP); 0 = never. Once per silence.
    speak_up_after_s: tuple = (4.5, 7.5)   # a fresh wait in this range each time (ninth call: a fixed wait "feels hard coded")
    guess_pause_s: float = 0.25          # a ready guess ("Superman?") is said at a pause this long (brain.guessing)
    # Being talked over. People's "mm-hm"s in the Dessa demos last a median 0.31 s
    # (70% under 0.45 s, 90% under 0.70 s); when Dessa is really interrupted it keeps
    # going a median 0.38 s (a quarter of the time 0.8 s or more). So: soften at 0.25 s,
    # stop at 0.65 s, and carry on at full voice if it was only a "mm-hm".
    duck_after_s: float = 0.25
    duck_gain: float = 0.25           # about -12 dB
    yield_after_s: float = 0.65
    fade_s: float = 0.2               # trail off rather than cut (third live call: "abrupt")
    # ...but if they start talking within this long of the voice starting (it came in during a
    # pause and they went on), stop after this much of them instead: in the benchmark most
    # times the voice gave way were such collisions, with 0.7-1 s of both talking.
    collision_window_s: float = 0.8
    collision_yield_s: float = 0.25
    collision_yield_speakers_s: float = 0.40   # with speakers: a few echo frames must not count
    # A second burst of their speech this soon after the first softening: stop (see conductor).
    second_burst_s: float = 2.5
    # After being talked over and stopping, wait this much silence (unless asked a question)
    # until it next gets a second of speech in: the person who cut in usually has more to say.
    # (Applied only after the voice's own early starts, it slowed every reply without fewer
    # back-to-back cut-offs; those came right after the person interrupted.)
    after_collision_hold_s: float = 0.5
    # With speakers: how much louder than the typical leftover echo the person must be.
    echo_margin_db: float = 10.0


@dataclass
class Voice:
    # The live chat always speaks in LFM3-Audio's built-in female voice; the short sounds are
    # made by its speech generator in that same voice.
    instructions: str = 'Respond with interleaved text and audio.'


@dataclass
class Brain:
    # 'split': the voice model hears and speaks, a small text model writes the words (app/brain.py).
    # 'audio': the voice model does it all; it slips into assistant mode about half the time
    # once the talk turns to itself (tools/probe_loops.py).
    mode: str = 'split'
    # Gemma 4 E2B (Google, Apache 2.0, built for phones) since the sixth live call: replayed on that
    # call's 41 turns it read the person's mood and criticism far better than LFM2.5-1.2B, LFM2-8B-A1B
    # or Qwen3-4B, with no apology loops (results/audit/probe-call6-*.txt). It uses ~1.6 GB of the
    # graphics card next to the voice model and writes ~55 word-pieces a second. E4B was better still
    # but needs 3.1 GB. The old brain: LFM2.5-1.2B-Instruct-Q4_K_M.gguf with text_template_kwargs ''.
    text_model: Path = ROOT / 'models' / 'text' / 'gemma-4-E2B-it-Q4_0.gguf'
    text_port: int = 9941
    text_context: int = 4096
    # Extra llama-server flags for the text model, e.g. '--n-cpu-moe 99': a mixture-of-experts
    # model's experts stay in system memory, so an 8B model needs ~0.7 GB of graphics memory.
    text_args: str = ''
    # Written at the start of every reply before the model goes on. '<think></think>' makes a
    # model that reasons first (LFM2.5-8B-A1B) answer at once, as if it had already thought.
    text_prefill: str = ''
    # Options for the text model's chat template, as JSON, e.g. '{"enable_thinking": false}' for
    # Gemma 4, which otherwise thinks before every reply.
    text_template_kwargs: str = '{"enable_thinking": false}'
    # Let the text model read the person's words while they are still talking (brain._read_along),
    # so that little is left to read when they stop.
    read_along: bool = True
    # "A fast mouth and a slow mind": while the person talks, a bigger model on the processor writes
    # short private notes on what they said (what they mean, a fact, an angle); the fast text model
    # above then answers the moment they stop, with those notes in front of it (brain._think).
    mind: bool = False
    mind_model: Path = ROOT / 'models' / 'text' / 'LFM2-8B-A1B-Q4_K_M.gguf'
    mind_port: int = 9951
    mind_args: str = '--device none -t 4'
    mind_tokens: int = 48
    temperature: float = 0.8
    max_sentences: int = 6          # room for a longer answer when asked (seventh call: "it doesn't know how to ramble")
    max_words: int = 90
    first_clause_words: int = 5     # a long first sentence starts sounding at its first comma after this many words
    history_turns: int = 10        # earlier exchanges the text model sees
    # The speech generator's built-in voice: 'female' (the one the chat uses, and the listening
    # sounds) or 'male'. LFM3-Audio ignores voice samples: the server logs "the model has no
    # voice conditioning, the speaker file is ignored".
    voice: str = 'female'
    # Who speaks: 'lfm3' (the voice server's /tts, on the graphics card) or 'pocket' (Kyutai's
    # Pocket TTS on the processor, phone-class: app/speech.py) in one of its built-in voices.
    speech: str = 'pocket'
    # Who hears: 'lfm3' (the voice server's live transcriber, on the graphics card) or 'moonshine'
    # (Moonshine Voice's streaming recognizer on the processor, phone-class: app/hearing.py).
    hearing: str = 'lfm3'
    moonshine_arch: str = 'small-streaming'     # tiny-streaming / small-streaming / medium-streaming
    name: str = ''                  # the voice's name; '' = the Pocket voice's own name ("Alba")
    aside_timeout_s: float = 30.0   # talk to Claude counts as such until they name the voice, ask it something, or this long
    guessing: bool = True           # guess out loud what they are describing (brain._maybe_guess)
    pocket_voice: str = 'alba'
    pocket_threads: int = 2
    # Pick the opening reaction ("Oh no,") from the live transcript while they are still talking,
    # so the turn can open the moment they stop: in the benchmark the median turn start fell from
    # 0.50 to 0.35 s, with the same replies otherwise.
    early_reaction: bool = True


@dataclass
class Config:
    paths: Paths = field(default_factory=Paths)
    server: Server = field(default_factory=Server)
    timing: Timing = field(default_factory=Timing)
    voice: Voice = field(default_factory=Voice)
    brain: Brain = field(default_factory=Brain)
    headphones: bool = False          # False: run the echo canceller for laptop speakers


def apply_settings(cfg, items, log=print):
    """Overrides from the command line: ['brain.early_reaction=1', 'timing.sure_hold_s=0.25', ...]."""
    for item in items:
        key, value = item.split('=', 1)
        section, name = key.split('.')
        obj = getattr(cfg, section)
        old = getattr(obj, name)
        new = value.lower() in ('1', 'true', 'yes') if isinstance(old, bool) else type(old)(value)
        setattr(obj, name, new)
        log(f'setting {key} = {new!r} (was {old!r})')
