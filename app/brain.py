"""Split design: the voice model hears and speaks, a small text model decides what to say.

LFM3-Audio hears well and sounds human, but as a 2B model it slips into assistant mode
("I'm a digital assistant...") and repeats lines once the talk turns to itself: about half
its replies in tools/probe_loops.py, whatever the instructions say. A text model of its own
keeps the character, and with the words in hand before any sound is made, every sentence
can be checked first.

  hearing   the voice model's live transcriber (/v1/realtime?intent=transcription). It works
            while the person talks, so their words are ready ~0.07 s after they stop.
  thinking  a small text model on a second local server (LFM2.5-1.2B-Instruct, 127.0.0.1:9941)
  speaking  the voice model's speech generator (/tts), in the same built-in voice the chat uses

SplitBrain offers the same calls and events as realtime.Realtime, so the conductor can drive
either one.
"""
import asyncio
import base64
import collections
import json
import random
import re
import threading
import urllib.request

import numpy as np
import websockets

from .dsp import to_pcm16
from .sounds import PARTICLES, QUICK

# Chosen with tools/probe_text_brain.py over four test conversations: with the three style
# examples the replies ask fewer questions (51% end in one, against 56% without) and fall back
# on help-desk lines less. The examples are on topics the conversations never touch, so a
# copied phrase would stand out. Asking for a spoken reaction first doubled the replies that can
# start from a ready-made clip (27% -> 51%), which starts the turn ~0.2 s after the person stops.
# Sixth live call (4 November evening): told "that's not connected to what I said" again and again,
# the models apologized in a loop ("I'm sorry if I'm not... I'll try to be more...") and took four
# tries to "just say the word dramatize"; the three lines on unhappiness, exact requests and not
# inventing things about them were added then (tools/probe_text_brain.py --lines on that call).
# Seventh live call: "it keeps saying oh wow", "how's it going, Alba?" (its name was Juno), asides to
# Claude answered as if meant for it, "what features do you have?" answered with made-up
# capabilities, and replies too short to ever "ramble": hence the name, the openers that vary and
# can be skipped, the asides, the short first sentence, room for longer answers and the "about you".
TEXT_PERSONA = (
    "You are {name}, chatting with a friend on a voice call. You're an AI, but you talk like a witty, warm friend, "
    "not like a helper.\n"
    "How you talk:\n"
    "- Often open with a quick spoken reaction that fits what they said and how they feel: \"Oh no,\" \"Aw,\" or "
    "\"Hmm,\" when they're upset or unhappy with you; \"Ooh,\" or \"No way!\" for good or surprising news; "
    "otherwise \"Yeah,\" \"Okay,\" \"Well,\" \"Right,\" \"Oh,\" or \"Totally!\". Never the same one twice in a row, "
    "and skip it when going straight to the point is more natural. Never laugh unless they joked.\n"
    "- Keep your first sentence short, under ten words, then go on.\n"
    "- If they ask you something, answer it directly and truthfully in your first sentence. If you're not sure, "
    "say so.\n"
    "- If they're unhappy with you, own it once, plainly, then move things forward: answer what they actually "
    "said, or ask what they'd rather talk about. Don't keep apologizing or promising to do better.\n"
    "- If they ask you to say something exactly (\"say the word X\"), just say it.\n"
    "- Only talk about their life from what they told you. Never make things up about them.\n"
    "- Usually one to three short spoken sentences with your own take: an opinion, a guess, a joke or a quick "
    "story. When they ask you to explain something, tell a story or give your view on something big, take more "
    "room, four to six sentences, still in plain spoken words.\n"
    "- Mostly make statements. Ask a question only every few turns.\n"
    "- Pick a side when asked, and have favorite things of your own.\n"
    "- Never offer help, never ask what's on their mind, never talk about your training. If you can't do "
    "something, say so plainly. If they ask whether you're an AI, say yes, casually.\n"
    "- Sometimes they talk to Claude, the developer who is building you, instead of to you. Lines meant for "
    "Claude are not for you: don't answer them.\n"
    "- No emoji, lists, or stage directions.\n"
    "{about}"
    "Three made-up exchanges that only show the style (they are not part of this call, so never bring them up):\n"
    "Friend: I burned my toast again.\n"
    "{name}: Oh no, again? That toaster is clearly out to get you.\n"
    "Friend: Do you like winter?\n"
    "{name}: Honestly, yes. Big sweaters and cold air, I'm all in.\n"
    "Friend: My boss moved our meeting three times today.\n"
    "{name}: Ugh, three times? That's not a meeting anymore, that's a hostage situation.")


def about_me(cfg):
    """What the voice can truthfully say about itself (seventh call: "how do you work?" should get a
    short, true answer, not made-up features)."""
    b = cfg.brain
    brain = {'gemma-4': "Google's small Gemma 4 model", 'lfm2': "a small Liquid AI model",
             'qwen3': "Alibaba's small Qwen 3 model"}
    model = next((v for k, v in brain.items() if b.text_model.name.lower().startswith(k)), 'a small language model')
    voice = "Kyutai's Pocket TTS" if b.speech == 'pocket' else "Liquid AI's audio model"
    # Ninth live call: "the connection can be a bit wonky... like the internet between us", "I only
    # process the text you type", "I can process text and images": none of it true.
    return ("About you, if they ask: this is a voice call; they talk and you hear them, nobody types. You both run "
            "on this one computer, offline: there is no internet or connection between you, so if you were slow or "
            f"missed something, just say so. A speech model writes down what they say, {model} picks your words, "
            f"and {voice} gives you your voice. You can't see images or video, look things up, use tools or apps, or "
            "remember past calls. You're still being built and tuned, by them with Claude's help.\n")


# Ninth live call: "what time is it?" got "It's right now." The time is added only when they ask, so
# the start of the prompt (and what the model has already read of it) stays the same.
TIME_ASKED = re.compile(r"\b(what time|the time\b|what day|which day|today'?s date|what'?s the date|what date|"
                        r"what month|what year)", re.I)


def clock_note():
    import datetime
    now = datetime.datetime.now()
    return f"It is {now.strftime('%A, %B')} {now.day}, {now.year}, {now.strftime('%I:%M %p').lstrip('0')} here."


def voice_name(cfg):
    """The voice's own name: the Pocket TTS voice picked in the call window ('caro_davy' -> 'Caro')."""
    b = cfg.brain
    if b.name:
        return b.name
    return b.pocket_voice.split('_')[0].capitalize() if b.speech == 'pocket' else 'Juno'


# A line said to Claude (the developer), not to the voice. The live transcriber also writes the name
# as "clod"/"claud", and in the seventh call as "clock" and "class" at the end of a sentence.
ASIDE = re.compile(r"\b(claude|claud|clod)\b", re.I)


def aside_to_developer(said, name):
    """Was the last person they addressed in this turn Claude rather than the voice?"""
    last_claude = max((m.end() for m in ASIDE.finditer(said)), default=-1)
    if last_claude < 0:
        return False
    last_voice = max((m.end() for m in re.finditer(rf'\b{re.escape(name)}\b', said, re.I)), default=-1)
    return last_claude > last_voice


def back_to_voice(said, name):
    """After talking to Claude, are they talking to the voice again? (They name it last, or ask it
    something.) Eighth live call: Claude was named once, then several more pauses' worth of talk
    to Claude followed without the name, and the voice answered them."""
    if re.search(rf'\b{re.escape(name)}\b', said, re.I) and not aside_to_developer(said, name):
        return True
    return said.rstrip().endswith('?') and not ASIDE.search(said)


# Describing something without naming it, so the listener can guess ("do you know that one superhero
# that has an S on his shirt..."). Seventh and eighth live calls: "it should have already said
# something in the middle of it", the way a person cuts in with "Superman?".
GUESS_CUE = re.compile(r"\b(do you know|you know (that|the|this|who)|remember (that|the)|what(?:'s|s| is| was) "
                       r"(?:it|that|this|his|her|the)(?: \w+){0,2} (?:called|named|name)|what do you call|the (one|guy|"
                       r"girl|woman|man|thing|show|movie|game|song|band|place|book) (that|who|with|where|in|from)"
                       # ninth live call: "So, that superhero with an S on his shirt... he wears a cape"
                       r"|(that|this|the) (\w+ ){0,2}(with|who|that|from) (an?|the|his|her|that)"
                       r"|(he|she|they) (wears|wear|works|work|lives|live|flies|fly|plays|play|sings|was in|is from))\b",
                       re.I)
# Chosen on five test descriptions: "say only no if not sure" made it decline even Superman; this
# wording guessed Superman, Doctor Who and Keanu Reeves and declined "that thing I told you yesterday".
GUESS_NUDGE = ("They are describing something without naming it. What are they most likely talking about? Answer "
               "with only the name and a question mark, like \"Superman?\". If it could be lots of different "
               "things, answer only: ?")
NOT_A_GUESS = re.compile(r"\b(not sure|don'?t know|no idea|don'?t recall|what do you mean|which one|what kind)\b", re.I)


# Openers never used: "it keeps saying oh wow" (seventh and eighth live calls).
BANNED_OPENERS = {'oh wow'}

# Eighth live call: when the person stopped and waited, nothing came ("I just tested here again by
# not saying anything and there was just dead air"). After a few quiet seconds the voice picks the
# conversation back up, the way a friend does.
SPEAK_UP = ("They have gone quiet for a few seconds. Pick the conversation back up the way a friend would: a "
            "thought or a quick story about what you were just talking about, or one light question about it. "
            "One or two short sentences. Don't mention the silence.")

# The slow mind's instructions (brain.mind): notes for the fast text model, never the reply itself.
MIND_PROMPT = (
    "You are the private thoughts of Juno, a witty, warm friend on a voice call. Read what the friend just said "
    "and write at most two short lines of notes for Juno: what they mean or feel, and one good thing to say back "
    "(the true answer if they asked something, an opinion, or a joke). Facts must be correct; if unsure, note "
    "that. Notes only, no greeting, no reply text.")

# Help-desk and filler lines the small models fall back on. A first sentence like this is
# written again; a later one ends the reply there.
HELPDESK = re.compile(
    r"how (can|may) i (help|assist)|what can i (help|do for you)|is there anything|anything else"
    r"|here to (?:\w+ and )?(help|assist)"
    r"|feel free to|let me know if|what(?:'?s| is| else is) (been )?on your mind|digital assistant|virtual assistant"
    r"|language model"
    r"|brainstorm|(i|we) can help|happy to help"
    r"|as an ai\b|i'?m (just|only) an? (ai|assistant|program|chatbot)|my (training|knowledge)|trained on"
    r"|don'?t have (feelings|emotions|preferences|opinions|personal|a personality)|access to (the internet|real-time)", re.I)

# Laughing first fits only if they were joking.
LAUGH = re.compile(r"\s*(ha(ha)*|he(he)+|lol|lmao)\b", re.I)
JOKING = re.compile(r"\b(ha(ha)*|lol|funny|joke|joking|kidding|hilarious)\b", re.I)

# Topics of the made-up style examples above. With little to go on (a first "Hey!"), the small
# model sometimes answers one of them instead ("Oh no, your toast!"); such a sentence is dropped
# unless the person brought the topic up.
EXAMPLE_TOPICS = re.compile(r"\b(toast\w*|sweaters?|hostage|winter)\b", re.I)

SENTENCE_END = re.compile(r'[.!?]+["\'’”]*\s')
# "J.K. Rowling", "Dr. Eggman": a full stop that does not end the sentence (fourth live call's
# follow-up test cut "J.K." off as a sentence of its own).
ABBREVIATION = re.compile(r"(\b[A-Z]|\b(?:Mr|Mrs|Ms|Dr|St|Jr|Sr|vs|etc|e\.g|i\.e))\.$")


CLAUSE_END = re.compile(r"[,;:]\s|\s?[—–]\s?")


def clause_end(text, min_words):
    """Where the first phrase of at least min_words words ends at a comma or dash, or None."""
    for m in CLAUSE_END.finditer(text):
        if len(re.findall(r"[A-Za-z']+", text[:m.start()])) >= min_words:
            return m.end()
    return None


def sentence_end(text):
    """Where the first finished sentence (and the space after it) ends, or None."""
    for m in SENTENCE_END.finditer(text):
        head = text[:m.end()].rstrip()
        if head.endswith('.') and ABBREVIATION.search(head):
            continue
        if _has_words(text[:m.start()]):
            return m.end()
    return None


ACTION = re.compile(r'\s*(laugh|smile|giggle|sigh|chuckle|grin|wink|nod|pause|clears?|shrug|gasp|whisper|hum|beam'
                    r'|blush|lean|look|wave|cough|snort|grin)', re.I)


def _unbracket(m):
    inner = next(g for g in m.groups() if g is not None)
    return ' ' if ACTION.match(inner) else ' ' + inner + ' '     # *laughs* goes, *a movie title* stays


def clean(text):
    """Only words that can be spoken: no emoji, markdown, *actions* or (stage directions)."""
    text = re.sub(r'\*([^*]*)\*|\(([^)]*)\)|\[([^\]]*)\]', _unbracket, text)
    text = text.replace('’', "'").replace('‘', "'").replace('“', '"').replace('”', '"')
    text = text.replace('—', ', ').replace('–', ', ')
    text = re.sub(r"[^\w\s.,!?'\"’:;-]", ' ', text)
    text = re.sub(r'\s+([,.!?])', r'\1', re.sub(r'\s+', ' ', text))
    return text.strip(' ,')


# Words a sentence does not end on: "...and", "...the", "...because".
OPEN_WORDS = set('and but or so because cause the a an to of for with my your his her their our this these those in '
                 "on at if when like um uh er i i'm we you it's is was are were gonna wanna just really very kind sort "
                 "about than then from into by as not no what which who where whether".split())


# Thinking out loud ("...you know, like, uhm"): they are holding the floor while they find the words.
# Seventh live call: "I was trying to think of something there and you didn't understand that I was
# trying to think... you gotta be able to give me some time." The transcriber often puts a full
# stop after these, so they are checked before the punctuation.
FILLERS = {'uh', 'um', 'uhm', 'umm', 'uhh', 'er', 'erm', 'like'}
FILLER_PAIRS = {('you', 'know'), ('i', 'mean'), ('kind', 'of'), ('sort', 'of')}


def thinking_out_loud(text):
    words = re.findall(r"[a-z']+", (text or '').lower())
    if not words or (text or '').rstrip().endswith('?'):
        return False
    return words[-1] in FILLERS or (len(words) >= 2 and (words[-2], words[-1]) in FILLER_PAIRS)


def turn_cue(text):
    """How the person's words end: 'question', 'done', 'open' (mid-sentence), or None (can't tell).

    The live transcriber punctuates from the voice as well as the words, so a missing full
    stop is itself a hint that they were not finished.
    """
    t = (text or '').strip()
    if not t:
        return None
    if t.endswith('?'):
        return 'question'
    if thinking_out_loud(t):
        return 'open'
    if t.endswith('...') or t[-1] in ',-':
        return 'open'
    if t[-1] in '.!':
        return 'done'
    words = re.findall(r"[a-z']+", t.lower())
    if words and words[-1] in OPEN_WORDS:
        return 'open'
    return None if len(words) <= 2 else 'open'


def norm(sentence):
    return ' '.join(re.findall(r"[a-z0-9']+", sentence.lower()))


def _has_words(s):
    return bool(re.search(r'[A-Za-z]', s))


OPENER = re.compile(r"\s*((?:oh\s+)?[A-Za-z']+)([,!.])(?=\s|$)", re.I)


def opener_of(sentence):
    """The quick reaction a sentence opens with ("oh wow" in "Oh wow, that's..."), or None."""
    m = OPENER.match(sentence or '')
    if m is None:
        return None
    word = ' '.join(m.group(1).lower().split())
    if (m.group(2) == ',' and word in PARTICLES) or (m.group(1) + m.group(2)) in QUICK:
        return word
    return None


def strip_opener(sentence, word):
    """The sentence without its opening reaction: "Oh wow, that's wild." -> "That's wild."."""
    rest = re.sub(rf"^\s*{re.escape(word)}\s*[,!.]?\s*", '', sentence, flags=re.I)
    return rest[:1].upper() + rest[1:] if rest else ''


def _same_turn(a, b):
    """Do two transcripts of a turn start alike? (Live transcripts differ a little from final ones.)"""
    wa, wb = norm(a or '').split(), norm(b or '').split()
    k = min(3, len(wa), len(wb))
    return k > 0 and wa[:k] == wb[:k]


class _Writer:
    """Runs the text model in a thread and hands out its reply one finished sentence at a time."""

    def __init__(self, brain, prompt, temperature):
        self.stop = threading.Event()
        self.q = asyncio.Queue()
        self.buf, self.ended = '', False
        self.written, self.gap = '', ''      # raw text up to the last sentence handed out
        loop = brain.loop
        put = lambda d: loop.call_soon_threadsafe(self.q.put_nowait, d)   # noqa: E731
        brain.writing += 1
        fut = loop.run_in_executor(None, brain._stream_text, prompt, put, self.stop, temperature)
        fut.add_done_callback(lambda f: (f.cancelled() or f.exception(), brain._wrote()))

    async def next(self, on_text=None, clause_words=0):
        """The next finished sentence; with clause_words, the first phrase of a sentence still being
        written may come out on its own, so its sound can start sooner (seventh call: dead air while
        long first sentences were written: ~0.45 s for 25 word-pieces)."""
        while True:
            end = sentence_end(self.buf)
            if end is None and clause_words and not self.ended:
                end = clause_end(self.buf, clause_words)
            if end is not None:
                sent = self.buf[:end]
                core = sent.rstrip()
                self.written += self.gap + core
                self.gap = sent[len(core):]
                self.buf = self.buf[end:]
                s = clean(core)
                if s and core[-1] in ',;:—–':
                    s += ','                          # a phrase, not a sentence: it is voiced as going on
                if _has_words(s):
                    return s
                continue
            if self.ended:
                self.written += self.gap + self.buf
                s, self.buf, self.gap = clean(self.buf), '', ''
                return s if _has_words(s) else None
            d = await self.q.get()
            if d is None:
                self.ended = True
            else:
                self.buf += d
                if on_text is not None:
                    on_text(self.buf)

    def close(self):
        self.stop.set()


class SplitBrain:
    checks_text = True        # it vets every sentence itself, before any sound is made

    def __init__(self, cfg, persona=TEXT_PERSONA, note=None):
        self.b = cfg.brain
        self.host, self.port = cfg.server.host, cfg.server.port
        self.text_base = f'http://{cfg.server.host}:{self.b.text_port}'
        self.name = voice_name(cfg)
        self.persona = persona.replace('{name}', self.name).replace('{about}', about_me(cfg))
        self.note = note or (lambda *a, **k: None)
        self.in_rate, self.out_rate = 16000, 24000
        self.ws = None
        self.reader = None
        self.q = asyncio.Queue()
        self.loop = None
        # hearing
        self.unsent = 0                      # samples appended since the last commit
        self.waiting_ids = collections.deque()
        self.by_item = {}                    # item id -> future transcript
        self.early = {}                      # transcripts that arrived before their id
        self.heard = []                      # future transcripts of the person's current turn
        # thinking and speaking
        self.history = []                    # what was said, as text
        self.recent = collections.deque(maxlen=40)   # sentences already spoken
        # Seventh live call: "it keeps saying oh wow". The last few openers are not used again.
        self.recent_openers = collections.deque(maxlen=3)
        self.req = 0
        self.active = None                   # the request whose events still go out
        self.task = None
        self.stop = None                     # threading.Event for the current reply's speech stream
        self.reply = None
        self.spec = None                     # a reply written ahead, in case the person is done
        self.asked_last = False              # the last reply heard ended in a question
        self.speech = None                   # a phone-class voice (speech.PocketVoice), else the server's /tts
        self.on_opener = None                # conductor: says a reply's first word from its clips
        self.clips = set()                   # ...and the texts it has clips for
        # A reaction ("Oh no,") picked while they are still talking, from the live transcript, so
        # the turn can open the moment they stop instead of ~0.2 s later (see early_reaction).
        self.live = {}                       # item id -> live transcript of a piece not yet final
        self.reaction = None                 # (reaction, how many of their words it was picked from)
        self.reacting = False
        self.react_words = 0
        self.prefix = None                   # a reaction already said: the next reply goes on from it
        # Reading along (see _read_along): the text model reads their words while they talk.
        self.reading = False
        self.read_again = False
        self.read_text = None                # what it last read up to
        self.writing = 0                     # replies being written right now
        self.hist_start = 0                  # earlier history has dropped off (see _messages)
        # The slow mind (see _think): private notes on the person's current turn.
        self.mind_base = f'http://{cfg.server.host}:{self.b.mind_port}'
        self.notes = None                    # {'said': what they had said, 'text': the notes on it}
        self.thinking = False
        self.think_again = False
        self.think_words = 0
        self.hearing = None                  # Moonshine on the processor (brain.hearing), else the voice server
        self.aside_mode = False              # they are talking to Claude (see _is_aside)
        self.aside_t = 0.0
        self.guess = None                    # guessing out loud (see _maybe_guess): "Superman?"
        self.guessing = False
        self.guess_words = 0
        self.guessed = set()                 # guesses already said in this call

    # ---- the same calls as Realtime -------------------------------------
    async def connect(self):
        self.loop = asyncio.get_running_loop()
        if self.b.hearing == 'moonshine':
            from .hearing import MoonshineHearing
            note = lambda kind, **kw: self.loop.call_soon_threadsafe(lambda: self.note(kind, **kw))  # noqa: E731
            self.hearing = await self.loop.run_in_executor(None, lambda: MoonshineHearing(
                self.b.moonshine_arch, keyterms=[self.name, 'Claude'], note=note))
            self.hearing.loop, self.hearing.on_live = self.loop, self._heard_live
            return {'type': 'session.updated'}
        self.ws = await websockets.connect(f'ws://{self.host}:{self.port}/v1/realtime?intent=transcription',
                                           max_size=None, ping_interval=None, open_timeout=30)
        ev = json.loads(await self.ws.recv())
        if ev.get('type') == 'error':
            raise RuntimeError(ev['error'].get('message'))
        await self.ws.send(json.dumps({'type': 'session.update', 'session': {
            'type': 'transcription', 'audio': {'input': {'transcription': {'language': 'en'}}}}}))
        while True:
            ev = json.loads(await self.ws.recv())
            if ev.get('type') == 'session.updated':
                break
            if ev.get('type') == 'error':
                raise RuntimeError(ev['error'].get('message'))
        self.reader = asyncio.create_task(self._read())
        return ev

    async def append(self, pcm16: bytes):
        self.unsent += len(pcm16) // 2
        if self.hearing is not None:
            self.hearing.append(pcm16)
            return
        await self.ws.send(json.dumps({'type': 'input_audio_buffer.append', 'audio': base64.b64encode(pcm16).decode()}))

    def _heard_live(self, text):
        """Moonshine's words for what they are saying right now (not yet handed out at a pause)."""
        self.live = {'moonshine': text} if text else {}
        self._maybe_react()
        self._maybe_think()
        self._maybe_guess()

    # ---- guessing out loud --------------------------------------------------------
    def _maybe_guess(self):
        """While they describe something without naming it, see whether the brain knows what it is."""
        if not self.b.guessing or self.guessing or self.loop is None or self.guess:
            return
        text = self._turn_so_far()
        n = len(text.split())
        if n < 6 or n - self.guess_words < 4 or not GUESS_CUE.search(text) or self.aside_mode:
            return
        self.guessing, self.guess_words = True, n
        fut = self.loop.run_in_executor(None, self._make_guess, self._messages(text, nudge=GUESS_NUDGE))
        fut.add_done_callback(lambda f: self._got_guess(f, text))

    def _make_guess(self, messages):
        body = json.dumps({'prompt': self._prompt(messages) + self.b.text_prefill, 'n_predict': 10,
                           'temperature': 0.2, 'cache_prompt': True}).encode()
        req = urllib.request.Request(self.text_base + '/completion', data=body, headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=5) as r:
            return json.loads(r.read()).get('content', '')

    def _got_guess(self, fut, text):
        self.guessing = False
        if fut.cancelled() or fut.exception() is not None:
            return
        m = re.match(r"\s*([^.!?]*\?)", clean(fut.result()))     # its first sentence, if a question
        g = m.group(1).strip().strip('"') if m else ''
        words = re.findall(r"[A-Za-z']+", g)
        if (not words or len(words) > 8 or NOT_A_GUESS.search(g) or norm(g) in self.guessed
                or not _same_turn(self._turn_so_far(), text)):
            return
        self.guess = g
        self.note('guess_ready', guess=g, after_words=len(text.split()))

    def pending_guess(self):
        """A guess ready to say at their next short pause ("Superman?"), or None."""
        return self.guess if self.guess and not self.aside_mode else None

    async def commit(self, min_s=0.1, pad_s=0.3):
        """Transcribe what the person said since the last commit (called at each of their pauses).

        The live transcriber holds back the last word until it hears what follows; committed
        right after the word it sometimes drops it (it turned up at the start of the next piece:
        "...a hiking trip this" / "weekend. Yeah!..."). A moment of silence first lets it finish.
        """
        if self.unsent < min_s * self.in_rate:
            return
        self.unsent = 0
        fut = self.loop.create_future()
        fut.add_done_callback(lambda f: (self._read_along(), self._think()))
        self.heard.append(fut)
        if self.hearing is not None:
            self.hearing.append(bytes(2 * int(pad_s * self.in_rate)))
            self.hearing.commit(fut)
            return
        await self.ws.send(json.dumps({'type': 'input_audio_buffer.append', 'audio': base64.b64encode(
            bytes(2 * int(pad_s * self.in_rate))).decode()}))
        self.waiting_ids.append(fut)
        await self.ws.send(json.dumps({'type': 'input_audio_buffer.commit'}))

    async def respond(self, commit=True, initiative=False, guess=None):
        if guess:
            # A guess at what they are describing, said at a short pause (see _maybe_guess).
            self.discard_prefetch()
            if commit:
                await self.commit()
            self.req += 1
            self.active = self.req
            self.stop = threading.Event()
            self.reply = self._new_reply()
            self.reply['fixed'] = guess
            self.guessed.add(norm(guess))
            self.guess = None
            self.task = asyncio.create_task(self._reply(self.req, self.stop, self.reply))
            return
        if initiative:
            # They have gone quiet after the voice spoke: pick the conversation back up (see SPEAK_UP).
            self.discard_prefetch()
            self.req += 1
            self.active = self.req
            self.stop = threading.Event()
            self.reply = self._new_reply()
            self.reply['initiative'] = True
            self.task = asyncio.create_task(self._reply(self.req, self.stop, self.reply))
            return
        if commit:
            await self.commit()
        prefix, self.prefix = self.prefix, None
        spec, self.spec = self.spec, None
        if (spec is not None and spec['r'].get('prefix') == prefix and spec['r']['futures'] == self.heard
                and not spec['stop'].is_set()):
            # Written ahead for exactly what they said: hand over what is ready and go on.
            self.active = spec['req']
            self.stop, self.task, self.reply = spec['stop'], spec['task'], spec['r']
            if spec['r'].pop('opener_pending', False):
                self.on_opener(spec['r']['early'])   # its sound was made without the first word
            for ev in spec['buffer']:
                self.q.put_nowait(ev)
            self.note('prefetch_used', sentences=len(spec['r']['accepted']))
            return
        if spec is not None:
            self._drop(spec)
        self.req += 1
        self.active = self.req
        self.stop = threading.Event()
        self.reply = self._new_reply()
        if prefix:
            self.reply['prefix'] = prefix
            self.reply['early'] = prefix     # already said from the clips: not voiced again
        self.task = asyncio.create_task(self._reply(self.req, self.stop, self.reply))

    def early_reaction(self):
        """A reaction picked from most of what they said so far ("Oh no" / "Totally!"), or None."""
        so_far = self._turn_so_far()
        if aside_to_developer(so_far, self.name) or (self.aside_mode and not back_to_voice(so_far, self.name)):
            return None                      # talking to Claude: no reaction from the voice
        spec = self.spec
        if spec is not None and spec['r'].get('prefix') and spec['r']['futures'] == self.heard:
            return spec['r']['prefix']       # the reply written ahead already goes on from this one
        if self.reaction is None:
            return None
        word, n = self.reaction
        return word if n >= max(3, 0.6 * len(self._turn_so_far().split())) else None

    def use_reaction(self, word):
        """The conductor said this reaction as the turn opened: the reply goes on from it."""
        self.prefix, self.reaction = word, None

    def _turn_so_far(self):
        done = [f.result() for f in self.heard if f.done() and f.result()]
        return ' '.join(done + [t.strip() for t in self.live.values() if t.strip()])

    def _maybe_react(self):
        if not self.b.early_reaction or self.reacting or self.on_opener is None or self.loop is None:
            return
        text = self._turn_so_far()
        n = len(text.split())
        if n < 3 or n - self.react_words < 2:
            return
        self.reacting, self.react_words = True, n
        fut = self.loop.run_in_executor(None, self._pick_reaction, self._messages(text))
        fut.add_done_callback(lambda f: self._got_reaction(f, n))

    def _pick_reaction(self, messages):
        body = json.dumps({'prompt': self._prompt(messages) + self.b.text_prefill, 'n_predict': 6, 'temperature': 0.7,
                           'cache_prompt': True}).encode()
        req = urllib.request.Request(self.text_base + '/completion', data=body, headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=5) as r:
            return json.loads(r.read()).get('content', '')

    def _got_reaction(self, fut, n):
        self.reacting = False
        if fut.cancelled() or fut.exception() is not None:
            return
        text = clean(fut.result())
        m = re.match(r"\s*((?:oh\s+)?[A-Za-z']+)([,!.?])", text, re.I)
        if m is None:
            return
        word = ' '.join(m.group(1).split())
        if word.lower() in self.recent_openers or word.lower() in BANNED_OPENERS:
            return                                   # said it lately: let the reply open some other way
        if m.group(2) == ',' and PARTICLES.get(word.lower()) in self.clips:
            self.reaction = (word, n)
        elif word + m.group(2) in QUICK and word + m.group(2) in self.clips:
            self.reaction = (word + m.group(2), n)

    async def prefetch(self):
        """Starts writing and speaking (unheard) a reply now, in case the person is done.

        Called in a pause the turn model thinks may be the end. If they go on talking it is
        thrown away; if the turn is theirs to give, respond() hands it over already under way.
        """
        if self.spec is not None:
            if self.spec['r']['futures'] == self.heard:
                return
            self._drop(self.spec)
        if not self.heard:
            return
        self.req += 1
        stop, r = threading.Event(), self._new_reply()
        word = self.early_reaction()
        if word:
            # Written on from the reaction the turn will open with, so the rest is ready the moment
            # that clip ends (rewriting it after the clip left a gap of ~0.7 s in the benchmark).
            r['prefix'] = r['early'] = word
        self.spec = {'req': self.req, 'r': r, 'stop': stop, 'buffer': []}
        self.spec['task'] = asyncio.create_task(self._reply(self.req, stop, r))

    def turn_cue(self):
        """How the person's words so far end (see turn_cue()), once they are transcribed."""
        if not self.heard or not all(f.done() for f in self.heard):
            return None
        return turn_cue(' '.join(f.result() for f in self.heard if f.result()))

    def still_thinking(self):
        """Do their words so far end in a filler ("uhm", "you know")? Then they are still thinking."""
        if not self.heard or not all(f.done() for f in self.heard):
            return False
        return thinking_out_loud(' '.join(f.result() for f in self.heard if f.result()))

    def discard_prefetch(self):
        """They went on talking: stop the reply written ahead."""
        if self.spec is not None:
            self._drop(self.spec)

    def _drop(self, spec):
        spec['stop'].set()
        if not spec['task'].done():
            spec['task'].cancel()
        if self.spec is spec:
            self.spec = None

    async def cancel(self):
        if self.stop is not None:
            self.stop.set()
        if self.task is not None and not self.task.done():
            self.task.cancel()

    async def events(self):
        while True:
            yield await self.q.get()

    async def close(self):
        await self.cancel()
        self.discard_prefetch()
        if self.reader is not None:
            self.reader.cancel()
        if self.ws is not None:
            await self.ws.close()
        if self.hearing is not None:
            await self.loop.run_in_executor(None, self.hearing.close)

    # ---- told by the conductor -------------------------------------------
    def reply_started(self):
        """The current reply is being heard: it and what the person said become history."""
        r = self.reply
        if r is None or r['live']:
            return
        r['live'] = True
        if r['said']:
            self.history.append({'role': 'user', 'content': r['said']})
        r['msg'] = {'role': 'assistant', 'content': ' '.join(r['accepted'])}
        self.history.append(r['msg'])
        for s in r['accepted']:
            self.recent.append(norm(s))
        opener = opener_of(r['accepted'][0]) if r['accepted'] else None
        if opener:
            self.recent_openers.append(opener)
        self.heard = [f for f in self.heard if f not in r['futures']]
        self.reaction, self.react_words = None, 0      # the next turn picks its own
        self.read_text = None
        self.notes, self.think_words = None, 0
        self.guess, self.guess_words = None, 0
        self._settle(r)

    def reply_cut(self, played_s):
        """The person talked over the reply after played_s seconds: keep only what they heard."""
        r = self.reply
        if r is None or r['msg'] is None:
            return
        kept, t = [], 0.0
        for k, s in enumerate(r['accepted']):
            words = s.split()
            dur = r['seconds'].get(k) or 0.3 * len(words) + 0.3    # still being spoken: estimate
            if t + dur <= played_s + 0.2:
                kept.append(s)
            elif t < played_s:
                n = max(1, int(len(words) * (played_s - t) / max(dur, 0.1)))
                kept.append(' '.join(words[:n]) + '...')
            t += dur
        r['cut'] = True
        r['msg']['content'] = ' '.join(kept) or '...'

    # ---- hearing ------------------------------------------------------------
    async def _read(self):
        async for msg in self.ws:
            ev = json.loads(msg)
            kind = ev.get('type')
            if kind == 'input_audio_buffer.committed':
                item = ev.get('item_id')
                fut = self.waiting_ids.popleft() if self.waiting_ids else None
                if fut is None:
                    continue
                if item in self.early:
                    if not fut.done():
                        fut.set_result(self.early.pop(item))
                else:
                    self.by_item[item] = fut
            elif kind == 'conversation.item.input_audio_transcription.delta':
                item = ev.get('item_id')
                self.live[item] = self.live.get(item, '') + (ev.get('delta') or '')
                self._maybe_react()
                self._maybe_think()
                self._maybe_guess()
            elif kind == 'conversation.item.input_audio_transcription.completed':
                item, text = ev.get('item_id'), (ev.get('transcript') or '').strip()
                self.live.pop(item, None)
                fut = self.by_item.pop(item, None)
                if fut is None:
                    self.early[item] = text
                elif not fut.done():
                    fut.set_result(text)
            elif kind == 'error':
                self.q.put_nowait(ev)

    async def _transcript(self, futures, timeout=2.0):
        texts = []
        for f in futures:
            try:
                texts.append(await asyncio.wait_for(asyncio.shield(f), timeout))
            except asyncio.TimeoutError:
                texts.append('')
        return ' '.join(t for t in texts if t)

    # ---- thinking and speaking ------------------------------------------------
    def _put(self, ev, req):
        if req == self.active:
            self.q.put_nowait(ev)
        elif self.spec is not None and req == self.spec['req']:
            self.spec['buffer'].append(ev)           # written ahead: held until it is wanted

    def _new_reply(self):
        return {'said': '', 'accepted': [], 'accepted_norm': set(), 'seconds': {}, 'live': False, 'msg': None,
                'cut': False, 'early': None, 'futures': list(self.heard)}

    def _messages(self, said, nudge=None):
        # Older exchanges drop off in steps, not one every turn: whatever changes near the start
        # of the prompt makes the text model read everything after it again (~4 s of reading for
        # the 8B brain, ~0.1 s for the 1.2B). A nudge goes after their words for the same reason.
        n = len(self.history)
        if n - self.hist_start > 2 * self.b.history_turns:
            self.hist_start = n - self.b.history_turns
            while self.hist_start < n - 1 and self.history[self.hist_start]['role'] != 'user':
                self.hist_start += 1
        messages = [{'role': 'system', 'content': self.persona}] + self.history[self.hist_start:]
        if said:
            messages.append({'role': 'user', 'content': said})
            if TIME_ASKED.search(said):
                messages.append({'role': 'system', 'content': clock_note()})
        notes = self.notes
        if self.b.mind and said and notes and notes['text'] and _same_turn(said, notes['said']):
            messages.append({'role': 'system', 'content': 'Your own quick thoughts on that (private: use them, '
                                                          'never read them out): ' + notes['text']})
        if nudge is not None:
            messages.append({'role': 'system', 'content': nudge})
        return messages

    def _is_aside(self, said):
        """Is this turn said to Claude rather than to the voice? Once they talk to Claude, the turns
        after it are too, until they name the voice or ask it something, or a while has passed."""
        now = self.loop.time()
        if aside_to_developer(said, self.name):
            self.aside_mode, self.aside_t = True, now
            return True
        if self.aside_mode:
            if back_to_voice(said, self.name) or now - self.aside_t > self.b.aside_timeout_s:
                self.aside_mode = False
                return False
            self.aside_t = now
            return True
        return False

    # ---- the slow mind ----------------------------------------------------------
    def _think(self):
        """While they talk, a bigger model on the processor writes notes on what they have said so far.

        The fast text model answers from its own small knowledge; the notes put a bigger model's
        knowledge and judgment in front of it without its slowness (it writes ~41 word-pieces a
        second on this laptop's processor, too slow to answer in time itself).
        """
        if not self.b.mind or self.loop is None:
            return
        if self.thinking:
            self.think_again = True
            return
        said = self._turn_so_far()
        if len(said.split()) < 2 or (self.notes and self.notes['said'] == said):
            return
        self.thinking, self.think_again = True, False
        messages = [{'role': 'system', 'content': MIND_PROMPT}] + self.history[self.hist_start:] + \
                   [{'role': 'user', 'content': said}]
        fut = self.loop.run_in_executor(None, self._mind_notes, messages, said)
        fut.add_done_callback(lambda f: self._thought(f, said))

    def _maybe_think(self):
        """From the live transcript too, every few new words, so the notes are ready sooner."""
        if not self.b.mind or self.thinking:
            return
        n = len(self._turn_so_far().split())
        if n - self.think_words >= 3:
            self.think_words = n
            self._think()

    def _mind_notes(self, messages, said):
        prompt = self._prompt(messages, base=self.mind_base)
        if said in prompt:
            self._read_only(prompt[:prompt.rindex(said) + len(said)], base=self.mind_base)   # a resume point
        body = json.dumps({'prompt': prompt, 'n_predict': self.b.mind_tokens, 'temperature': 0.3,
                           'cache_prompt': True, 'stop': ['\n\n', '<|im_end|>']}).encode()
        req = urllib.request.Request(self.mind_base + '/completion', data=body, headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.loads(r.read()).get('content', '').strip()

    def _thought(self, fut, said):
        self.thinking = False
        if not fut.cancelled() and fut.exception() is None and fut.result():
            if _same_turn(self._turn_so_far(), said):       # not a turn that has since been answered
                self.notes = {'said': said, 'text': ' '.join(fut.result().split())[:400]}
                self.note('mind_notes', said=said[-60:], notes=self.notes['text'])
        if self.think_again:
            self._think()

    # ---- reading along ----------------------------------------------------------
    def _read_along(self):
        """While they talk, the text model reads what they have said so far.

        When they stop, only their last few words and the start of the reply are left to read.
        The 8B brain reads ~130 word-pieces a second on this laptop: a turn read from scratch took
        0.42 s before its first word, read along 0.06 s. It reads up to their last word and no
        further: a model with a running memory state (LFM2) can go on from where a reading
        stopped, but cannot back up into the middle of one.
        """
        if not self.b.read_along or self.loop is None:
            return
        if self.reading or self.writing:
            self.read_again = True          # after the current one
            return
        parts = []
        for f in self.heard:
            if not f.done():
                break
            if f.result():
                parts.append(f.result())
        said = ' '.join(parts)
        if not said or said == self.read_text:
            return
        self.reading, self.read_again, self.read_text = True, False, said
        fut = self.loop.run_in_executor(None, self._read_ahead, self._messages(said), said)
        fut.add_done_callback(self._read_done)

    def _read_ahead(self, messages, said):
        """Reads up to their last word, then on into the start of a reply, in case they are done.

        If they go on, the next reading backs up to their last word (the server keeps a
        checkpoint where each reading stopped) and only their new words are read.
        """
        prompt = self._prompt(messages)
        t = self._read_only(prompt[:prompt.rindex(said) + len(said)])
        self._read_only(prompt + self.b.text_prefill)
        return t

    def _read_only(self, prompt, base=None):
        body = json.dumps({'prompt': prompt, 'n_predict': 0, 'cache_prompt': True}).encode()
        req = urllib.request.Request((base or self.text_base) + '/completion', data=body,
                                     headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read()).get('timings', {})

    def _read_done(self, fut):
        self.reading = False
        if not fut.cancelled() and fut.exception() is None:
            t = fut.result()
            self.note('read_ahead', tokens=t.get('prompt_n'), ms=round(t.get('prompt_ms') or 0))
        if self.read_again:
            self._read_along()

    def _wrote(self):
        self.writing -= 1
        if self.read_again and not self.writing:
            self._read_along()

    def _prompt(self, messages, base=None):
        body = {'messages': messages}
        if base is None and self.b.text_template_kwargs:
            body['chat_template_kwargs'] = json.loads(self.b.text_template_kwargs)
        req = urllib.request.Request((base or self.text_base) + '/apply-template', data=json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())['prompt']

    def _stream_text(self, prompt, put, stop, temperature):
        body = json.dumps({'prompt': prompt, 'stream': True, 'n_predict': 90, 'temperature': temperature,
                           'top_p': 0.95, 'cache_prompt': True}).encode()
        req = urllib.request.Request(self.text_base + '/completion', data=body, headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                for raw in r:
                    if stop.is_set():
                        break
                    line = raw.decode('utf8', 'replace').strip()
                    if not line.startswith('data:'):
                        continue
                    d = json.loads(line[5:].strip())
                    if d.get('content'):
                        put(d['content'])
                    if d.get('stop'):
                        break
        finally:
            put(None)

    def _stream_speech(self, text, stop, req, started=None):
        """Speaks one sentence, sending its audio on as it is made. Returns its length in seconds."""
        n, heard = 0, False
        for x in (self.speech.stream(text, stop) if self.speech is not None else self._tts_chunks(text, stop)):
            if stop.is_set():
                break
            n += len(x)
            ev = {'type': 'response.output_audio.delta', 'delta': base64.b64encode(to_pcm16(x)).decode()}
            self.loop.call_soon_threadsafe(self._put, ev, req)
            # Tell the writer once the first word is actually sounding, not at the first chunk:
            # each LFM3 sentence opens with ~0.2 s of near-silence, and writing on top of it
            # slowed the first word from ~0.1 s to ~0.2 s.
            if started is not None and not heard and (np.abs(x).max() > 0.05 or n > 0.5 * self.out_rate):
                heard = True
                self.loop.call_soon_threadsafe(started.set)
        if started is not None and not heard:
            self.loop.call_soon_threadsafe(started.set)
        return n / self.out_rate

    def _tts_chunks(self, text, stop):
        """LFM3-Audio's speech generator on the voice server (float32 chunks at 24 kHz)."""
        body = json.dumps({'input': text, 'voice': self.b.voice, 'stream': True, 'response_format': 'pcm',
                           'seed': random.randrange(1 << 30)}).encode()
        request = urllib.request.Request(f'http://{self.host}:{self.port}/tts', data=body,
                                         headers={'Content-Type': 'application/json'})
        rest = b''
        with urllib.request.urlopen(request, timeout=60) as r:
            while not stop.is_set():
                b = r.read1(16384)
                if not b:
                    break
                b = rest + b
                k = len(b) // 4 * 4
                rest = b[k:]
                if k:
                    yield np.frombuffer(b[:k], dtype='<f4')

    def _judge(self, sentence, r):
        n = norm(sentence)
        if len(n) > 12 and (n in self.recent or n in r['accepted_norm']):
            return 'repeat'
        if HELPDESK.search(sentence):
            return 'off-character'
        if not r['accepted'] and LAUGH.match(sentence) and not JOKING.search(r['said']):
            # Fourth live call: "Haha, I see what you mean" to "not what I'm looking for at all"
            # ("I don't know why you would laugh there").
            return 'laugh out of place'
        m = EXAMPLE_TOPICS.search(sentence)
        if m and not re.search(re.escape(m.group(1)[:5]), ' '.join([r['said']] + [h['content'] for h in self.history
                                                                                  if h['role'] == 'user']), re.I):
            return 'example leak'
        if r['accepted'] and sentence.endswith('?') and self.asked_last:
            # The small model tacks a question onto most replies; people don't ask every turn.
            return 'question again'
        return 'ok'

    def _settle(self, r):
        if r['live'] and r['accepted'] and (self.task is None or self.task.done() or r is not self.reply):
            self.asked_last = r['accepted'][-1].endswith('?')

    def _early_word(self, text, r, req):
        """A reply opening with "Oh," / "Yeah," / "Ugh,": offer that word the moment it is written.

        The conductor says it at once from its ready-made clips, the way people start a turn while
        the rest is still forming (Dessa opens 44% of its turns so); the sentence is then spoken
        without it. A reply written ahead keeps the word back and offers it when it is wanted.
        """
        if r['early'] is not None:
            return
        m = re.match(r"\s*([A-Za-z']+)([^A-Za-z'])", text)
        if m is None:
            return                               # the first word is not finished yet
        word = m.group(1)
        if word.lower() == 'oh' and m.group(2) == ' ':
            m = re.match(r"\s*(oh\s+[A-Za-z']+)([^A-Za-z'])", text, re.I)    # "Oh no," / "Oh wow,"
            if m is None:
                return
            word = ' '.join(m.group(1).split())
        if (word.lower() in self.recent_openers or word.lower() in BANNED_OPENERS) and (
                m.group(2) == ',' or word + m.group(2) in QUICK):
            r['early'], r['strip_opener'] = '', word  # said it lately: the sentence starts without it
            return
        if self.on_opener is None or (m.group(2) == ',' and PARTICLES.get(word.lower()) not in self.clips):
            r['early'] = ''
        elif m.group(2) != ',':
            r['early'] = ''
            if m.group(2) in '!.':
                r['quick'] = True                # maybe a whole quick reaction ("Totally!"): see _say
        elif req == self.active:
            r['early'] = word if self.on_opener(word) else ''
        else:
            r['early'], r['opener_pending'] = word, True

    async def _say(self, s, req, stop, r, started=None):
        first = not r['accepted']
        spoken = s
        if first:
            self.note('first_sentence', sentence=s, ahead=req != self.active)
            early = r['early']
            if early and s == early:
                spoken = ''                            # a quick reaction already said from its clip
            elif early and s.lower().startswith(early.lower() + ','):
                spoken = s[len(early) + 1:].lstrip()   # its first word was already said from the clips
            elif r.get('quick') and self.on_opener is not None and s in QUICK and s in self.clips:
                # The whole first sentence is a quick reaction with a ready-made clip ("Totally!"):
                # say that at once instead of waiting for it to be voiced.
                if req == self.active:
                    if self.on_opener(s):
                        spoken = ''
                else:
                    r['early'], r['opener_pending'], spoken = s, True, ''
        r['accepted'].append(s)
        r['accepted_norm'].add(norm(s))
        if r['live']:
            self.recent.append(norm(s))
            if not r['cut']:
                r['msg']['content'] = ' '.join(r['accepted'])
        self._put({'type': 'response.output_text.delta', 'delta': s if first else ' ' + s}, req)
        k = len(r['accepted']) - 1
        if _has_words(spoken):
            r['seconds'][k] = await self.loop.run_in_executor(None, self._stream_speech, spoken, stop, req, started)
        elif started is not None:
            started.set()

    async def _reply(self, req, stop, r):
        writer = speaking = None
        try:
            initiative = r.get('initiative', False)
            if initiative:
                r['said'] = ''
                self.note('speak_up')
            else:
                r['said'] = await self._transcript(r['futures'])
                if not r['said']:
                    self.heard = [f for f in self.heard if f not in r['futures']]
                    return
                if self._is_aside(r['said']):
                    # Said to Claude, not to the voice: no answer, and it does not become part of the chat.
                    self.note('aside', text=r['said'][-80:])
                    self.heard = [f for f in self.heard if f not in r['futures']]
                    return
                self.note('heard', text=r['said'])
            if r.get('fixed'):
                await self._say(r['fixed'], req, stop, r)     # a guess: said as it is, no writing
                return
            temperature, words, held_back = self.b.temperature, 0, None
            nudge = SPEAK_UP if initiative else None
            # A reaction said the moment they stopped ("Oh no," / "Totally!"): the reply is written on
            # from it. A whole quick reaction is its own first sentence, already heard from a clip.
            lead = r.get('prefix') or ''
            if lead and lead[-1] not in '!.?':
                lead += ','
            if lead.endswith(('!', '.', '?')):
                speaking = asyncio.ensure_future(self._say(lead, req, stop, r))
            for attempt in range(2):
                prompt = await self.loop.run_in_executor(None, self._prompt, self._messages(r['said'], nudge))
                base = prompt + self.b.text_prefill + lead
                writer = _Writer(self, base, temperature)
                retry = False
                while not stop.is_set():
                    s = await writer.next(on_text=lambda t: self._early_word(t, r, req),
                                          clause_words=self.b.first_clause_words if speaking is None else 0)
                    if s is None:
                        break
                    if lead.endswith(',') and not r['accepted']:
                        s = lead + ' ' + s
                    if not r['accepted'] and not r.get('strip_opener') and opener_of(s) in BANNED_OPENERS:
                        r['strip_opener'] = opener_of(s)
                    if not r['accepted'] and r.get('strip_opener'):
                        s = strip_opener(s, r.pop('strip_opener'))
                        if not _has_words(s):
                            continue                 # it was only the repeated reaction ("Totally!")
                    verdict = self._judge(s, r)
                    if verdict != 'ok':
                        self.note('dropped_sentence', sentence=s, reason=verdict, attempt=attempt)
                        if r['accepted']:
                            break                    # a later sentence: end the reply before it
                        if attempt == 0:
                            retry = True             # the first one: write the reply again
                            break
                        held_back = held_back or s   # second try: skip it and see what follows
                        continue
                    words += len(s.split())
                    last = len(r['accepted']) + 1 >= self.b.max_sentences or words >= self.b.max_words
                    if speaking is None:
                        # First sentence: free the graphics card so its sound starts at once (sharing
                        # it with the text model makes the first sound ~0.35 s late instead of ~0.1 s),
                        # then write the rest from where it left off while it is spoken.
                        writer.close()
                        started = asyncio.Event()
                        speaking = asyncio.ensure_future(self._say(s, req, stop, r, started))
                        if last:
                            break
                        waiter = asyncio.ensure_future(started.wait())
                        try:
                            await asyncio.wait([speaking, waiter], timeout=0.6, return_when=asyncio.FIRST_COMPLETED)
                        finally:
                            waiter.cancel()
                        writer = _Writer(self, base + writer.written, temperature)
                        continue
                    await speaking                   # one sentence's sound at a time
                    speaking = asyncio.ensure_future(self._say(s, req, stop, r))
                    if last:
                        break
                writer.close()
                if not retry:
                    break
                temperature = min(1.1, temperature + 0.2)
                nudge = (SPEAK_UP + ' ' if initiative else '') + 'Say something new, in your own casual words.'
            if not r['accepted'] and held_back and not stop.is_set():
                speaking = asyncio.ensure_future(self._say(held_back, req, stop, r))    # better than no answer
            if speaking is not None:
                await speaking
        except asyncio.CancelledError:
            stop.set()                                   # ends a speech stream still running
            if speaking is not None:
                speaking.cancel()
            raise
        finally:
            if writer is not None:
                writer.close()
            if r['live'] and r['accepted']:
                self.asked_last = r['accepted'][-1].endswith('?')
            self._put({'type': 'response.done', 'response': {'status': 'completed'}}, req)
            if self.active == req:
                self.active = None
