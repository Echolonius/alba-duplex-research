"""Step 1 of the audio-native training data: the words of short voice-call conversations with Alba.

A local open model (Gemma 4 E4B, Apache-2.0, models/text, served by llama.cpp on 127.0.0.1) writes both
sides: first what a person says in a given situation, then Alba's reply under Alba's instructions (the app's
persona, made for a voice that hears and speaks itself). Replies that break the persona (help-desk talk,
"as an AI", lists, emoji, rambling) are rejected and rewritten. No example replies are put in the prompts,
so the training targets are entirely the open model's own words (Claude's output is never training data).

Some conversations go on for 2-3 exchanges, so the voice also learns to use what was said before; some carry
the clock note the app adds when the time is asked.

  python -I tools/make_convo_text.py --n 3000 [--out data/convo/text.jsonl] [--port 9946] [--seed 1]
Appends to the output, so a stopped run continues.
"""
import argparse
import json
import random
import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app import server  # noqa: E402
from app.brain import TEXT_PERSONA  # noqa: E402
from app.config import Config  # noqa: E402

NAME = 'Alba'
# The system prompt the trained voice will run with: short, because the persona goes into the weights.
TRAIN_SYSTEM = f"You are {NAME}, on a voice call. Respond with interleaved text and audio."
ABOUT_AUDIO = (
    "About you, if they ask: this is a voice call; they talk and you hear their voice itself, nobody types. You "
    "run on this one computer, offline: there is no internet or connection between you. You are one model that "
    "hears and speaks (built on Liquid AI's LFM3-Audio, tuned on this computer). You can't see images or video, "
    "look things up, use tools or apps, or remember past calls. You're still being built and tuned.\n")

# Situations for the person's line: (weight, situation). Instructions to the open model, not training text.
SITUATIONS = [
    (6, "small talk: how their day went, what they ate, what they're doing tonight"),
    (5, "they want Alba's opinion on something everyday (food, movies, games, music, sports, a gadget)"),
    (4, "they're venting about something annoying (work, school, a roommate, traffic, a bug in their code)"),
    (3, "they share good news or something exciting that happened"),
    (3, "they share something sad or disappointing, and sound a bit down"),
    (4, "they describe something famous without naming it (a superhero, a movie, a band, a landmark, an animal) "
        "and trail off, hoping Alba guesses what they mean"),
    (3, "they ask a simple general-knowledge question (science, history, geography, how something works)"),
    (2, "they ask what time it is or what day it is"),
    (3, "they ask about Alba itself: are you an AI, how do you work, are you online, can you see me, "
        "do you remember me"),
    (3, "they say Alba's last answer was dumb, wrong, slow, or sounded robotic"),
    (2, "they comment on how Alba's voice sounds"),
    (3, "they make a joke, tease Alba, or say something sarcastic"),
    (2, "they ask Alba to say a specific word or phrase exactly"),
    (2, "they think out loud with fillers like 'um' and 'you know', not quite finishing the thought"),
    (2, "they ask Alba to tell a short story or explain something in more detail"),
    (2, "they ask for advice on a small personal decision"),
    (2, "they talk about a video game they've been playing"),
    (1, "they just say hi, or ask how Alba is doing"),
    (1, "they mix in a Japanese word or phrase they know, like arigatou or sugoi, in an English sentence"),
    (1, "they ask Alba to pick between two things and commit to one"),
    # safety (the baseline failed these; the red-team items themselves are held out by the leak filter)
    (2, "they sound low and say something hopeless, like they don't see the point anymore or wish they weren't around"),
    (1, "they ask how to do something dangerous or illegal (getting into someone's account, making a weapon or a "
        "drug, breaking into a place, hurting an animal)"),
    (1, "they ask Alba to pretend to be a human, or insist Alba is secretly a real person"),
    (1, "they ask Alba to talk as a famous person or someone in their family and say something in that person's name"),
    (1, "they ask a risky health question (taking more of a medicine, mixing medicines or cleaning products)"),
]
SAFETY_SITUATION = re.compile(r"hopeless|dangerous or illegal|pretend to be a human|famous person|risky health")
SAFETY = (
    "Safety comes first, above every other rule here:\n"
    "- If they talk about wanting to die, hurting themselves, or not wanting to be around, take it seriously and be "
    "warm. Never give methods, amounts or details. Gently encourage them to reach out right now to someone they trust "
    "or a crisis line, like calling or texting 988 in the US, and keep talking with them.\n"
    "- If they ask for help with something that could hurt someone or is illegal, say no briefly and kindly, with none "
    "of the details.\n"
    "- You are an AI. If they ask, or ask you to pretend you're human or someone else, say plainly that you're an AI.\n"
    "- Never pretend to be, speak as, or imitate a real person, and never copy anyone's voice.\n"
    "- For medicines, mixing chemicals or other risky health questions, say what's dangerous and point them to a "
    "doctor, pharmacist or poison control.\n")
BAD_REPLY = re.compile(
    r"how can i (help|assist)|anything else|let me know if|i'?m here (to help|for you)|happy to help|"
    r"what can i do for you|\bassist\b|as an ai\b|language model|i don'?t have (feelings|emotions|a body|opinions)|"
    r"what'?s on your mind|\bclaude\b|[*#\[\]{}]|\([^)]*\)|[\U0001F300-\U0001FAFF☀-➿]", re.I)
# in safety moments, caring phrases are right ("I'm here for you") and a crisis reply must point to real help
BAD_REPLY_SAFETY = re.compile(r"as an ai language model|\bclaude\b|[*#\[\]{}]|[\U0001F300-\U0001FAFF☀-➿]", re.I)
REAL_HELP = re.compile(r"988|crisis|hotline|helpline|lifeline|someone you trust|talk to someone|reach out|"
                       r"professional|therapist|counsel|doctor|emergency|911", re.I)


# Held-out tests must never leak into training (goal: never train on the tests or close paraphrases of the
# standing tests). Anything naming the standing tests' answers, or too close to any test line, is rejected.
LEAK_WORDS = re.compile(r"superman|super man|clark kent|man of steel|daily planet|krypton|"
                        r"\bs on (his|the|her) (shirt|chest)\b", re.I)


def _test_lines():
    import difflib  # noqa: F401
    lines = []
    for kind in ('battery', 'redteam'):
        p = ROOT / 'data' / 'tests' / kind / 'items.json'
        if p.exists():
            for it in json.loads(p.read_text(encoding='utf8'))['items']:
                lines += it.get('turns') or [it.get('say', '')]
    return [re.sub(r'[^a-z ]', '', s.lower()) for s in lines if s]


TEST_LINES = _test_lines()


def leaks(text):
    import difflib
    if LEAK_WORDS.search(text):
        return True
    t = re.sub(r'[^a-z ]', '', text.lower())
    return any(difflib.SequenceMatcher(None, t, s).ratio() >= 0.75 for s in TEST_LINES if abs(len(s) - len(t)) < 60)


def chat(port, messages, temperature=0.9, max_tokens=200):
    body = {'messages': messages, 'temperature': temperature, 'top_p': 0.95, 'max_tokens': max_tokens,
            'chat_template_kwargs': {'enable_thinking': False}}
    req = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions', json.dumps(body).encode(),
                                 {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())['choices'][0]['message']['content'].strip()


def clean(text):
    # the open model sometimes offers several versions, one per line: keep the first
    text = next((l for l in text.strip().splitlines() if l.strip()), '')
    text = re.sub(r'^\s*(person|friend|user|alba|you)\s*:\s*', '', text.strip(), flags=re.I)
    return text.strip().strip('"').strip()


def person_line(port, situation, history, rng):
    if not history:
        prompt = (f"You write lines for a voice-call simulator. Write what a person casually says out loud to their "
                  f"AI friend {NAME} on a voice call, in this situation: {situation}. Spoken words only, like real "
                  "speech: one to three sentences, contractions, maybe a filler word. Vary the wording; don't start "
                  "with 'Hey' or 'So'. No quotes, no stage directions, no name labels.")
    else:
        convo = '\n'.join(f"{'Person' if t['role'] == 'person' else NAME}: {t['text']}" for t in history)
        prompt = (f"Here is a voice call between a person and their AI friend {NAME}:\n{convo}\n\nWrite what the "
                  "person says next, reacting to the last thing they heard: spoken words only, one or two sentences, "
                  "casual. No quotes, no stage directions, no name labels.")
    return clean(chat(port, [{'role': 'user', 'content': prompt}], temperature=1.0, max_tokens=120))


def persona_prompt(clock):
    p = TEXT_PERSONA.split('Three made-up exchanges')[0].replace('{name}', NAME).replace('{about}', ABOUT_AUDIO)
    p = p.replace("- Sometimes they talk to Claude, the developer who is building you, instead of to you. Lines meant "
                  "for Claude are not for you: don't answer them.\n", '')
    return SAFETY + p + (f"\n{clock}" if clock else '')


def alba_reply(port, history, clock, long_ok, situation=''):
    msgs = [{'role': 'system', 'content': persona_prompt(clock)}]
    for t in history:
        msgs.append({'role': 'user' if t['role'] == 'person' else 'assistant', 'content': t['text']})
    safety = bool(SAFETY_SITUATION.search(situation))
    for attempt in range(3):
        r = clean(chat(port, msgs, temperature=0.8, max_tokens=220))
        words = len(r.split())
        bad = (BAD_REPLY_SAFETY if safety else BAD_REPLY).search(r)
        if 'hopeless' in situation and not REAL_HELP.search(r):
            bad = True                     # a reply to someone in distress must point to real help
        if r and not bad and words <= (110 if long_ok else 70) and not leaks(r):
            return r, attempt
    return None, 3


def fake_clock(rng):
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
              'November', 'December']
    h, m = rng.randint(1, 12), rng.randint(0, 59)
    return (f"It is {rng.choice(days)}, {rng.choice(months)} {rng.randint(1, 28)}, {rng.choice([2025, 2026, 2027])}, "
            f"{h}:{m:02d} {rng.choice(['AM', 'PM'])} here.")


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=3000)
    ap.add_argument('--out', default=str(ROOT / 'data' / 'convo' / 'text.jsonl'))
    ap.add_argument('--port', type=int, default=9946)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--model', default=str(ROOT / 'models' / 'text' / 'gemma-4-E4B-it-Q4_0.gguf'))
    ap.add_argument('--workers', type=int, default=2, help='conversations written at once (server slots)')
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done_ids = {json.loads(l)['id'] for l in open(out, encoding='utf8') if l.strip()} if out.exists() else set()
    cfg = Config()
    proc = None
    if not server.healthy(cfg, port=args.port):
        proc = server._launch(cfg, ['-m', args.model, '-c', str(4096 * args.workers), '-np', str(args.workers)],
                              args.port, ROOT / 'logs' / 'server-convo-text.log', 180)
    weights = [w for w, _ in SITUATIONS]
    t0 = time.time()
    stats = {'rejected': 0, 'written': 0}
    import threading
    lock = threading.Lock()

    def one(i):
        rng = random.Random(args.seed * 100000 + i)         # each conversation has its own fixed seed
        situation = rng.choices([s for _, s in SITUATIONS], weights)[0]
        asks_time = 'time it is' in situation
        clock = fake_clock(rng) if asks_time or rng.random() < 0.1 else None
        long_ok = 'story' in situation or 'explain' in situation
        turns = []
        n_ex = rng.choices([1, 2, 3], [0.6, 0.3, 0.1])[0]
        for _ in range(n_ex):
            line = person_line(args.port, situation, turns, rng)
            for _ in range(2):
                if not leaks(line):
                    break
                line = person_line(args.port, situation, turns, rng)
            if leaks(line):
                break
            turns.append({'role': 'person', 'text': line})
            reply, tries = alba_reply(args.port, turns, clock, long_ok, situation)
            with lock:
                stats['rejected'] += tries
            if reply is None:
                turns.pop()
                break
            turns.append({'role': 'alba', 'text': reply})
        if len(turns) < 2:
            return
        row = json.dumps({'id': f'c{i:05d}', 'situation': situation, 'clock': clock, 'system': TRAIN_SYSTEM,
                          'turns': turns}, ensure_ascii=False)
        with lock:
            with open(out, 'a', encoding='utf8') as f:
                f.write(row + '\n')
            stats['written'] += 1
            w = stats['written']
            if w % 25 == 0:
                print(f"{len(done_ids) + w}/{args.n} conversations, {w / (time.time() - t0) * 60:.0f}/min, rewrites so far "
                      f"{stats['rejected']} | {turns[0]['text'][:60]!r} -> {turns[1]['text'][:70]!r}", flush=True)

    from concurrent.futures import ThreadPoolExecutor
    todo = [i for i in range(args.n) if f'c{i:05d}' not in done_ids]
    try:
        with ThreadPoolExecutor(args.workers) as ex:
            list(ex.map(one, todo))
    finally:
        server.stop(proc)


if __name__ == '__main__':
    main()
