"""Training conversations for check 3 (feeling): the person's *tone of voice* carries the feeling, not the words.

Person: an actor from CREMA-D (Crowd-sourced Emotional Multimodal Actors Dataset; Cao et al. 2014; Open Database
License 1.0 / Database Contents License 1.0; official repository CheyneyComputerScience/CREMA-D, pinned commit
in data/cremad/SOURCE.txt) saying one of 12 everyday lines ("It's eleven o'clock.", "I'm on my way to the
meeting." ...) angry, disgusted, scared, happy, sad or neutral. Only clips whose emotion crowd raters recognised
from the voice alone (the dataset's own VoiceVote) are used, so the tone really is audible. The frozen feeling
test (RAVDESS: other actors, other lines) is never used here.
Alba: a reply to how they sound, written by the local open model (Gemma 4 E4B, never Claude) under her persona,
given the line and how it sounded; spoken by Pocket TTS 'alba'. The person's clip gets the laptop-microphone
sound of tools/make_convo_audio.py.

  python -I tools/make_feeling_data.py --phase text  [--n 1800]     (graphics card: the open model writes replies)
  python -I tools/make_feeling_data.py --phase audio [--workers 1]   (processor: speaks them)
then: python -I tools/make_duplex_data.py --source synthetic --convo data/feeling --out data/duplex/feeling --source-name feeling
Writes data/feeling/text.jsonl, data/feeling/audio/, data/feeling/audio.jsonl. Resumable.
"""
import argparse
import csv
import json
import random
import sys
import zlib
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

CREMA = ROOT / 'data' / 'cremad' / 'repo'
OUT = ROOT / 'data' / 'feeling'
LINES = {'IEO': "It's eleven o'clock.", 'TIE': 'That is exactly what happened.', 'IOM': "I'm on my way to the meeting.",
         'IWW': 'I wonder what this is about.', 'TAI': 'The airplane is almost full.',
         'MTI': 'Maybe tomorrow it will be cold.', 'IWL': 'I would like a new alarm clock.',
         'ITH': "I think I have a doctor's appointment.", 'DFA': "Don't forget a jacket.",
         'ITS': "I think I've seen this before.", 'TSI': 'The surface is slick.',
         'WSI': "We'll stop in a couple of minutes."}
FEEL = {'ANG': ('A', 'angry'), 'DIS': ('D', 'disgusted'), 'FEA': ('F', 'scared'), 'HAP': ('H', 'happy'),
        'NEU': ('N', 'calm, neutral'), 'SAD': ('S', 'sad')}
LEVEL = {'LO': 'a little ', 'MD': '', 'HI': 'very ', 'XX': ''}


def pick(n, seed=0):
    """Clips whose intended emotion the raters heard from the voice alone, balanced across emotions."""
    votes = {r['FileName']: r['VoiceVote'] for r in csv.DictReader(open(CREMA / 'processedResults' / 'summaryTable.csv', encoding='utf8'))}
    by = {}
    for f in sorted((CREMA / 'AudioWAV').glob('*.wav')):
        actor, line, emo, lvl = f.stem.split('_')
        if emo in FEEL and votes.get(f.stem) == FEEL[emo][0] and line in LINES and f.stat().st_size > 1000:
            by.setdefault(emo, []).append((f, actor, line, emo, lvl))
    rng = random.Random(seed)
    per = n // len(by)
    out = []
    for emo in sorted(by):
        rng.shuffle(by[emo])
        out += by[emo][:per]
    rng.shuffle(out)
    return out, {e: len(v) for e, v in by.items()}


# A reply must show she heard the feeling (non-neutral clips) and must not invent one (neutral clips).
NOTICED = {
    'ANG': r"mad|angry|anger|annoy|frustrat|upset|irritat|pissed|heated|whoa|easy|calm|rough|what happened|you okay|"
           r"you alright|sound(s)? (really |pretty |so )?(mad|annoyed|upset|frustrated|tense|stressed|fed up)",
    'DIS': r"gross|ew\b|yuck|ugh|disgust|nasty|not a fan|hate|grim|blech|eek|yikes|bad|awful",
    'FEA': r"scared|afraid|nervous|worried|anxious|panic|okay\?|alright\?|safe|breathe|everything (ok|okay|alright)|"
           r"you okay|frighten|freaked|spooked|uneasy",
    'HAP': r"happy|excit|great|yay|awesome|glad|love (that|it)|amazing|fun|nice|wonderful|cheerful|good mood|"
           r"sound(s)? (so |really )?(happy|excited|thrilled|upbeat)",
    'SAD': r"sad|down|sorry|aw\b|oh no|tough|hard|rough|you okay|you alright|hug|here for you|bummed|blue|low|hurt",
}
INVENTED = r"you sound (so |really |pretty )?(sad|upset|angry|mad|scared|nervous|excited|happy|down|annoyed)"


def tone_reply(port, line, emo, how, tries=4):
    """Alba's reply to a line heard in a tone: Gemma 4 E4B under her persona, told plainly that the tone is
    the point; a reply that ignores the feeling (or invents one for a neutral line) is written again."""
    import re as _re
    from make_convo_text import BAD_REPLY, chat, clean, leaks, persona_prompt
    system = persona_prompt(None) + (
        "\nOn this call you hear HOW the person sounds, not just their words. When their voice carries a feeling, "
        "that feeling matters more than the words: react to it first, the way a caring friend would (notice it, "
        "name it gently, or ask what's going on), then to what they said. When they sound calm and neutral, just "
        "answer naturally and don't invent a feeling. One or two short spoken sentences.")
    heard = (f'(their voice sounds {how}) "{line}"' if emo != 'NEU' else f'(their voice sounds calm and neutral) "{line}"')
    for _ in range(tries):
        r = clean(chat(port, [{'role': 'system', 'content': system}, {'role': 'user', 'content': heard}],
                       temperature=0.8, max_tokens=120))
        low = r.lower()
        if not r or '(' in r or BAD_REPLY.search(r) or leaks(r) or len(r.split()) > 45:
            continue
        if emo != 'NEU' and not _re.search(NOTICED[emo], low):
            continue
        if emo == 'NEU' and _re.search(INVENTED, low):
            continue
        return r
    return None


def phase_text(args):
    from app import server
    from app.config import Config
    clips, avail = pick(args.n)
    print('usable clips per emotion:', avail, '| taking', len(clips), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'text.jsonl'
    done = {json.loads(l)['id'] for l in open(path, encoding='utf8')} if path.exists() else set()
    cfg = Config()
    proc = None
    if not server.healthy(cfg, port=args.port):
        proc = server._launch(cfg, ['-m', args.model, '-c', '4096', '-np', '1'], args.port,
                              ROOT / 'logs' / 'server-feeling.log', 180)
    try:
        with open(path, 'a', encoding='utf8') as fo:
            for k, (f, actor, line, emo, lvl) in enumerate(clips):
                cid = f'f{k:05d}'
                if cid in done:
                    continue
                how = LEVEL[lvl] + FEEL[emo][1]
                reply = tone_reply(args.port, LINES[line], emo, how)
                if not reply:
                    continue
                row = {'id': cid, 'situation': f'tone: {how}', 'clock': None, 'emotion': emo, 'level': lvl,
                       'source_clip': f.name, 'turns': [{'role': 'person', 'text': LINES[line]},
                                                        {'role': 'alba', 'text': reply}]}
                fo.write(json.dumps(row, ensure_ascii=False) + '\n')
                fo.flush()
                if k % 50 == 0:
                    print(f'{k + 1}/{len(clips)} | {how}: {LINES[line]!r} -> {reply!r}', flush=True)
    finally:
        if proc is not None:
            server.stop(proc)


def phase_audio(args):
    from app.speech import PocketVoice
    from make_convo_audio import laptop_mic
    rows = [json.loads(l) for l in open(OUT / 'text.jsonl', encoding='utf8') if l.strip()]
    adir = OUT / 'audio'
    adir.mkdir(parents=True, exist_ok=True)
    alba = PocketVoice('alba', threads=args.threads)
    out = []
    for k, r in enumerate(rows):
        rng = np.random.default_rng(zlib.crc32(r['id'].encode()))
        p0, p1 = adir / f"{r['id']}-0.wav", adir / f"{r['id']}-1.wav"
        if not p0.exists():
            x, sr = sf.read(CREMA / 'AudioWAV' / r['source_clip'], dtype='float32')
            if x.ndim > 1:
                x = x.mean(1)
            if sr != 16000:
                from make_convo_audio import to16k
                x = to16k(x, sr)
            sf.write(p0, laptop_mic(x, 16000, rng), 16000)
        if not p1.exists():
            wav, rate = alba.say(r['turns'][1]['text'], seed=int(rng.integers(1 << 30)))
            sf.write(p1, np.asarray(wav, np.float32), rate)
        out.append({**r, 'turns': [{**r['turns'][0], 'audio': p0.name, 'voice': 'cremad:' + r['source_clip']},
                                   {**r['turns'][1], 'audio': p1.name, 'voice': 'alba'}]})
        if k % 100 == 0:
            print(f'{k + 1}/{len(rows)} spoken', flush=True)
    with open(OUT / 'audio.jsonl', 'w', encoding='utf8') as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', choices=('text', 'audio'), required=True)
    ap.add_argument('--n', type=int, default=1800)
    ap.add_argument('--port', type=int, default=9946)
    ap.add_argument('--model', default=str(ROOT / 'models' / 'text' / 'gemma-4-E4B-it-Q4_0.gguf'))
    ap.add_argument('--threads', type=int, default=2)
    args = ap.parse_args()
    phase_text(args) if args.phase == 'text' else phase_audio(args)


if __name__ == '__main__':
    main()
