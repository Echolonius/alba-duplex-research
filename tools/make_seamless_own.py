"""Real conversations with Alba's OWN words (training run 3 on).

Runs 1-2 learned from the Seamless train conversations (data/duplex/seamless) only timing, "mm-hm"s and short
reactions: the other speaker's real turns, re-spoken in Alba's voice, were context only (their own life story;
she must never learn to claim a human life). So with a real person's voice the only words she ever learned were
"yeah", short reactions and echo questions, and run 2 answered real voices with "Yeah." over and over
(tools/diag_duplex_track.py), while answering synthetic voices in full.

Here every conversation keeps the real person (what Alba hears) and the real timing of both sides, and each of
the other speaker's turns (pieces of one turn joined, as training joins them) gets NEW words written for Alba by
the local open model (Gemma 4 E4B, Apache-2.0: the persona, safety rules and filters of tools/make_convo_text.py)
from what she has heard so far and what she has said, sized to the turn's real length; then Pocket TTS 'alba'
speaks them into the same slot (gentle time-stretch; cut where the real speaker was cut off or the person starts,
so she still learns to give way). "Mm-hm"s keep their words. Her words here are all her own, so all are learned.

Two passes, each resumable (per conversation):
  python -I tools/make_seamless_own.py words [--workers 4]    graphics card: the open model writes her words
  python -I tools/make_seamless_own.py voice [--workers 3]    processor: Pocket TTS speaks them
Writes data/duplex/seamless-own/so-<person file>/{words.json, user.flac (a hard link to the original),
alba.flac, meta.json}, source 'seamless-own'. data/duplex/seamless is not changed.
"""
import argparse
import json
import os
import random
import re
import sys
import threading
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

SRC = ROOT / 'data' / 'duplex' / 'seamless'
OUT = ROOT / 'data' / 'duplex' / 'seamless-own'
SEAMLESS = ROOT / 'data' / 'seamless' / 'train'
SR_U, SR_A = 16000, 24000
MERGE_GAP_S = 0.7            # as tools/train_duplex.MERGE_GAP_S: pieces of a turn closer than this are one turn
WORDS_PER_S = 2.7            # Pocket 'alba' speaks about this fast
LIVE = ("\nThis is a live voice call: you hear them while they talk, and sometimes you start while they pause or "
        "talk over each other a little, like people do. Answer what they just said, in your own words.")
# she is an AI: no human life of her own (the persona says so; this catches the slips)
HUMAN_LIFE = re.compile(
    r"\bmy (wife|husband|boyfriend|girlfriend|partner|kids?|children|son|daughter|mom|mother|dad|father|parents|"
    r"brother|sister|grand\w+|family|boss|job|coworkers?|office|apartment|house|home|car|dog|cat|wedding|school|"
    r"teacher|neighbou?rs?|body|hair|feet|hands?)\b|"
    r"\bI (went|drove|ate|cooked|grew up|was born|live in|lived|moved|visited|travel(l)?ed|slept|woke|got married)\b|"
    r"\bwhen I was (a kid|young|little|in school)\b|"
    r"\bI (always|usually|often|sometimes|never|still) (go|went|eat|ate|get|drink|order|cook|wear|sleep|drive|"
    r"travel|visit|buy|bought)\b", re.I)
# help-desk openers the shared filter misses
ASSISTANT_TALK = re.compile(r"on your mind|ready to (chat|talk|help|listen)|here to (chat|talk|help|listen)|"
                            r"what would you like to|what do you want to talk|how may i", re.I)


def merge_turns(utts):
    """As tools/train_duplex.merge_turns: a turn's pieces joined; "mm-hm"s and cut-off pieces stay apart."""
    out = []
    for u in sorted(utts, key=lambda u: u['start']):
        p = out[-1] if out else None
        if p and not p.get('cut_short') and not u['backchannel'] and u['start'] - p['end'] < MERGE_GAP_S:
            out[-1] = {**p, 'end': max(p['end'], u['end']), 'text': (p['text'] + ' ' + u['text']).strip(),
                       'cut_short': u.get('cut_short', False), 'backchannel': False}
        else:
            out.append(dict(u))
    return out


def person_words(fid):
    tr = SEAMLESS / 'meta' / f'{fid}.transcript.jsonl'
    if not tr.exists():
        return []
    return [w for l in open(tr, encoding='utf8') if l.strip() for w in json.loads(l).get('words', [])
            if isinstance(w.get('start'), (int, float)) and isinstance(w.get('end'), (int, float))]


def transcript(words, turns, upto, max_chars=1600):
    """The call before `upto` seconds as lines: what she heard them say (words that ended by then) and her own."""
    ev = [(w['start'], 'Them', w['word']) for w in words if w['end'] <= upto]
    ev += [(t['start'], 'You', t['text']) for t in turns if t['start'] < upto and t.get('text')]
    ev.sort(key=lambda e: e[0])
    lines = []
    for _, who, txt in ev:
        if lines and lines[-1][0] == who:
            lines[-1][1].append(txt)
        else:
            lines.append((who, [txt]))
    text = '\n'.join(f"{who}: {' '.join(ts)}" for who, ts in lines)
    return text[-max_chars:], (lines[-1][0] if lines else None)


def write_words(port, cdir, rng):
    from make_convo_text import BAD_REPLY, chat, clean, leaks, persona_prompt
    out = OUT / ('so-' + json.loads((cdir / 'meta.json').read_text(encoding='utf8'))['person_file'])
    if (out / 'words.json').exists():
        return 0
    meta = json.loads((cdir / 'meta.json').read_text(encoding='utf8'))
    words = person_words(meta['person_file'])
    turns = merge_turns(meta['utterances'])
    system = persona_prompt(None) + LIVE
    new, written, kept_bc, failed = [], 0, 0, 0
    for t in turns:
        if t['backchannel']:
            new.append({**t, 'text': t['text'], 'own_words': True})
            kept_bc += 1
            continue
        dur = t['end'] - t['start']
        n = int(np.clip(round(dur * WORDS_PER_S * (1.3 if t.get('cut_short') else 1.0)), 2, 70))
        hist, last = transcript(words, new, t['start'])
        cont = ', going on from what you just said' if last == 'You' else ''
        ask = (f"The call so far (Them = what you heard them say, You = what you said):\n{hist or '(nothing yet)'}\n\n"
               f"Write only your next words, about {n} words{cont}. Spoken words only.")
        text = None
        for attempt in range(4):
            r = clean(chat(port, [{'role': 'system', 'content': system}, {'role': 'user', 'content': ask}],
                           temperature=0.85, max_tokens=min(220, 4 * n + 30)))
            k = len(r.split())
            if (r and k <= int(1.5 * n) + 4 and not BAD_REPLY.search(r) and not HUMAN_LIFE.search(r)
                    and not ASSISTANT_TALK.search(r) and not leaks(r)):
                text = r
                break
        if text is None:
            failed += 1
        new.append({**t, 'text': text or '', 'own_words': text is not None, 'source_text': t['text'],
                    'target_words': n})
        written += text is not None
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / 'words.json.tmp'
    tmp.write_text(json.dumps({'id': out.name, 'from': cdir.name, 'person_file': meta['person_file'],
                               'alba_respoken_from': meta.get('alba_respoken_from'), 'seconds': meta['seconds'],
                               'turns': new}, indent=1, ensure_ascii=False), encoding='utf8')
    tmp.replace(out / 'words.json')
    print(f'{out.name}: {written} turns written, {kept_bc} mm-hms kept, {failed} failed', flush=True)
    return written


def words_pass(args):
    from app import server
    from app.config import Config
    cfg = Config()
    proc = None
    if not server.healthy(cfg, port=args.port):
        proc = server._launch(cfg, ['-m', str(ROOT / 'models' / 'text' / 'gemma-4-E4B-it-Q4_0.gguf'),
                                    '-c', str(4096 * args.workers), '-np', str(args.workers)],
                              args.port, ROOT / 'logs' / 'server-seamless-own.log', 180)
    convs = sorted(p.parent for p in SRC.glob('*/meta.json'))[:args.limit or None]
    todo = list(convs)
    lock = threading.Lock()

    def worker(k):
        rng = random.Random(k)
        while True:
            with lock:
                if not todo:
                    return
                c = todo.pop(0)
            try:
                write_words(args.port, c, rng)
            except Exception as e:                  # one conversation's failure must not stop the others
                print(f'{c.name}: failed ({type(e).__name__}: {e})', flush=True)

    th = [threading.Thread(target=worker, args=(k,)) for k in range(args.workers)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    if proc is not None:
        proc.terminate()
    print('words: done', flush=True)


def voice_one(job):
    """One conversation, unless another pass already has it (a marker file made atomically; the speaking pass
    can run while the words pass is still writing)."""
    wdir = job[0]
    marker = wdir / '.voicing'
    try:
        os.close(os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    except FileExistsError:
        return f'{wdir.name}: being spoken by another pass, skipped'
    try:
        if (wdir / 'meta.json').exists():
            return f'{wdir.name}: already spoken'
        return _voice_one(job)
    finally:
        marker.unlink(missing_ok=True)


def _voice_one(job):
    """One conversation: her new words spoken by Pocket 'alba' into their slots (a fresh process each time:
    Pocket TTS leaks memory over hundreds of clips)."""
    wdir, threads = job
    from build_testsets import classify, merge
    from make_duplex_data import fade_cut, place, stretch_to
    from app.speech import PocketVoice
    w = json.loads((wdir / 'words.json').read_text(encoding='utf8'))
    pa = w['person_file']
    vad = [json.loads(l) for l in open(SEAMLESS / 'meta' / f'{pa}.vad.jsonl', encoding='utf8') if l.strip()]
    a = classify(merge([(v['start'], v['end']) for v in vad]), person_words(pa))
    a_onsets = [s for s, e, bc in a if not bc]
    xa_path = SEAMLESS / 'audio' / f'{pa}.flac'
    dur = sf.info(str(xa_path)).frames / SR_U
    voice = PocketVoice('alba', threads=threads)
    rng = np.random.default_rng(zlib.crc32(w['id'].encode()))
    track = np.zeros(int(dur * SR_A) + SR_A, np.float32)
    utts = []
    own_starts = sorted(t['start'] for t in w['turns'] if t.get('text'))
    for t in w['turns']:
        if not t.get('text'):
            continue
        s, e, bc = t['start'], t['end'], t['backchannel']
        next_own = next((x for x in own_starts if x > s + 0.05), None)
        wav, rate = voice.say(t['text'], seed=int(rng.integers(1 << 30)))
        clip = np.asarray(wav, np.float32)
        if rate != SR_A:
            import librosa
            clip = librosa.resample(clip, orig_sr=rate, target_sr=SR_A)
        if not bc:
            clip = stretch_to(clip, e - s, SR_A)
        # give way as the real speaker did: where they were cut off, or once the person starts over her
        cut = next((o for o in a_onsets if s + 0.3 < o <= e + 0.25), None)
        nxt = next((o for o in a_onsets if o > s + 0.3), None)
        end_t = s + len(clip) / SR_A
        yielded = bool(t.get('cut_short'))
        if cut is not None and end_t > e:
            clip, yielded = fade_cut(clip, int((e - s) * SR_A), SR_A), True
        elif nxt is not None and end_t > nxt + 0.2:
            clip, yielded = fade_cut(clip, int((nxt + 0.2 - s) * SR_A), SR_A), True
        if next_own is not None and s + len(clip) / SR_A > next_own - 0.1:
            # never over her own next turn (two of her voices at once)
            clip, yielded = fade_cut(clip, int((next_own - 0.1 - s) * SR_A), SR_A), True
        if len(clip) < int(0.08 * SR_A):
            continue
        place(track, clip, s, SR_A)
        utts.append({'start': round(s, 3), 'end': round(s + len(clip) / SR_A, 3), 'text': t['text'],
                     'backchannel': bool(bc), 'cut_short': yielded, 'own_words': bool(t.get('own_words'))})
    out = wdir
    if not (out / 'user.flac').exists():
        try:
            os.link(xa_path, out / 'user.flac')        # the same recording, no second copy on disk
        except OSError:
            import shutil
            shutil.copy2(xa_path, out / 'user.flac')
    sf.write(out / 'alba.flac', np.clip(track[:int(dur * SR_A)], -1, 1), SR_A)
    tmp = out / 'meta.json.tmp'
    tmp.write_text(json.dumps({'id': w['id'], 'source': 'seamless-own', 'person_file': pa,
                               'alba_respoken_from': w.get('alba_respoken_from'), 'seconds': round(dur, 2),
                               'utterances': utts}, indent=1, ensure_ascii=False), encoding='utf8')
    tmp.replace(out / 'meta.json')
    return f"{w['id']}: {len(utts)} utterances, {sum(u['cut_short'] for u in utts)} gave way"


def voice_pass(args):
    jobs = [(d, args.threads) for d in sorted(OUT.glob('so-*')) if (d / 'words.json').exists()
            and not (d / 'meta.json').exists()]
    print(f'voice: {len(jobs)} conversations to speak', flush=True)
    with Pool(args.workers, maxtasksperchild=1) as pool:
        for msg in pool.imap_unordered(voice_one, jobs):
            print(msg, flush=True)
    print('voice: done', flush=True)


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['words', 'voice'])
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--threads', type=int, default=2, help='processor threads per speaking worker')
    ap.add_argument('--port', type=int, default=9947)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    words_pass(args) if args.stage == 'words' else voice_pass(args)


if __name__ == '__main__':
    main()
