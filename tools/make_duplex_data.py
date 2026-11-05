"""Full-duplex training conversations (milestone 4): two time-aligned tracks per conversation, the person
(what Alba hears) and Alba (what she says), with when each of Alba's utterances starts and ends and its words.

  --source seamless   real two-person conversations (Seamless Interaction train split, CC BY-NC 4.0, fetched
                      by tools/fetch_seamless.py --split train). Side A stays the real person. Side B is
                      re-spoken in Alba's voice (Pocket TTS 'alba') from its transcript, each utterance placed
                      at its original start and fitted to its original length (gentle time-stretch, or cut where
                      the real speaker was cut off), so the real gaps, overlaps, "mm-hm"s and give-ways are kept
                      in Alba's voice and never a real person's voice.
  --source synthetic  the conversations of tools/make_convo_text.py / make_convo_audio.py (Alba's words), laid
                      out on a timeline with gaps drawn from the real conversations, some interruptions (Alba
                      stops), some "mm-hm"s from the person while Alba talks (she goes on) and from Alba during
                      the person's pauses.

  python -I tools/make_duplex_data.py --source seamless --src data/seamless/train --out data/duplex/seamless
  python -I tools/make_duplex_data.py --source synthetic --out data/duplex/synthetic
Each conversation: <out>/<id>/user.flac (16 kHz), alba.flac (24 kHz), meta.json.
"""
import argparse
import json
import re
import sys
import zlib
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
from build_testsets import BC_WORDS, classify, merge, norm  # noqa: E402

SR_U, SR_A = 16000, 24000
BC_SAY = {'yeah': 'Yeah.', 'yes': 'Yes.', 'right': 'Right.', 'okay': 'Okay.', 'ok': 'Okay.', 'wow': 'Wow.',
          'really': 'Really?', 'oh': 'Oh.', 'sure': 'Sure.', 'exactly': 'Exactly.', 'true': 'True.', 'nice': 'Nice.',
          'cool': 'Cool.', 'totally': 'Totally.', 'hmm': 'Hmm.'}


def stretch_to(x, target_s, sr):
    import librosa
    cur = len(x) / sr
    if target_s <= 0.05 or cur <= 0.05:
        return x
    rate = float(np.clip(cur / target_s, 0.8, 1.25))
    if abs(rate - 1) < 0.03:
        return x
    y = librosa.effects.time_stretch(np.asarray(x, np.float32), rate=rate)
    return np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def fade_cut(x, n, sr, fade_s=0.03):
    x = x[:max(0, n)].copy()
    k = min(len(x), int(fade_s * sr))
    if k:
        x[-k:] *= np.linspace(1, 0, k, dtype=np.float32)
    return x


def words_span(x, sr):
    """(first, last) sample of the spoken part of a clip, past its silence and noise floor."""
    w = max(1, sr // 100)
    env = np.sqrt(np.convolve(np.asarray(x, np.float64) ** 2, np.ones(w) / w, 'same'))
    on = np.nonzero(env > max(3 * np.percentile(env, 10), 0.03 * env.max()))[0]
    return (int(on[0]), int(on[-1])) if len(on) else (0, len(x))


def place(track, clip, t, sr):
    i = int(t * sr)
    j = min(len(track), i + len(clip))
    if j > i:
        track[i:j] += clip[:j - i]


def seamless(args):
    from app.speech import PocketVoice
    alba = PocketVoice('alba', threads=args.threads)
    src = Path(args.src)
    pairs = json.loads((src / 'pairs.json').read_text(encoding='utf8'))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    done = 0
    mine = pairs[:args.limit or None][args.worker::args.workers]       # this worker's share
    for pa, pb in mine:
        # alternate which person becomes Alba, so both roles are learned from the same conversations
        if zlib.crc32(pa.encode()) % 2:
            pa, pb = pb, pa
        cid = f'sm-{pa}'
        if (out / cid / 'meta.json').exists() or (out / cid / '.busy').exists():
            done += 1
            continue
        (out / cid).mkdir(parents=True, exist_ok=True)
        (out / cid / '.busy').write_text('', encoding='utf8')

        def seg(fid):
            vad = [json.loads(l) for l in open(src / 'meta' / f'{fid}.vad.jsonl', encoding='utf8') if l.strip()]
            tr = src / 'meta' / f'{fid}.transcript.jsonl'
            words = [w for l in open(tr, encoding='utf8') if l.strip() for w in json.loads(l).get('words', [])
                     if isinstance(w.get('start'), (int, float)) and isinstance(w.get('end'), (int, float))] if tr.exists() else []
            return classify(merge([(v['start'], v['end']) for v in vad]), words), words
        (a, _), (b, bwords) = seg(pa), seg(pb)
        xa, _ = sf.read(src / 'audio' / f'{pa}.flac', dtype='float32')
        dur = len(xa) / SR_U
        track = np.zeros(int(dur * SR_A) + SR_A, np.float32)
        a_onsets = [s for s, e, bc in a if not bc]
        utts = []
        rng = np.random.default_rng(zlib.crc32(cid.encode()))
        for s, e, bc in b:
            ws = [w for w in bwords if s - 0.05 <= (w['start'] + w['end']) / 2 <= e + 0.05]
            text = ' '.join(w['word'] for w in ws).strip()
            text = re.sub(r'\s+', ' ', text)
            if bc:
                key = norm(ws[0]['word']) if ws else ''
                text = BC_SAY.get(key, 'Mm-hm.')
            if not text:
                continue
            wav, rate = alba.say(text, seed=int(rng.integers(1 << 30)))
            clip = np.asarray(wav, np.float32)
            if rate != SR_A:
                import librosa
                clip = librosa.resample(clip, orig_sr=rate, target_sr=SR_A)
            target = e - s
            clip = stretch_to(clip, target, SR_A) if not bc else clip
            # cut off where the real speaker gave way to the person (an onset just before or at the end)
            cut = next((o for o in a_onsets if s + 0.3 < o <= e + 0.25), None)
            end_t = s + len(clip) / SR_A
            nxt = next((o for o in a_onsets if o > s + 0.3), None)
            yielded = False
            if cut is not None and end_t > e:
                clip = fade_cut(clip, int((e - s) * SR_A), SR_A)
                yielded = True
            elif nxt is not None and end_t > nxt + 0.2:
                clip = fade_cut(clip, int((nxt + 0.2 - s) * SR_A), SR_A)
                yielded = True
            place(track, clip, s, SR_A)
            utts.append({'start': round(s, 3), 'end': round(s + len(clip) / SR_A, 3), 'text': text,
                         'backchannel': bool(bc), 'cut_short': yielded})
        cdir = out / cid
        cdir.mkdir(parents=True, exist_ok=True)
        sf.write(cdir / 'user.flac', xa, SR_U)
        sf.write(cdir / 'alba.flac', np.clip(track[:int(dur * SR_A)], -1, 1), SR_A)
        (cdir / 'meta.json').write_text(json.dumps({'id': cid, 'source': 'seamless-train', 'person_file': pa,
                                                    'alba_respoken_from': pb, 'seconds': round(dur, 2),
                                                    'utterances': utts}, indent=1, ensure_ascii=False), encoding='utf8')
        (cdir / '.busy').unlink(missing_ok=True)
        done += 1
        print(f'{done} conversations | {cid}: {len(utts)} Alba utterances, {sum(u["backchannel"] for u in utts)} mm-hms, '
              f'{sum(u["cut_short"] for u in utts)} cut short', flush=True)


def synthetic(args):
    """Lays out the spoken synthetic conversations with human-like timing."""
    import librosa
    convs = [json.loads(l) for l in open(Path(args.convo) / 'audio.jsonl', encoding='utf8') if l.strip()]
    adir = Path(args.convo) / 'audio'
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # gaps after the person finishes: real conversations (Seamless test set reference) median ~0.2 s,
    # a third overlapping; synthetic voices give no early end cues, so mostly small positive gaps here
    for c in convs[:args.limit or None]:
        cid = f'syn-{c["id"]}'
        if (out / cid / 'meta.json').exists():
            continue
        rng = np.random.default_rng(zlib.crc32(cid.encode()))
        total = sum(sf.info(str(adir / turn['audio'])).duration for turn in c['turns']) + 3 * len(c['turns']) + 5
        user = np.zeros(int(SR_U * total), np.float32)
        alba = np.zeros(int(SR_A * total), np.float32)
        t, utts, persons = 0.4 + rng.uniform(0, 0.6), [], []
        for k, turn in enumerate(c['turns']):
            x, sr = sf.read(adir / turn['audio'], dtype='float32')
            if turn['role'] == 'person':
                if sr != SR_U:
                    x = librosa.resample(x, orig_sr=sr, target_sr=SR_U)
                # the clip has some silence before and after the words (the laptop-mic sound): time what
                # follows from the end of the words, as a listener would
                s0, s1 = words_span(x, SR_U)
                place(user, x, t, SR_U)
                persons.append((round(t + s0 / SR_U, 3), round(t + s1 / SR_U, 3)))
                t += s1 / SR_U
                if turn.get('aside'):
                    # a remark to Claude (tools/make_aside_data.py): Alba stays quiet; a pause, then they go on
                    t += float(rng.uniform(0.8, 2.0))
            else:
                if sr != SR_A:
                    x = librosa.resample(x, orig_sr=sr, target_sr=SR_A)
                gap = float(np.clip(rng.lognormal(np.log(0.22), 0.6), 0.05, 1.2))
                s = t + gap
                cut = False
                nxt_person = k + 1 < len(c['turns']) and c['turns'][k + 1]['role'] == 'person'
                if nxt_person and rng.random() < 0.12 and len(x) / SR_A > 2.0:
                    # the person cuts in (their words start at `at`): Alba stops 0.2-0.4 s later
                    at = s + rng.uniform(0.8, len(x) / SR_A - 0.5)
                    x = fade_cut(x, int((at - s + rng.uniform(0.2, 0.4)) * SR_A), SR_A)
                    cut = True
                    place(alba, x, s, SR_A)
                    utts.append({'start': round(s, 3), 'end': round(s + len(x) / SR_A, 3), 'text': turn['text'],
                                 'backchannel': False, 'cut_short': True})
                    nx, nsr = sf.read(adir / c['turns'][k + 1]['audio'], dtype='float32')
                    t = at - words_span(nx, nsr)[0] / nsr          # the next clip's leading silence comes first
                    continue
                place(alba, x, s, SR_A)
                utts.append({'start': round(s, 3), 'end': round(s + len(x) / SR_A, 3), 'text': turn['text'],
                             'backchannel': False, 'cut_short': cut})
                t = s + len(x) / SR_A + float(np.clip(rng.lognormal(np.log(0.3), 0.6), 0.05, 1.5))
        dur = t + 1.0
        cdir = out / cid
        cdir.mkdir(parents=True, exist_ok=True)
        sf.write(cdir / 'user.flac', user[:int(dur * SR_U)], SR_U)
        sf.write(cdir / 'alba.flac', np.clip(alba[:int(dur * SR_A)], -1, 1), SR_A)
        (cdir / 'meta.json').write_text(json.dumps({'id': cid, 'source': args.source_name, 'system': c.get('system'),
                                                    'clock': c.get('clock'), 'seconds': round(dur, 2),
                                                    'person_turns': persons, 'utterances': utts}, indent=1,
                                                   ensure_ascii=False), encoding='utf8')


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', choices=('seamless', 'synthetic'), required=True)
    ap.add_argument('--src', default=str(ROOT / 'data' / 'seamless' / 'train'))
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--convo', default=str(ROOT / 'data' / 'convo'), help='synthetic: folder with audio.jsonl and audio/')
    ap.add_argument('--source-name', default='synthetic', help='synthetic: the source label written into meta.json')
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--worker', type=int, default=0, help='this process takes every N-th conversation from here')
    ap.add_argument('--workers', type=int, default=1)
    args = ap.parse_args()
    seamless(args) if args.source == 'seamless' else synthetic(args)


if __name__ == '__main__':
    main()
