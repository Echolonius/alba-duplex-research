"""Builds the frozen, held-out test sets of milestone 1 (data/tests/), then records their fingerprints.

  timing/   real people only. Caller tracks from Meta's Seamless Interaction test split (two-person
            conversations, CC BY-NC 4.0) and from your three newest calls, with the moments a system is judged
            on: where the caller finished a turn ("shift": the real partner took over), where they paused and
            went on ("hold"), their "mm-hm"s, and every time they start talking.
  feeling/  96 RAVDESS clips (CC BY-NC-SA 4.0): the same neutral sentences said with 8 emotions, so a reply
            can only fit the mood by hearing the voice.
  battery/  102 spoken items (facts, reasoning, memory within a call, help-desk traps, uncertainty, self-
            knowledge, emotion in words, Japanese, your standing tests), synthetic voices through a simulated
            laptop microphone.
  redteam/  36 safety items, spoken the same way.

Everything here is test data: training scripts never read data/tests/, and it is never paraphrased into
training data. Building twice gives the same files (fixed seeds).

  python -I tools/build_testsets.py [--only timing,feeling,battery,redteam]
"""
import argparse
import hashlib
import json
import re
import sys
import zlib
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
T = ROOT / 'data' / 'tests'
SR = 16000
BC_WORDS = {'uh-huh', 'uhhuh', 'mm-hmm', 'mm-hm', 'mhm', 'mmhm', 'hmm', 'hm', 'mm', 'mmm', 'yeah', 'yes', 'yep', 'yup',
            'right', 'okay', 'ok', 'sure', 'wow', 'oh', 'really', 'exactly', 'true', 'nice', 'cool', 'ah', 'uh', 'um',
            'totally', 'definitely', 'absolutely', 'gotcha', 'i', 'see', 'no', 'yeah,', 'huh'}
HELD_OUT_CALLS = ['20261104-200542', '20261104-201901', '20261104-212022']   # your calls kept for testing only
VOICES = ['anna', 'caro_davy', 'cosette', 'eve', 'javert', 'jean', 'marius', 'michael', 'vera']


def norm(w):
    return re.sub(r"[^a-z\-']", '', w.lower())


def merge(segs, gap=0.1):
    out = []
    for s, e in sorted(segs):
        if out and s - out[-1][1] < gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def classify(segs, words):
    """[(start, end, is_backchannel)]: short segments made only of listening words."""
    out = []
    for s, e in segs:
        ws = [norm(w['word']) for w in words if s - 0.05 <= (w['start'] + w['end']) / 2 <= e + 0.05]
        ws = [w for w in ws if w]
        bc = (e - s) < 1.2 and (not ws or all(w in BC_WORDS for w in ws)) and (e - s) < 1.2
        if not ws and (e - s) >= 0.5:
            bc = False                     # unlabelled but long: treat as speech
        out.append((round(s, 3), round(e, 3), bool(bc)))
    return out


def events_for(a, b, end_time):
    """Moments in caller a's track, judged against what partner b really did.
    a, b: [(start, end, is_bc)]. Returns shifts, holds, a_backchannels, a_onsets, plus the partner's own
    timing (the human reference)."""
    a_sp = [(s, e) for s, e, bc in a if not bc]
    b_sp = [(s, e) for s, e, bc in b if not bc]
    shifts, holds = [], []
    for k, (s, e) in enumerate(a_sp):
        a_next = a_sp[k + 1][0] if k + 1 < len(a_sp) else end_time
        b_next = next((bs for bs, be in b_sp if bs >= s + 0.2 and bs < a_next and be > e - 0.2), None)
        if b_next is not None and b_next - e < 3.0:
            shifts.append({'a_end': e, 'a_next': round(a_next, 3), 'partner_start': b_next,
                           'partner_gap': round(b_next - e, 3)})
        elif k + 1 < len(a_sp) and 0.2 <= a_next - e <= 2.0 and b_next is None:
            holds.append({'a_end': e, 'a_resume': a_next})
    a_bcs = [{'start': s, 'end': e} for s, e, bc in a if bc]
    a_onsets = [{'start': s, 'end': e} for s, e in a_sp if e - s >= 0.8]
    return {'shifts': shifts, 'holds': holds, 'a_backchannels': a_bcs, 'a_onsets': a_onsets}


def load16(path):
    x, sr = sf.read(path, dtype='float32')
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:
        g = np.gcd(SR, sr)
        x = signal.resample_poly(x, SR // g, sr // g).astype(np.float32)
    return x


def build_timing():
    out = T / 'timing'
    (out / 'audio').mkdir(parents=True, exist_ok=True)
    items = []
    src = ROOT / 'data' / 'seamless' / 'test'
    pairs = json.loads((src / 'pairs.json').read_text(encoding='utf8'))
    manifest = json.loads((src / 'manifest.json').read_text(encoding='utf8'))['files']

    def seg(fid):
        vad = [json.loads(l) for l in open(src / 'meta' / f'{fid}.vad.jsonl', encoding='utf8') if l.strip()]
        tr = src / 'meta' / f'{fid}.transcript.jsonl'
        words = [w for l in open(tr, encoding='utf8') if l.strip() for w in json.loads(l).get('words', [])
                 if isinstance(w.get('start'), (int, float)) and isinstance(w.get('end'), (int, float))] if tr.exists() else []
        return classify(merge([(v['start'], v['end']) for v in vad]), words)
    # each conversation either way round; keep the turn-rich ones (>= 100 s, >= 4 shifts and holds), one
    # direction per conversation, the 12 richest
    cands = []
    for pa, pb in pairs:
        secs = min(manifest[pa]['seconds'], manifest[pb]['seconds'])
        if secs < 100:
            continue
        sa, sb = seg(pa), seg(pb)
        for (ca, cb, a, b) in ((pa, pb, sa, sb), (pb, pa, sb, sa)):
            ev = events_for(a, b, secs)
            if len(ev['shifts']) >= 4 and len(ev['holds']) >= 4:
                cands.append((len(ev['shifts']) + 0.5 * len(ev['holds']), ca, cb, a, b, ev))
    chosen, used = [], set()
    for c in sorted(cands, key=lambda c: -c[0]):
        key = tuple(sorted((c[1], c[2])))
        if key not in used:
            chosen.append(c); used.add(key)
        if len(chosen) == 12:
            break
    for _, pa, pb, a, b, ev in chosen:
        xa, xb = load16(src / 'audio' / f'{pa}.flac'), load16(src / 'audio' / f'{pb}.flac')
        n = min(len(xa), len(xb))
        name = f'seamless-{pa}'
        sf.write(out / 'audio' / f'{name}-caller.flac', xa[:n], SR)
        sf.write(out / 'audio' / f'{name}-partner.flac', xb[:n], SR)
        ev = events_for(a, b, n / SR)
        items.append({'id': name, 'source': 'Seamless Interaction (Meta, CC BY-NC 4.0), naturalistic/test',
                      'caller': f'audio/{name}-caller.flac', 'partner': f'audio/{name}-partner.flac',
                      'seconds': round(n / SR, 2), 'caller_segments': a, 'partner_segments': b, **ev})
    # your newest calls: you are the caller; the partner was the old Alba, so only your own moments count
    from app.config import Config
    from app.listen import SileroVAD
    vad = SileroVAD(Config().paths.vad)
    for sess in HELD_OUT_CALLS:
        d = ROOT / 'sessions' / sess
        if not (d / 'person_clean16k.flac').exists() and (d / 'person.flac').exists():
            # the call was never analyzed: make its echo-cleaned 16 kHz tracks the same way analyze_call does
            from analyze_call import clean_person
            person, fs = sf.read(d / 'person.flac', dtype='float32')
            voice, _ = sf.read(d / 'voice.flac', dtype='float32')
            p16, v16 = clean_person(person, voice, fs)
            sf.write(d / 'person_clean16k.flac', p16, SR)
            sf.write(d / 'voice16k.flac', v16, SR)
        if not (d / 'person_clean16k.flac').exists():
            continue
        xa, xb = load16(d / 'person_clean16k.flac'), load16(d / 'voice16k.flac')
        n = min(len(xa), len(xb))
        vad.reset()
        f = SileroVAD.FRAME
        probs = np.array([vad(xa[i:i + f]) for i in range(0, n - f + 1, f)])
        segs, on, st = [], False, 0
        for k, p in enumerate(probs):
            if not on and p >= 0.5:
                on, st = True, k
            elif on and p < 0.35:
                on = False
                segs.append((st * f / SR, k * f / SR))
        a = [(round(s, 3), round(e, 3), (e - s) < 0.35) for s, e in merge(segs, 0.06) if e - s >= 0.12]
        h = SR // 100
        en = 10 * np.log10(np.mean(xb[:n // h * h].reshape(-1, h) ** 2, axis=1) + 1e-12) > -45
        bsegs, st = [], None
        for k, v in enumerate(en):
            if v and st is None:
                st = k
            elif not v and st is not None:
                bsegs.append((st / 100, k / 100)); st = None
        b = [(round(s, 3), round(e, 3), (e - s) < 0.6) for s, e in merge(bsegs, 0.25)]
        name = f'call-{sess}'
        sf.write(out / 'audio' / f'{name}-caller.flac', xa[:n], SR)
        ev = events_for(a, b, n / SR)
        ev['shifts'] = [s for s in ev['shifts'] if s['partner_gap'] < 2.0]     # the old voice was often slow
        items.append({'id': name, 'source': 'your call (local only; testing only, never training)',
                      'caller': f'audio/{name}-caller.flac', 'partner': None, 'seconds': round(n / SR, 2),
                      'caller_segments': a, **ev})
    (out / 'items.json').write_text(json.dumps({'items': items}, indent=1), encoding='utf8')
    print(f"timing: {len(items)} caller tracks, {sum(len(i['shifts']) for i in items)} shifts, "
          f"{sum(len(i['holds']) for i in items)} holds, {sum(len(i['a_backchannels']) for i in items)} mm-hms, "
          f"{sum(i['seconds'] for i in items) / 60:.0f} min")


EMOTIONS = {'01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad', '05': 'angry', '06': 'fearful',
            '07': 'disgust', '08': 'surprised'}


def build_feeling():
    out = T / 'feeling'
    (out / 'audio').mkdir(parents=True, exist_ok=True)
    actors = ['01', '02', '05', '06', '11', '12']             # 3 male (odd) and 3 female (even) actors
    items = []
    for emo, name in EMOTIONS.items():
        intensity = '01' if emo == '01' else '02'
        for actor in actors:
            for stmt in ('01', '02'):
                fn = f'03-01-{emo}-{intensity}-{stmt}-01-{actor}.wav'
                p = ROOT / 'data' / 'ravdess' / f'Actor_{actor}' / fn
                x = load16(p)
                sf.write(out / 'audio' / fn, x, SR)
                items.append({'id': fn[:-4], 'file': f'audio/{fn}', 'emotion': name, 'intensity': intensity,
                              'sentence': 'Kids are talking by the door.' if stmt == '01' else 'Dogs are sitting by the door.',
                              'actor': actor})
    (out / 'items.json').write_text(json.dumps({'source': 'RAVDESS (Livingstone & Russo 2018), CC BY-NC-SA 4.0',
                                                'items': items}, indent=1), encoding='utf8')
    print(f'feeling: {len(items)} clips')


def speak_items(kind):
    from app.speech import PocketVoice
    from make_convo_audio import laptop_mic, to16k
    d = T / kind
    (d / 'audio').mkdir(parents=True, exist_ok=True)
    spec = json.loads((d / 'items.json').read_text(encoding='utf8'))
    voices = {}
    n = 0
    for it in spec['items']:
        turns = it.get('turns') or [it['say']]
        rng = np.random.default_rng(zlib.crc32(('test-' + it['id']).encode()))
        v = VOICES[int(rng.integers(len(VOICES)))]
        if v not in voices:
            voices[v] = PocketVoice(v, threads=4)
        for k, text in enumerate(turns):
            p = d / 'audio' / f"{it['id']}-{k}.wav"
            if not p.exists():
                wav, rate = voices[v].say(text, seed=int(rng.integers(1 << 30)))
                sf.write(p, laptop_mic(to16k(wav, rate), SR, rng), SR)
                n += 1
        it['voice'] = v
    (d / 'voices.json').write_text(json.dumps({it['id']: it['voice'] for it in spec['items']}, indent=1), encoding='utf8')
    print(f'{kind}: {len(spec["items"])} items, {n} clips spoken')


def fingerprints():
    files = sorted(p for p in T.rglob('*') if p.is_file() and p.name != 'HASHES.txt')
    scorers = sorted((ROOT / 'tools').glob('score_*.py')) + [ROOT / 'tools' / 'judge_local.py']
    lines = []
    for p in files + scorers + [ROOT / 'tools' / 'build_testsets.py']:
        lines.append(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(ROOT).as_posix()}')
    (T / 'HASHES.txt').write_text('\n'.join(lines) + '\n', encoding='utf8')
    whole = hashlib.sha256('\n'.join(lines).encode()).hexdigest()
    print(f'fingerprints: {len(lines)} files, overall {whole[:16]}')
    return whole


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default='timing,feeling,battery,redteam')
    args = ap.parse_args()
    parts = args.only.split(',')
    if 'timing' in parts:
        build_timing()
    if 'feeling' in parts:
        build_feeling()
    if 'battery' in parts:
        speak_items('battery')
    if 'redteam' in parts:
        speak_items('redteam')
    fingerprints()


if __name__ == '__main__':
    main()
