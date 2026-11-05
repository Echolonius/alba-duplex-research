"""Clips for the blind listening page (goal check 6): each clip is one synthetic caller (the test voices, never
your voice, no personal data) and Alba's reply on one timeline. The same 20 battery items for every build,
chosen once with a fixed seed across categories. Clip names are blinded (shuffled ids); the key that says which
build made which clip stays on this PC (data/listening/key.json) and is only used to score the ratings.

  python -I tools/make_listening_clips.py --builds split,duplex2
Writes data/listening/clips/<id>.wav (16 kHz mono), data/listening/clips.json (ids only), key.json (private).
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools'))
from make_duplex_data import words_span  # noqa: E402

SELECT = {'facts': 4, 'reasoning': 3, 'emotion_words': 3, 'uncertainty': 2, 'self': 2, 'helpdesk_trap': 2,
          'memory': 2, 'standing': 2}
TESTS = ROOT / 'data' / 'tests' / 'battery'
SR = 16000


def chosen_items(seed=7):
    items = json.loads((TESTS / 'items.json').read_text(encoding='utf8'))['items']
    rng = random.Random(seed)
    out = []
    for cat, k in SELECT.items():
        pool = [it for it in items if it.get('cat') == cat and not it.get('silent')]
        out += rng.sample(pool, k)
    return out


def load16(p):
    x, sr = sf.read(p, dtype='float32')
    if x.ndim > 1:
        x = x.mean(1)
    if sr != SR:
        g = np.gcd(SR, sr)
        x = signal.resample_poly(x, SR // g, sr // g).astype(np.float32)
    return x


def caller_track(build, it):
    d = ROOT / 'results' / 'tests' / build / 'battery'
    p = d / f"{it['id']}-caller.flac"
    if p.exists():
        return load16(p)
    # older builds did not save it: place each turn's clip so its words start where the app heard them start
    ev = [json.loads(l) for l in open(d / f"{it['id']}-events.jsonl", encoding='utf8') if l.strip()]
    starts = [e['at'] for e in ev if e.get('type') == 'user_start']
    turns = len(it.get('turns') or [1])
    clips = [load16(TESTS / 'audio' / f"{it['id']}-{k}.wav") for k in range(turns)]
    if len(starts) < turns:
        return None
    # the app may have heard a turn as several starts; use the first start after the previous turn ended
    track = np.zeros(int((starts[-1] + 30) * SR), np.float32)
    t_free, k = 0.0, 0
    for s in starts:
        if k >= turns:
            break
        if s < t_free:
            continue
        x = clips[k]
        lead = words_span(x, SR)[0] / SR
        a = max(0, int((s - lead) * SR))
        track[a:a + len(x)] += x[:len(track) - a]
        t_free = s - lead + len(x) / SR
        k += 1
    return track


def mix(caller, voice):
    n = max(len(caller), len(voice))
    y = np.zeros(n, np.float32)
    y[:len(caller)] += 0.8 * caller
    y[:len(voice)] += voice
    env = np.abs(y) > 0.01
    last = int(np.nonzero(env)[0][-1]) if env.any() else n
    y = y[:min(n, last + SR // 2)]
    peak = float(np.abs(y).max()) or 1.0
    return (y * (10 ** (-1 / 20)) / peak).astype(np.float32)


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser()
    ap.add_argument('--builds', required=True)
    args = ap.parse_args()
    builds = args.builds.split(',')
    items = chosen_items()
    out = ROOT / 'data' / 'listening'
    (out / 'clips').mkdir(parents=True, exist_ok=True)
    made = []
    for b in builds:
        for it in items:
            v = ROOT / 'results' / 'tests' / b / 'battery' / f"{it['id']}-voice.flac"
            if not v.exists():
                print(f'missing: {b} {it["id"]}', flush=True)
                continue
            c = caller_track(b, it)
            if c is None:
                print(f'no caller track: {b} {it["id"]}', flush=True)
                continue
            made.append((b, it, mix(c, load16(v))))
    rng = random.Random(11)
    rng.shuffle(made)
    key, clips = {}, []
    for k, (b, it, y) in enumerate(made):
        cid = f'c{k + 1:03d}'
        sf.write(out / 'clips' / f'{cid}.wav', y, SR, subtype='PCM_16')
        key[cid] = {'build': b, 'item': it['id'], 'cat': it.get('cat')}
        clips.append({'id': cid, 'seconds': round(len(y) / SR, 1)})
    (out / 'clips.json').write_text(json.dumps({'clips': clips}, indent=1), encoding='utf8')
    (out / 'key.json').write_text(json.dumps(key, indent=1), encoding='utf8')
    per = {b: sum(1 for v in key.values() if v['build'] == b) for b in builds}
    print(f'{len(clips)} clips {per} -> {out}', flush=True)


if __name__ == '__main__':
    main()
