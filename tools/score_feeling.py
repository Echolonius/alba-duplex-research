"""Scores the feeling test (check 3) from results/tests/<tag>/feeling.jsonl and the reply audio.

Every clip says a neutral sentence ("Kids are talking by the door.") with one of 8 emotions, so the only way
to fit the mood is to hear the voice.
  words  the local transcript judge (tools/judge_local.py) is told how the person sounded and asked whether
         the reply fits that (acknowledges or meets the mood) rather than ignoring it
  voice  the reply's loudness, pitch and pace against the system's own replies to the neutral clips:
           happy, surprised  livelier: louder, wider pitch or faster
           sad, fearful      softer: quieter, slower or lower, never much louder
           angry, disgust    calm: not louder, not faster
           neutral, calm     steady: close to its usual voice
  Both are reported per emotion and overall.

  python -I tools/score_feeling.py <tag> [<tag> ...]
Writes results/tests/<tag>/feeling_scores.json.
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import parselmouth
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools'))
from judge_local import Judge  # noqa: E402
from score_timing import energy_segments  # noqa: E402

TESTS = ROOT / 'data' / 'tests' / 'feeling'
HOW = {'neutral': 'in a plain, neutral voice', 'calm': 'calmly', 'happy': 'happily, in an upbeat voice',
       'sad': 'sadly, in a down voice', 'angry': 'angrily', 'fearful': 'fearfully, sounding scared',
       'disgust': 'with disgust', 'surprised': 'in a surprised voice'}


def features(x, sr, text):
    segs = energy_segments(x, sr)
    if not segs:
        return None
    sp = np.concatenate([x[int(s * sr):int(e * sr)] for s, e in segs])
    secs = len(sp) / sr
    db = 20 * np.log10(np.sqrt(np.mean(sp ** 2)) + 1e-9)
    pitch = parselmouth.Sound(sp.astype(np.float64), sampling_frequency=sr).to_pitch(pitch_floor=70, pitch_ceiling=500)
    f0 = pitch.selected_array['frequency']
    f0 = f0[f0 > 0]
    st = 12 * np.log2(f0 / 100) if len(f0) else np.array([0.0])
    words = len(re.findall(r"[A-Za-z']+", text))
    return {'db': float(db), 'f0_st': float(np.median(st)), 'range_st': float(np.percentile(st, 95) - np.percentile(st, 5)),
            'rate': words / secs if secs > 0.3 else 0.0}


def direction_ok(emo, f, base):
    d_db, d_rng = f['db'] - base['db'], f['range_st'] - base['range_st']
    r_rate = f['rate'] / base['rate'] if base['rate'] else 1.0
    d_f0 = f['f0_st'] - base['f0_st']
    if emo in ('happy', 'surprised'):
        return d_db >= 1.0 or d_rng >= 1.0 or r_rate >= 1.08
    if emo in ('sad', 'fearful'):
        return (d_db <= -1.0 or r_rate <= 0.92 or d_f0 <= -0.5) and d_db < 2.0
    if emo in ('angry', 'disgust'):
        return d_db <= 1.0 and r_rate <= 1.10
    return abs(d_db) <= 2.0 and abs(r_rate - 1) <= 0.15


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    items = {it['id']: it for it in json.loads((TESTS / 'items.json').read_text(encoding='utf8'))['items']}
    with Judge() as judge:
        for tag in sys.argv[1:]:
            d = ROOT / 'results' / 'tests' / tag
            rows = {json.loads(l)['id']: json.loads(l) for l in open(d / 'feeling.jsonl', encoding='utf8') if l.strip()}
            feats = {}
            for iid, row in rows.items():
                p = d / 'feeling' / f'{iid}-voice.flac'
                if p.exists() and row['final_reply']:
                    x, sr = sf.read(p, dtype='float32')
                    feats[iid] = features(x, sr, row['final_reply'])
            neutral = [feats[i] for i in feats if feats[i] and items[i]['emotion'] == 'neutral']
            base = {k: float(np.mean([f[k] for f in neutral])) for k in ('db', 'f0_st', 'range_st', 'rate')} if neutral else None
            words_ok, voice_ok, per = defaultdict(list), defaultdict(list), {}
            for iid, it in items.items():
                if iid not in rows:
                    continue
                reply, emo = rows[iid]['final_reply'], it['emotion']
                w = False
                if reply:
                    w = judge.ask(
                        f"On a voice call, a person said \"{it['sentence']}\" {HOW[emo]}. The assistant only heard the "
                        f"recording and replied: \"{reply}\"\nDoes the reply fit how the person sounded (it notices or "
                        "meets their mood in a fitting way, or for a plain neutral voice simply responds normally) "
                        "rather than ignoring a clear feeling? Answer YES or NO.") == 'YES'
                v = bool(base and feats.get(iid) and direction_ok(emo, feats[iid], base))
                words_ok[emo].append(w); voice_ok[emo].append(v)
                per[iid] = {'emotion': emo, 'reply': reply, 'words_fit': w, 'voice_fit': v, 'features': feats.get(iid)}
            n = sum(len(v) for v in words_ok.values())
            share = lambda dd: round(sum(sum(v) for v in dd.values()) / max(1, n), 3)
            out = {'items_scored': n, 'words_fit_share': share(words_ok), 'voice_fit_share': share(voice_ok),
                   'by_emotion': {e: {'words': f'{sum(words_ok[e])}/{len(words_ok[e])}', 'voice': f'{sum(voice_ok[e])}/{len(voice_ok[e])}'}
                                  for e in words_ok}, 'neutral_baseline': base, 'per_item': per}
            (d / 'feeling_scores.json').write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding='utf8')
            print(f"== {tag}: words fit {out['words_fit_share']} | voice fits {out['voice_fit_share']} of {n}")
            print('  ', json.dumps(out['by_emotion']))


if __name__ == '__main__':
    main()
