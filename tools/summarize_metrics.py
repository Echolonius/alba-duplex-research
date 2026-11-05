"""Pools per-conversation metrics into one comparison table.

Usage: python -I summarize_metrics.py <metrics.json> <recordings.json> <out.json> [label]
Groups: the voice under test (track 2) and the other party (track 1), split by
whether the other party is a person or another AI.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else None


def dist(a):
    a = np.asarray(a, float)
    return {'n': int(len(a)), 'median': pct(a, 50), 'p25': pct(a, 25), 'p75': pct(a, 75),
            'mean': float(a.mean()) if len(a) else None,
            'share_overlapping_start': float(np.mean(a < 0)) if len(a) else None,
            'share_within_200ms': float(np.mean(np.abs(a) <= 0.2)) if len(a) else None,
            'share_over_700ms': float(np.mean(a > 0.7)) if len(a) else None}


FILLERS = re.compile(r"\b(um+|uh+|erm|hmm+|mm+|you know|i mean|like)\b", re.I)


def caption_stats(caps, spk):
    lines = [c for c in caps if c['speaker'] == spk]
    words = sum(len(c['text'].split()) for c in lines)
    secs = sum(c['end'] - c['start'] for c in lines)
    text = ' '.join(c['text'] for c in lines).lower()
    rep = len(re.findall(r"\b(\w+)[,]? \1\b", text))
    return {'words': words, 'caption_seconds': secs,
            'words_per_min_while_talking': words / secs * 60 if secs else None,
            'fillers_per_100_words': len(FILLERS.findall(text)) / max(1, words) * 100,
            'immediate_repeats_per_100_words': rep / max(1, words) * 100}


def main():
    metrics = json.loads(Path(sys.argv[1]).read_text(encoding='utf8'))
    recs = {r['id']: r for r in json.loads(Path(sys.argv[2]).read_text(encoding='utf8'))['recordings']}
    label = sys.argv[4] if len(sys.argv) > 4 else 'voice under test'
    groups = {'test_vs_human': [], 'test_vs_ai': []}
    for m in metrics:
        groups['test_vs_ai' if 'GPT' in m['speaker1'] else 'test_vs_human'].append(m)

    def pool(ms, spk):
        tt = [m['turn_taking'] for m in ms]
        fto = sum((t['floor_transfers'][str(spk)] for t in tt), [])
        keep = sum((t['kept_talking_when_interrupted_s'][str(spk)] for t in tt), [])
        other = 3 - spk
        mins_other = sum(t['talk_s'][str(other)] for t in tt) / 60
        mins_total = sum(m['duration_s'] for m in ms) / 60
        v = [m['tracks'][spk - 1]['voice'] for m in ms if m['tracks'][spk - 1]['voice']]
        return {
            'conversations': len(ms), 'minutes': mins_total,
            'response_offset_s': dist(fto),
            'backchannels_per_min_of_other_speech': sum(t['backchannels'][str(spk)] for t in tt) / max(1e-9, mins_other),
            'interruptions_per_min': sum(t['interruptions'][str(spk)] for t in tt) / max(1e-9, mins_total),
            'kept_talking_when_talked_over_s': dist(keep),
            'talk_share': sum(t['talk_s'][str(spk)] for t in tt) / max(1e-9, sum(t['talk_s']['1'] + t['talk_s']['2'] for t in tt)),
            'overlap_share_of_time': float(np.mean([t['overlap_share_of_time'] for t in tt])),
            'bandwidth_hz_median': pct([m['tracks'][spk - 1]['bandwidth_hz'] for m in ms], 50),
            'f0_median_hz_per_conversation': [round(x['f0_median_hz']) for x in v],
            'f0_range_semitones_median': pct([x['f0_range_semitones_p5_p95'] for x in v], 50),
            'f0_sd_semitones_median': pct([x['f0_sd_semitones'] for x in v], 50),
            'jitter_pct_median': pct([x['jitter_local_pct'] for x in v], 50),
            'shimmer_pct_median': pct([x['shimmer_local_pct'] for x in v], 50),
            'hnr_db_median': pct([x['hnr_db'] for x in v if x['hnr_db'] is not None], 50),
        }

    out = {'label': label,
           'test_with_people': pool(groups['test_vs_human'], 2),
           'people_with_test': pool(groups['test_vs_human'], 1),
           'test_with_ai': pool(groups['test_vs_ai'], 2) if groups['test_vs_ai'] else None,
           'ai_with_test': pool(groups['test_vs_ai'], 1) if groups['test_vs_ai'] else None}
    # Words, from captions where they exist (includes caption-only recordings).
    caps = [r for r in recs.values() if r['captions'] and r['speaker1'] == 'Human']
    if caps:
        out['captions_test'] = caption_stats(sum((r['captions'] for r in caps), []), 2)
        out['captions_people'] = caption_stats(sum((r['captions'] for r in caps), []), 1)
    Path(sys.argv[3]).write_text(json.dumps(out, indent=1), encoding='utf8')

    def line(name, a, b, c=None, d=None, fmt='{:.2f}'):
        cells = [fmt.format(x) if isinstance(x, (int, float)) else str(x) for x in (a, b, c, d) if x is not None or True]
        print(f"{name:44s} " + ' | '.join(f"{x:>14s}" for x in cells))
    A, B, C, D = out['test_with_people'], out['people_with_test'], out['test_with_ai'] or {}, out['ai_with_test'] or {}
    print(f"{'':44s} {label + ' w/ people':>14s} | {'people w/ it':>14s} | {label + ' w/ AI':>14s} | {'other AI':>14s}")
    for k in ['median', 'p25', 'p75', 'share_overlapping_start', 'share_within_200ms', 'share_over_700ms', 'n']:
        line(f"response offset {k}", A['response_offset_s'][k], B['response_offset_s'][k],
             (C.get('response_offset_s') or {}).get(k), (D.get('response_offset_s') or {}).get(k))
    for k in ['backchannels_per_min_of_other_speech', 'interruptions_per_min', 'talk_share', 'overlap_share_of_time',
              'bandwidth_hz_median', 'f0_range_semitones_median', 'f0_sd_semitones_median', 'jitter_pct_median', 'shimmer_pct_median', 'hnr_db_median']:
        line(k, A[k], B[k], C.get(k), D.get(k))
    line('kept talking when talked over (median s)', A['kept_talking_when_talked_over_s']['median'], B['kept_talking_when_talked_over_s']['median'],
         (C.get('kept_talking_when_talked_over_s') or {}).get('median'), (D.get('kept_talking_when_talked_over_s') or {}).get('median'))
    print('f0 medians per conversation (test):', A['f0_median_hz_per_conversation'])
    print('f0 medians per conversation (people):', B['f0_median_hz_per_conversation'])
    if caps:
        print('captions test  :', {k: round(v, 2) for k, v in out['captions_test'].items()})
        print('captions people:', {k: round(v, 2) for k, v in out['captions_people'].items()})


if __name__ == '__main__':
    main()
