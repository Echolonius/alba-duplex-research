"""What the duplex training data teaches, per source, before any training run (no graphics card, no model).

Every conversation in the data folder is laid out exactly as tools/train_duplex.py lays it out for training (its
own timeline(), spans_of() and build(), so a change there shows here), and Alba's side is counted per source:
  kinds          "mm-hm"s, short reactions (1-3 words) and longer turns, and how many last 0.8 s or more, the
                 timing scorer's line between a listening sound and a turn (tools/score_timing.BC_MAX)
  words learned  whether her words carry loss (runs 1-2 kept a real speaker's own story as context only)
  first words    what her learned utterances open with. An "mm-hm" and an answer start with the same marker and part
                 ways at the plan's first word; run 2 answered real voices with a planned "Yeah." almost every time
                 (results/diag_track_alba-duplex2-0.json)
  text loss      what the text loss is made of: LISTEN steps kept (LISTEN_KEEP of them on average), starts, words
                 learned and words given as context only; next to it the speech frames the audio loss learns
Then the same for the training mix of --shares, each source's 40 s windows weighted as train_duplex's Data.pick
draws them, split into the real conversations (a real person's recording is what she hears) and the rest.

  .venv-train/Scripts/python.exe -I tools/diag_duplex_data.py [--data data/duplex]
      [--shares '{"seamless-own": 0.33, "synthetic": 0.45, "feeling": 0.14, "asides": 0.08}']
Writes results/diag_duplex_data.json. Words are counted as words and punctuation marks, not the model's own
word-pieces, so word counts are approximate (in the same way for every source); everything else is exact.
"""
import argparse
import json
import random
import re
import sys
import zlib
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
import train_duplex as TD  # noqa: E402  (the training layout itself)
from build_testsets import norm  # noqa: E402
from score_timing import BC_MAX  # noqa: E402

WORD = re.compile(r"[\w'-]+|[^\w\s]")
KINDS = ('mm-hm', 'short', 'turn')
LOSS = ('listen_kept', 'starts', 'words_learned', 'words_context', 'frames')


class Words:
    """Stands in for train_duplex.Grammar: the same layout, with words and punctuation marks as the word-pieces and
    the layout's markers as ids that no word has."""
    audio_in, audio_out, mixed_start, text_end, listen, eos_code = -1, -2, -3, -4, -5, -6

    def __init__(self):
        self.vocab = {}

    def tok(self, text, add_special_tokens=False):
        return {'input_ids': [self.vocab.setdefault(w, len(self.vocab)) for w in WORD.findall(text)]}

    def head(self, clock=None, tools=None):
        return []


def kind_of(u):
    return 'mm-hm' if u.get('backchannel') else ('short' if len(u['text'].split()) <= 3 else 'turn')


def conversation(meta, persons, G, seed=0, delay=TD.DELAY, plan_chunks='count'):
    """One conversation as training lays it out: {'seconds', 'utterances': [...], 'loss': Counter}. persons: the
    person's segments, as train_duplex.person_segments gives them."""
    utts = TD.timeline(meta, persons, np.random.default_rng(seed))
    n = max(1, int(round(meta['seconds'] / TD.STEP_S)) - delay)
    spans = [sp for sp in TD.spans_of(utts, delay) if sp[0] < n]
    codes = np.zeros((n + delay, 12), np.int64)
    # every LISTEN label kept here: the count is scaled by LISTEN_KEEP below, so the numbers are not random
    ids, labels, _ = TD.build(G, spans, codes, 0, n, random.Random(seed), delay=delay, listen_keep=1.0,
                              plan_chunks=plan_chunks, tools=meta.get('tools'))
    loss = Counter()
    for i, lab in zip(ids, labels):
        if i == G.listen:
            loss['listen_kept'] += TD.LISTEN_KEEP
        elif i == G.mixed_start:
            loss['starts'] += 1
        elif i == G.audio_out:
            loss['frames'] += 1
        elif i == G.text_end or i >= 0:
            loss['words_learned' if lab != -100 else 'words_context'] += 1
    rows = []
    for s, e, u in spans:
        words = u['text'].split()
        rows.append({'kind': kind_of(u), 'learned': bool(u.get('learn_words', True)),
                     'first': norm(words[0]) if words else '', 'turn_long': (e - s) * TD.STEP_S >= BC_MAX})
    return {'seconds': float(meta['seconds']), 'utterances': rows, 'loss': loss}


def summarize(convs, weights=None):
    """convs: conversation() results; weights: how much each counts (default 1 each, i.e. plain totals)."""
    weights = [1.0] * len(convs) if weights is None else weights
    rows = [(w, r) for c, w in zip(convs, weights) for r in c['utterances']]
    total = sum(w for w, _ in rows)
    learned = [(w, r) for w, r in rows if r['learned']]
    total_l = sum(w for w, _ in learned)
    share = lambda part, whole: round(part / whole, 3) if whole else None
    first = Counter()
    for w, r in learned:
        first[r['first']] += w
    loss = Counter()
    for c, w in zip(convs, weights):
        for k in LOSS:
            loss[k] += w * c['loss'][k]
    text = sum(loss[k] for k in LOSS if k != 'frames')
    return {
        'utterances': round(total, 2),
        'kinds': {k: share(sum(w for w, r in rows if r['kind'] == k), total) for k in KINDS},
        'heard_as_turn_share': share(sum(w for w, r in rows if r['turn_long']), total),
        'words_learned_share': share(total_l, total),
        'words_learned_share_by_kind': {k: share(sum(w for w, r in learned if r['kind'] == k),
                                                 sum(w for w, r in rows if r['kind'] == k)) for k in KINDS},
        'learned_open_with_yeah_share': share(first['yeah'], total_l),
        'learned_first_words': [[word, share(n, total_l)] for word, n in first.most_common(8)],
        'loss_counts': {k: round(loss[k], 2) for k in LOSS},
        'text_loss_shares': {k: share(loss[k], text) for k in LOSS if k != 'frames'},
    }


def mix(by_source, shares, window_s):
    """The training mix: a source is drawn by its share, a conversation by min(600 s, its length) (train_duplex's
    Data.pick), and a window of window_s seconds from it; returns each conversation's weight, so that the counts
    of summarize() are expected numbers per training window."""
    used = {s: v for s, v in shares.items() if v > 0 and by_source.get(s)}
    z = sum(used.values())
    out = {}
    for s, p in used.items():
        convs = by_source[s]
        pick = [min(600.0, c['seconds']) for c in convs]
        for c, w in zip(convs, pick):
            out[id(c)] = p / z * w / sum(pick) * min(1.0, window_s / max(c['seconds'], 1e-6))
    return out


def report(stats):
    pct = lambda v: '-' if v is None else f'{100 * v:.0f}%'
    lines = [f"{'source':20s} {'convs':>5s} {'hours':>6s} {'utt/min':>7s} {'mm-hm':>6s} {'short':>6s} {'turn':>6s} "
             f"{'>=0.8s':>6s} {'learned':>7s} {'yeah 1st':>8s}  learned utterances open with"]
    for name, s in stats.items():
        k = s['kinds']
        top = ', '.join(f'{w or "?"} {pct(v)}' for w, v in s['learned_first_words'][:4])
        per_min = f"{s['utterances_per_min']:.1f}" if s.get('utterances_per_min') is not None else '-'
        lines.append(f"{name:20s} {s.get('conversations', '-')!s:>5s} {s.get('hours', '-')!s:>6s} {per_min:>7s} "
                     f"{pct(k['mm-hm']):>6s} {pct(k['short']):>6s} {pct(k['turn']):>6s} "
                     f"{pct(s['heard_as_turn_share']):>6s} {pct(s['words_learned_share']):>7s} "
                     f"{pct(s['learned_open_with_yeah_share']):>8s}  {top}")
    lines.append('')
    lines.append(f"{'text loss is made of':20s} {'LISTEN':>7s} {'starts':>7s} {'learned':>8s} {'context':>8s}"
                 f"  frames per text position")
    for name, s in stats.items():
        t = s['text_loss_shares']
        n_text = sum(v for k, v in s['loss_counts'].items() if k != 'frames')
        ratio = f"{s['loss_counts']['frames'] / n_text:.1f}" if n_text else '-'
        lines.append(f"{name:20s} {pct(t['listen_kept']):>7s} {pct(t['starts']):>7s} {pct(t['words_learned']):>8s} "
                     f"{pct(t['words_context']):>8s}  {ratio}")
    return '\n'.join(lines)


def main(argv=None):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=str(ROOT / 'data' / 'duplex'))
    ap.add_argument('--shares', default='{"seamless-own": 0.33, "synthetic": 0.45, "feeling": 0.14, "asides": 0.08}',
                    help='the training mix to weigh (as train_duplex --shares; run 2: seamless-train 0.33 instead)')
    ap.add_argument('--window-s', type=float, default=40.0)
    ap.add_argument('--delay', type=int, default=TD.DELAY)
    ap.add_argument('--plan-chunks', default='count', choices=['count', 'predicted'])
    ap.add_argument('--out', default=str(ROOT / 'results' / 'diag_duplex_data.json'))
    args = ap.parse_args(argv)
    G = Words()
    by_source, missing = {}, 0
    for p in sorted(Path(args.data).rglob('meta.json')):
        meta = json.loads(p.read_text(encoding='utf8'))
        try:
            persons = TD.person_segments(meta)
        except OSError:
            # a real conversation whose Seamless files are not here: only the give-way trims need them
            persons, missing = [], missing + 1
        c = conversation(meta, persons, G, seed=zlib.crc32(p.parent.name.encode()), delay=args.delay,
                         plan_chunks=args.plan_chunks)
        by_source.setdefault(meta.get('source', '?'), []).append(c)
    if not by_source:
        raise SystemExit(f'no meta.json under {args.data}')
    stats = {}
    for name, convs in sorted(by_source.items()):
        minutes = sum(c['seconds'] for c in convs) / 60
        s = {'conversations': len(convs), 'hours': round(minutes / 60, 2), **summarize(convs)}
        s['utterances_per_min'] = round(s['utterances'] / minutes, 2) if minutes else None
        stats[name] = s
    shares = json.loads(args.shares)
    w = mix(by_source, shares, args.window_s)
    groups = {'mix: real voices': [s for s in by_source if s in TD.REAL_SOURCES],
              'mix: other sources': [s for s in by_source if s not in TD.REAL_SOURCES]}
    mixed = {}
    for g, names in groups.items():
        convs = [c for s in names for c in by_source[s] if id(c) in w]
        if convs:
            mixed[g] = summarize(convs, [w[id(c)] for c in convs])
    print(report({**stats, **mixed}), flush=True)
    if missing:
        print(f'({missing} real conversations without their Seamless files: give-way trims skipped)', flush=True)
    out = {'data': Path(args.data).name, 'shares': shares, 'window_s': args.window_s, 'delay_steps': args.delay,
           'plan_chunks': args.plan_chunks, 'listen_keep': TD.LISTEN_KEEP, 'turn_from_s': BC_MAX,
           'words_counted_as': 'words and punctuation marks', 'sources': stats,
           'mix_per_window': mixed}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding='utf8')
    print('wrote', Path(args.out).name, flush=True)
    return out


if __name__ == '__main__':
    main()
