"""Does the split design's text side stay in character? Plays the benchmark conversations
(as text) through app/brain.py's real reply logic, with speech switched off, and counts the
same things as probe_loops.py plus reply length, how often a reply ends in a question, and
help-desk phrases.

  python -I probe_text_brain.py <label> [--trials 2] [--scripts L,A,B,C] [--temp 0.8]
                                [--persona app|old] [--no-guard]
Needs the text server on 127.0.0.1:9941 (app.server.start_text).
"""
import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
from app.brain import HELPDESK, MIND_PROMPT, TEXT_PERSONA, SplitBrain   # noqa: E402
from app.sounds import PARTICLES, QUICK                       # noqa: E402
from app.config import Config, apply_settings                 # noqa: E402
from app.main import PERSONA                                  # noqa: E402
from benchmark import SCRIPTS                                 # noqa: E402
from probe_loops import ASSIST, STOCK, sentences              # noqa: E402

PERSONAS = {'app': TEXT_PERSONA,
            'old': PERSONA + 'Your replies are spoken aloud, so never use emoji, lists or markdown.'}
TRY = Path(__file__).resolve().parent / 'persona_try.txt'     # a candidate wording to compare
if TRY.exists():
    PERSONAS['try'] = TRY.read_text(encoding='utf8').strip()


def quick_start(text):
    """Does the reply open with something said from a ready-made clip ("Oh," / "Totally!")?"""
    m = re.match(r"\s*([A-Za-z']+(?: [A-Za-z']+)?),", text)
    if m and m.group(1).lower() in PARTICLES:
        return True
    m = re.match(r"\s*([A-Za-z']+),", text)
    if m and m.group(1).lower() in PARTICLES:
        return True
    first = re.match(r"\s*(.+?[.!?])(\s|$)", text)
    return bool(first and first.group(1) in QUICK)


async def converse(cfg, persona, lines, guard, read_along='none'):
    brain = SplitBrain(cfg, persona=persona)
    brain.loop = asyncio.get_running_loop()
    def silent(text, stop, req, started=None):                                     # no sound
        if started is not None:
            brain.loop.call_soon_threadsafe(started.set)
        return 0.3 * len(text.split()) + 0.3
    brain._stream_speech = silent
    dropped = []
    brain.note = lambda kind, **kw: dropped.append(kw) if kind == 'dropped_sentence' else None
    if not guard:
        brain._judge = lambda s: 'ok'
    replies, secs, notes = [], [], []
    converse.notes = notes
    for line in lines:
        fut = brain.loop.create_future()
        fut.set_result(line)
        brain.heard.append(fut)
        if cfg.brain.mind:
            # The slow mind's notes, as if written while they talked (the best case: all of the line).
            msgs = [{'role': 'system', 'content': MIND_PROMPT}] + brain.history[brain.hist_start:] + \
                   [{'role': 'user', 'content': line}]
            text = await brain.loop.run_in_executor(None, brain._mind_notes, msgs, line)
            brain.notes = {'said': line, 'text': ' '.join(text.split())[:400]}
            notes.append(brain.notes['text'])
        if read_along != 'none':
            # As if read while they talked: all of it ('full', the best case), or all but their
            # last three words ('partial': the last piece arrives when they stop).
            words = line.split()
            ahead = ' '.join(words[:-3]) if read_along == 'partial' and len(words) > 5 else line
            await brain.loop.run_in_executor(None, brain._read_ahead, brain._messages(ahead), ahead)
        t0 = time.perf_counter()
        await brain.respond(commit=False)
        text, first = '', None
        while True:
            ev = await brain.q.get()
            if ev['type'] == 'response.output_text.delta':
                if first is None:
                    first = time.perf_counter() - t0
                    brain.reply_started()
                text += ev['delta']
            elif ev['type'] == 'response.done':
                break
        replies.append(text.strip())
        secs.append(first if first is not None else float('nan'))
    return replies, secs, dropped


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser()
    ap.add_argument('label')
    ap.add_argument('--trials', type=int, default=2)
    ap.add_argument('--scripts', default='L')
    ap.add_argument('--persona', default='app', choices=sorted(PERSONAS))
    ap.add_argument('--temp', type=float, default=None)
    ap.add_argument('--no-guard', action='store_true', help="don't drop help-desk or repeated sentences")
    ap.add_argument('--quiet', action='store_true')
    ap.add_argument('--read-along', default='none', choices=('none', 'full', 'partial'),
                    help='let the brain read each line first, as it would while they talk')
    ap.add_argument('--set', action='append', default=[], metavar='SECTION.NAME=VALUE',
                    help='override a setting from app/config.py, e.g. brain.text_prefill=<think></think>')
    ap.add_argument('--lines', help='a JSON list of what a person said, turn by turn (e.g. from a live call), '
                                    'played as script X instead of --scripts')
    ap.add_argument('--first', type=int, default=0, help='with --lines: only the first N turns')
    args = ap.parse_args()
    cfg = Config()
    apply_settings(cfg, args.set)
    if args.temp is not None:
        cfg.brain.temperature = args.temp
    if args.lines:
        SCRIPTS['X'] = json.loads(Path(args.lines).read_text(encoding='utf8'))[:args.first or None]
        args.scripts = 'X'
    rows = []
    for name in args.scripts.split(','):
        lines = [s.replace(' | ', ' ').replace('|', '') for s in SCRIPTS[name]]
        for trial in range(args.trials):
            replies, secs, dropped = asyncio.run(converse(cfg, PERSONAS[args.persona], lines, not args.no_guard, args.read_along))
            seen, rep = set(), 0
            for t in replies:
                ss = sentences(t)
                rep += any(x in seen for x in ss)
                seen.update(ss)
            ok = [s for s in secs if s == s]
            r = {'script': name, 'n': len(replies), 'repeats': rep,
                 'stock': sum(bool(STOCK.search(t)) for t in replies),
                 'assistant': sum(bool(ASSIST.search(t) or HELPDESK.search(t)) for t in replies),
                 'empty': sum(not t for t in replies),
                 'words': sum(len(t.split()) for t in replies) / len(replies),
                 'questions': sum(t.rstrip().endswith('?') for t in replies) / len(replies),
                 'quick': sum(quick_start(t) for t in replies) / len(replies),
                 'first': sorted(ok)[len(ok) // 2] if ok else float('nan'), 'dropped': len(dropped)}
            rows.append(r)
            print(f"[{args.label} {name}#{trial}] repeats {r['repeats']}/{r['n']} | stock {r['stock']} | assistant-mode "
                  f"{r['assistant']} | empty {r['empty']} | words {r['words']:.1f} | ends in a question {r['questions']:.0%} "
                  f"| opens with a quick clip {r['quick']:.0%} | first sentence after {r['first']:.2f}s "
                  f"| sentences dropped {r['dropped']}")
            if not args.quiet:
                notes = getattr(converse, 'notes', [])
                for k, (line, t) in enumerate(zip(lines, replies)):
                    print(f'   > {line}')
                    if k < len(notes):
                        print(f'     [notes: {notes[k]}]')
                    print(f'     {t}')
                for d in dropped:
                    print(f"   (dropped, {d['reason']}: {d['sentence']})")
    n = sum(r['n'] for r in rows)
    print(f"== {args.label}: per 10 replies: repeats {10 * sum(r['repeats'] for r in rows) / n:.1f}, "
          f"stock {10 * sum(r['stock'] for r in rows) / n:.1f}, assistant-mode {10 * sum(r['assistant'] for r in rows) / n:.1f}, "
          f"empty {10 * sum(r['empty'] for r in rows) / n:.1f} | words {sum(r['words'] * r['n'] for r in rows) / n:.1f} "
          f"| questions {sum(r['questions'] * r['n'] for r in rows) / n:.0%} "
          f"| opens with a quick clip {sum(r['quick'] * r['n'] for r in rows) / n:.0%}")


if __name__ == '__main__':
    main()
