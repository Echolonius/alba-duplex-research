"""What the voice said in benchmark runs, and how its turns went, from the event logs.

  python -I bench_content.py <bench_dir> <tag> [<tag> ...] [--show]

Per run: replies, assistant-mode and stock lines, replies repeating an earlier sentence,
how long after the person stopped the voice was heard (median), how often it was talked over
and gave way, and (split design) how often the reply had been written ahead and how many of the
person's words never reached the text model (word error of what it heard against the script).
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark import SCRIPTS                         # noqa: E402
from intelligibility import wer                       # noqa: E402
from probe_loops import ASSIST, STOCK, sentences      # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.brain import HELPDESK                        # noqa: E402


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    show = '--show' in sys.argv
    bench, tags = Path(args[0]), args[1:]
    for tag in tags:
        files = sorted(bench.glob(f'{tag}-*-events.jsonl'))
        ev = [json.loads(line) for f in files for line in open(f, encoding='utf8')]
        texts = [e['text'] for e in ev if e['type'] == 'reply_text']
        seen, rep = set(), 0
        for t in texts:
            ss = sentences(t)
            rep += any(x in seen for x in ss)
            seen.update(ss)
        first = [e['delay_from_user_end'] for e in ev if e['type'] == 'first_audio' and e.get('delay_from_user_end') is not None]
        n = max(1, len(texts))
        err = words = 0
        for f in files:
            script = f.name[len(tag) + 1:].split('-')[0]
            fe = [json.loads(line) for line in open(f, encoding='utf8')]
            heard, last = [], None
            for e in fe:
                if e['type'] == 'heard':
                    last = e['text']
                elif e['type'] == 'first_audio' and last:
                    heard.append(last)
                    last = None
            if (heard or last) and script in SCRIPTS:
                e_, n_ = wer(' '.join(x.replace('|', ' ') for x in SCRIPTS[script]), ' '.join(heard + ([last] if last else [])))
                err, words = err + e_, words + n_
        missed = f' | words missed {err / words:.1%}' if words else ''
        print(f"{tag}: {len(files)} conversations | replies {len(texts)} | assistant-mode "
              f"{sum(bool(ASSIST.search(t) or HELPDESK.search(t)) for t in texts)} | stock {sum(bool(STOCK.search(t)) for t in texts)} "
              f"| repeating {rep} | words {sum(len(t.split()) for t in texts) / n:.1f} "
              f"| heard after {np.median(first) if first else float('nan'):.2f}s (median) "
              f"| gave way {sum(e['type'] == 'yield' and not e.get('immediate') for e in ev)} "
              f"| took back {sum(e['type'] == 'yield' and e.get('immediate') for e in ev)} "
              f"| written ahead {sum(e['type'] == 'prefetch_used' for e in ev)} "
              f"| sentences dropped {sum(e['type'] == 'dropped_sentence' for e in ev)}{missed}")
        if show:
            for t in texts:
                print(f'      {t}')


if __name__ == '__main__':
    main()
