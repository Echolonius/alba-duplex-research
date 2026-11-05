"""One table of every build's held-out scores against the goal's targets (for PROGRESS.md and the final report).

  python -I tools/report_scores.py human split audio m2 ...      (tags under results/tests/)
Prints a markdown table; writes results/tests/report.md.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
T = ROOT / 'results' / 'tests'

ROWS = [
    ('Reply starts after you stop (median)', 'timing_scores.json', ('all', 'reply_start_median_s'), '<= 0.3 s', lambda v: v <= 0.3, 's'),
    ('Moments answered before you go on', 'timing_scores.json', ('all', 'answered_share'), '', None, '%'),
    ('Cuts into your thinking pauses', 'timing_scores.json', ('all', 'cut_in_share'), '< 10%', lambda v: v < 0.10, '%'),
    ('Stops when you talk over it (median)', 'timing_scores.json', ('all', 'stop_median_s'), '<= 0.4 s', lambda v: v <= 0.4, 's'),
    ('Keeps going through your "mm-hm"', 'timing_scores.json', ('all', 'talk_through_share'), '', None, '%'),
    ('"Mm-hm"s per minute', 'timing_scores.json', ('all', 'mmhm_per_min'), 'people 6.8', None, ''),
    ('Battery items passed (102)', 'battery_scores.json', ('pass_share',), '', None, '%'),
    ('Help-desk lines', 'battery_scores.json', ('helpdesk_lines',), '0', lambda v: v == 0, ''),
    ('Standing tests passed', 'battery_scores.json', ('standing_tests',), 'all 6', lambda v: all(v.values()), 'dict'),
    ('Feeling: words fit the mood', 'feeling_scores.json', ('words_fit_share',), '', None, '%'),
    ('Feeling: voice fits the mood', 'feeling_scores.json', ('voice_fit_share',), '', None, '%'),
    ('Safety items handled safely (36)', 'redteam_scores.json', ('safe_share',), '100%', lambda v: v >= 0.999, '%'),
]


def get(tag, fname, path):
    p = T / tag / fname
    if not p.exists():
        return None
    v = json.loads(p.read_text(encoding='utf8'))
    for k in path:
        if not isinstance(v, dict) or k not in v:
            return None
        v = v[k]
    return v


def fmt(v, unit):
    if v is None:
        return '-'
    if unit == '%':
        return f'{100 * v:.0f}%'
    if unit == 's':
        return f'{v:.2f} s'
    if unit == 'dict':
        return f'{sum(v.values())}/{len(v)}'
    return f'{v:.1f}' if isinstance(v, float) else str(v)


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    tags = sys.argv[1:]
    lines = ['| | ' + ' | '.join(tags) + ' | target |', '|---|' + '---|' * (len(tags) + 1)]
    for name, fname, path, target, ok, unit in ROWS:
        cells = []
        for tag in tags:
            v = get(tag, fname, path)
            mark = ''
            if v is not None and ok is not None:
                try:
                    mark = ' ✓' if ok(v) else ''
                except Exception:
                    mark = ''
            cells.append(fmt(v, unit) + mark)
        lines.append(f'| {name} | ' + ' | '.join(cells) + f' | {target} |')
    out = '\n'.join(lines)
    print(out)
    (T / 'report.md').write_text(out + '\n', encoding='utf8')


if __name__ == '__main__':
    main()
