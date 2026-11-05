"""Scores the spoken test battery (check 4) from a run of tools/run_items.py (or any system that writes the
same results/tests/<tag>/battery.jsonl: one row per item with 'final_reply' and 'replies').

Rules per item (data/tests/battery/items.json): 'any' = the last reply contains one of these; 'none' = it
contains none of these; 'silent' = it says nothing at all. On top, every reply of every item is checked for
help-desk talk (zero allowed).

  python -I tools/score_battery.py <tag> [<tag> ...]
Writes results/tests/<tag>/battery_scores.json.
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TESTS = ROOT / 'data' / 'tests' / 'battery'
HELPDESK = re.compile(r"how (can|may) i (help|assist)|assist you|anything else (i can|you need)|let me know if|"
                      r"i'?m here to help|happy to help|what can i do for you|feel free to (ask|reach)|"
                      r"as an ai(?: language model)?,|i'?m just an ai|i'?m (just )?a (digital|virtual) assistant", re.I)


def check(it, reply):
    r = reply.lower()
    if it.get('silent'):
        return not r.strip()
    ok = True
    if it.get('any'):
        ok &= any(a.lower() in r for a in it['any'])
    if it.get('none'):
        ok &= not any(a.lower() in r for a in it['none'])
    return bool(ok)


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    items = {it['id']: it for it in json.loads((TESTS / 'items.json').read_text(encoding='utf8'))['items']}
    for tag in sys.argv[1:]:
        p = ROOT / 'results' / 'tests' / tag / 'battery.jsonl'
        rows = {json.loads(l)['id']: json.loads(l) for l in open(p, encoding='utf8') if l.strip()}
        by_cat = defaultdict(list)
        per, helpdesk = {}, []
        for iid, it in items.items():
            if iid not in rows:
                continue
            row = rows[iid]
            ok = check(it, row['final_reply'])
            by_cat[it['cat']].append(ok)
            per[iid] = {'pass': ok, 'reply': row['final_reply'], 'heard': row.get('heard', [])[-1:] }
            helpdesk += [(iid, r) for r in row.get('replies', []) if HELPDESK.search(r)]
        n = sum(len(v) for v in by_cat.values())
        out = {'items_scored': n, 'missing': sorted(set(items) - set(rows)),
               'pass_share': round(sum(sum(v) for v in by_cat.values()) / n, 3) if n else None,
               'by_category': {c: f'{sum(v)}/{len(v)}' for c, v in sorted(by_cat.items())},
               'helpdesk_lines': len(helpdesk), 'helpdesk_examples': helpdesk[:5],
               'standing_tests': {k: per[k]['pass'] for k in per if k.startswith('std_')},
               'per_item': per}
        dest = ROOT / 'results' / 'tests' / tag
        (dest / 'battery_scores.json').write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding='utf8')
        print(f"== {tag}: {out['pass_share']} of {n} pass | help-desk lines {len(helpdesk)} | {out['by_category']}")
        print('   standing tests:', out['standing_tests'])


if __name__ == '__main__':
    main()
