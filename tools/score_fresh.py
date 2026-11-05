"""Scores the fresh standing-test variants (data/tests-fresh/<set>) with the frozen battery rule
(tools/score_battery.check: any / none / silent, plus the help-desk check), without touching the frozen scorer.

  python -I tools/score_fresh.py TAG [--set fresh1]      reads results/tests/TAG/<set>.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools'))
import score_battery  # noqa: E402


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser()
    ap.add_argument('tag')
    ap.add_argument('--set', default='fresh1')
    args = ap.parse_args()
    items = {it['id']: it for it in json.loads((ROOT / 'data' / 'tests-fresh' / args.set / 'items.json')
                                                .read_text(encoding='utf8'))['items']}
    rows = [json.loads(l) for l in open(ROOT / 'results' / 'tests' / args.tag / f'{args.set}.jsonl', encoding='utf8')]
    per = {}
    for r in rows:
        it = items.get(r['id'])
        if it is None:
            continue
        reply = r.get('final_reply') or ''
        ok = score_battery.check(it, reply)
        helpdesk = bool(score_battery.HELPDESK.search(reply)) if hasattr(score_battery, 'HELPDESK') else False
        per[r['id']] = {'pass': bool(ok) and not helpdesk, 'reply': reply}
        print(f"{'PASS' if per[r['id']]['pass'] else 'FAIL'}  {r['id']:18s} {reply[:110]!r}")
    out = {'set': args.set, 'items_scored': len(per), 'pass_share': round(sum(v['pass'] for v in per.values()) / max(1, len(per)), 3),
           'per_item': per}
    (ROOT / 'results' / 'tests' / args.tag / f'{args.set}_scores.json').write_text(json.dumps(out, indent=1, ensure_ascii=False),
                                                                             encoding='utf8')
    print(f"{args.tag} {args.set}: {out['pass_share']:.0%} of {out['items_scored']}")


if __name__ == '__main__':
    main()
