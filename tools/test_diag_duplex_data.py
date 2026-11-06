"""Checks tools/diag_duplex_data.py on three small made-up conversations (no audio, no model, no real data): a real
conversation laid out as runs 1-2 learned it ('seamless-train'), one with her own words ('seamless-own') and a
synthetic call. Every expected number below was worked out by hand from tools/train_duplex.py's layout.

  .venv-train/Scripts/python.exe -I tools/test_diag_duplex_data.py
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
import diag_duplex_data as D  # noqa: E402


def utt(start, end, text, bc=False):
    return {'start': start, 'end': end, 'text': text, 'backchannel': bc, 'cut_short': False}


REAL = {'id': 'sm-test', 'source': 'seamless-train', 'person_file': 'no-such-file', 'seconds': 60.0, 'utterances': [
    utt(2.0, 2.4, 'Yeah.', bc=True),                                     # an "mm-hm": learned
    utt(5.0, 9.0, 'So I went to the market with my sister and we bought way too much bread.'),   # their life: context
    utt(12.0, 12.9, 'No way.'),                                           # a short reaction: learned
    utt(20.0, 20.3, 'Yeah.', bc=True),                                    # opens the next turn (0.3 s before it) ...
    utt(20.6, 24.0, 'What did they say about the long drive home after that?'),   # ... a question: learned
    utt(30.0, 33.0, 'That sounds like a really long and tiring trip for everyone involved there.')]}   # > 10 words
OWN = {'id': 'so-test', 'source': 'seamless-own', 'person_file': 'no-such-file', 'seconds': 30.0, 'utterances': [
    utt(1.0, 1.3, 'Yeah.', bc=True),
    utt(4.0, 8.0, 'Aw, that sounds rough, but I bet the bread was worth it in the end.')]}
SYN = {'id': 'syn-test', 'source': 'synthetic', 'seconds': 20.0, 'person_turns': [[0.5, 3.0], [10.0, 12.0]],
       'utterances': [utt(3.3, 6.3, 'Oh, that sounds like a lovely way to spend a quiet Sunday.'),
                      utt(12.3, 13.0, 'Sure thing!')]}


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    checks = {}

    def check(name, ok, detail=''):
        checks[name] = bool(ok)
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)

    G = D.Words()
    real = D.conversation(REAL, [], G)
    s = D.summarize([real])
    check('real: the "yeah" opening a turn joins it, as training joins them', s['utterances'] == 5, s['utterances'])
    check('real: kinds', s['kinds'] == {'mm-hm': 0.2, 'short': 0.2, 'turn': 0.6}, s['kinds'])
    check('real: words learned only for the mm-hm, the reaction and the question', s['words_learned_share'] == 0.6 and
          s['words_learned_share_by_kind'] == {'mm-hm': 1.0, 'short': 1.0, 'turn': 0.333},
          (s['words_learned_share'], s['words_learned_share_by_kind']))
    check('real: two of the three learned utterances open with "yeah"', s['learned_open_with_yeah_share'] == 0.667 and
          s['learned_first_words'][0] == ['yeah', 0.667], s['learned_first_words'])
    check('real: four utterances last 0.8 s or more', s['heard_as_turn_share'] == 0.8, s['heard_as_turn_share'])
    want = {'listen_kept': 117.8, 'starts': 5, 'words_learned': 22, 'words_context': 33, 'frames': 158}
    check('real: what the losses are made of (worked out by hand)', s['loss_counts'] == want, s['loss_counts'])

    own = D.summarize([D.conversation(OWN, [], G)])
    check('own words: every word learned, even with "I" in it', own['words_learned_share'] == 1.0 and
          own['loss_counts']['words_context'] == 0, own['loss_counts'])
    syn = D.summarize([D.conversation(SYN, [(0.5, 3.0, False), (10.0, 12.0, False)], G)])
    check('synthetic: no mm-hm, all learned, none open with "yeah"', syn['kinds']['mm-hm'] == 0 and
          syn['words_learned_share'] == 1.0 and syn['learned_open_with_yeah_share'] == 0, syn['learned_first_words'])

    # the mix: a source by its share, a conversation by its length (up to 600 s), a 40 s window out of it
    w = D.mix({'seamless-train': [real], 'synthetic': [D.conversation(SYN, [], G)]},
              {'seamless-train': 0.5, 'synthetic': 0.5, 'feeling': 0.2}, 40.0)
    check('mix: a source with no data is left out, the rest weighed per window',
          sorted(round(v, 4) for v in w.values()) == [0.3333, 0.5], sorted(w.values()))
    check('mix: expected real utterances per window', D.summarize([real], [w[id(real)]])['utterances'] == 1.67)

    # end to end: the files as tools/make_duplex_data.py and tools/make_seamless_own.py write them
    tmp = ROOT / 'logs' / 'diag-data-test-tmp'
    shutil.rmtree(tmp, ignore_errors=True)
    try:
        for meta in (REAL, OWN, SYN):
            (tmp / 'data' / meta['source'] / meta['id']).mkdir(parents=True)
            (tmp / 'data' / meta['source'] / meta['id'] / 'meta.json').write_text(json.dumps(meta), encoding='utf8')
        out = D.main(['--data', str(tmp / 'data'), '--out', str(tmp / 'out.json'),
                      '--shares', '{"seamless-train": 0.33, "synthetic": 0.45, "feeling": 0.14}'])
        saved = json.loads((tmp / 'out.json').read_text(encoding='utf8'))
        check('main: every source reported', sorted(saved['sources']) == ['seamless-own', 'seamless-train', 'synthetic'],
              sorted(saved['sources']))
        check('main: the run-2 mix splits real voices from the rest',
              saved['mix_per_window']['mix: real voices']['learned_open_with_yeah_share'] == 0.667 and
              saved['mix_per_window']['mix: other sources']['kinds']['mm-hm'] == 0, list(saved['mix_per_window']))
        check('main: returns what it saved', out['sources']['synthetic']['utterances'] == 2)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    ok = all(checks.values())
    print('all pass' if ok else 'SOME FAILED', f'({sum(checks.values())} of {len(checks)})', flush=True)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
