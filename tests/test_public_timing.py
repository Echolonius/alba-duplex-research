"""The public timing set's plumbing on a made-up conversation (no download, no model, processor only): the builder's
--public-only mode never reads sessions/ or writes data/tests/, the frozen build stops without the held-out calls,
and the scorer keeps the public set's results apart from the frozen set's.

  python -I -m unittest discover -s tests -v
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools'))
import build_testsets  # noqa: E402
import score_timing  # noqa: E402

A, B = 'V09_S0001_I00000001_P0001', 'V09_S0001_I00000001_P0002'


def fake_seamless(root):
    """One 105 s conversation: every 10 s the caller (A) talks, pauses 0.5 s and goes on (a hold), then the
    partner (B) takes over 0.3 s after A stops (a shift)."""
    src = root / 'data' / 'seamless' / 'test'
    (src / 'meta').mkdir(parents=True)
    (src / 'audio').mkdir()
    turns = {A: [], B: []}
    for k in range(10):
        t = 10.0 * k
        turns[A] += [(t, t + 3.0), (t + 3.5, t + 5.0)]
        turns[B] += [(t + 5.3, t + 8.0)]
    for fid, segs in turns.items():
        with open(src / 'meta' / f'{fid}.vad.jsonl', 'w', encoding='utf8') as f:
            f.writelines(json.dumps({'start': s, 'end': e}) + '\n' for s, e in segs)
        with open(src / 'meta' / f'{fid}.transcript.jsonl', 'w', encoding='utf8') as f:
            f.writelines(json.dumps({'words': [{'word': 'tomorrow', 'start': s, 'end': e}]}) + '\n' for s, e in segs)
        sf.write(src / 'audio' / f'{fid}.flac', np.zeros(105 * 16000, np.float32), 16000)
    (src / 'manifest.json').write_text(json.dumps({'files': {A: {'seconds': 105.0}, B: {'seconds': 105.0}}}))
    pub = root / 'data' / 'tests-public' / 'timing'
    pub.mkdir(parents=True)
    (pub / 'sources.json').write_text(json.dumps({'label': 'naturalistic', 'split': 'test', 'pairs': [[A, B]]}))


class PublicBuild(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        fake_seamless(self.root)
        self.patch = mock.patch.multiple(build_testsets, ROOT=self.root, T=self.root / 'data' / 'tests',
                                         PUBLIC=self.root / 'data' / 'tests-public')
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    def test_public_only_builds_the_seamless_tracks_alone(self):
        build_testsets.build_timing(public_only=True)
        out = self.root / 'data' / 'tests-public' / 'timing'
        items = json.loads((out / 'items.json').read_text(encoding='utf8'))['items']
        self.assertEqual([it['id'] for it in items], [f'seamless-{A}'])
        self.assertEqual((len(items[0]['shifts']), len(items[0]['holds'])), (10, 10))
        self.assertNotIn(b'\r', (out / 'items.json').read_bytes())      # same bytes on every system
        self.assertTrue((out / 'audio' / f'seamless-{A}-caller.flac').exists())
        self.assertTrue((out / 'audio' / f'seamless-{A}-partner.flac').exists())
        (self.root / 'tools').mkdir()
        for name in ('score_timing.py', 'score_fresh.py', 'judge_local.py', 'build_testsets.py'):
            (self.root / 'tools' / name).write_text('# stand-in\n')
        build_testsets.fingerprints(build_testsets.PUBLIC)
        hashes = (self.root / 'data' / 'tests-public' / 'HASHES.txt').read_bytes()
        self.assertIn(b'  data/tests-public/timing/items.json\n', hashes)
        self.assertIn(b'  tools/score_timing.py\n', hashes)
        self.assertNotIn(b'score_fresh.py', hashes)      # only the scorer this set uses
        self.assertNotIn(b'judge_local.py', hashes)
        self.assertNotIn(b'\r', hashes)
        self.assertFalse((self.root / 'data' / 'tests').exists())      # the frozen folder is never written
        self.assertFalse((self.root / 'sessions').exists())

    def test_frozen_build_stops_without_the_held_out_calls(self):
        with self.assertRaises(SystemExit):
            build_testsets.build_timing()
        self.assertFalse((self.root / 'data' / 'tests').exists())      # stopped before writing anything


class PublicScores(unittest.TestCase):
    def test_public_scores_are_kept_apart(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_seamless(root)
            with mock.patch.multiple(build_testsets, ROOT=root, T=root / 'data' / 'tests',
                                     PUBLIC=root / 'data' / 'tests-public'):
                build_testsets.build_timing(public_only=True)
            # a "system" that answers every hand-off 0.2 s after the caller stops
            voice = np.zeros(105 * 16000, np.float32)
            for k in range(10):
                voice[(10000 * k + 5200) * 16:(10000 * k + 7500) * 16] = 0.1      # milliseconds -> samples
            out = root / 'results' / 'tests-public' / 'sys' / 'timing'
            out.mkdir(parents=True)
            sf.write(out / f'seamless-{A}-voice.flac', voice, 16000)
            with mock.patch.multiple(score_timing, ROOT=root, TESTS=root / 'data' / 'tests' / 'timing',
                                     PUBLIC=root / 'data' / 'tests-public' / 'timing'), \
                    mock.patch.object(sys, 'argv', ['score_timing.py', '--public', 'sys', 'human']):
                score_timing.main()
            res = root / 'results' / 'tests-public'
            sys_scores = json.loads((res / 'sys' / 'timing_scores.json').read_text(encoding='utf8'))
            human = json.loads((res / 'human' / 'timing_scores.json').read_text(encoding='utf8'))
            self.assertEqual(sys_scores['all']['answered_share'], 1.0)
            self.assertEqual(sys_scores['all']['reply_start_median_s'], 0.2)
            self.assertEqual(human['all']['reply_start_median_s'], 0.3)
            self.assertEqual(human['all']['shifts'], 10)
            self.assertFalse((root / 'results' / 'tests').exists())   # the frozen results are never written


if __name__ == '__main__':
    unittest.main()
