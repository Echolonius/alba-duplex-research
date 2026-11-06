"""Real two-person conversations from Meta's Seamless Interaction dataset (CC BY-NC 4.0: free for personal,
non-commercial use; attribution: Meta FAIR, "Seamless Interaction", 2025). Each person has a separate,
speaker-bleed-denoised track, so turn-taking, overlaps and "mm-hm"s are real.

Only the audio and the small transcript/VAD files are fetched, one file at a time, from Meta's public file
host (the same URLs as Meta's own download script; no account). The audio is stored as 16 kHz mono FLAC.
Meta publishes no per-file checksums for these, so the SHA-256 of every original file is recorded in
manifest.json (our pinned copy).

  python -I tools/fetch_seamless.py --label naturalistic --split test --pairs 40 --out data/seamless/test
Resumable: finished files are skipped.

The public timing set (data/tests-public/timing) needs only its own 12 conversations (24 files, about 0.9 GB);
--pairs-from fetches exactly those and checks every file against the SHA-256 recorded there (no file list needed):

  python -I tools/fetch_seamless.py --pairs-from data/tests-public/timing/sources.json
"""
import argparse
import csv
import hashlib
import io
import json
import random
import re
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

ROOT = Path(__file__).resolve().parent.parent
BASE = 'https://dl.fbaipublicfiles.com/seamless_interaction'
UA = {'User-Agent': 'localvoice-fetch'}
FILE_ID = re.compile(r'V(\d+)_S(\d+)_I(\d+)_P(\w+)')


def get(url, tries=3):
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
                data = r.read()
                n = int(r.headers.get('Content-Length') or len(data))
                if len(data) != n:
                    raise IOError(f'short read {len(data)} of {n}')
                return data
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if k == tries - 1:
                raise
        except Exception:
            if k == tries - 1:
                raise
        time.sleep(2 * (k + 1))


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser()
    ap.add_argument('--label', default='naturalistic')
    ap.add_argument('--split', default='test')
    ap.add_argument('--pairs', type=int, default=40)
    ap.add_argument('--out', default=str(ROOT / 'data' / 'seamless' / 'test'))
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--pairs-from', default='', help='fetch exactly the pairs listed in this file and check every '
                    'file against the SHA-256 recorded there (e.g. data/tests-public/timing/sources.json)')
    args = ap.parse_args()
    out = Path(args.out)
    (out / 'audio').mkdir(parents=True, exist_ok=True)
    (out / 'meta').mkdir(parents=True, exist_ok=True)
    want = {}                                  # file id -> the SHA-256 each original file must have
    if args.pairs_from:
        spec = json.loads(Path(args.pairs_from).read_text(encoding='utf8'))
        args.label, args.split, pairs, want = spec['label'], spec['split'], spec['pairs'], spec.get('files', {})
    else:
        rows = [r for r in csv.DictReader(open(ROOT / 'data' / 'seamless' / 'filelist.csv', encoding='utf8'))
                if r['label'] == args.label and r['split'] == args.split]
        groups = defaultdict(list)
        for r in rows:
            m = FILE_ID.match(r['file_id'])
            if m:
                groups[m.groups()[:3]].append(r['file_id'])
        pairs = sorted(sorted(v) for v in groups.values() if len(v) == 2)
        random.Random(args.seed).shuffle(pairs)
        pairs = pairs[:args.pairs]

    def check(fid, key, digest):
        if fid in want and want[fid].get(key) != digest:
            sys.exit(f'{fid}: {key} {digest} differs from the recorded {want[fid].get(key)}; '
                     'the source file changed, so the set cannot be rebuilt as it was')
    man_path = out / 'manifest.json'
    manifest = json.loads(man_path.read_text(encoding='utf8')) if man_path.exists() else {'source': BASE, 'license': 'CC BY-NC 4.0', 'files': {}}
    done_pairs = []
    t0 = time.time()
    for k, pair in enumerate(pairs):
        ok = True
        for fid in pair:
            flac = out / 'audio' / f'{fid}.flac'
            if fid not in manifest['files'] or not flac.exists():
                url = f'{BASE}/{args.label}/{args.split}/audio/{fid}.wav'
                data = get(url)
                check(fid, 'sha256', None if data is None else hashlib.sha256(data).hexdigest())
                if data is None:
                    ok = False
                    break
                x, sr = sf.read(io.BytesIO(data), dtype='float32')
                if len(x) < sr * 5:                       # empty or near-empty upstream file: skip the pair
                    ok = False
                    break
                if x.ndim > 1:
                    x = x.mean(axis=1)
                g = np.gcd(16000, sr)
                y = signal.resample_poly(x, 16000 // g, sr // g).astype(np.float32)
                sf.write(flac, np.clip(y, -1, 1), 16000)
                manifest['files'][fid] = {'url': url, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(),
                                          'rate': sr, 'seconds': round(len(x) / sr, 2)}
            for kind in ('transcript', 'vad'):
                p = out / 'meta' / f'{fid}.{kind}.jsonl'
                if not p.exists():
                    data = get(f'{BASE}/{args.label}/{args.split}/metadata/{kind}/{fid}.jsonl')
                    check(fid, f'{kind}_sha256', None if data is None else hashlib.sha256(data).hexdigest())
                    if data is not None:
                        p.write_bytes(data)
                        manifest['files'][fid][f'{kind}_sha256'] = hashlib.sha256(data).hexdigest()
            for key in ('sha256', 'transcript_sha256', 'vad_sha256'):     # files fetched earlier
                if key in manifest['files'][fid]:
                    check(fid, key, manifest['files'][fid][key])
        if ok:
            done_pairs.append(pair)
        man_path.write_text(json.dumps(manifest, indent=1), encoding='utf8')
        secs = sum(manifest['files'][f]['seconds'] for p in done_pairs for f in p) / 2
        print(f'{k + 1}/{len(pairs)} pairs, {len(done_pairs)} complete, {secs / 3600:.2f} h of conversation, '
              f'{time.time() - t0:.0f}s', flush=True)
    if not args.pairs_from:              # a fixed list leaves the folder's own pair list as it was
        (out / 'pairs.json').write_text(json.dumps(done_pairs, indent=1), encoding='utf8')


if __name__ == '__main__':
    main()
