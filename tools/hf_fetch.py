"""Downloads files of a Hugging Face repository at a pinned revision, and checks every one.

Large (LFS) files are checked against the SHA-256 the site publishes, small ones against their git
blob id. A partial download is resumed. Files whose check fails are kept as <name>.bad, never used.

  python -I tools/hf_fetch.py <repo> <revision> <out_dir> [--only PATTERN] [--skip PATTERN]
e.g. python -I tools/hf_fetch.py LiquidAI/LFM2.5-Audio-1.5B c362a06... models/lfm25-audio
"""
import argparse
import fnmatch
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

UA = {'User-Agent': 'localvoice-fetch'}


def git_blob_sha1(path):
    data = path.read_bytes()
    return hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def fetch(url, dest, size=None):
    dest.parent.mkdir(parents=True, exist_ok=True)
    have = dest.stat().st_size if dest.exists() else 0
    if size is not None and have == size:
        return
    if size is not None and have > size:
        dest.unlink()        # longer than the real file: start again (it is a partial download of ours)
        have = 0
    req = urllib.request.Request(url, headers={**UA, **({'Range': f'bytes={have}-'} if have else {})})
    with urllib.request.urlopen(req, timeout=60) as r, open(dest, 'ab' if have and r.status == 206 else 'wb') as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser()
    ap.add_argument('repo')
    ap.add_argument('revision')
    ap.add_argument('out')
    ap.add_argument('--only', default='*')
    ap.add_argument('--skip', default='')
    args = ap.parse_args()
    out = Path(args.out)
    api = f'https://huggingface.co/api/models/{args.repo}/revision/{args.revision}?blobs=true'
    with urllib.request.urlopen(urllib.request.Request(api, headers=UA), timeout=60) as r:
        info = json.loads(r.read())
    files = [s for s in info['siblings'] if fnmatch.fnmatch(s['rfilename'], args.only)
             and not (args.skip and fnmatch.fnmatch(s['rfilename'], args.skip)) and not s['rfilename'].startswith('.')]
    bad = 0
    for s in files:
        name = s['rfilename']
        dest = out / name
        lfs = s.get('lfs') or {}
        fetch(f'https://huggingface.co/{args.repo}/resolve/{args.revision}/{name}', dest, s.get('size'))
        ok = sha256(dest) == lfs['sha256'] if lfs.get('sha256') else git_blob_sha1(dest) == s.get('blobId')
        print(f"{'OK  ' if ok else 'BAD '} {name} ({s.get('size') or dest.stat().st_size:,} bytes)", flush=True)
        if not ok:
            bad += 1
            dest.replace(dest.with_name(dest.name + '.bad'))
    manifest = out / 'MANIFEST.json'
    manifest.write_text(json.dumps({'repo': args.repo, 'revision': args.revision,
                                    'files': {s['rfilename']: (s.get('lfs') or {}).get('sha256') or s.get('blobId')
                                              for s in files}}, indent=1), encoding='utf8')
    print(f'{len(files) - bad} of {len(files)} files verified' + (f', {bad} kept as .bad' if bad else ''))
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
