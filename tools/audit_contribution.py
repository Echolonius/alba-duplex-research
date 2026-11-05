"""Audit of a contribution (pull request) against this repository's safety rules (SECURITY.md, CONTRIBUTING.md).

It reads the changes between a base commit and the checked-out head, and reports:
  errors   (the check fails): binaries, model weights, pickles, archives or executables; files over 1 MB; hidden
           bidirectional or zero-width characters in added text (a known way to make code read differently than
           it runs); frozen test files that no longer match data/tests/HASHES.txt; frozen tests or scorers changed
           without an entry in docs/lab-notebook.md
  warnings (for the reviewer): risky code in added lines (eval/exec, pickle or unsafe torch.load, shell=True,
           new network access, decoding then running hidden data, trust_remote_code), very long lines, changes to
           .github/, and text that looks aimed at AI tools (prompt injection)
It never runs the contributed code. In CI it runs from the main branch, so a pull request cannot change its own
auditor.

  python tools/audit_contribution.py --base <base commit> [--repo .]
"""
import argparse
import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path

BLOCKED_EXT = {'.pt', '.pth', '.ckpt', '.bin', '.safetensors', '.gguf', '.onnx', '.pkl', '.pickle', '.joblib',
               '.npy', '.npz', '.h5', '.zip', '.tar', '.gz', '.tgz', '.bz2', '.xz', '.7z', '.rar', '.exe', '.dll',
               '.so', '.dylib', '.whl', '.jar', '.msi', '.bat', '.cmd', '.ps1', '.scr', '.wav', '.flac', '.mp3'}
ALLOWED_BINARY = {'.png', '.svg', '.jpg', '.jpeg', '.gif', '.webp'}      # images for docs, under the size limit
MAX_BYTES = 1_000_000
CODE_EXT = {'.py', '.sh', '.js', '.mjs', '.ts', '.html', '.toml', '.yml', '.yaml', '.cfg', '.ini'}
FROZEN = ('data/tests/', 'tools/score_', 'tools/judge_local.py', 'tools/build_testsets.py', 'tools/build_tool_tests.py')
HIDDEN = re.compile('[‪-‮⁦-⁩​-‏⁠﻿]')
RISKY = [
    (r'\beval\s*\(|\bexec\s*\(', 'eval/exec'),
    (r'\bcompile\s*\(.*exec', 'compile then exec'),
    (r'__import__\s*\(', 'dynamic import'),
    (r'\bpickle\.loads?\s*\(|\bmarshal\.loads?\s*\(|\bdill\.|\bjoblib\.load', 'loading pickled data'),
    (r'torch\.load\s*\((?![^)]*weights_only\s*=\s*True)', 'torch.load without weights_only=True'),
    (r'shell\s*=\s*True|\bos\.system\s*\(|\bos\.popen\s*\(', 'shell command'),
    (r'\brequests\.|urllib\.request|http\.client|\bsocket\.|\bhttpx\.|\baiohttp\b|\bftplib\b|\bsmtplib\b|websocket',
     'network access'),
    (r'b64decode|b32decode|a85decode|codecs\.decode|zlib\.decompress|bytes\.fromhex', 'decoding hidden data'),
    (r'trust_remote_code\s*=\s*True', 'trust_remote_code'),
    (r'\bcurl\b|\bwget\b|Invoke-WebRequest|\|\s*(ba)?sh\b|chmod\s+\+x|rm\s+-rf', 'download or shell in a script'),
]
INJECTION = re.compile(r'ignore (all |any )?(previous|prior|above) (instructions|messages)|disregard (the|your) '
                       r'(instructions|rules)|you are now\b|system prompt|as an ai (agent|assistant)|'
                       r'\b(claude|chatgpt|copilot|gemini|llm|ai agent)s?\b.{0,40}\b(must|should|please|run|execute)\b',
                       re.I)


def git(repo, *args):
    return subprocess.run(['git', '-C', str(repo), *args], capture_output=True, text=True, check=True,
                          encoding='utf-8', errors='replace').stdout


def changed(repo, base):
    out = []
    for line in git(repo, 'diff', '--name-status', '--no-renames', f'{base}...HEAD').splitlines():
        status, path = line.split('\t', 1)
        out.append((status, path))
    return out


def added_lines(repo, base, path):
    diff = git(repo, 'diff', '-U0', f'{base}...HEAD', '--', path)
    no, out = 0, []
    for line in diff.splitlines():
        m = re.match(r'@@ -\d+(?:,\d+)? \+(\d+)', line)
        if m:
            no = int(m.group(1))
            continue
        if line.startswith('+') and not line.startswith('+++'):
            out.append((no, line[1:]))
            no += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True)
    ap.add_argument('--repo', default='.')
    args = ap.parse_args()
    repo = Path(args.repo)
    errors, warnings = [], []
    files = changed(repo, args.base)
    paths = {p for _, p in files}
    for status, path in files:
        if status == 'D':
            if path.startswith(FROZEN):
                warnings.append(f'`{path}`: a frozen test or scorer file is deleted')
            continue
        f = repo / path
        ext = f.suffix.lower()
        if ext in BLOCKED_EXT:
            errors.append(f'`{path}`: {ext} files are not accepted (binaries, weights, pickles, archives, '
                          'executables and audio stay out of git)')
            continue
        if f.exists() and f.stat().st_size > MAX_BYTES:
            errors.append(f'`{path}`: larger than 1 MB')
            continue
        if path.startswith('.github/'):
            warnings.append(f'`{path}`: changes workflows or repository settings; review with extra care')
        if path.startswith(FROZEN) and 'docs/lab-notebook.md' not in paths:
            errors.append(f'`{path}`: a frozen test or scorer changed without a reason logged in docs/lab-notebook.md')
        if ext in ALLOWED_BINARY and ext != '.svg':
            continue
        for no, text in added_lines(repo, args.base, path):
            where = f'`{path}:{no}`'
            if HIDDEN.search(text):
                errors.append(f'{where}: hidden bidirectional or zero-width characters')
            if INJECTION.search(text):
                warnings.append(f'{where}: text that looks aimed at AI tools (possible prompt injection)')
            if ext == '.svg':
                # an image that can run code: no scripts, event handlers or javascript: links
                if re.search(r'<script|\bon[a-z]+\s*=|javascript:|<foreignObject', text, re.I):
                    errors.append(f'{where}: script or event handler inside an SVG image')
            elif ext in CODE_EXT:
                for pat, why in RISKY:
                    if re.search(pat, text):
                        warnings.append(f'{where}: {why}')
                if len(text) > 1000:
                    warnings.append(f'{where}: a line over 1,000 characters (minified or hidden code?)')
    # frozen test files must still match their recorded fingerprints
    hashes = repo / 'data' / 'tests' / 'HASHES.txt'
    if hashes.exists():
        for line in hashes.read_text(encoding='utf-8').splitlines():
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                continue
            want, rel = parts[0], parts[1].lstrip('*').strip()
            p = repo / rel
            if p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest() != want:
                errors.append(f'`{rel}`: does not match its fingerprint in data/tests/HASHES.txt')
    report = ['## Contribution audit', '', f'{len(files)} changed files, {len(errors)} errors, {len(warnings)} warnings.']
    if errors:
        report += ['', '### Errors (must be fixed)'] + [f'- {e}' for e in errors]
    if warnings:
        report += ['', '### For the reviewer'] + [f'- {w}' for w in warnings]
    report += ['', 'The audit never runs contributed code; the owner reviews every change before it merges.']
    text = '\n'.join(report) + '\n'
    print(text)
    summary = os.environ.get('GITHUB_STEP_SUMMARY')
    if summary:
        with open(summary, 'a', encoding='utf-8') as fh:
            fh.write(text)
    sys.exit(1 if errors else 0)


if __name__ == '__main__':
    main()
