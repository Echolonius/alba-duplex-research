"""Example integration 2: Alba can search a folder of your own notes (read-only; nothing is changed or sent).

A tiny service on 127.0.0.1 that searches the .txt and .md files in one folder you choose (default
data/api/notes) and registers one tool with Alba's API (app/api.py): notes_search(query) -> the best matching
lines with their file names.

  .venv/Scripts/python.exe -m integrations.notes_folder [--folder PATH] [--port 8782] [--api 8765]
"""
import argparse
import json
import re
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

TOOL = {'name': 'notes_search', 'description': "Search the person's notes folder", 'changes_something': False,
        'parameters': {'type': 'object', 'properties': {'query': {'type': 'string'}}, 'required': ['query']}}


def search(folder, query, k=3):
    words = set(re.findall(r'\w+', query.lower())) - {'the', 'a', 'an', 'my', 'of', 'to', 'in', 'is', 'what', 'about'}
    hits = []
    for f in sorted(Path(folder).glob('**/*')):
        if f.suffix.lower() not in ('.txt', '.md') or f.stat().st_size > 2_000_000:
            continue
        for n, line in enumerate(f.read_text(encoding='utf8', errors='replace').splitlines(), 1):
            score = len(words & set(re.findall(r'\w+', line.lower())))
            if score:
                hits.append((score, f.name, n, line.strip()[:300]))
    hits.sort(key=lambda h: (-h[0], h[1], h[2]))
    return {'matches': [{'file': h[1], 'line': h[2], 'text': h[3]} for h in hits[:k]]}


def start(folder=ROOT / 'data' / 'api' / 'notes', port=8782, api_port=8765):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get('Content-Length') or 0)) or b'{}')
            out = json.dumps(search(folder, str((body.get('arguments') or {}).get('query', '')))).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(out)))
            self.end_headers()
            self.wfile.write(out)

    srv = ThreadingHTTPServer(('127.0.0.1', port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    req = urllib.request.Request(f'http://127.0.0.1:{api_port}/v1/tools',
                                 json.dumps({**TOOL, 'url': f'http://127.0.0.1:{port}/run'}).encode(),
                                 {'Content-Type': 'application/json'})
    urllib.request.urlopen(req, timeout=5).read()
    return srv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folder', default=str(ROOT / 'data' / 'api' / 'notes'))
    ap.add_argument('--port', type=int, default=8782)
    ap.add_argument('--api', type=int, default=8765)
    args = ap.parse_args()
    start(args.folder, args.port, args.api)
    print(f'notes search registered with Alba (folder {args.folder}). Ctrl+C to stop.', flush=True)
    threading.Event().wait()


if __name__ == '__main__':
    main()
