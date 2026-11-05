"""Example integration 1: a to-do list Alba can read and change (changes need your spoken yes).

A tiny service on 127.0.0.1 that keeps data/api/todo.json and registers three tools with Alba's API
(app/api.py): todo_list (read-only), todo_add and todo_done (they change the list, so Alba asks first).

  .venv/Scripts/python.exe -m integrations.todo_list [--port 8781] [--api 8765]
"""
import argparse
import json
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

TOOLS = [
    {'name': 'todo_list', 'description': 'Read the to-do list', 'changes_something': False,
     'parameters': {'type': 'object', 'properties': {}}},
    {'name': 'todo_add', 'description': 'Add an item to the to-do list', 'changes_something': True,
     'parameters': {'type': 'object', 'properties': {'item': {'type': 'string'}}, 'required': ['item']}},
    {'name': 'todo_done', 'description': 'Tick off an item on the to-do list', 'changes_something': True,
     'parameters': {'type': 'object', 'properties': {'item': {'type': 'string'}}, 'required': ['item']}},
]


class TodoList:
    def __init__(self, path):
        self.path, self.lock = Path(path), threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self):
        return json.loads(self.path.read_text(encoding='utf8')) if self.path.exists() else []

    def run(self, name, args):
        with self.lock:
            items = self._load()
            if name == 'todo_list':
                return {'open': [i['item'] for i in items if not i['done']]}
            item = str(args.get('item', '')).strip()[:200]
            if not item:
                return {'error': 'no item'}
            if name == 'todo_add':
                items.append({'item': item, 'done': False})
            elif name == 'todo_done':
                hit = next((i for i in items if not i['done'] and item.lower() in i['item'].lower()), None)
                if hit is None:
                    return {'error': f'no open item like {item!r}'}
                hit['done'] = True
            self.path.write_text(json.dumps(items, indent=1, ensure_ascii=False), encoding='utf8')
            return {'ok': True, 'open': [i['item'] for i in items if not i['done']]}


def start(port=8781, api_port=8765, path=ROOT / 'data' / 'api' / 'todo.json'):
    todo = TodoList(path)

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get('Content-Length') or 0)) or b'{}')
            out = json.dumps(todo.run(body.get('name'), body.get('arguments') or {})).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(out)))
            self.end_headers()
            self.wfile.write(out)

    srv = ThreadingHTTPServer(('127.0.0.1', port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    for t in TOOLS:
        req = urllib.request.Request(f'http://127.0.0.1:{api_port}/v1/tools',
                                     json.dumps({**t, 'url': f'http://127.0.0.1:{port}/run'}).encode(),
                                     {'Content-Type': 'application/json'})
        urllib.request.urlopen(req, timeout=5).read()
    return srv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8781)
    ap.add_argument('--api', type=int, default=8765)
    args = ap.parse_args()
    start(args.port, args.api)
    print(f'to-do list tools registered with Alba (service on 127.0.0.1:{args.port}). Ctrl+C to stop.', flush=True)
    threading.Event().wait()


if __name__ == '__main__':
    main()
