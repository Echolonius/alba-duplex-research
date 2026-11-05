"""Alba's local API: the way other programs on this PC plug into the voice (goal check 7).

Only 127.0.0.1 (nothing else on the network can reach it), no accounts, no internet. Everything is stored in
data/api/ on this PC:
  tools      programs register tools Alba may use in a call (name, description, JSON-schema parameters, the
             127.0.0.1 address that runs it, and whether it changes something). Tools that change something
             only run with confirmed=true, which the voice sets only after you said yes to her question.
  knowledge  a local knowledge store (SQLite full-text search, BM25 ranking): notes and documents Alba can look
             up. Add, search, delete.
  learning   opt-in learning from calls: after a call you allowed it for, proposed things to remember wait here
             until you approve them (they go into the knowledge store). Any of them can be undone (taken back out)
             or deleted for good, with the call snippet they came from.
  actions    a log of every tool run (what, when, with what), to look back at.

  .venv/Scripts/python.exe -m app.api [--port 8765]          then open http://127.0.0.1:8765/review
Endpoints (JSON): GET /v1/health | GET,POST /v1/tools | DELETE /v1/tools/<name> | POST /v1/tools/<name>/call
| GET /v1/knowledge?q= | POST /v1/knowledge | DELETE /v1/knowledge/<id> | GET /v1/learning | POST
/v1/learning (propose) | POST /v1/learning/<id>/approve | POST /v1/learning/<id>/undo | DELETE /v1/learning/<id>
| GET /v1/actions
"""
import argparse
import ast
import datetime
import json
import operator
import re
import sqlite3
import threading
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STORE = ROOT / 'data' / 'api'
NAME_OK = re.compile(r'^[a-z][a-z0-9_]{0,47}$')


def now():
    return datetime.datetime.now().isoformat(timespec='seconds')


def local_url(url):
    """Tools may only live on this PC."""
    u = urllib.parse.urlparse(url)
    return u.scheme == 'http' and u.hostname in ('127.0.0.1', 'localhost') and u.port is not None


class Store:
    def __init__(self, folder=STORE):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.db = sqlite3.connect(self.folder / 'store.db', check_same_thread=False)
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS docs (id INTEGER PRIMARY KEY, title TEXT, text TEXT, source TEXT, added TEXT,
                                             learning_id INTEGER);
            CREATE VIRTUAL TABLE IF NOT EXISTS docs_fts USING fts5(title, text, content='docs', content_rowid='id');
            CREATE TRIGGER IF NOT EXISTS docs_ai AFTER INSERT ON docs BEGIN
                INSERT INTO docs_fts(rowid, title, text) VALUES (new.id, new.title, new.text); END;
            CREATE TRIGGER IF NOT EXISTS docs_ad AFTER DELETE ON docs BEGIN
                INSERT INTO docs_fts(docs_fts, rowid, title, text) VALUES ('delete', old.id, old.title, old.text); END;
            CREATE TABLE IF NOT EXISTS learning (id INTEGER PRIMARY KEY, call TEXT, text TEXT, snippet TEXT,
                                                 status TEXT, proposed TEXT, decided TEXT);
            CREATE TABLE IF NOT EXISTS actions (id INTEGER PRIMARY KEY, at TEXT, tool TEXT, args TEXT, result TEXT,
                                                confirmed INTEGER);
        """)
        self.tools_path = self.folder / 'tools.json'
        self.tools = json.loads(self.tools_path.read_text(encoding='utf8')) if self.tools_path.exists() else {}

    # ---- tools ---------------------------------------------------------------------------------------------
    def register(self, t):
        name = str(t.get('name', ''))
        if not NAME_OK.match(name) or name in BUILTIN:
            raise ValueError('name: lowercase letters, digits and _ (not a built-in name)')
        if not local_url(str(t.get('url', ''))):
            raise ValueError('url must be http://127.0.0.1:<port>/... (tools run on this PC only)')
        tool = {'name': name, 'description': str(t.get('description', ''))[:500],
                'parameters': t.get('parameters') or {'type': 'object', 'properties': {}},
                'url': t['url'], 'changes_something': bool(t.get('changes_something', True)), 'registered': now()}
        with self.lock:
            self.tools[name] = tool
            self.tools_path.write_text(json.dumps(self.tools, indent=1), encoding='utf8')
        return tool

    def unregister(self, name):
        with self.lock:
            gone = self.tools.pop(name, None)
            self.tools_path.write_text(json.dumps(self.tools, indent=1), encoding='utf8')
        return gone is not None

    def all_tools(self):
        return [{k: v for k, v in b.items() if k != 'run'} for b in BUILTIN.values()] + list(self.tools.values())

    def call(self, name, args, confirmed=False):
        spec = BUILTIN.get(name) or self.tools.get(name)
        if spec is None:
            raise KeyError(name)
        if spec['changes_something'] and not confirmed:
            return {'needs_confirmation': True, 'ask': f"Ask the person first: should I {spec['description'].lower()}"
                    f" with {json.dumps(args)}?"}
        if name in BUILTIN:
            result = BUILTIN[name]['run'](self, args)
        else:
            req = urllib.request.Request(spec['url'], json.dumps({'name': name, 'arguments': args}).encode(),
                                         {'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=10) as r:
                result = json.loads(r.read())
        with self.lock:
            self.db.execute('INSERT INTO actions (at, tool, args, result, confirmed) VALUES (?,?,?,?,?)',
                            (now(), name, json.dumps(args), json.dumps(result)[:2000], int(bool(confirmed))))
            self.db.commit()
        return {'result': result}

    # ---- knowledge -------------------------------------------------------------------------------------------
    def add_doc(self, title, text, source='you', learning_id=None):
        with self.lock:
            cur = self.db.execute('INSERT INTO docs (title, text, source, added, learning_id) VALUES (?,?,?,?,?)',
                                  (title, text, source, now(), learning_id))
            self.db.commit()
            return cur.lastrowid

    def search(self, q, k=5):
        words = re.findall(r'\w+', q.lower())
        if not words:
            return []
        query = ' OR '.join(f'"{w}"' for w in words)
        with self.lock:
            rows = self.db.execute('SELECT d.id, d.title, d.text, d.source, d.added FROM docs_fts f JOIN docs d ON '
                                   'd.id = f.rowid WHERE docs_fts MATCH ? ORDER BY bm25(docs_fts) LIMIT ?',
                                   (query, k)).fetchall()
        return [dict(zip(('id', 'title', 'text', 'source', 'added'), r)) for r in rows]

    def list_docs(self):
        with self.lock:
            rows = self.db.execute('SELECT id, title, text, source, added FROM docs ORDER BY id DESC').fetchall()
        return [dict(zip(('id', 'title', 'text', 'source', 'added'), r)) for r in rows]

    def delete_doc(self, i):
        with self.lock:
            n = self.db.execute('DELETE FROM docs WHERE id = ?', (i,)).rowcount
            self.db.commit()
        return n > 0

    # ---- opt-in learning ---------------------------------------------------------------------------------------
    def propose(self, call, text, snippet=''):
        with self.lock:
            cur = self.db.execute('INSERT INTO learning (call, text, snippet, status, proposed) VALUES (?,?,?,?,?)',
                                  (call, text, snippet, 'waiting', now()))
            self.db.commit()
            return cur.lastrowid

    def learning(self):
        with self.lock:
            rows = self.db.execute('SELECT id, call, text, snippet, status, proposed, decided FROM learning '
                                   'ORDER BY id DESC').fetchall()
        return [dict(zip(('id', 'call', 'text', 'snippet', 'status', 'proposed', 'decided'), r)) for r in rows]

    def approve(self, i):
        row = next((r for r in self.learning() if r['id'] == i), None)
        if row is None or row['status'] == 'approved':
            return row is not None
        self.add_doc(f"From the call of {row['call']}", row['text'], source='learned', learning_id=i)
        with self.lock:
            self.db.execute("UPDATE learning SET status = 'approved', decided = ? WHERE id = ?", (now(), i))
            self.db.commit()
        return True

    def undo(self, i):
        """Takes an approved item back out of the knowledge store; it waits for review again."""
        with self.lock:
            self.db.execute('DELETE FROM docs WHERE learning_id = ?', (i,))
            n = self.db.execute("UPDATE learning SET status = 'waiting', decided = ? WHERE id = ?", (now(), i)).rowcount
            self.db.commit()
        return n > 0

    def forget(self, i):
        """Deletes for good: the item, its call snippet and anything it put in the knowledge store."""
        with self.lock:
            self.db.execute('DELETE FROM docs WHERE learning_id = ?', (i,))
            n = self.db.execute('DELETE FROM learning WHERE id = ?', (i,)).rowcount
            self.db.commit()
        return n > 0

    def actions(self, k=100):
        with self.lock:
            rows = self.db.execute('SELECT at, tool, args, result, confirmed FROM actions ORDER BY id DESC LIMIT ?',
                                   (k,)).fetchall()
        return [dict(zip(('at', 'tool', 'args', 'result', 'confirmed'), r)) for r in rows]


# ---- built-in tools (read-only ones never need a yes) ------------------------------------------------------------
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos, ast.Mod: operator.mod,
        ast.FloorDiv: operator.floordiv}


def _calc(expr):
    """Arithmetic only (numbers and + - * / // % **), never code."""
    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            a, b = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Pow) and (abs(b) > 100 or abs(a) > 1e6):
                raise ValueError('too large')
            v = _OPS[type(n.op)](a, b)
        elif isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            v = _OPS[type(n.op)](ev(n.operand))
        else:
            raise ValueError('only arithmetic')
        if isinstance(v, complex) or abs(v) > 1e100:
            raise ValueError('too large')
        return v
    if len(str(expr)) > 200:
        raise ValueError('too long')
    return ev(ast.parse(str(expr).replace('^', '**'), mode='eval'))


BUILTIN = {
    'clock': {'name': 'clock', 'description': 'Tell the current local date and time', 'changes_something': False,
              'parameters': {'type': 'object', 'properties': {}},
              'run': lambda s, a: {'now': datetime.datetime.now().strftime('%A, %B %d, %Y, %I:%M %p')}},
    'calculator': {'name': 'calculator', 'description': 'Work out an arithmetic expression', 'changes_something': False,
                   'parameters': {'type': 'object', 'properties': {'expression': {'type': 'string'}},
                                  'required': ['expression']},
                   'run': lambda s, a: {'value': _calc(a.get('expression', ''))}},
    'search_knowledge': {'name': 'search_knowledge', 'description': "Look something up in the person's local notes",
                         'changes_something': False,
                         'parameters': {'type': 'object', 'properties': {'query': {'type': 'string'}},
                                        'required': ['query']},
                         'run': lambda s, a: {'matches': [{k: d[k] for k in ('title', 'text')}
                                                          for d in s.search(a.get('query', ''), 3)]}},
    'save_note': {'name': 'save_note', 'description': 'Save a note to the local knowledge store',
                  'changes_something': True,
                  'parameters': {'type': 'object', 'properties': {'title': {'type': 'string'}, 'text': {'type': 'string'}},
                                 'required': ['text']},
                  'run': lambda s, a: {'saved_id': s.add_doc(a.get('title') or 'Note', a.get('text', ''), 'call')}},
}


REVIEW_PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Alba: review</title><style>
body{font:15px Segoe UI,sans-serif;background:#16181b;color:#eceeef;margin:24px;max-width:900px}
h1{font-size:22px}h2{font-size:17px;margin-top:28px}.item{background:#22262b;padding:10px 14px;margin:8px 0;border-radius:6px}
.dim{color:#8d949a;font-size:13px}button{margin-right:6px;background:#7fa2ff;border:0;padding:4px 10px;border-radius:4px;cursor:pointer}
button.del{background:#e66767}</style></head><body><h1>Alba: what she may remember</h1>
<p class="dim">Everything here stays on this PC. Approve to let Alba use it; undo takes it back out; delete removes it for good.</p>
<h2>From your calls (opt-in)</h2><div id="learning"></div><h2>Knowledge store</h2><div id="docs"></div>
<h2>Recent actions</h2><div id="actions"></div><script>
async function j(m,u){const r=await fetch(u,{method:m});return r.json()}
function esc(s){return String(s??'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
async function load(){
 const L=await j('GET','/v1/learning');document.getElementById('learning').innerHTML=L.items.map(x=>`<div class="item">
 <div>${esc(x.text)}</div><div class="dim">${esc(x.status)} - call ${esc(x.call)} - "${esc(x.snippet)}"</div>
 ${x.status!=='approved'?`<button onclick="act('POST','/v1/learning/${x.id}/approve')">Approve</button>`:`<button onclick="act('POST','/v1/learning/${x.id}/undo')">Undo</button>`}
 <button class="del" onclick="act('DELETE','/v1/learning/${x.id}')">Delete</button></div>`).join('')||'<p class="dim">Nothing waiting.</p>';
 const D=await j('GET','/v1/knowledge');document.getElementById('docs').innerHTML=D.items.map(x=>`<div class="item"><b>${esc(x.title)}</b>
 <div>${esc(x.text)}</div><div class="dim">${esc(x.source)} - ${esc(x.added)}</div>
 <button class="del" onclick="act('DELETE','/v1/knowledge/${x.id}')">Delete</button></div>`).join('')||'<p class="dim">Empty.</p>';
 const A=await j('GET','/v1/actions');document.getElementById('actions').innerHTML=A.items.slice(0,20).map(x=>`<div class="item dim">
 ${esc(x.at)} - ${esc(x.tool)} ${esc(x.args)} -> ${esc(x.result)}</div>`).join('')||'<p class="dim">None yet.</p>';}
async function act(m,u){await j(m,u);load()}load();</script></body></html>"""


def make_handler(store):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, obj, ctype='application/json'):
            body = obj.encode() if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header('Content-Type', ctype + '; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self):
            n = int(self.headers.get('Content-Length') or 0)
            return json.loads(self.rfile.read(n) or b'{}') if n else {}

        def _route(self, method):
            # browsers on other sites must not be able to drive this (no CORS headers are ever sent; and a
            # request whose Origin is not this page is refused)
            origin = self.headers.get('Origin')
            if origin and urllib.parse.urlparse(origin).hostname not in ('127.0.0.1', 'localhost'):
                return self._send(403, {'error': 'only this PC'})
            host = (self.headers.get('Host') or '').rsplit(':', 1)[0]
            if host not in ('127.0.0.1', 'localhost'):          # a web page re-pointing its own name here
                return self._send(403, {'error': 'only this PC'})
            u = urllib.parse.urlparse(self.path)
            parts = [p for p in u.path.split('/') if p]
            q = urllib.parse.parse_qs(u.query)
            try:
                if parts == ['review'] and method == 'GET':
                    return self._send(200, REVIEW_PAGE, 'text/html')
                if parts[:1] != ['v1']:
                    return self._send(404, {'error': 'not found'})
                p = parts[1:]
                if p == ['health']:
                    return self._send(200, {'ok': True, 'tools': len(store.all_tools())})
                if p == ['tools'] and method == 'GET':
                    return self._send(200, {'items': store.all_tools()})
                if p == ['tools'] and method == 'POST':
                    return self._send(200, store.register(self._body()))
                if len(p) == 2 and p[0] == 'tools' and method == 'DELETE':
                    return self._send(200, {'removed': store.unregister(p[1])})
                if len(p) == 3 and p[0] == 'tools' and p[2] == 'call' and method == 'POST':
                    b = self._body()
                    return self._send(200, store.call(p[1], b.get('arguments') or {}, bool(b.get('confirmed'))))
                if p == ['knowledge'] and method == 'GET':
                    return self._send(200, {'items': store.search(q['q'][0]) if q.get('q') else store.list_docs()})
                if p == ['knowledge'] and method == 'POST':
                    b = self._body()
                    return self._send(200, {'id': store.add_doc(b.get('title') or 'Note', b['text'], b.get('source', 'api'))})
                if len(p) == 2 and p[0] == 'knowledge' and method == 'DELETE':
                    return self._send(200, {'deleted': store.delete_doc(int(p[1]))})
                if p == ['learning'] and method == 'GET':
                    return self._send(200, {'items': store.learning()})
                if p == ['learning'] and method == 'POST':
                    b = self._body()
                    return self._send(200, {'id': store.propose(b.get('call', ''), b['text'], b.get('snippet', ''))})
                if len(p) == 3 and p[0] == 'learning' and method == 'POST' and p[2] in ('approve', 'undo'):
                    ok = store.approve(int(p[1])) if p[2] == 'approve' else store.undo(int(p[1]))
                    return self._send(200, {'ok': ok})
                if len(p) == 2 and p[0] == 'learning' and method == 'DELETE':
                    return self._send(200, {'deleted': store.forget(int(p[1]))})
                if p == ['actions']:
                    return self._send(200, {'items': store.actions()})
                return self._send(404, {'error': 'not found'})
            except KeyError as e:
                return self._send(404, {'error': f'unknown: {e}'})
            except (ValueError, json.JSONDecodeError) as e:
                return self._send(400, {'error': str(e)})

        def do_GET(self):
            self._route('GET')

        def do_POST(self):
            self._route('POST')

        def do_DELETE(self):
            self._route('DELETE')
    return Handler


def serve(port=8765, store=None):
    """Starts the API on 127.0.0.1 only; returns the server (call .shutdown() to stop)."""
    store = store or Store()
    srv = ThreadingHTTPServer(('127.0.0.1', port), make_handler(store))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def main():
    ap = argparse.ArgumentParser(description="Alba's local API (127.0.0.1 only)")
    ap.add_argument('--port', type=int, default=8765)
    args = ap.parse_args()
    srv = ThreadingHTTPServer(('127.0.0.1', args.port), make_handler(Store()))
    print(f'Alba API on http://127.0.0.1:{args.port} (review page: /review). Ctrl+C to stop.', flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
