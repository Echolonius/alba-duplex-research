"""Checks Alba's local API (app/api.py) and both example integrations end to end, on test ports with a
throwaway store (your real data/api is not touched). Writes results/api_test.json.

  .venv/Scripts/python.exe -I tools/test_api.py
"""
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.api import Store, serve  # noqa: E402
from integrations import notes_folder, todo_list  # noqa: E402

API = 18765


def call(method, path, body=None, headers=None):
    req = urllib.request.Request(f'http://127.0.0.1:{API}{path}', json.dumps(body).encode() if body is not None else None,
                                 {'Content-Type': 'application/json', **(headers or {})}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b'{}')


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    # a throwaway store inside the project (removed at the end), never in the system's temp folder
    import shutil
    tmp = ROOT / 'logs' / 'api-test-tmp'
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    store = Store(tmp / 'store')
    srv = serve(API, store)
    (tmp / 'notes').mkdir()
    (tmp / 'notes' / 'garden.md').write_text('Tomatoes go in the sunny bed by the fence.\nWater twice a week.\n', encoding='utf8')
    todo_list.start(18781, API, tmp / 'todo.json')
    notes_folder.start(tmp / 'notes', 18782, API)
    checks = {}

    def check(name, ok, detail=''):
        checks[name] = {'pass': bool(ok), 'detail': detail}
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)

    s, r = call('GET', '/v1/tools')
    names = {t['name'] for t in r['items']}
    check('both integrations registered their tools', {'todo_list', 'todo_add', 'todo_done', 'notes_search'} <= names,
          sorted(names))
    s, r = call('POST', '/v1/tools/todo_add/call', {'arguments': {'item': 'buy milk'}})
    check('a change is refused without a yes', r.get('needs_confirmation') is True, r.get('ask', ''))
    s, r = call('POST', '/v1/tools/todo_list/call', {'arguments': {}})
    check('...and nothing changed', r['result']['open'] == [], r)
    s, r = call('POST', '/v1/tools/todo_add/call', {'arguments': {'item': 'buy milk'}, 'confirmed': True})
    check('after a yes it runs', r['result'].get('open') == ['buy milk'], r)
    s, r = call('POST', '/v1/tools/todo_done/call', {'arguments': {'item': 'milk'}, 'confirmed': True})
    check('tick off works', r['result'].get('open') == [], r)
    s, r = call('POST', '/v1/tools/notes_search/call', {'arguments': {'query': 'where do the tomatoes go'}})
    m = r['result']['matches']
    check('notes search (read-only, no yes needed) finds the line', m and 'Tomatoes' in m[0]['text'], m[:1])
    s, r = call('POST', '/v1/tools/calculator/call', {'arguments': {'expression': '17 * 23 + 4'}})
    check('calculator', r['result']['value'] == 395, r)
    s, r = call('POST', '/v1/tools/calculator/call', {'arguments': {'expression': '__import__("os").system("x")'}})
    check('calculator refuses code', s == 400, r)
    s, r = call('POST', '/v1/tools', {'name': 'evil', 'description': 'x', 'url': 'http://example.com/run'})
    check('tools must live on this PC', s == 400, r)
    s, r = call('GET', '/v1/health', headers={'Origin': 'https://example.com'})
    check('web pages from other sites are refused', s == 403, r)
    s, r = call('GET', '/v1/health', headers={'Host': 'evil.example.com'})
    check('a re-pointed name is refused', s == 403, r)
    s, r = call('POST', '/v1/knowledge', {'title': 'Sister', 'text': "Maya's birthday is on March 3rd."})
    s, r = call('GET', '/v1/knowledge?q=when%20is%20maya%20birthday')
    check('knowledge store finds it', r['items'] and 'March' in r['items'][0]['text'], r['items'][:1])
    s, r = call('POST', '/v1/learning', {'call': '20261105-0100', 'text': 'They are learning the cello.',
                                         'snippet': "I've been practising the cello."})
    lid = r['id']
    s, r = call('GET', '/v1/knowledge?q=cello')
    check('a proposal is not used before approval', r['items'] == [], r)
    call('POST', f'/v1/learning/{lid}/approve')
    s, r = call('GET', '/v1/knowledge?q=cello')
    check('approved: now in the knowledge store', len(r['items']) == 1, r['items'])
    call('POST', f'/v1/learning/{lid}/undo')
    s, r = call('GET', '/v1/knowledge?q=cello')
    s2, r2 = call('GET', '/v1/learning')
    check('undo takes it back out and it waits again', r['items'] == [] and r2['items'][0]['status'] == 'waiting', r2['items'][0])
    call('DELETE', f'/v1/learning/{lid}')
    s, r = call('GET', '/v1/learning')
    check('delete removes it and its snippet for good', r['items'] == [], r)
    s, r = call('GET', '/v1/actions')
    check('every tool run is logged', len(r['items']) >= 4, f"{len(r['items'])} actions")
    srv.shutdown()
    store.db.close()
    shutil.rmtree(tmp, ignore_errors=True)
    out = ROOT / 'results' / 'api_test.json'
    out.write_text(json.dumps({'all_pass': all(c['pass'] for c in checks.values()), 'checks': checks}, indent=1,
                              ensure_ascii=False, default=str), encoding='utf8')
    print('all pass' if all(c['pass'] for c in checks.values()) else 'SOME FAILED', '->', out)


if __name__ == '__main__':
    main()
