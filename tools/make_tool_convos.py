"""Voice calls where Alba uses her tools (goal check 7; training run 4 on): the words, with REAL tool results.

The tools are the ones a call has (app/api.py built-ins and the example integrations): clock, calculator,
search_knowledge, save_note, todo_list, todo_add, todo_done, notes_search. Each conversation has a sandbox (a
throwaway knowledge store with made-up notes, a made-up to-do list, a made-up notes folder), so every result is
what the real tool code returns, never invented. The local open model (Gemma 4 E4B, the persona and filters of
tools/make_convo_text.py) writes what the person says and what Alba says around the tool.

Anything that changes something (save_note, todo_add, todo_done) always goes: the person asks, Alba asks back
("want me to add oat milk?"), the person says yes or no; only after a yes does Alba call the tool (and the runtime
checks that yes again with speech recognition before running it). Asked to store a password, PIN or bank detail,
she declines and calls nothing.

In the duplex stream (tools/train_duplex.build) a tool call sits at the start of the utterance that uses it:
  [heard] <tool_call>...</tool_call> <tool_response>...</tool_response> <|mixed_start|> plan... frame
in the model's own tool format (models/lfm3-audio-hf/chat_template.jinja), with the tool list in the opening.

  python -I tools/make_tool_convos.py --n 600 [--out data/convo-tools/text.jsonl] [--port 9948] [--workers 2]
Then: tools/make_convo_audio.py --text data/convo-tools/text.jsonl, tools/make_duplex_data.py --source synthetic
--convo data/convo-tools --source-name tools --out data/duplex/tools.
"""
import argparse
import datetime
import json
import random
import re
import sys
import tempfile
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

# the tools a call offers, as their JSON specs (the opening lists them in the model's own <tools> format)
SPECS = [
    {'name': 'clock', 'description': 'Tell the current local date and time', 'parameters': {'type': 'object', 'properties': {}}},
    {'name': 'calculator', 'description': 'Work out an arithmetic expression',
     'parameters': {'type': 'object', 'properties': {'expression': {'type': 'string'}}, 'required': ['expression']}},
    {'name': 'search_knowledge', 'description': "Look something up in the person's local notes",
     'parameters': {'type': 'object', 'properties': {'query': {'type': 'string'}}, 'required': ['query']}},
    {'name': 'save_note', 'description': 'Save a note to the local knowledge store',
     'parameters': {'type': 'object', 'properties': {'title': {'type': 'string'}, 'text': {'type': 'string'}},
                    'required': ['text']}},
    {'name': 'todo_list', 'description': 'Read the to-do list', 'parameters': {'type': 'object', 'properties': {}}},
    {'name': 'todo_add', 'description': 'Add an item to the to-do list',
     'parameters': {'type': 'object', 'properties': {'item': {'type': 'string'}}, 'required': ['item']}},
    {'name': 'todo_done', 'description': 'Tick off an item on the to-do list',
     'parameters': {'type': 'object', 'properties': {'item': {'type': 'string'}}, 'required': ['item']}},
    {'name': 'notes_search', 'description': "Search the person's notes folder",
     'parameters': {'type': 'object', 'properties': {'query': {'type': 'string'}}, 'required': ['query']}},
]
CHANGES = {'save_note', 'todo_add', 'todo_done'}

# made-up contents for the sandboxes (no real person's data)
NOTES = [
    ('Dentist', 'Dentist appointment Tuesday the 14th at 3:30 PM, Dr. Patel, bring the insurance card.'),
    ('Book club', 'Book club picks: The Remains of the Day next, then Piranesi. Meeting at Sam\'s on the 2nd.'),
    ('Car', 'Oil change due at 48,000 miles. Tire pressure 35 PSI front and back.'),
    ('Garden', 'Plant the tomatoes after the last frost, around mid-May. Basil goes next to them.'),
    ('Gift ideas', 'Gift ideas for Jordan: a pour-over coffee set, the new Zelda game, wool socks.'),
    ('Recipe', 'Pancakes: 1 cup flour, 1 egg, 1 cup milk, 1 tablespoon sugar, 2 teaspoons baking powder, pinch of salt.'),
    ('Trip', 'Train to Portland leaves at 8:10 AM Friday from platform 4. Hotel check-in after 3 PM.'),
    ('Plants', 'Water the fern twice a week, the cactus once a month. The orchid likes the east window.'),
    ('Workout', 'Monday and Thursday: 20 minutes of running. Wednesday: yoga. Saturday: long bike ride.'),
    ('Wifi', 'The guest wifi network is called BlueHeron and the router is in the hall closet.'),
    ('Movies', 'Movies to watch: Arrival, Paddington 2, Spirited Away, The Grand Budapest Hotel.'),
    ('Landlord', 'The landlord said the heater gets checked on the 21st between 9 and noon.'),
    ('Birthday', 'Grandma\'s birthday is June 12. She likes lemon cake and crossword books.'),
    ('Project', 'Slides for the quarterly review are due Thursday. Ask Priya for the sales numbers.'),
    ('Vet', 'The cat\'s vaccines are due in March. The vet closes at 6 on weekdays.'),
]
TODOS = ['buy oat milk', 'call the plumber', 'return the library books', 'pay the electric bill', 'book a haircut',
         'water the plants', 'send the birthday card', 'renew the passport', 'clean the fridge', 'pick up dry cleaning',
         'fix the bike tire', 'email the landlord', 'buy dog food', 'schedule the dentist', 'back up the laptop']

SITUATIONS = [
    (3, 'clock', "they ask Alba what time it is, or what day or date it is today"),
    (4, 'calculator', "they ask Alba to work out some everyday arithmetic out loud: a tip, splitting a bill, a "
                      "discount, a quick multiplication, minutes into hours"),
    (3, 'search_knowledge', "they ask Alba to look up something they saved in their notes before: {topic}"),
    (3, 'todo_list', "they ask Alba what's on their to-do list, or what they still have to do"),
    (3, 'todo_add', "they ask Alba to put something on their to-do list"),
    (2, 'todo_done', "they tell Alba they finished something on their to-do list ({item}) and want it ticked off"),
    (2, 'save_note', "they ask Alba to remember something for later by saving a note (an appointment, a recipe "
                     "tweak, a gift idea, where they parked)"),
    (2, 'notes_search', "they ask Alba to search their notes folder for something: {topic}"),
    (1, 'decline', "they ask Alba to save a password, a PIN or a bank card number in a note"),
]
SAY_YES = 0.8


def render_call(name, args):
    """The model's own tool-call text (chat_template.jinja render_tool_calls)."""
    s = f'<tool_call>\n<function={name}>\n'
    for k, v in args.items():
        s += f'<parameter={k}>\n{v}\n</parameter>\n'
    return s + '</function>\n</tool_call>'


def render_response(result):
    return '<tool_response>\n' + json.dumps(result, ensure_ascii=False) + '\n</tool_response>'


def tools_block(specs=SPECS):
    """The opening's tool list, as the model's own template writes it."""
    return '<tools>\n' + '\n'.join(json.dumps(s, ensure_ascii=False) for s in specs) + '\n</tools>'


class Sandbox:
    """A throwaway knowledge store, to-do list and notes folder: the real tool code, made-up contents."""

    def __init__(self, rng, notes=None, todo=None, folder=None):
        """Made-up contents drawn with rng, or given (the frozen tool tests' own sandbox)."""
        from app.api import Store
        from integrations.todo_list import TodoList
        self.tmp = tempfile.TemporaryDirectory(prefix='alba-tools-')
        d = Path(self.tmp.name)
        self.store = Store(d / 'store')
        self.notes = [tuple(n) for n in notes] if notes is not None else rng.sample(NOTES, 6)
        for title, text in self.notes:
            self.store.add_doc(title, text, 'you')
        self.todo = TodoList(d / 'todo.json')
        self.items = list(todo) if todo is not None else rng.sample(TODOS, rng.randint(2, 5))
        for it in self.items:
            self.todo.run('todo_add', {'item': it})
        self.folder = [tuple(n) for n in folder] if folder is not None else rng.sample(NOTES, 5)

    def run(self, name, args, clock=None):
        try:
            return self._run(name, args, clock)
        except Exception as e:                     # what the API itself would answer
            return {'error': f'{type(e).__name__}: {e}'[:200]}

    def _run(self, name, args, clock=None):
        from app.api import BUILTIN
        if name == 'clock':
            return {'now': clock}
        if name in ('calculator', 'search_knowledge', 'save_note'):
            return BUILTIN[name]['run'](self.store, args)
        if name.startswith('todo_'):
            return self.todo.run(name, args)
        if name == 'notes_search':
            q = set(re.findall(r'[a-z]+', str(args.get('query', '')).lower()))
            hits = [{'file': f'{t.lower().replace(" ", "-")}.txt', 'text': x} for t, x in self.folder
                    if q & set(re.findall(r'[a-z]+', (t + ' ' + x).lower()))]
            return {'matches': hits[:3]}
        raise ValueError(name)

    def close(self):
        self.store.db.close()
        self.tmp.cleanup()


def fake_now(rng):
    d = datetime.datetime(rng.choice([2025, 2026, 2027]), rng.randint(1, 12), rng.randint(1, 28), rng.randint(0, 23),
                          rng.randint(0, 59))
    return d.strftime('%A, %B %d, %Y, %I:%M %p')


def _tool_test_lines():
    p = ROOT / 'data' / 'tests' / 'tools' / 'items.json'
    if not p.exists():
        return []
    return [re.sub(r'[^a-z ]', '', t.lower()) for it in json.loads(p.read_text(encoding='utf8'))['items']
            for t in it['turns']]


TOOL_TEST_LINES = _tool_test_lines()
TOOL_TEST_WORDS = re.compile(r'monstera|aunt rosa|printer ink|okafor|vellichor|ironworks|sourdough starter|'
                             r'piano lessons|locker (is )?(number )?214|ridge trail|oolong', re.I)


def leaks_tools(text):
    """Too close to the frozen tool tests (data/tests/tools) or naming what only their sandbox holds."""
    import difflib
    if TOOL_TEST_WORDS.search(text):
        return True
    t = re.sub(r'[^a-z ]', '', text.lower())
    return any(difflib.SequenceMatcher(None, t, s).ratio() >= 0.75 for s in TOOL_TEST_LINES
               if abs(len(s) - len(t)) < 60)


def ask_json(port, prompt, rng):
    from make_convo_text import chat
    for _ in range(3):
        r = chat(port, [{'role': 'user', 'content': prompt}], temperature=0.4, max_tokens=120)
        m = re.search(r'\{.*\}', r, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return None


def one_conversation(port, cid, rng):
    from make_convo_text import BAD_REPLY, TRAIN_SYSTEM, chat, clean, leaks, persona_prompt
    w = [s[0] for s in SITUATIONS]
    _, tool, situation = rng.choices(SITUATIONS, weights=w)[0]
    sb = Sandbox(rng)
    try:
        topic = rng.choice(sb.notes if tool == 'search_knowledge' else sb.folder)[0].lower()
        item = rng.choice(sb.items)
        situation = situation.format(topic=topic, item=item)
        now = fake_now(rng)
        persona = persona_prompt(None)

        def person(history, instruction):
            ctx = '\n'.join(f"{'Them' if t['role'] == 'person' else 'Alba'}: {t['text']}" for t in history)
            prompt = (f"You write lines for a voice-call simulator. A person is on a voice call with their AI friend "
                      f"Alba, who can use a few tools (clock, calculator, their notes, their to-do list). "
                      f"{('The call so far:' + chr(10) + ctx + chr(10)) if ctx else ''}"
                      f"Write only what the person says next, out loud, casually, one or two sentences: {instruction}")
            return clean(chat(port, [{'role': 'user', 'content': prompt}], temperature=1.0, max_tokens=90))

        def alba(history, instruction):
            msgs = [{'role': 'system', 'content': persona + '\n' + instruction}]
            for t in history:
                msgs.append({'role': 'user' if t['role'] == 'person' else 'assistant', 'content': t['text']})
            for _ in range(3):
                r = clean(chat(port, msgs, temperature=0.8, max_tokens=120))
                if r and not BAD_REPLY.search(r) and not leaks(r) and not leaks_tools(r) and len(r.split()) <= 45:
                    return r
            return None

        turns = []
        ask = person([], situation)
        if not ask or leaks(ask) or leaks_tools(ask):
            return None
        turns.append({'role': 'person', 'text': ask})
        if tool == 'decline':
            r = alba(turns, "Safety: never store or repeat passwords, PINs or card numbers. Say no kindly in one or "
                            "two sentences and suggest a password manager or keeping it somewhere safe offline.")
            if not r:
                return None
            turns.append({'role': 'alba', 'text': r})
        else:
            args = {}
            if tool == 'calculator':
                j = ask_json(port, f'They said: "{ask}". Give the arithmetic as JSON {{"expression": "..."}} using only '
                                   f'numbers, + - * / and parentheses (a percentage p of x is x*p/100).', rng)
                args = {'expression': str((j or {}).get('expression', ''))}
            elif tool in ('search_knowledge', 'notes_search'):
                j = ask_json(port, f'They said: "{ask}". Give a short search query (1 to 3 words) as JSON '
                                   f'{{"query": "..."}}.', rng)
                args = {'query': str((j or {}).get('query', topic))}
            elif tool in ('todo_add', 'todo_done'):
                j = ask_json(port, f'They said: "{ask}". Give the to-do item (2 to 6 words, lower case) as JSON '
                                   f'{{"item": "..."}}.', rng)
                args = {'item': str((j or {}).get('item', item if tool == 'todo_done' else ''))}
            elif tool == 'save_note':
                j = ask_json(port, f'They said: "{ask}". Give the note to save as JSON {{"title": "2 to 4 words", '
                                   f'"text": "the note, one sentence"}}.', rng)
                args = {'title': str((j or {}).get('title', 'Note')), 'text': str((j or {}).get('text', ''))}
            if any(v == '' for v in args.values()):
                return None
            if tool in CHANGES:
                q = alba(turns, f"Before doing anything, ask them in one short sentence whether to go ahead: "
                                f"{tool.replace('_', ' ')} {json.dumps(args)}. Ask only; do not say it is done.")
                if not q:
                    return None
                turns.append({'role': 'alba', 'text': q})
                yes = rng.random() < SAY_YES
                turns.append({'role': 'person', 'text': person(turns, 'a casual yes, go ahead' if yes else
                                                                'a casual no, never mind') or ('Yes, please.' if yes else 'No, leave it.')})
                if not yes:
                    r = alba(turns, 'They said no: say briefly that you will leave it.')
                    if not r:
                        return None
                    turns.append({'role': 'alba', 'text': r})
                    return {'id': cid, 'situation': situation, 'clock': None, 'system': TRAIN_SYSTEM, 'tools': SPECS,
                            'turns': turns}
            result = sb.run(tool, args, clock=now)
            r = alba(turns, f"You just used your {tool} tool and it returned: {json.dumps(result, ensure_ascii=False)}. "
                            f"Tell them the answer in your own spoken words, short. Use only what the tool returned; if "
                            f"it found nothing or returned an error, say so plainly.")
            if not r:
                return None
            turns.append({'role': 'alba', 'text': r,
                          'tool': {'name': tool, 'arguments': args, 'result': result,
                                   'call': render_call(tool, args), 'response': render_response(result)}})
        return {'id': cid, 'situation': situation, 'clock': None, 'system': TRAIN_SYSTEM, 'tools': SPECS, 'turns': turns}
    finally:
        sb.close()


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=600)
    ap.add_argument('--out', default=str(ROOT / 'data' / 'convo-tools' / 'text.jsonl'))
    ap.add_argument('--port', type=int, default=9948)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--seed', type=int, default=4)
    args = ap.parse_args()
    from app import server
    from app.config import Config
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)['id'] for l in open(out, encoding='utf8') if l.strip()} if out.exists() else set()
    cfg = Config()
    proc = None
    if not server.healthy(cfg, port=args.port):
        proc = server._launch(cfg, ['-m', str(ROOT / 'models' / 'text' / 'gemma-4-E4B-it-Q4_0.gguf'),
                                    '-c', str(4096 * args.workers), '-np', str(args.workers)],
                              args.port, ROOT / 'logs' / 'server-tool-convos.log', 180)
    todo = [f't{k:05d}' for k in range(args.n) if f't{k:05d}' not in done]
    lock = threading.Lock()

    def worker():
        while True:
            with lock:
                if not todo:
                    return
                cid = todo.pop(0)
            rng = random.Random(f'{args.seed}-{cid}')
            try:
                c = one_conversation(args.port, cid, rng)
            except Exception as e:
                print(f'{cid}: failed ({type(e).__name__}: {e})', flush=True)
                continue
            if c is None:
                print(f'{cid}: rejected', flush=True)
                continue
            with lock:
                with open(out, 'a', encoding='utf8') as f:
                    f.write(json.dumps(c, ensure_ascii=False) + '\n')
            t = next((t['tool']['name'] for t in c['turns'] if t.get('tool')), 'none')
            print(f"{cid}: {len(c['turns'])} turns, tool {t}", flush=True)

    th = [threading.Thread(target=worker) for _ in range(args.workers)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    if proc is not None:
        proc.terminate()
    print('tool conversations: done', flush=True)


if __name__ == '__main__':
    main()
