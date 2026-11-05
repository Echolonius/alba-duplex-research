"""A local, offline transcript judge for the scorers: Gemma 4 E4B (Apache-2.0, models/text) served by
llama.cpp on 127.0.0.1, asked one yes/no style question at a time, deterministically (temperature 0).
It reads text only (it cannot hear). Its verdicts are scores, never training data.

  with Judge() as j: j.ask("...question... Answer YES or NO.", choices=('YES', 'NO'))
"""
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app import server  # noqa: E402
from app.config import Config  # noqa: E402

PORT = 9946
MODEL = ROOT / 'models' / 'text' / 'gemma-4-E4B-it-Q4_0.gguf'


class Judge:
    def __enter__(self):
        self.cfg = Config()
        self.proc = None
        if not server.healthy(self.cfg, port=PORT):
            self.proc = server._launch(self.cfg, ['-m', str(MODEL), '-c', '4096', '-np', '1'], PORT,
                                       ROOT / 'logs' / 'server-judge.log', 180)
        return self

    def __exit__(self, *a):
        server.stop(self.proc)

    def ask(self, prompt, choices=('YES', 'NO')):
        body = {'messages': [{'role': 'user', 'content': prompt}], 'temperature': 0, 'max_tokens': 8,
                'chat_template_kwargs': {'enable_thinking': False}}
        req = urllib.request.Request(f'http://127.0.0.1:{PORT}/v1/chat/completions', json.dumps(body).encode(),
                                     {'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=120) as r:
            text = json.loads(r.read())['choices'][0]['message']['content'].strip().upper()
        return next((c for c in choices if text.startswith(c)), next((c for c in choices if c in text), None))
