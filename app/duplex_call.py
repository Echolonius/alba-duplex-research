"""Call window for Alba Duplex: one model listens and talks at the same time (app/duplex.py), live.

Sound path: microphone (48 kHz, 10 ms blocks) -> echo canceller (laptop speakers) -> 16 kHz -> the model every
80 ms -> Alba's sound, each utterance started exactly DELAY steps after the step that began it (the timing
she was trained with) -> speakers. Both sides are recorded to sessions/<date-time>/ (person.flac, voice.flac,
events.jsonl) only while a call runs; session.json when it ends; feedback.txt from the box after the call.

  .venv-train\\Scripts\\pythonw.exe -m app.duplex_call [--model models/alba-duplex] [--headphones]
  .venv-train\\Scripts\\python.exe -m app.duplex_call --simulate data/tests/timing/<caller>.flac   (no mic)
"""
import argparse
import collections
import json
import queue
import re
import threading
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path

import numpy as np

from .audio import RATE, LiveAudio, Playback, SimAudio
from .dsp import EchoCanceller, Resampler
from .duplex import DELAY, FIRST_STEP, STEP, AlbaDuplex, system_prompt
from .recorder import DiskRecorder

ROOT = Path(__file__).resolve().parent.parent
BG, FG, DIM, ACCENT, LIVE, WARN = '#16181b', '#eceeef', '#8d949a', '#7fa2ff', '#3cc23c', '#fab219'


YES = re.compile(r"\b(yes|yeah|yep|yup|sure|okay|ok|go ahead|do it|please do|sounds good|absolutely|definitely|"
                 r"correct|right|please)\b", re.I)
NO = re.compile(r"\b(no|nope|nah|don'?t|do not|never ?mind|cancel|wait|stop|not yet)\b", re.I)


class ToolRail:
    """Her tools in a live call: the local API's list (app/api.py on 127.0.0.1), each call run through the API.
    A tool that changes something runs only with confirmed=true, and that is set only when the last few
    seconds you said, transcribed on the processor (Moonshine), are a plain yes with no no/wait/cancel in them:
    she asks first (as trained), and this checks your answer again before anything changes."""

    def __init__(self, api='http://127.0.0.1:8765', log=print):
        import urllib.request
        self.api, self.log, self.urllib = api, log, urllib.request
        self.tools, self.tr = {}, None
        self.recent = collections.deque(maxlen=int(5.0 * 16000 / STEP))    # your last ~5 s as heard (16 kHz)
        try:
            with urllib.request.urlopen(f'{api}/v1/tools', timeout=2) as r:
                self.tools = {t['name']: t for t in json.loads(r.read())['items']}
        except Exception:
            self.tools = {}                       # no API running: no tools in this call

    def heard_yes(self):
        if self.tr is None:
            import os
            os.environ.setdefault('MOONSHINE_VOICE_CACHE', str(ROOT / 'models' / 'moonshine'))
            from moonshine_voice import Transcriber, get_model_for_language
            from moonshine_voice.moonshine_api import string_to_model_arch
            mp, arch = get_model_for_language('en', string_to_model_arch('small-streaming'))
            self.tr = Transcriber(model_path=mp, model_arch=arch)
        x = np.concatenate(list(self.recent)) if self.recent else np.zeros(0, np.float32)
        r = self.tr.transcribe_without_streaming(x.tolist(), 16000) if len(x) else None
        text = ' '.join((l.text or '').strip() for l in r.lines) if r else ''
        return bool(YES.search(text)) and not NO.search(text), text

    def run(self, name, args):
        spec = self.tools.get(name)
        if spec is None:
            return {'error': f'there is no tool called {name}'}
        confirmed, heard = False, None
        if spec.get('changes_something'):
            confirmed, heard = self.heard_yes()
            if not confirmed:
                self.log({'type': 'tool_blocked', 'tool': name, 'args': args, 'heard': heard})
                return {'error': 'not done: this changes something and needs a clear yes from them first'}
        req = self.urllib.Request(f'{self.api}/v1/tools/{name}/call',
                                  json.dumps({'arguments': args, 'confirmed': confirmed}).encode(),
                                  {'Content-Type': 'application/json'})
        with self.urllib.urlopen(req, timeout=10) as r:
            out = json.loads(r.read())
        self.log({'type': 'tool', 'tool': name, 'args': args, 'confirmed': confirmed, 'heard': heard, 'result': out})
        return out

    def close(self):
        if self.tr is not None:
            self.tr.close()


class DuplexEngine:
    """Runs one call: audio callback (10 ms) -> step worker (80 ms) -> playback."""

    def __init__(self, model_dir, headphones=False, post=print):
        md = Path(model_dir)
        st = json.loads((md / 'duplex.json').read_text(encoding='utf8')) if (md / 'duplex.json').exists() else {}
        self.delay = int(st.get('delay_steps', DELAY))
        self.bias = float(st.get('listen_bias', 0.0))
        self.tools_trained = bool(st.get('tools'))            # set by training run 4 on (tool-call data)
        self.post = post
        self.alba = AlbaDuplex(md / 'model-Q4_K_M.gguf', md / 'hf')
        self.headphones = headphones
        self.playback = Playback()
        self.mic_rs, self.ref_rs, self.out_rs = Resampler(RATE, 16000), Resampler(RATE, 16000), Resampler(24000, RATE)
        self.aec = EchoCanceller()
        self.echo_state = ROOT / 'models' / 'echo_path.npy'
        if not headphones and self.echo_state.exists():
            try:
                self.aec.load(self.echo_state)
            except Exception:
                pass
        self.aec_removed = collections.deque(maxlen=500)
        self.aec_on = False
        self.inq = queue.Queue()
        self.pulled = 0
        self.running = False
        self.events = []
        self.step_ms = collections.deque(maxlen=250)

    # ---- audio thread, every 10 ms --------------------------------------------------------------------
    def on_block(self, mic48, out48):
        self.pulled += len(out48)
        m, r = self.mic_rs(mic48), self.ref_rs(out48)
        e = m
        if not self.headphones:
            a, _ = self.aec(m, r)
            if float(np.dot(r, r)) > 1e-6 * len(r):
                self.aec_removed.append(10 * np.log10((np.dot(m, m) + 1e-12) / (np.dot(a, a) + 1e-12)))
            removed = float(np.median(self.aec_removed)) if len(self.aec_removed) >= 100 else 0.0
            self.aec_on = removed >= (2.0 if self.aec_on else 4.0)    # many laptop mics cancel on their own
            e = a if self.aec_on else m
        self.inq.put(np.asarray(e, np.float32))

    # ---- step worker ------------------------------------------------------------------------------------
    def run(self, log):
        self.running = True
        # tools only for a model trained to use them (its duplex.json says so) and only with the local API running
        rail = ToolRail(log=log) if self.tools_trained else None
        tools = list(rail.tools.values()) if rail and rail.tools else None
        self.alba.start(system=system_prompt(), tools=tools, tool_runner=rail.run if tools else None)
        if tools:
            log({'t': 0.0, 'type': 'tools', 'names': sorted(rail.tools)})
        buf, k, said_at = np.zeros(0, np.float32), 0, 0
        last_bias = None
        while self.running:
            if self.bias != last_bias:                 # the listen setting, moved with the slider during the call
                last_bias = self.bias
                log({'t': round(k * 0.08, 2), 'type': 'listen_bias', 'value': last_bias})
            need = FIRST_STEP if k == 0 else STEP
            while len(buf) < need and self.running:
                try:
                    buf = np.concatenate([buf, self.inq.get(timeout=0.2)])
                except queue.Empty:
                    pass
            if not self.running:
                break
            pcm, buf = buf[:need], buf[need:]
            if rail is not None:
                rail.recent.append(pcm)
            was = self.alba.state
            t0 = time.perf_counter()
            y = self.alba.step(pcm, listen_bias=self.bias)
            self.step_ms.append(1000 * (time.perf_counter() - t0))
            if was == 'listen' and self.alba.state == 'speak':
                log({'t': round(k * 0.08, 2), 'type': 'start'})
            if was == 'speak' and self.alba.state == 'listen':
                text = self.alba.tok.decode(self.alba.said[said_at:], skip_special_tokens=True).strip()
                said_at = len(self.alba.said)
                log({'t': round(k * 0.08, 2), 'type': 'stop', 'planned': text})
                self.post('said', text)
            if y.size:
                y48 = self.out_rs(y).astype(np.float32)
                if was == 'listen':
                    # a new utterance is heard DELAY steps after the step that began it (as trained)
                    target = int((k + self.delay) * 0.08 * RATE)
                    with self.playback.lock:
                        queued = sum(len(c) for c in self.playback.main)
                    gap = target - (self.pulled + queued)
                    if gap > 0:
                        self.playback.push(np.zeros(gap, np.float32))
                    elif gap < -RATE // 50:
                        log({'t': round(k * 0.08, 2), 'type': 'late', 'ms': round(-1000 * gap / RATE)})
                self.playback.push(y48)
            k += 1
            if k % 25 == 0:
                backlog = (len(buf) + self.inq.qsize() * 160) / 16000
                self.post('speed', (float(np.mean(self.step_ms)), backlog, self.aec_on))
        if rail is not None:
            rail.close()

    def stop(self):
        self.running = False
        if not self.headphones:
            try:
                self.aec.save(self.echo_state)
            except Exception:
                pass


class FilePerson:
    """--simulate: a recorded caller as the microphone (no real mic or speakers)."""

    def __init__(self, path, tail_s=5.0):
        import soundfile as sf
        from scipy import signal
        x, sr = sf.read(path, dtype='float32')
        g = np.gcd(RATE, sr)
        self.x = signal.resample_poly(x, RATE // g, sr // g).astype(np.float32)
        self.n, self.end = 0, len(self.x) + int(tail_s * RATE)

    def next_block(self, out, now):
        if self.n >= self.end:
            return None
        b = self.x[self.n:self.n + 480]
        self.n += 480
        return np.pad(b, (0, 480 - len(b)))


class DuplexCallApp:
    def __init__(self, root, model_dir, headphones=False, simulate=None, listen_bias=None):
        self.root, self.model_dir, self.simulate = root, model_dir, simulate
        st = Path(model_dir) / 'duplex.json'
        self.start_bias = float(listen_bias if listen_bias is not None else
                                (json.loads(st.read_text(encoding='utf8')).get('listen_bias', 0.0) if st.exists() else 0.0))
        self.msgs = queue.Queue()
        self.state, self.engine, self.folder, self.t_live = 'ready', None, None, None
        root.title('Alba (full duplex)')
        root.configure(bg=BG)
        root.geometry('760x520')
        self.main = tk.Frame(root, bg=BG)
        self.main.pack(fill='both', expand=True, padx=28, pady=20)
        tk.Label(self.main, text='Alba', font=('Segoe UI', 26, 'bold'), bg=BG, fg=FG).pack(anchor='w')
        tk.Label(self.main, text='Full duplex: Alba listens while talking. Talk over Alba, pause, go "mm-hm".',
                 font=('Segoe UI', 11), bg=BG, fg=DIM).pack(anchor='w', pady=(0, 12))
        # the listen setting, live: left = Alba starts talking more readily, right = Alba waits and listens more
        self.bias_var = tk.DoubleVar(value=self.start_bias)
        row = tk.Frame(self.main, bg=BG)
        row.pack(anchor='w', fill='x', pady=(0, 8))
        tk.Label(row, text='Alba talks more', font=('Segoe UI', 10), bg=BG, fg=DIM).pack(side='left')
        tk.Scale(row, from_=0.0, to=2.4, resolution=0.1, orient='horizontal', length=260, showvalue=True,
                 variable=self.bias_var, command=self.on_bias, bg=BG, fg=FG, troughcolor='#22262b',
                 highlightthickness=0, font=('Segoe UI', 9)).pack(side='left', padx=8)
        tk.Label(row, text='Alba listens more', font=('Segoe UI', 10), bg=BG, fg=DIM).pack(side='left')
        self.headphones = tk.IntVar(value=1 if headphones else 0)
        self.hp_box = tk.Checkbutton(self.main, text='I use headphones', variable=self.headphones, bg=BG, fg=FG,
                                     selectcolor=BG, activebackground=BG, activeforeground=FG, font=('Segoe UI', 11))
        self.hp_box.pack(anchor='w')
        self.learn = tk.IntVar(value=0)            # off unless you tick it, for each call
        self.learn_box = tk.Checkbutton(self.main, text='Let Alba suggest things to remember from this call '
                                        '(you review each one; nothing is used until you approve it)',
                                        variable=self.learn, bg=BG, fg=FG, selectcolor=BG, activebackground=BG,
                                        activeforeground=FG, font=('Segoe UI', 10))
        self.learn_box.pack(anchor='w')
        self.button = tk.Button(self.main, text='Start call', font=('Segoe UI', 14, 'bold'), bg=ACCENT, fg='#0b0d10',
                                relief='flat', padx=22, pady=8, command=self.on_button)
        self.button.pack(anchor='w', pady=14)
        self.status = tk.Label(self.main, text='Ready.', font=('Segoe UI', 12), bg=BG, fg=FG, anchor='w', justify='left')
        self.status.pack(anchor='w', fill='x')
        self.timer = tk.Label(self.main, text='0:00', font=('Segoe UI', 12), bg=BG, fg=DIM)
        self.timer.pack(anchor='w')
        self.said = tk.Label(self.main, text='', font=('Segoe UI', 12), bg=BG, fg=DIM, anchor='w', justify='left',
                             wraplength=680)
        self.said.pack(anchor='w', fill='x', pady=(10, 0))
        self.speed = tk.Label(self.main, text='', font=('Segoe UI', 9), bg=BG, fg=DIM, anchor='w')
        self.speed.pack(anchor='w', fill='x')
        self.fb_frame = tk.Frame(self.main, bg=BG)
        tk.Label(self.fb_frame, text='How did it feel? (optional)', font=('Segoe UI', 11), bg=BG, fg=FG).pack(anchor='w')
        self.fb_text = tk.Text(self.fb_frame, height=5, font=('Segoe UI', 11), bg='#22262b', fg=FG, insertbackground=FG,
                               relief='flat', wrap='word')
        self.fb_text.pack(fill='both', expand=True, pady=6)
        self.fb_button = tk.Button(self.fb_frame, text='Send feedback', font=('Segoe UI', 11), bg=ACCENT, fg='#0b0d10',
                                   relief='flat', command=self.submit_feedback)
        self.fb_button.pack(anchor='w')
        root.protocol('WM_DELETE_WINDOW', self.on_close)
        self.close_when_saved = False
        self.opt_in = False
        root.after(100, self.poll)

    def post(self, kind, value=None):
        self.msgs.put((kind, value))

    def on_bias(self, value):
        self.start_bias = float(value)
        if self.engine is not None:
            self.engine.bias = float(value)        # read by the step worker every 80 ms (and logged there)

    def on_button(self):
        if self.state == 'ready':
            self.state = 'starting'
            self.button.config(state='disabled', text='Starting…')
            self.hp_box.config(state='disabled')
            self.learn_box.config(state='disabled')
            self.opt_in = bool(self.learn.get())       # read here: Tk is not thread-safe
            threading.Thread(target=self.call, args=(bool(self.headphones.get()),), daemon=True).start()
        elif self.state == 'live':
            self.end_call()

    def call(self, headphones):
        try:
            self.post('status', 'Getting Alba ready (about 20 seconds)…')
            self.engine = eng = DuplexEngine(self.model_dir, headphones=headphones, post=self.post)
            eng.bias = self.start_bias                 # the slider's setting
            self.folder = ROOT / 'sessions' / datetime.now().strftime('%Y%m%d-%H%M%S')
            self.folder.mkdir(parents=True, exist_ok=True)
            rec = DiskRecorder(self.folder, RATE)
            ev = open(self.folder / 'events.jsonl', 'a', encoding='utf8')

            def log(e):
                ev.write(json.dumps(e, ensure_ascii=False) + '\n')
                ev.flush()
            log({'t': 0, 'type': 'session', 'design': 'duplex', 'model': str(self.model_dir), 'delay_steps': eng.delay,
                 'listen_bias': eng.bias, 'headphones': headphones})
            if self.simulate:
                audio = SimAudio(eng.playback, eng.on_block, rec, FilePerson(self.simulate))
            else:
                audio = LiveAudio(eng.playback, eng.on_block, rec)
            worker = threading.Thread(target=eng.run, args=(log,), daemon=True)
            worker.start()
            devices = audio.start()
            self.stop_call = lambda: (audio.stop(), eng.stop())
            self.post('live', f'Mic: {devices[0]} | Speakers: {devices[1]}')
            if self.simulate:
                audio.done.wait()
                self.post('end')
            while eng.running:
                time.sleep(0.2)
            audio.stop()
            worker.join(timeout=5)
            rec.close()
            secs = rec.seconds
            ev.close()
            (self.folder / 'session.json').write_text(json.dumps({
                'design': 'duplex', 'model': str(self.model_dir), 'seconds': round(secs, 1),
                'ended': datetime.now().isoformat(timespec='seconds'), 'delay_steps': eng.delay,
                'listen_bias': eng.bias, 'headphones': headphones, 'learning_opt_in': self.opt_in,
                'step_ms_mean': round(float(np.mean(eng.step_ms)), 1) if eng.step_ms else None}, indent=1), encoding='utf8')
            eng.alba.close()
            if self.opt_in:
                # proposals to review (tools/learn_from_call.py), made on this PC after the call
                import subprocess
                import sys as _sys
                py = Path(_sys.executable).with_name('python.exe')
                lv = ROOT / '.venv' / 'Scripts' / 'python.exe'
                subprocess.Popen([str(lv if lv.exists() else py), '-I', str(ROOT / 'tools' / 'learn_from_call.py'),
                                  str(self.folder)], cwd=str(ROOT),
                                 stdout=open(self.folder / 'learning.log', 'w'), stderr=subprocess.STDOUT)
            self.post('saved')
        except Exception as e:      # shown in the window; the call folder keeps what was recorded
            self.post('error', f'{type(e).__name__}: {e}')

    def end_call(self):
        if self.state == 'live' and self.engine is not None:
            self.state = 'saving'
            self.status.config(text='Ending the call and saving…', fg=WARN)
            self.button.config(state='disabled', text='Saving…')
            self.engine.stop()

    def submit_feedback(self):
        text = self.fb_text.get('1.0', 'end').strip() or '(No typed feedback: it is in the call itself.)'
        folder = self.folder or (ROOT / 'sessions' / datetime.now().strftime('%Y%m%d-%H%M%S-feedback'))
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'feedback.txt').write_text(f"submitted {datetime.now().isoformat(timespec='seconds')}\n\n{text}\n",
                                            encoding='utf8')
        self.fb_text.config(state='disabled', fg=DIM)
        self.fb_button.config(state='disabled', text='Feedback sent. Thanks!')
        self.status.config(text='Call and feedback saved. You can close this window.', fg=LIVE)

    def on_close(self):
        if self.state in ('live', 'starting', 'saving'):
            self.close_when_saved = True
            self.end_call()
        else:
            self.root.destroy()

    def poll(self):
        while not self.msgs.empty():
            kind, value = self.msgs.get()
            if kind == 'status':
                self.status.config(text=value, fg=FG)
            elif kind == 'live':
                self.state, self.t_live = 'live', time.time()
                self.status.config(text='Live. Just talk.', fg=LIVE)
                self.button.config(state='normal', text='End call', bg='#e66767')
                self.speed.config(text=value)
            elif kind == 'said':
                self.said.config(text=f'Alba: {value}' if value else '')
            elif kind == 'speed':
                ms, backlog, aec = value
                warn = '  (falling behind)' if backlog > 0.3 else ''
                self.speed.config(text=f'{ms:.0f} ms per 80 ms step | backlog {backlog:.2f} s | echo canceller '
                                       f'{"on" if aec else "off"}{warn}', fg=WARN if warn else DIM)
            elif kind == 'end':
                self.end_call()
            elif kind == 'saved':
                self.state = 'done'
                self.status.config(text='Call saved. Tell me how it felt below (optional).', fg=LIVE)
                self.button.config(text='Call ended', bg='#2a2e33', fg=DIM)
                self.fb_frame.pack(fill='both', expand=True, pady=(12, 0))
                if self.close_when_saved:
                    self.root.destroy()
                    return
            elif kind == 'error':
                self.state = 'done'
                self.status.config(text=f'Something went wrong: {value}', fg=WARN)
        if self.state == 'live' and self.t_live:
            s = int(time.time() - self.t_live)
            self.timer.config(text=f'{s // 60}:{s % 60:02d}', fg=FG)
        self.root.after(100, self.poll)


def main():
    ap = argparse.ArgumentParser(description='Alba Duplex call window')
    ap.add_argument('--model', default=str(ROOT / 'models' / 'alba-duplex'))
    ap.add_argument('--headphones', action='store_true')
    ap.add_argument('--simulate', default='', help='a recorded caller (any rate, mono) instead of the microphone')
    ap.add_argument('--listen-bias', type=float, default=None, help="the slider's start (default: the model's own)")
    ap.add_argument('--auto-start', action='store_true', help='tests: start the call at once')
    args = ap.parse_args()
    root = tk.Tk()
    app = DuplexCallApp(root, args.model, headphones=args.headphones, simulate=args.simulate or None,
                        listen_bias=args.listen_bias)
    if args.auto_start:
        root.after(500, app.on_button)
    root.mainloop()


if __name__ == '__main__':
    main()
