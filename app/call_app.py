"""A small call window: Start call, a status line, a call timer, End call.

Everything about starting the voice is handled after the click. Both sides of the
call are written to sessions/<date-time>/ while it happens (person.flac, voice.flac,
events.jsonl), and session.json is written when the call ends cleanly.

  .venv\\Scripts\\pythonw.exe -m app.call_app          (no console window)
  python -m app.call_app --simulate                    (self-test: no mic, no speakers)
"""
import argparse
import asyncio
import json
import queue
import re
import sys
import threading
import time
import tkinter as tk
from datetime import datetime

from . import server, sounds, speech
from .audio import LiveAudio, Playback, SimAudio
from .conductor import Conductor
from .config import ROOT, Config, apply_settings
from .main import PERSONA, voice_instructions
from .recorder import DiskRecorder

BG, FG, DIM, ACCENT, LIVE, WARN = '#16181b', '#eceeef', '#8d949a', '#7fa2ff', '#3cc23c', '#fab219'

# The voices to choose from: Pocket TTS's built-in voices (phone-class, on the processor) and the
# voice model's own voice used until November 2026. A new voice makes its short sounds on first use.
VOICES = {'Alba': 'alba', 'Anna': 'anna', 'Caro': 'caro_davy', 'Cosette': 'cosette', 'Eve': 'eve',
          'Javert': 'javert', 'Jean': 'jean', 'Marius': 'marius', 'Michael': 'michael', 'Vera': 'vera',
          'Old voice (LFM3)': None}


class CallApp:
    def __init__(self, root, simulate=False, settings=(), voice='Alba'):
        self.root, self.simulate = root, simulate
        self.settings = list(settings)          # --set overrides for app/config.py
        self.msgs = queue.Queue()
        self.state = 'ready'           # ready | starting | live | saving | saved | failed
        self.stop_call = None
        self.conductor = None
        self.t_live = None
        self.folder = None
        self.close_when_saved = False

        root.title('LocalVoice')
        root.configure(bg=BG)
        root.geometry('900x620')
        root.minsize(700, 520)
        font = 'Segoe UI'
        # Things Claude would like tried in the next call (sessions/try_next.txt, written before each
        # test call): eighth live call, "add a little annotation bar in the top left".
        tips = tk.Frame(root, bg='#1d2024', width=250)
        tips_path = ROOT / 'sessions' / 'try_next.txt'
        lines = []
        if tips_path.exists():
            lines = [ln.strip() for ln in tips_path.read_text(encoding='utf8').splitlines() if ln.strip()]
        if lines:
            tips.pack(side='left', fill='y', anchor='n')
            tips.pack_propagate(False)
            tk.Label(tips, text='Claude would like you to try', bg='#1d2024', fg=ACCENT, font=(font, 10, 'bold'),
                     anchor='w', justify='left').pack(fill='x', padx=14, pady=(16, 6))
            for ln in lines:
                tk.Label(tips, text='• ' + ln, bg='#1d2024', fg=FG, font=(font, 9), anchor='w', justify='left',
                         wraplength=220).pack(fill='x', padx=14, pady=(0, 8))
        main = tk.Frame(root, bg=BG)
        main.pack(side='left', fill='both', expand=True)
        self.main = main
        tk.Label(main, text='LocalVoice', bg=BG, fg=FG, font=(font, 20, 'bold')).pack(pady=(22, 0))
        tk.Label(main, text='Runs on this computer only · the call is recorded here', bg=BG, fg=DIM,
                 font=(font, 10)).pack()
        self.status = tk.Label(main, text='Ready when you are.', bg=BG, fg=FG, font=(font, 13))
        self.status.pack(pady=(26, 4))
        self.timer = tk.Label(main, text='00:00', bg=BG, fg=DIM, font=('Consolas', 34, 'bold'))
        self.timer.pack()
        self.level = tk.Canvas(main, width=300, height=8, bg='#2a2e33', highlightthickness=0)
        self.level.pack(pady=(6, 0))
        self.level_bar = self.level.create_rectangle(0, 0, 0, 8, fill=LIVE, width=0)
        self.headphones = tk.BooleanVar(value=False)
        self.hp_box = tk.Checkbutton(main, text="I'm wearing headphones", variable=self.headphones, bg=BG, fg=FG,
                                     selectcolor=BG, activebackground=BG, activeforeground=FG, font=(font, 10))
        self.hp_box.pack(pady=(14, 0))
        row = tk.Frame(main, bg=BG)
        row.pack(pady=(8, 0))
        tk.Label(row, text='Voice', bg=BG, fg=FG, font=(font, 10)).pack(side='left', padx=(0, 8))
        self.voice = tk.StringVar(value=voice if voice in VOICES else 'Alba')
        self.voice_menu = tk.OptionMenu(row, self.voice, *VOICES)
        self.voice_menu.config(bg='#22262b', fg=FG, activebackground='#2a2e33', activeforeground=FG,
                               highlightthickness=0, relief='flat', font=(font, 10), width=16)
        self.voice_menu['menu'].config(bg='#22262b', fg=FG, activebackground=ACCENT, activeforeground='#0b0d10',
                                       font=(font, 10))
        self.voice_menu.pack(side='left')
        self.button = tk.Button(main, text='Start call', command=self.on_button, font=(font, 14, 'bold'),
                                bg=ACCENT, fg='#0b0d10', activebackground=FG, relief='flat', padx=28, pady=8,
                                cursor='hand2')
        self.button.pack(pady=(14, 0))
        # What was just said, both sides ("You: ..." / "Alba: ..."), wrapped to the window's width
        # (seventh call: "change it from saying voice to the voice's name... so it looks cleaner").
        self.talk = tk.Frame(main, bg=BG)
        self.talk.pack(fill='x', padx=28, pady=(18, 0))
        self.you_line = tk.Label(self.talk, text='', bg=BG, fg=DIM, font=(font, 12), justify='left', anchor='w')
        self.you_line.pack(fill='x')
        self.voice_line = tk.Label(self.talk, text='', bg=BG, fg=FG, font=(font, 12), justify='left', anchor='w')
        self.voice_line.pack(fill='x', pady=(6, 0))
        self.said = tk.Label(main, text='', bg=BG, fg=DIM, font=(font, 9), justify='center')   # devices, saved to...
        self.said.pack(pady=(12, 0))
        root.bind('<Configure>', self.on_resize)
        # Feedback, shown once the call has ended: nothing in it is read until Submit is clicked.
        self.fb_frame = tk.Frame(main, bg=BG)
        tk.Label(self.fb_frame, text='How did it feel? Anything off? (type, then Submit)', bg=BG, fg=FG,
                 font=(font, 10)).pack(anchor='w')
        self.fb_text = tk.Text(self.fb_frame, height=6, width=56, wrap='word', bg='#22262b', fg=FG,
                               insertbackground=FG, relief='flat', font=(font, 11), padx=8, pady=6, undo=True)
        self.fb_text.pack(pady=(4, 6), fill='both', expand=True)
        # Usual editing keys: Ctrl/Alt+Backspace and Ctrl+Delete remove a whole word, Ctrl+A selects all,
        # Ctrl+Z / Ctrl+Y undo and redo.
        for key in ('<Control-BackSpace>', '<Alt-BackSpace>'):
            self.fb_text.bind(key, _delete_word_back)
        self.fb_text.bind('<Control-Delete>', _delete_word_forward)
        self.fb_text.bind('<Control-a>', _select_all)
        self.fb_text.bind('<Control-y>', lambda e: (e.widget.event_generate('<<Redo>>'), 'break')[1])
        self.fb_button = tk.Button(self.fb_frame, text='Submit feedback', command=self.submit_feedback,
                                   font=(font, 11, 'bold'), bg=ACCENT, fg='#0b0d10', activebackground=FG,
                                   relief='flat', padx=18, pady=4, cursor='hand2')
        self.fb_button.pack()
        root.protocol('WM_DELETE_WINDOW', self.on_close)
        root.after(100, self.poll)

    def on_resize(self, event):
        if event.widget is self.root:
            width = max(300, self.main.winfo_width() - 60)
            for label in (self.you_line, self.voice_line, self.said):
                label.config(wraplength=width)

    def show_feedback(self):
        if self.root.state() != 'zoomed' and self.root.winfo_height() < 780:
            self.root.geometry(f'{max(900, self.root.winfo_width())}x780')
        self.fb_frame.pack(pady=(14, 14), padx=28, fill='both', expand=True)
        self.fb_text.focus_set()

    def submit_feedback(self):
        text = self.fb_text.get('1.0', 'end').strip() or '(No typed feedback: it is in the call itself.)'
        folder = self.folder or (ROOT / 'sessions' / datetime.now().strftime('%Y%m%d-%H%M%S-feedback'))
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'feedback.txt').write_text(f"submitted {datetime.now().isoformat(timespec='seconds')}\n\n{text}\n",
                                            encoding='utf8')
        self.fb_text.config(state='disabled', fg=DIM)
        self.fb_button.config(state='disabled', text='Feedback sent. Thanks!', bg='#2a2e33', fg=DIM)
        self.status.config(text='Call and feedback saved. You can close this window.', fg=LIVE)

    # ---- button and window --------------------------------------------------
    def on_button(self):
        if self.state == 'ready':
            self.state = 'starting'
            self.button.config(state='disabled', text='Starting…')
            self.hp_box.config(state='disabled')
            self.voice_menu.config(state='disabled')
            headphones = bool(self.headphones.get())      # read here: Tk is not thread-safe
            voice = VOICES.get(self.voice.get())
            threading.Thread(target=lambda: asyncio.run(self.call(headphones, voice)), daemon=True).start()
        elif self.state == 'live':
            self.end_call()

    def end_call(self):
        if self.stop_call is not None and self.state == 'live':
            self.state = 'saving'
            self.status.config(text='Ending the call and saving…', fg=WARN)
            self.button.config(state='disabled', text='Saving…')
            self.stop_call()

    def on_close(self):
        if self.state in ('live', 'starting', 'saving'):
            self.close_when_saved = True
            self.end_call()
        else:
            self.root.destroy()

    # ---- updates from the call thread ----------------------------------------
    def post(self, kind, value=None):
        self.msgs.put((kind, value))

    def poll(self):
        while not self.msgs.empty():
            kind, value = self.msgs.get()
            if kind == 'status':
                self.status.config(text=value, fg=FG)
            elif kind == 'live':
                self.state, self.t_live = 'live', time.time()
                self.status.config(text='Live. Just talk.', fg=LIVE)
                self.timer.config(fg=FG)
                self.button.config(state='normal', text='End call', bg='#e66767')
                self.said.config(text=value)
            elif kind == 'reply':
                line = ' '.join(value.split())
                if line.startswith('You:'):
                    self.you_line.config(text=line[:400])
                else:
                    self.voice_line.config(text=line[:400])
            elif kind == 'saved':
                self.state = 'saved'
                self.status.config(text='Call saved.', fg=LIVE)
                self.button.config(state='disabled', text='Done', bg='#2a2e33', fg=DIM)
                self.said.config(text=value)
                if self.close_when_saved:
                    self.root.after(600, self.root.destroy)
                else:
                    self.show_feedback()
            elif kind == 'failed':
                self.state = 'failed'
                self.status.config(text='Something went wrong (details saved in logs).', fg='#e66767')
                self.button.config(state='disabled', text='Stopped')
                self.said.config(text=value[:300])
                if self.close_when_saved:
                    self.root.after(1500, self.root.destroy)
                else:
                    self.show_feedback()
        if self.state == 'live' and self.t_live:
            s = int(time.time() - self.t_live)
            self.timer.config(text=f'{s // 60:02d}:{s % 60:02d}')
            if self.conductor is not None:
                lvl = max(0.0, min(1.0, (self.conductor.mic_level_db + 70) / 55))
                self.level.coords(self.level_bar, 0, 0, int(300 * lvl), 8)
        self.root.after(100, self.poll)

    # ---- the call itself (own thread, own event loop) -------------------------
    async def call(self, headphones, voice='alba'):
        proc = text_proc = mind_proc = audio = recorder = None
        logs = ROOT / 'logs'
        logs.mkdir(exist_ok=True)
        try:
            cfg = Config()
            apply_settings(cfg, self.settings, log=lambda s: None)
            if voice:
                cfg.brain.speech, cfg.brain.pocket_voice = 'pocket', voice
            else:
                cfg.brain.speech = 'lfm3'
            cfg.voice.instructions = voice_instructions(cfg)
            if server.needs_voice(cfg):
                self.post('status', 'Starting the voice model…')
                proc = await asyncio.get_running_loop().run_in_executor(None, server.start, cfg, logs / 'server.log')
            if cfg.brain.mode == 'split':
                self.post('status', 'Starting the conversation model…')
                text_proc = await asyncio.get_running_loop().run_in_executor(
                    None, server.start_text, cfg, logs / 'server-text.log')
                if cfg.brain.mind:
                    self.post('status', 'Starting the thinking model…')
                    mind_proc = await asyncio.get_running_loop().run_in_executor(
                        None, server.start_mind, cfg, logs / 'server-mind.log')
            speaker = None
            if cfg.brain.mode == 'split' and cfg.brain.speech == 'pocket':
                self.post('status', 'Loading the voice…')
                speaker = await asyncio.get_running_loop().run_in_executor(None, speech.make, cfg)
            bank, particles = await asyncio.get_running_loop().run_in_executor(
                None, lambda: sounds.prepare(cfg, speaker, server.base_url(cfg), status=lambda s: self.post('status', s)))
            self.folder = cfg.paths.sessions / datetime.now().strftime('%Y%m%d-%H%M%S')
            recorder = DiskRecorder(self.folder)
            playback = Playback()
            self.conductor = Conductor(cfg, playback, bank, headphones=headphones,
                                       log=lambda s: self.post('reply', s.strip()),
                                       echo_state=ROOT / 'models' / 'echo_path.npy',
                                       events_path=self.folder / 'events.jsonl', particles=particles, speaker=speaker)
            stop = asyncio.Event()
            loop = asyncio.get_running_loop()
            self.stop_call = lambda: loop.call_soon_threadsafe(stop.set)
            self.post('status', 'Connecting…')
            task = asyncio.create_task(self.conductor.run(stop))
            t0 = time.time()
            while self.conductor.reply_rs is None:
                if task.done():
                    task.result()
                if time.time() - t0 > 30:
                    raise TimeoutError('the voice model did not answer')
                await asyncio.sleep(0.05)
            if self.simulate:
                audio = SimAudio(playback, self.conductor.on_block, recorder, _SilentPerson(), seconds_limit=8)
            else:
                audio = LiveAudio(playback, self.conductor.on_block, recorder)
            mic, spk = audio.start()
            started = time.time()
            self.post('live', f'Microphone: {mic}\nSpeakers: {spk}')
            if self.simulate:
                await asyncio.sleep(6)
                self.stop_call()
            await task
            audio.stop()
            audio = None
            recorder.close()
            meta = {'finished': True, 'started': datetime.fromtimestamp(started).isoformat(timespec='seconds'),
                    'seconds_recorded': round(recorder.seconds, 2), 'headphones': headphones,
                    'microphone': mic, 'speakers': spk, 'model': cfg.paths.model.name, 'design': cfg.brain.mode,
                    'text_model': cfg.brain.text_model.name if cfg.brain.mode == 'split' else None,
                    'speech': f'pocket-{cfg.brain.pocket_voice}' if speaker is not None else 'lfm3',
                    'hearing': cfg.brain.hearing,
                    'events': len(self.conductor.events),
                    'replies': sum(e['type'] == 'respond' for e in self.conductor.events)}
            (self.folder / 'session.json').write_text(json.dumps(meta, indent=1), encoding='utf8')
            self.post('saved', f'Saved {meta["seconds_recorded"] / 60:.1f} min to sessions\\{self.folder.name}')
        except Exception as e:   # keep the window honest about failures
            (logs / 'call_app_error.txt').write_text(f'{datetime.now().isoformat()}\n{e!r}\n', encoding='utf8')
            self.post('failed', repr(e))
        finally:
            if audio is not None:
                audio.stop()
            if recorder is not None and recorder.running:
                recorder.close()
            server.stop(mind_proc)
            server.stop(text_proc)
            server.stop(proc)


def _delete_word_back(event):
    w = event.widget
    if w.tag_ranges('sel'):
        w.delete('sel.first', 'sel.last')
        return 'break'
    m = re.search(r"(\w+['’]?\w*\s*|[^\w\s]+\s*|\s+)$", w.get('1.0', 'insert'))
    if m:
        w.delete(f'insert -{len(m.group(0))}c', 'insert')
    return 'break'


def _delete_word_forward(event):
    w = event.widget
    if w.tag_ranges('sel'):
        w.delete('sel.first', 'sel.last')
        return 'break'
    m = re.match(r"\s*\w+['’]?\w*|\s*[^\w\s]+|\s+", w.get('insert', 'end-1c'))
    if m:
        w.delete('insert', f'insert +{len(m.group(0))}c')
    return 'break'


def _select_all(event):
    event.widget.tag_add('sel', '1.0', 'end-1c')
    return 'break'


class _SilentPerson:
    """Self-test stand-in for a microphone: quiet room noise."""

    def __init__(self):
        import numpy as np
        self.np = np
        self.rng = np.random.default_rng(0)

    def next_block(self, out, now):
        return (self.rng.standard_normal(480) * 1e-4).astype(self.np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--simulate', action='store_true', help='self-test without microphone or speakers')
    ap.add_argument('--voice', default='Alba', choices=sorted(VOICES), help='voice picked when the window opens')
    ap.add_argument('--set', action='append', default=[], metavar='SECTION.NAME=VALUE',
                    help='override a setting from app/config.py, e.g. brain.early_reaction=0')
    args = ap.parse_args()
    root = tk.Tk()
    app = CallApp(root, simulate=args.simulate, settings=args.set, voice=args.voice)
    if args.simulate:
        root.after(500, app.on_button)
    root.mainloop()


if __name__ == '__main__':
    main()
