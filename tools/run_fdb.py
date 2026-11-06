"""Full-Duplex-Bench for Alba Duplex (app/duplex.py): feeds every input of the benchmark's offline tasks (v1.0 and
v1.5 of github.com/DanielLin94144/Full-Duplex-Bench) through the duplex model in 80 ms steps and writes what Alba
says where the benchmark's own scorers read it, so her scores sit on the same scale as Moshi's and PersonaPlex's
(for a run that keeps up with real time: see late frames below).

The streaming is tools/run_timing_duplex.py's own (run_track), so the clock is the same: a fresh call per input,
and Alba's sound from step k placed DELAY steps (240 ms) after the step began, when a listener would hear it.

  in    every input.wav (v1.0 and v1.5) and clean_input.wav (v1.5: the same turn without the overlap) under --data,
        at any sample rate (mixed to mono, resampled to 16 kHz)
  out   output.wav or clean_output.wav next to each: mono, 16-bit, 16 kHz, exactly as long as its input (sound due
        after the input ends is cut, as the benchmark's own model scripts do)
  log   results/fdb/<tag>/speed.jsonl: one row per finished input (real-time factor, slowest step, late frames,
        starts, seed) and what made it: the model's fingerprint, the device and the settings. A rerun skips
        finished inputs, so a stopped run picks up where it left off; a log made with another model, device or
        setting stops the run instead of mixing its outputs in (--force continues it anyway).

Late frames: a step slower than its budget makes its frame late (a gap a listener would hear), but the sound is
still placed on time, as in the timing harness. Such an input is logged with real_time false and the run ends with
a warning: its timing is what the model would do on a machine fast enough for it, not what this one gives (--cpu).

Repeatable: every input opens the call with the same words (the trained opening without its clock line) and draws
its random numbers from its own seed, after a warm-up that does the fast paths' one-time set-up (which also draws
random numbers), so a rerun on the same machine gives the same outputs, however the inputs are split between runs.

Scoring is the benchmark's own (tools/README.md has the steps): v1_v1.5/get_transcript/asr.py, then
v1_v1.5/evaluation (evaluate.py per task, and get_timing.py for v1.5). Its v2 and v3 talk to a live model (an
examiner over WebRTC, tool calls over LiveKit), so they are not covered here.

  .venv-train/Scripts/python.exe -I tools/run_fdb.py --model models/alba-duplex --data data/fdb/v1_0 --tag duplex
"""
import argparse
import hashlib
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy import signal

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
from app.duplex import DELAY, FIRST_STEP, STEP, AlbaDuplex, system_prompt  # noqa: E402
from run_items_duplex import to16  # noqa: E402
from run_timing_duplex import model_settings, run_track  # noqa: E402

INPUTS = ('input.wav', 'clean_input.wav')
RATE = 16000           # the outputs' sample rate: every scorer of the benchmark works at 16 kHz
OPENING = system_prompt(clock='')    # the trained opening without its clock line, as most training calls had it
MODEL_FILES = ('model-Q4_K_M.gguf', 'trained.safetensors', 'duplex.json')    # and every file under hf/


def find_inputs(data):
    """Every benchmark input under data: the whole download, one version, one task or one sample folder."""
    return sorted(p for name in INPUTS for p in Path(data).rglob(name))


def output_of(p):
    """input.wav -> output.wav, clean_input.wav -> clean_output.wav, in the same folder."""
    return p.with_name(p.name.replace('input.wav', 'output.wav'))


def key_of(p):
    """task/ID/file: unique across v1.0 and v1.5 (their task names differ), and no local paths in the log."""
    return '/'.join(p.parts[-3:])


def seed_of(p):
    """An input's own random seed, from its task and sample folder: the same in every run, and shared by both inputs
    of a v1.5 sample, so any difference between their two outputs comes from the overlap."""
    return zlib.crc32('/'.join(p.parts[-3:-1]).encode('utf8'))


def fingerprint(md):
    """The model's identity: a hash over the name and SHA-256 of every file the runtime loads from its folder (the
    GGUF, trained.safetensors, duplex.json and hf/), so two exports of the same size still differ. Names are
    relative to the folder: no local paths. The untouched base release a slim export builds on is the same for
    every export and is left out."""
    md = Path(md)
    files = [md / name for name in MODEL_FILES if (md / name).is_file()]
    if (md / 'hf').is_dir():
        files += sorted(p for p in (md / 'hf').rglob('*') if p.is_file())
    if not files:
        raise SystemExit(f'no model files in {md.name}/ ({", ".join(MODEL_FILES)} or hf/): see tools/export_alba.py')
    whole = hashlib.sha256()
    for p in files:
        with open(p, 'rb') as f:
            digest = hashlib.file_digest(f, 'sha256').hexdigest()
        whole.update(f'{p.relative_to(md).as_posix()} {digest}\n'.encode('utf8'))
    return whole.hexdigest()[:16]


def respond(alba, x, sr, delay, listen_bias, seed=0):
    """One input (samples, or samples x channels, at any rate) -> (Alba's side at 16 kHz on the input's clock and
    exactly as long as the input; the speed of the run). The model is anything with AlbaDuplex's streaming
    interface, as run_track uses it: start(), step(pcm, listen_bias=...) -> the sound made in that step (24 kHz),
    state ('listen' while quiet) and events (('start', step, ...) when it begins to talk); plus flush() (ends an
    utterance: its held-back sound, and a fresh sound decoder) and rng (its random draws, here started from seed)."""
    x = np.asarray(x, np.float32)
    n_out = round(len(x) * RATE / sr)
    x16 = to16(x if x.ndim == 1 else x.mean(axis=1), sr)
    if len(x16) < FIRST_STEP:               # less than one step: nothing heard, nothing said
        return np.zeros(n_out, np.float32), {'steps': 0, 'rtf': 0.0, 'p99_step_ms': 0.0, 'max_step_ms': 0.0,
                                             'late_frames': 0, 'real_time': True, 'starts': 0}
    alba.rng = np.random.default_rng(seed)  # listen or talk, and the plan's words
    torch.manual_seed(seed)                 # the speech frames (every device's generator, CUDA graphs included)
    voice24, t, late = run_track(alba, x16, delay, listen_bias)
    # a reply the input's end cut off: its held-back sound is dropped (due after the end anyway) and the sound
    # decoder starts afresh. start() does not reset the fast decoder, so without this the next input's first reply
    # would begin 20 ms late, after this one's leftover sound.
    alba.flush()
    y = signal.resample_poly(voice24, 2, 3)[:n_out]
    speed = {'steps': len(t), 'rtf': round(float(t.sum() / (len(t) * 0.08)), 4),
             'p99_step_ms': round(float(np.percentile(t, 99) * 1000), 1),
             'max_step_ms': round(float(t.max() * 1000), 1), 'late_frames': late, 'real_time': late == 0,
             'starts': sum(1 for e in alba.events if e[0] == 'start')}
    return np.pad(y, (0, n_out - len(y))).astype(np.float32), speed


def run_fdb(make_model, data, log, delay=DELAY, listen_bias=0.0, overwrite=False, limit=0, model='', device='',
            force=False):
    """Answers every input under data that has no output from this run yet; make_model() builds the model, only
    if there is something to do. model and device (the model's fingerprint, 'cuda' or 'cpu') are logged with the
    settings for every output, and a log made with others is not continued unless force. Returns the log rows
    written this time."""
    data, log = Path(data), Path(log)
    inputs = find_inputs(data)
    if not inputs:
        raise SystemExit(f'no input.wav or clean_input.wav under {data}')
    made = {}                   # sample -> its latest row, which says what made its output (--overwrite adds rows)
    if log.exists():
        for line in log.read_text(encoding='utf8').splitlines():
            if line.strip():
                r = json.loads(line)
                made[r['sample']] = r
    this = {'model': model, 'device': device, 'delay_steps': delay, 'listen_bias': listen_bias}
    other = sorted({f'{k} {r.get(k)}, now {v}' for r in made.values() for k, v in this.items() if r.get(k) != v})
    if other:
        if not force:
            raise SystemExit(f'{log} was made with another model, device or setting ({"; ".join(other)}): use '
                             'another --tag and a fresh copy of the data, or --force to continue it anyway')
        print(f'--force: continuing a log made with another model, device or setting ({"; ".join(other)})',
              flush=True)
    todo = []
    for p in inputs:
        if output_of(p).exists() and not overwrite:
            if key_of(p) not in made:
                raise SystemExit(f'{key_of(p)}: {output_of(p).name} is already there but not in {log}, so it comes '
                                 'from another model or run: use a fresh copy of the data, or --overwrite')
            continue
        todo.append(p)
    todo = todo[:limit or None]
    log.parent.mkdir(parents=True, exist_ok=True)
    alba, new = None, []
    try:
        for i, p in enumerate(todo):
            if alba is None:
                alba = make_model()
            x, sr = sf.read(p, dtype='float32')
            y, speed = respond(alba, x, sr, delay, listen_bias, seed_of(p))
            row = {'sample': key_of(p), 'seconds': round(len(x) / sr, 2), **speed, **this, 'seed': seed_of(p),
                   'ran_at': time.strftime('%Y-%m-%d %H:%M:%S')}
            # written whole, then logged, then put in place: a stopped run never leaves half an output, or an
            # output without its row (which a rerun would take for another model's)
            tmp = output_of(p).with_suffix('.tmp.wav')
            sf.write(tmp, np.clip(y, -1, 1), RATE, subtype='PCM_16')
            with open(log, 'a', encoding='utf8') as f:
                f.write(json.dumps(row) + '\n')
            tmp.replace(output_of(p))
            new.append(row)
            print(f"[{i + 1}/{len(todo)} {row['sample']}] {row['seconds']:.0f}s | real-time factor {row['rtf']:.2f} | "
                  f"p99 step {row['p99_step_ms']:.0f} ms | late frames {row['late_frames']}"
                  f"{'' if row['real_time'] else ' (not real-time)'} | starts {row['starts']}", flush=True)
    finally:
        if alba is not None:
            alba.close()
    have = [p for p in inputs if output_of(p).exists()]
    print(f'{len(new)} inputs answered now; {len(have)} of {len(inputs)} have an output', flush=True)
    made.update((r['sample'], r) for r in new)
    late = sum(1 for p in have if made.get(key_of(p), {}).get('late_frames', 0) > 0)
    if late:
        print(f'not real-time: {late} of these outputs had late frames (steps slower than their budget). Their sound '
              'is placed as if on time, so their timing is what the model would do on a machine fast enough for it, '
              'not what this one gives (tools/README.md).', flush=True)
    return new


class BenchAlba(AlbaDuplex):
    """AlbaDuplex as the benchmark runs it: every input opens the call with the same words (OPENING, not the day
    and time of the run), and warm_up() does the fast paths' one-time set-up before the first input."""

    def start(self, system=None, **kw):
        super().start(system=OPENING if system is None else system, **kw)

    def warm_up(self):
        """A few steps of silence, the last three with a forced reply, so that capturing the CUDA graphs (slow, and
        it draws random numbers) happens here rather than in an input: no input's output or late frames then
        depend on whether it came first in a run."""
        self.start()
        self.step(np.zeros(FIRST_STEP, np.float32))
        for _ in range(3):
            self.step(np.zeros(STEP, np.float32), force_start=True)
        self.flush()


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, help='folder made by tools/export_alba.py: model-Q4_K_M.gguf, '
                    'trained.safetensors (or hf/) and duplex.json')
    ap.add_argument('--data', required=True, help="the benchmark's data (the whole download, one version, one task "
                    "or one sample folder); outputs are written into it, so use one copy per model or setting")
    ap.add_argument('--tag', default='duplex', help='names the log: results/fdb/<tag>/speed.jsonl')
    ap.add_argument('--listen-bias', type=float, default=None, help="default: the trained model's own (duplex.json)")
    ap.add_argument('--limit', type=int, default=0, help='answer at most this many inputs this time')
    ap.add_argument('--overwrite', action='store_true', help='answer again inputs that already have an output')
    ap.add_argument('--force', action='store_true', help='continue a log made with another model, device or '
                    'setting (its outputs are kept, unless --overwrite)')
    ap.add_argument('--cpu', action='store_true', help='processor only: no graphics card at all (not real-time: '
                    'expect late replies, see tools/README.md)')
    args = ap.parse_args()
    md = ROOT / args.model
    st = model_settings(md)
    delay = int(st.get('delay_steps', DELAY))
    bias = args.listen_bias if args.listen_bias is not None else float(st.get('listen_bias', 0.0))
    device = 'cpu' if args.cpu else 'cuda'
    model = fingerprint(md)
    print(f'model fingerprint {model}, device {device}, delay {delay} steps, listen bias {bias}', flush=True)

    def make_model():
        if args.cpu:
            alba = BenchAlba(md / 'model-Q4_K_M.gguf', md / 'hf', device='cpu', gpu_layers=0, threads=6)
        else:
            alba = BenchAlba(md / 'model-Q4_K_M.gguf', md / 'hf')
        alba.warm_up()
        return alba

    run_fdb(make_model, ROOT / args.data, ROOT / 'results' / 'fdb' / args.tag / 'speed.jsonl', delay, bias,
            args.overwrite, args.limit, model, device, args.force)


if __name__ == '__main__':
    main()
