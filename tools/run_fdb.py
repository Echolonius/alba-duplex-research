"""Full-Duplex-Bench for Alba Duplex (app/duplex.py): feeds every input of the benchmark's offline tasks (v1.0 and
v1.5 of github.com/DanielLin94144/Full-Duplex-Bench) through the duplex model in 80 ms steps and writes what Alba
says where the benchmark's own scorers read it, so her scores sit on the same scale as Moshi's and PersonaPlex's.

The streaming is tools/run_timing_duplex.py's own (run_track), so the clock is the same: a fresh call per input,
and Alba's sound from step k placed DELAY steps (240 ms) after the step began, when a listener would hear it.

  in    every input.wav (v1.0 and v1.5) and clean_input.wav (v1.5: the same turn without the overlap) under --data,
        at any sample rate (mixed to mono, resampled to 16 kHz)
  out   output.wav or clean_output.wav next to each: mono, 16-bit, 16 kHz, exactly as long as its input (sound due
        after the input ends is cut, as the benchmark's own model scripts do)
  log   results/fdb/<tag>/speed.jsonl: one row per finished input (real-time factor, slowest step, late frames,
        starts). A rerun skips finished inputs, so a stopped run picks up where it left off.

Scoring is the benchmark's own: v1_v1.5/get_transcript/asr.py, then v1_v1.5/evaluation (evaluate.py per task, and
get_timing.py for v1.5). Its v2 and v3 talk to a live model (an examiner over WebRTC, tool calls over LiveKit), so
they are not covered here.

  .venv-train/Scripts/python.exe -I tools/run_fdb.py --model models/alba-duplex --data data/fdb/v1_0 --tag duplex
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
from app.duplex import DELAY, FIRST_STEP, AlbaDuplex  # noqa: E402
from run_items_duplex import to16  # noqa: E402
from run_timing_duplex import model_settings, run_track  # noqa: E402

INPUTS = ('input.wav', 'clean_input.wav')
RATE = 16000           # the outputs' sample rate: every scorer of the benchmark works at 16 kHz


def find_inputs(data):
    """Every benchmark input under data: the whole download, one version, one task or one sample folder."""
    return sorted(p for name in INPUTS for p in Path(data).rglob(name))


def output_of(p):
    """input.wav -> output.wav, clean_input.wav -> clean_output.wav, in the same folder."""
    return p.with_name(p.name.replace('input.wav', 'output.wav'))


def key_of(p):
    """task/ID/file: unique across v1.0 and v1.5 (their task names differ), and no local paths in the log."""
    return '/'.join(p.parts[-3:])


def respond(alba, x, sr, delay, listen_bias):
    """One input (samples, or samples x channels, at any rate) -> (Alba's side at 16 kHz on the input's clock and
    exactly as long as the input; the speed of the run). The model is anything with AlbaDuplex's streaming
    interface, as run_track uses it: start(), step(pcm, listen_bias=...) -> the sound made in that step (24 kHz),
    state ('listen' while quiet) and events (('start', step, ...) when it begins to talk)."""
    x = np.asarray(x, np.float32)
    n_out = round(len(x) * RATE / sr)
    x16 = to16(x if x.ndim == 1 else x.mean(axis=1), sr)
    if len(x16) < FIRST_STEP:               # less than one step: nothing heard, nothing said
        return np.zeros(n_out, np.float32), {'steps': 0, 'rtf': 0.0, 'p99_step_ms': 0.0, 'max_step_ms': 0.0,
                                             'late_frames': 0, 'starts': 0}
    voice24, t, late = run_track(alba, x16, delay, listen_bias)
    y = signal.resample_poly(voice24, 2, 3)[:n_out]
    speed = {'steps': len(t), 'rtf': round(float(t.sum() / (len(t) * 0.08)), 4),
             'p99_step_ms': round(float(np.percentile(t, 99) * 1000), 1),
             'max_step_ms': round(float(t.max() * 1000), 1), 'late_frames': late,
             'starts': sum(1 for e in alba.events if e[0] == 'start')}
    return np.pad(y, (0, n_out - len(y))).astype(np.float32), speed


def run_fdb(make_model, data, log, delay=DELAY, listen_bias=0.0, overwrite=False, limit=0):
    """Answers every input under data that has no output from this run yet; make_model() builds the model, only
    if there is something to do. Returns the log rows written this time."""
    data, log = Path(data), Path(log)
    inputs = find_inputs(data)
    if not inputs:
        raise SystemExit(f'no input.wav or clean_input.wav under {data}')
    rows = []
    if log.exists():
        rows = [json.loads(line) for line in log.read_text(encoding='utf8').splitlines() if line.strip()]
    other = {(r['delay_steps'], r['listen_bias']) for r in rows} - {(delay, listen_bias)}
    if other:
        raise SystemExit(f'{log} was made with other settings (delay steps, listen bias: {sorted(other)}); '
                         'use another --tag')
    done = {r['sample'] for r in rows}
    todo = []
    for p in inputs:
        if output_of(p).exists() and not overwrite:
            if key_of(p) not in done:
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
            y, speed = respond(alba, x, sr, delay, listen_bias)
            row = {'sample': key_of(p), 'seconds': round(len(x) / sr, 2), **speed, 'delay_steps': delay,
                   'listen_bias': listen_bias, 'ran_at': time.strftime('%Y-%m-%d %H:%M:%S')}
            # written whole, then logged, then put in place: a stopped run never leaves half an output, or an
            # output without its row (which a rerun would take for another model's)
            tmp = output_of(p).with_suffix('.tmp.wav')
            sf.write(tmp, np.clip(y, -1, 1), RATE, subtype='PCM_16')
            with open(log, 'a', encoding='utf8') as f:
                f.write(json.dumps(row) + '\n')
            tmp.replace(output_of(p))
            new.append(row)
            print(f"[{i + 1}/{len(todo)} {row['sample']}] {row['seconds']:.0f}s | real-time factor {row['rtf']:.2f} | "
                  f"p99 step {row['p99_step_ms']:.0f} ms | late frames {row['late_frames']} | starts {row['starts']}",
                  flush=True)
    finally:
        if alba is not None:
            alba.close()
    print(f'{len(new)} inputs answered now; {sum(output_of(p).exists() for p in inputs)} of {len(inputs)} have an '
          f'output', flush=True)
    return new


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True,
                    help='folder with model-Q4_K_M.gguf, hf/ and duplex.json (tools/export_alba.py)')
    ap.add_argument('--data', required=True, help="the benchmark's data (the whole download, one version, one task "
                    "or one sample folder); outputs are written into it, so use one copy per model or setting")
    ap.add_argument('--tag', default='duplex', help='names the log: results/fdb/<tag>/speed.jsonl')
    ap.add_argument('--listen-bias', type=float, default=None, help="default: the trained model's own (duplex.json)")
    ap.add_argument('--limit', type=int, default=0, help='answer at most this many inputs this time')
    ap.add_argument('--overwrite', action='store_true', help='answer again inputs that already have an output')
    ap.add_argument('--cpu', action='store_true', help='processor only: no graphics card at all')
    args = ap.parse_args()
    md = ROOT / args.model
    st = model_settings(md)
    delay = int(st.get('delay_steps', DELAY))
    bias = args.listen_bias if args.listen_bias is not None else float(st.get('listen_bias', 0.0))

    def make_model():
        if args.cpu:
            return AlbaDuplex(md / 'model-Q4_K_M.gguf', md / 'hf', device='cpu', gpu_layers=0, threads=6)
        return AlbaDuplex(md / 'model-Q4_K_M.gguf', md / 'hf')

    run_fdb(make_model, ROOT / args.data, ROOT / 'results' / 'fdb' / args.tag / 'speed.jsonl', delay, bias,
            args.overwrite, args.limit)


if __name__ == '__main__':
    main()
