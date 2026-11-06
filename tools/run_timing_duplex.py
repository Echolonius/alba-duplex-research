"""Timing test (check 2) and speed test (check 5) for Alba Duplex (app/duplex.py): feeds each held-out caller
track (real people, data/tests/timing) through the duplex model in 80 ms steps and records what Alba says on the
same clock, so tools/score_timing.py scores it exactly like the other systems.

Honest clock: step k's sound is complete 95 ms after the step began (80 ms + the encoder's 15 ms window);
Alba's sound from step k is placed DELAY steps after the step began (as the live app plays it). A virtual
wall clock runs the measured compute time of every step one after another: a frame that would not be ready by
its play time counts as late (a gap you would hear), and is reported with the real-time factor (compute time /
audio time; goal <= 0.7).

With --speakers, Alba's own voice is mixed back into what she hears (laptop speakers; check 1: she must not
answer her own echo), as a 40 ms delayed, -10 dB, band-limited copy of what is playing at that moment.

  .venv-train/Scripts/python.exe -I tools/run_timing_duplex.py --model models/alba-duplex --tag duplex [--speakers]

--public runs the public timing set instead (data/tests-public/timing: the frozen set's 12 Seamless tracks, which
anyone can rebuild) and writes to results/tests-public/<tag>/; score it with tools/score_timing.py --public <tag>.
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
from app.duplex import DELAY, FIRST_STEP, OUT_STEP, STEP, AlbaDuplex  # noqa: E402

TESTS = ROOT / 'data' / 'tests' / 'timing'
PUBLIC = ROOT / 'data' / 'tests-public' / 'timing'


def folders(args):
    """(test folder, results folder): the frozen set's, or with --public the public set's (kept apart)."""
    return (PUBLIC, ROOT / 'results' / 'tests-public') if args.public else (TESTS, ROOT / 'results' / 'tests')


def model_settings(md):
    p = Path(md) / 'duplex.json'
    return json.loads(p.read_text(encoding='utf8')) if p.exists() else {}


def echo_path():
    rng = np.random.default_rng(0)
    h = np.zeros(int(0.2 * 16000)); d0 = int(0.04 * 16000)
    h[d0:] = rng.standard_normal(len(h) - d0) * np.exp(-np.arange(len(h) - d0) / (0.03 * 16000))
    h = signal.lfilter(*signal.butter(2, [250 / 8000, 7000 / 8000], 'band'), h)
    return h * (10 ** (-10 / 20)) / np.sqrt(np.sum(h ** 2))


def run_track(alba, x, delay, listen_bias, h=None):
    """One caller track -> (Alba's voice at 24 kHz on the caller's clock, per-step compute times, late frames)."""
    alba.start()
    n = (len(x) - FIRST_STEP) // STEP + 1
    voice24 = np.zeros((n + delay + 4) * OUT_STEP, np.float32)
    cursor, times, late, busy_until = 0, [], 0, 0.0
    for k in range(n):
        lo, hi = (0, FIRST_STEP) if k == 0 else (FIRST_STEP + (k - 1) * STEP, FIRST_STEP + k * STEP)
        pcm = x[lo:hi].copy()
        if h is not None:
            # what the speakers play while this audio is recorded (already final: it was made >= 2 steps ago)
            a, b = max(0, lo - len(h)), hi
            played = signal.resample_poly(voice24[int(a * 1.5):int(b * 1.5)], 2, 3)[:b - a]
            e = np.convolve(np.tanh(2 * played) / 2, h)[:b - a]
            pcm += e[lo - a:].astype(np.float32)[:len(pcm)]
        was = alba.state
        t0 = time.perf_counter()
        y = alba.step(pcm, listen_bias=listen_bias)
        dt = time.perf_counter() - t0
        times.append(dt)
        # virtual wall clock: the step can start once its sound exists and the previous step is done
        done_at = max(hi / 16000, busy_until) + dt
        busy_until = done_at
        if was == 'listen' and y.size:
            cursor = (k + delay) * OUT_STEP
        if y.size:
            if done_at > (k + delay) * 0.08 + 1e-4:
                late += 1
            voice24[cursor:cursor + len(y)] = y[:len(voice24) - cursor]
            cursor += len(y)
    return voice24[:(n + delay) * OUT_STEP], np.array(times), late


def ten_minutes(alba, delay, bias, args, minutes=10.0):
    """One continuous call of `minutes` (held-out callers back to back, 1 s apart): the real-time factor over the
    whole call, as check 5 asks, with the model's memory growing the whole time."""
    tests, results = folders(args)
    items = json.loads((tests / 'items.json').read_text(encoding='utf8'))['items']
    parts, total = [], 0
    for it in items:
        x, sr = sf.read(tests / it['caller'], dtype='float32')
        parts += [x, np.zeros(16000, np.float32)]
        total += len(x) + 16000
        if total >= minutes * 60 * 16000:
            break
    x = np.concatenate(parts)[:int(minutes * 60 * 16000)]
    voice24, t, late = run_track(alba, x, delay, bias)
    tag = args.tag + ('-cpu' if args.cpu else '')
    out = results / tag
    out.mkdir(parents=True, exist_ok=True)
    res = {'minutes': minutes, 'device': 'cpu' if args.cpu else 'cuda', 'rtf': float(t.sum() / (len(t) * 0.08)),
           'rtf_last_minute': float(t[-750:].sum() / (min(750, len(t)) * 0.08)),
           'p99_step_ms': float(np.percentile(t, 99) * 1000), 'max_step_ms': float(t.max() * 1000),
           'late_frames': late, 'steps': len(t), 'memory_rebuilds': sum(1 for e in alba.events if e[0] == 'rebuilt')}
    (out / 'speed_10min.json').write_text(json.dumps(res, indent=1), encoding='utf8')
    print(json.dumps(res), flush=True)
    alba.close()


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, help='folder with model-Q4_K_M.gguf, hf/ and duplex.json (tools/export_alba.py)')
    ap.add_argument('--tag', default='duplex')
    ap.add_argument('--only', default='')
    ap.add_argument('--speakers', action='store_true')
    ap.add_argument('--listen-bias', type=float, default=None, help="default: the trained model's own (duplex.json)")
    ap.add_argument('--ten-minutes', action='store_true',
                    help='check 5: one continuous 10-minute call (held-out callers back to back); speed only')
    ap.add_argument('--cpu', action='store_true', help='processor only (the path to phones): no graphics card at all')
    ap.add_argument('--public', action='store_true',
                    help='the public timing set (data/tests-public/timing), into results/tests-public/<tag>/')
    args = ap.parse_args()
    tests, results = folders(args)
    md = ROOT / args.model
    st = model_settings(md)
    delay = int(st.get('delay_steps', DELAY))
    bias = args.listen_bias if args.listen_bias is not None else float(st.get('listen_bias', 0.0))
    if args.cpu:
        alba = AlbaDuplex(md / 'model-Q4_K_M.gguf', md / 'hf', device='cpu', gpu_layers=0, threads=6)
    else:
        alba = AlbaDuplex(md / 'model-Q4_K_M.gguf', md / 'hf')
    if args.ten_minutes:
        return ten_minutes(alba, delay, bias, args)
    items = json.loads((tests / 'items.json').read_text(encoding='utf8'))['items']
    if args.only:
        items = [i for i in items if i['id'] in args.only.split(',')]
    out = results / (args.tag + ('-speakers' if args.speakers else '')) / 'timing'
    out.mkdir(parents=True, exist_ok=True)
    h = echo_path() if args.speakers else None
    speed = {}
    for it in items:
        x, sr = sf.read(tests / it['caller'], dtype='float32')
        assert sr == 16000, sr
        voice24, t, late = run_track(alba, x, delay, bias, h)
        sf.write(out / f"{it['id']}-voice.flac", signal.resample_poly(voice24, 2, 3).astype(np.float32), 16000)
        n = len(t)
        speed[it['id']] = {'rtf': float(t.sum() / (n * 0.08)), 'p99_step_ms': float(np.percentile(t, 99) * 1000),
                           'max_step_ms': float(t.max() * 1000), 'steps': n, 'late_frames': late}
        print(f"[{it['id']}] {n * 0.08:.0f}s | real-time factor {speed[it['id']]['rtf']:.2f} | p99 step "
              f"{speed[it['id']]['p99_step_ms']:.0f} ms | late frames {late} | starts "
              f"{sum(1 for e in alba.events if e[0] == 'start')}", flush=True)
    allt = [v['rtf'] for v in speed.values()]
    (out.parent / 'speed.json').write_text(json.dumps({'rtf_mean': float(np.mean(allt)), 'delay_steps': delay,
                                                       'listen_bias': bias, 'per_item': speed}, indent=1), encoding='utf8')
    alba.close()


if __name__ == '__main__':
    main()
