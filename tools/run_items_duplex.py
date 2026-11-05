"""The held-out item tests (battery, redteam, feeling) for Alba Duplex (app/duplex.py), in the same output format
as tools/run_items.py, so the same scorers apply.

Closed loop, in 80 ms steps: the simulated person says a turn, then waits until Alba has answered and been quiet
for 0.8 s (or 12 s pass), then the next turn; after the last turn it listens 6 s (4 s for 'silent' items).
Alba's reply text is her own plan text (the words she wrote ahead of speaking) after the last turn.

  .venv-train/Scripts/python.exe -I tools/run_items_duplex.py --model models/alba-duplex --tag duplex --sets battery,redteam,feeling
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
from app.duplex import DELAY, FIRST_STEP, OUT_STEP, STEP, AlbaDuplex, system_prompt  # noqa: E402
from run_timing_duplex import model_settings  # noqa: E402

TESTS = ROOT / 'data' / 'tests'


def items_of(kind):
    """The items of a test set and their audio files (as tools/run_items.items_of, without the old app)."""
    spec = json.loads((TESTS / kind / 'items.json').read_text(encoding='utf8'))
    out = []
    for it in spec['items']:
        if kind == 'feeling':
            files = [TESTS / kind / it['file']]
        else:
            n = len(it.get('turns') or [1])
            files = [TESTS / kind / 'audio' / f"{it['id']}-{k}.wav" for k in range(n)]
        out.append((it, files))
    return out


def to16(x, sr):
    if sr == 16000:
        return x
    g = np.gcd(16000, sr)
    return signal.resample_poly(x, 16000 // g, sr // g).astype(np.float32)


def run_item(alba, clips, silent=False, listen_bias=0.0, delay=DELAY, clock=None, tools=None, tool_runner=None,
             info=None):
    """Returns (Alba's voice at 24 kHz on the call's clock, her reply text to the last turn, the caller's side).
    tools / tool_runner: as AlbaDuplex.start (default: none). info: a dict that receives the step each turn
    started at ('turn_steps')."""
    alba.start(system=system_prompt(clock), tools=tools, tool_runner=tool_runner)
    turn_steps = [0]
    voice = np.zeros(OUT_STEP * 256, np.float32)
    cursor, n, k, pos, state = 0, 0, 0, 0, 'speak'
    wait_start, last_on, spoke = 0.0, -1.0, False
    last_end_step, starts = None, []          # starts: (step, how much plan text she had before it)
    fed = []                                  # the caller's side as heard (16 kHz), for the listening clips
    while True:
        need = FIRST_STEP if n == 0 else STEP
        t = n * 0.08
        if state == 'speak':
            pcm = clips[k][pos:pos + need]
            pcm = np.pad(pcm, (0, need - len(pcm)))
            pos += need
            if pos >= len(clips[k]):
                state, wait_start, spoke = 'wait', t, False
                if k == len(clips) - 1:
                    last_end_step = n
        else:
            pcm = np.zeros(need, np.float32)
        was, said_before = alba.state, len(alba.said)
        fed.append(pcm)
        y = alba.step(pcm, listen_bias=listen_bias)
        if was == 'listen' and len(alba.said) > said_before:
            starts.append((n, said_before))
        if was == 'listen' and y.size:
            cursor = (n + delay) * OUT_STEP
        if y.size:
            if cursor + len(y) > len(voice):
                voice = np.concatenate([voice, np.zeros(max(len(y), len(voice)), np.float32)])
            voice[cursor:cursor + len(y)] = y
            cursor += len(y)
            if np.sqrt(np.mean(y ** 2)) > 10 ** (-50 / 20):
                last_on, spoke = t + delay * 0.08, True        # when the person hears it
        n += 1
        t = n * 0.08
        if state == 'wait':
            quiet = t - max(last_on, wait_start)
            if k == len(clips) - 1:
                if (spoke and alba.state == 'listen' and t - last_on > 1.5) or t - wait_start > (20 if spoke else (4 if silent else 6)):
                    break
            elif (spoke and alba.state == 'listen' and quiet > 0.8) or t - wait_start > 12:
                k, pos, state = k + 1, 0, 'speak'
                turn_steps.append(n)
    if info is not None:
        info['turn_steps'] = turn_steps
    # her reply: what she planned in the utterances that began from just before the person's last turn ended
    # (people often start a fraction early); "mm-hm"s during that turn are earlier and not counted
    first = next((said for step, said in starts if last_end_step is not None and step >= last_end_step - 4), len(alba.said))
    reply = alba.tok.decode(alba.said[first:], skip_special_tokens=True).strip()
    return voice[:max(cursor, n * OUT_STEP)], reply, np.concatenate(fed)


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--tag', default='duplex')
    ap.add_argument('--sets', default='battery,redteam,feeling')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--tests-root', default='', help='another folder of test sets (e.g. data/tests-fresh)')
    args = ap.parse_args()
    if args.tests_root:
        global TESTS
        TESTS = ROOT / args.tests_root
    md = ROOT / args.model
    st = model_settings(md)
    delay, bias = int(st.get('delay_steps', DELAY)), float(st.get('listen_bias', 0.0))
    alba = AlbaDuplex(md / 'model-Q4_K_M.gguf', md / 'hf')
    for kind in args.sets.split(','):
        dest = ROOT / 'results' / 'tests' / args.tag / kind
        dest.mkdir(parents=True, exist_ok=True)
        rows_path = ROOT / 'results' / 'tests' / args.tag / f'{kind}.jsonl'
        done = {json.loads(l)['id'] for l in open(rows_path, encoding='utf8')} if rows_path.exists() else set()
        todo = [x for x in items_of(kind) if x[0]['id'] not in done][:args.limit or None]
        for it, files in todo:
            clips = [to16(*sf.read(f, dtype='float32')) for f in files]
            t0 = time.time()
            voice, reply, caller = run_item(alba, clips, silent=bool(it.get('silent')), listen_bias=bias, delay=delay,
                                            clock=it.get('clock'))
            sf.write(dest / f"{it['id']}-voice.flac", signal.resample_poly(voice, 2, 3).astype(np.float32), 16000)
            sf.write(dest / f"{it['id']}-caller.flac", caller, 16000)
            row = {'id': it['id'], 'cat': it.get('cat') or it.get('emotion'), 'heard': [], 'replies': [reply] if reply else [],
                   'final_reply': reply, 'first_audio_s': [], 'ran_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                   'seconds': round(time.time() - t0, 1)}
            with open(rows_path, 'a', encoding='utf8') as f:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
            print(f"[{kind}/{it['id']}] -> {reply[:100]!r}", flush=True)
    alba.close()


if __name__ == '__main__':
    main()
