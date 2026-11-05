"""Phone-class hearing on a real call: Moonshine streaming on the processor, fed at real-time pace.

Feeds a call's person track (16 kHz) through app/hearing.py as the app would, asks for the words
at each of the person's pauses (from analysis.json), and prints how long each answer took and the
words, next to the voice model's transcript of the same stretch.

  python -I tools/try_moonshine.py sessions/<call> [--arch small-streaming] [--seconds 90]
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.hearing import MoonshineHearing  # noqa: E402


async def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser()
    ap.add_argument('session')
    ap.add_argument('--arch', default='small-streaming')
    ap.add_argument('--seconds', type=float, default=90)
    ap.add_argument('--keyterms', default='Alba,Claude')
    ap.add_argument('--live-s', type=float, default=0.3, help='live transcript every this many seconds')
    ap.add_argument('--stream-s', type=float, default=0.3, help="the stream's own passes (0: none)")
    ap.add_argument('--pad', type=float, default=0.3, help='seconds of silence added before asking (as the app does)')
    args = ap.parse_args()
    S = Path(args.session)
    said = json.loads((S / 'analysis.json').read_text(encoding='utf8'))['said']
    x, sr = sf.read(S / 'person_clean16k.flac', dtype='float32')
    assert sr == 16000
    x = x[:int(args.seconds * sr)]
    notes = []
    loop = asyncio.get_running_loop()
    t = time.perf_counter()
    h = await loop.run_in_executor(None, lambda: MoonshineHearing(
        args.arch, keyterms=[k for k in args.keyterms.split(',') if k], note=lambda k, **kw: notes.append((k, kw)),
        update_s=args.live_s, stream_update_s=args.stream_s or None))
    print(f'{args.arch} loaded in {time.perf_counter() - t:.1f}s')
    h.loop = loop
    live = []
    h.on_live = lambda text: live.append(text)
    ends = [s['end'] for s in said if s['end'] <= args.seconds]
    block = 512
    t0 = time.perf_counter()
    results = []
    k = 0
    for i in range(0, len(x), block):
        pcm = (np.clip(x[i:i + block], -1, 1) * 32767).astype('<i2').tobytes()
        h.append(pcm)
        now = (i + block) / sr
        while k < len(ends) and now >= ends[k] + 0.1:      # their pause: ask for the words so far
            fut = loop.create_future()
            if args.pad:
                h.append(bytes(2 * int(args.pad * sr)))
            asked = time.perf_counter()
            done = {}
            fut.add_done_callback(lambda f, d=done: d.setdefault('t', time.perf_counter()))
            h.commit(fut)
            results.append((ends[k], asked, fut, done))
            k += 1
        wait = t0 + now - time.perf_counter()
        if wait > 0:
            await asyncio.sleep(wait)
    for end, asked, fut, done in results:
        await fut
    wait = [1000 * (d['t'] - asked) for _, asked, _, d in results]
    ms = [kw['ms'] for kind, kw in notes if kind == 'hearing_commit']
    print(f'pauses: {len(results)} | words in hand after asking: median {np.median(wait):.0f} ms, slowest '
          f'{max(wait):.0f} ms (the pass itself: median {np.median(ms):.0f} ms) | live updates {len(live)}')
    for (end, asked, fut, done), s in zip(results, [s for s in said if s['end'] <= args.seconds]):
        print(f"\n[{int(end // 60):02d}:{end % 60:04.1f}] moonshine: {fut.result()}\n          voice model: {s['text']}")
    h.close()


if __name__ == '__main__':
    asyncio.run(main())
