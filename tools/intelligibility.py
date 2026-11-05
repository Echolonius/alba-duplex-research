"""How clearly does the voice say what it meant to say?

For each reply that played to the end in a benchmark run, the voice's audio is
transcribed by the model's own speech recognition, and compared with the text
the model generated alongside that audio. Reports the word error rate.

  python -I intelligibility.py <bench_dir> <tag> [<tag> ...]
"""
import asyncio
import json
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.dsp import Resampler, to_pcm16   # noqa: E402
import websockets                        # noqa: E402


def words(s):
    s = s.lower().replace("'", '')
    return re.findall(r"[a-z0-9]+", s)


def wer(ref, hyp):
    r, h = words(ref), words(hyp)
    d = np.arange(len(h) + 1)
    for i, rw in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, hw in enumerate(h, 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (rw != hw))
            prev, d[j] = d[j], cur
    return d[len(h)], max(1, len(r))


async def transcribe(clips):
    out = []
    async with websockets.connect('ws://127.0.0.1:9931/v1/realtime?intent=transcription', max_size=None) as ws:
        json.loads(await ws.recv())
        await ws.send(json.dumps({'type': 'session.update', 'session': {
            'type': 'transcription', 'audio': {'input': {'transcription': {'language': 'en'}}}}}))
        while json.loads(await ws.recv()).get('type') != 'session.updated':
            pass
        for clip in clips:
            for i in range(0, len(clip), 1600):
                await ws.send(json.dumps({'type': 'input_audio_buffer.append',
                                          'audio': __import__('base64').b64encode(to_pcm16(clip[i:i + 1600])).decode()}))
            await ws.send(json.dumps({'type': 'input_audio_buffer.commit'}))
            text = ''
            while True:
                ev = json.loads(await ws.recv())
                if ev.get('type') == 'conversation.item.input_audio_transcription.completed':
                    text = ev.get('transcript', '')
                    break
                if ev.get('type') == 'error':
                    break
            out.append(text)
    return out


def clips_for(bench, tag):
    pairs = []
    for f in sorted(bench.glob(f'{tag}-*-events.jsonl')):
        sid = f.name[:-len('-events.jsonl')]
        x, fs = sf.read(bench / f'{sid}-voice.flac', dtype='float32')
        ev = [json.loads(l) for l in f.open(encoding='utf8')]
        start = None
        for k, e in enumerate(ev):
            if e['type'] == 'first_audio':
                start = e['t']
            elif e['type'] == 'yield':
                start = None
            elif e['type'] == 'agent_end' and start is not None:
                text = next((g['text'] for g in reversed(ev[:k]) if g['type'] == 'reply_text'), None)
                if text:
                    seg = x[int(start * fs): int((e['t'] + 0.2) * fs)]
                    pairs.append((text, Resampler(fs, 16000)(seg).astype(np.float32)))
                start = None
    return pairs


def main():
    bench = Path(sys.argv[1])
    for tag in sys.argv[2:]:
        pairs = clips_for(bench, tag)
        heard = asyncio.run(transcribe([c for _, c in pairs]))
        errs = [wer(t, h) for (t, _), h in zip(pairs, heard)]
        total = sum(e for e, _ in errs) / max(1, sum(n for _, n in errs))
        print(f"[{tag}] {len(pairs)} replies, word error rate {total * 100:.1f}%")
        for (t, _), h, (e, n) in list(zip(pairs, heard, errs))[:4]:
            print(f"   meant: {t[:90]!r}\n   heard: {h[:90]!r}  ({e}/{n} words off)")


if __name__ == '__main__':
    main()
