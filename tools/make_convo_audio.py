"""Step 2 of the audio-native training data: speak the conversations from tools/make_convo_text.py.

The person's lines are spoken by one of nine Pocket TTS voices per conversation (never Alba's), then made to
sound like a laptop microphone in a room: a random level, band limits, a little reverb and a noise floor, so
the trained voice copes with real calls. Alba's lines are spoken by Pocket TTS 'alba' (24 kHz), the voice the
trained model learns to speak with. Pocket TTS weights: kyutai/pocket-tts-without-voice-cloning, CC-BY-4.0.

  python -I tools/make_convo_audio.py [--text data/convo/text.jsonl] [--workers 3] [--threads 2]
Writes data/convo/audio/<id>-<turn>.wav and data/convo/audio.jsonl; skips what is already done.
"""
import argparse
import json
import multiprocessing as mp
import zlib
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PERSON_VOICES = ['anna', 'caro_davy', 'cosette', 'eve', 'javert', 'jean', 'marius', 'michael', 'vera']
ALBA_VOICE = 'alba'


def laptop_mic(x, rate, rng):
    """A clean 16 kHz clip -> as heard by a laptop microphone in a small room."""
    x = np.asarray(x, np.float64)
    # room: a short decaying noise tail, mixed in lightly
    rt60 = rng.uniform(0.12, 0.45)
    n = int(rt60 * rate)
    ir = rng.standard_normal(n) * np.exp(-6.9 * np.arange(n) / n)
    ir = ir / np.sqrt(np.sum(ir ** 2))
    wet = signal.fftconvolve(x, ir)[:len(x)]
    x = x + rng.uniform(0.05, 0.3) * wet
    # band limits of small microphones
    lo, hi = rng.uniform(80, 200), rng.uniform(5500, 7800)
    b, a = signal.butter(2, [lo / (rate / 2), hi / (rate / 2)], 'band')
    x = signal.lfilter(b, a, x)
    # level: speech at -34 .. -18 dBFS (real calls here are about -30)
    voiced = x[np.abs(x) > 1e-3]
    rms = np.sqrt(np.mean(voiced ** 2)) if len(voiced) else 1e-3
    x = x * (10 ** (rng.uniform(-34, -18) / 20) / rms)
    # noise floor
    x = x + rng.standard_normal(len(x)) * 10 ** (rng.uniform(-66, -48) / 20)
    pad = lambda s: np.zeros(int(s * rate))
    x = np.concatenate([pad(rng.uniform(0.05, 0.35)), x, pad(rng.uniform(0.15, 0.45))])
    return np.clip(x, -1, 1).astype(np.float32)


def to16k(x, rate):
    g = np.gcd(16000, rate)
    return signal.resample_poly(np.asarray(x, np.float64), 16000 // g, rate // g).astype(np.float32)


def save(path, x, rate, tries=5):
    """Writes a clip; a passing system error (a moment of full disk or a scanner holding the file, seen 5 Nov)
    is retried instead of ending the whole run. A half-written file is never left behind."""
    import time
    for k in range(tries):
        try:
            tmp = path.with_suffix('.tmp.wav')
            sf.write(tmp, x, rate)
            tmp.replace(path)
            return
        except Exception:
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass
            if k == tries - 1:
                raise
            time.sleep(5 * (k + 1))


def work(args):
    shard, convs, out_dir, threads = args
    from app.speech import PocketVoice
    voices = {}

    def voice(name):
        if name not in voices:
            voices[name] = PocketVoice(name, threads=threads)
        return voices[name]

    rows = []
    for c in convs:
        rng = np.random.default_rng(zlib.crc32(c["id"].encode()))
        pv = PERSON_VOICES[int(rng.integers(len(PERSON_VOICES)))]
        turns = []
        for k, t in enumerate(c['turns']):
            path = out_dir / f"{c['id']}-{k}.wav"
            if t['role'] == 'person':
                if not path.exists():
                    wav, rate = voice(pv).say(t['text'], seed=int(rng.integers(1 << 30)))
                    save(path, laptop_mic(to16k(wav, rate), 16000, rng), 16000)
                turns.append({**t, 'audio': path.name, 'voice': pv})
            else:
                if not path.exists():
                    wav, rate = voice(ALBA_VOICE).say(t['text'], seed=int(rng.integers(1 << 30)))
                    save(path, np.asarray(wav, np.float32), rate)
                turns.append({**t, 'audio': path.name, 'voice': ALBA_VOICE})
        rows.append({**c, 'turns': turns})
    return rows


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser()
    ap.add_argument('--text', default=str(ROOT / 'data' / 'convo' / 'text.jsonl'))
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--threads', type=int, default=2)
    args = ap.parse_args()
    convs = [json.loads(line) for line in open(args.text, encoding='utf8') if line.strip()]
    out_dir = ROOT / 'data' / 'convo' / 'audio'
    out_dir.mkdir(parents=True, exist_ok=True)
    # small parts, and a fresh process for each (the speech library's memory grows with use: ~9 GB per worker
    # after ~400 conversations on 5 Nov), so memory stays bounded
    chunks = [convs[i::args.workers * 16] for i in range(args.workers * 16)]
    jobs = [(i, ch, out_dir, args.threads) for i, ch in enumerate(chunks) if ch]
    rows = []
    with mp.get_context('spawn').Pool(args.workers, maxtasksperchild=1) as pool:
        for k, part in enumerate(pool.imap_unordered(work, jobs)):
            rows += part
            print(f'{k + 1}/{len(jobs)} parts done, {len(rows)} conversations spoken', flush=True)
    rows.sort(key=lambda r: r['id'])
    with open(ROOT / 'data' / 'convo' / 'audio.jsonl', 'w', encoding='utf8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
