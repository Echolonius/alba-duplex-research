"""Collects the numbers the report charts use into one JSON file.

  python -I report_data.py <localvoice_metrics.json> <bench_dir> <out.json> [tag]
"""
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

ROOT = Path(__file__).resolve().parent.parent
dessa = json.loads((ROOT / 'results' / 'dessa_metrics.json').read_text(encoding='utf8'))
local = json.loads(Path(sys.argv[1]).read_text(encoding='utf8'))
bench = Path(sys.argv[2])

bins = np.round(np.arange(-1.0, 1.61, 0.1), 2)


def hist(vals):
    v = np.clip(np.asarray(vals, float), bins[0], bins[-1] - 1e-6)
    h, _ = np.histogram(v, bins=bins)
    return (h / max(1, h.sum())).round(4).tolist()


human = [m for m in dessa if m['speaker1'] == 'Human']
d_resp = sum((m['turn_taking']['floor_transfers']['2'] for m in human), [])
h_resp = sum((m['turn_taking']['floor_transfers']['1'] for m in human), [])
l_resp = sum((m['turn_taking']['floor_transfers']['2'] for m in local), [])


def ltas(path, seconds=120):
    x, fs = sf.read(path, dtype='float64')
    x = x[: int(seconds * fs)]
    f, t, S = signal.stft(x, fs, nperseg=4096, noverlap=2048)
    P = np.abs(S) ** 2
    e = P.sum(axis=0)
    loud = e > np.percentile(e, 95) * 10 ** (-25 / 10)
    l = 10 * np.log10(P[:, loud].mean(axis=1) + 1e-30)
    l -= l[(f > 200) & (f < 4000)].max()
    keep = f <= 16000
    step = 4
    return f[keep][::step].round(0).tolist(), np.maximum(l[keep][::step], -110).round(1).tolist()


fd, ld = ltas(ROOT / 'reference' / 'dessa-audio' / 'life-advice-speaker2.flac')
tag = sys.argv[4] if len(sys.argv) > 4 else ''
voice_file = sorted(bench.glob(f'{tag}*-voice.flac'))[0]
fl, ll = ltas(voice_file)
out = {'bins': bins.tolist(),
       'response_hist': {'Dessa': hist(d_resp), 'People talking to Dessa': hist(h_resp), 'LocalVoice': hist(l_resp)},
       'response_n': {'Dessa': len(d_resp), 'People talking to Dessa': len(h_resp), 'LocalVoice': len(l_resp)},
       'ltas': {'Dessa': {'f': fd, 'db': ld}, 'LocalVoice': {'f': fl, 'db': ll}},
       'dessa_f0_by_conversation': [round(m['tracks'][1]['voice']['f0_median_hz']) for m in human if m['tracks'][1]['voice']],
       'people_f0_by_conversation': [round(m['tracks'][0]['voice']['f0_median_hz']) for m in human if m['tracks'][0]['voice']]}
Path(sys.argv[3]).write_text(json.dumps(out), encoding='utf8')
print('response offsets:', out['response_n'])
