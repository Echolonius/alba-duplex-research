"""Finds clicks and dropouts in what the voice played.

dropout: inside a stretch of speech, the sound falls to near-zero for 15-150 ms and comes back
click:   a sample-to-sample jump far larger than the surrounding audio would produce
  python -I audio_glitches.py <voice.flac> [events.jsonl]
"""
import json, sys
import numpy as np, soundfile as sf
x, fs = sf.read(sys.argv[1], dtype='float64')
ev = [json.loads(l) for l in open(sys.argv[2], encoding='utf8')] if len(sys.argv) > 2 else []
w = int(0.005 * fs)                                  # 5 ms blocks
n = len(x) // w
rms = np.sqrt(np.mean(x[:n * w].reshape(n, w) ** 2, axis=1))
loud = rms > 0.01
drop = []
k = 0
while k < n:
    if not loud[k] and k > 4 and loud[k - 4:k].all():
        j = k
        while j < n and not loud[j]: j += 1
        dur = (j - k) * 0.005
        if 0.015 <= dur <= 0.15 and j < n and loud[j:j + 4].all() and rms[k:j].max() < 0.002:
            drop.append((round(k * 0.005, 3), round(dur, 3)))
        k = j
    k += 1
d = np.abs(np.diff(x))
env = np.convolve(d, np.ones(441) / 441, 'same') + 1e-4
clicks = np.where((d > 0.08) & (d > 25 * env))[0]
groups = []
for c in clicks:
    if not groups or c - groups[-1][-1] > int(0.01 * fs): groups.append([c])
    else: groups[-1].append(c)
clk = [round(g[0] / fs, 3) for g in groups]
print(f'dropouts inside speech: {len(drop)}  {drop[:15]}')
print(f'clicks: {len(clk)}  {clk[:15]}')
def near(t, kinds=('first_audio', 'yield', 'duck', 'unduck', 'agent_end', 'opener', 'opener_word_skipped', 'backchannel', 'trimmed_tail')):
    return [(e['t'], e['type']) for e in ev if abs(e['t'] - t) < 0.25 and e['type'] in kinds]
for t, d_ in drop[:10]:
    print(f'  dropout {t}s ({d_ * 1000:.0f} ms) near events {near(t)}')
for t in clk[:10]:
    print(f'  click {t}s near events {near(t)}')
