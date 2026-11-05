"""Measures the "how it talks" fingerprint of a two-track conversation.

Each conversation is two separate mono tracks (speaker 1 = the other party,
speaker 2 = the voice under test). Everything here is plain signal processing,
so the same numbers can be produced for Dessa's demos and for our own model.

Measured per conversation:
  - bandwidth: where the speech spectrum stops (reveals the true sample rate)
  - levels: speech loudness and the noise floor in pauses
  - timing: talkspurts, gaps, overlaps, response delays, backchannels,
    interruptions and how fast each side yields when talked over
  - voice: pitch level and range, jitter, shimmer, harmonics-to-noise ratio

Usage:
  python -I conversation_metrics.py <recordings.json> <audio_dir> <out.json>
  (recordings.json lists {id, files:[track1, track2], speaker1, speaker2})
"""
import json
import sys
from pathlib import Path

import numpy as np
import parselmouth
import soundfile as sf
from scipy import signal

FS = 16000        # analysis rate; the demos carry no content above 8 kHz
HOP = 0.01        # 10 ms analysis frames
WIN = 0.025
MIN_PAUSE = 0.18  # silences shorter than this stay inside a talkspurt
MIN_SPURT = 0.08  # shorter blips are ignored
BACKCHANNEL_MAX = 1.0


def load(path):
    x, fs = sf.read(path, dtype='float64', always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x, fs


def to16k(x, fs):
    if fs == FS:
        return x
    g = np.gcd(int(fs), FS)
    return signal.resample_poly(x, FS // g, int(fs) // g)


def frame_db(x):
    w, h = int(WIN * FS), int(HOP * FS)
    n = 1 + max(0, (len(x) - w) // h)
    idx = np.arange(w)[None, :] + h * np.arange(n)[:, None]
    return 10 * np.log10(np.mean(x[idx] ** 2, axis=1) + 1e-20)


def bandwidth(x, fs):
    """Frequency where the speech spectrum falls 20 dB below its 4-6 kHz level."""
    f, t, S = signal.stft(x, fs, nperseg=4096, noverlap=3072)
    P = np.abs(S) ** 2
    e = 10 * np.log10(P.sum(axis=0) + 1e-20)
    loud = e > np.percentile(e, 95) - 25
    ltas = 10 * np.log10(P[:, loud].mean(axis=1) + 1e-30)
    ref = ltas[(f >= 4000) & (f < 6000)].mean()
    above = np.where((ltas > ref - 20) & (f > 3000))[0]
    cut = float(f[above[-1]]) if len(above) else float('nan')
    plateau = float(ltas[(f > cut + 1500) & (f < min(fs / 2, cut + 6000))].mean() - ref) if cut == cut else float('nan')
    return cut, plateau


def remove_crosstalk(h, d):
    """Subtracts the part of track h that is a linear echo of track d.

    Some demo tracks carry the other voice too (speaker echo ~0.1 s late, or a
    direct mix with no delay). A single Wiener filter (lags -5 ms .. +300 ms),
    estimated over the whole conversation, predicts that echo from d.
    Returns the cleaned track and how many dB of echo were removed.
    """
    n = min(len(h), len(d))
    h, d = h[:n], d[:n]
    nper = 16384
    _, pdd = signal.welch(d, FS, nperseg=nper, noverlap=nper // 2)
    _, phd = signal.csd(d, h, FS, nperseg=nper, noverlap=nper // 2)
    ir = np.fft.irfft(phd / (pdd + 1e-14 * pdd.max()), nper)
    pre, post = int(0.005 * FS), int(0.3 * FS)
    kernel = np.concatenate([ir[-pre:], ir[:post]])
    echo = signal.fftconvolve(d, kernel)[pre:pre + n]
    cleaned = h - echo
    w = int(0.05 * FS)
    env = np.convolve(d ** 2, np.ones(w) / w, 'same')
    loud = env > np.percentile(env, 90) * 0.1
    before = np.mean(h[loud] ** 2) + 1e-20
    after = np.mean(cleaned[loud] ** 2) + 1e-20
    removed = 10 * np.log10(before / after)
    if removed < 0.3:          # nothing worth removing: keep the original
        return h, 0.0
    return cleaned, float(removed)


def vad(db):
    """Two-threshold energy VAD tuned per track (tracks are separately recorded)."""
    floor, ref = np.percentile(db, 10), np.percentile(db, 97)
    on = max(floor + 12, ref - 32)
    off = on - 6
    act = np.zeros(len(db), bool)
    state = False
    for i, v in enumerate(db):
        state = v > on if not state else v > off
        act[i] = state
    return act, float(on)


def spurts(act):
    """Frame activity -> list of (start_s, end_s) talkspurts."""
    d = np.diff(np.concatenate([[0], act.astype(int), [0]]))
    s, e = np.where(d == 1)[0] * HOP, np.where(d == -1)[0] * HOP
    merged = []
    for a, b in zip(s, e):
        if merged and a - merged[-1][1] < MIN_PAUSE:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged if b - a >= MIN_SPURT]


def active_at(sp, t0, t1):
    """Seconds of speech in sp overlapping [t0, t1]."""
    return sum(max(0.0, min(b, t1) - max(a, t0)) for a, b in sp)


def turn_taking(sp1, sp2, total):
    """Gaps/overlaps between speakers (Heldner & Edlund style) plus backchannels,
    interruptions and yielding. Speaker labels: 1 = other party, 2 = voice under test."""
    ev = sorted([(a, b, 1) for a, b in sp1] + [(a, b, 2) for a, b in sp2])
    # Backchannels: short spurts that start while the other speaker holds the floor
    # and do not take it over (the other is still talking 0.5 s after they end).
    sp = {1: sp1, 2: sp2}
    bc = {1: [], 2: []}
    for a, b, s in ev:
        o = sp[3 - s]
        if b - a <= BACKCHANNEL_MAX and active_at(o, a - 0.6, a) > 0.15 and active_at(o, b, b + 0.8) > 0.15:
            bc[s].append((a, b))
    bcset = {(a, b, s) for s in (1, 2) for a, b in bc[s]}
    # Floor-holding spurts only (backchannels removed) define turns.
    floor = [(a, b, s) for a, b, s in ev if (a, b, s) not in bcset]
    turns = []
    for a, b, s in floor:
        if turns and turns[-1][2] == s and a - turns[-1][1] < 1.5:
            turns[-1][1] = max(turns[-1][1], b)
        else:
            turns.append([a, b, s])
    # Floor transfers: offset = next turn start - previous turn end (negative = overlap).
    fto = {1: [], 2: []}   # keyed by the speaker who takes the floor
    for (a0, b0, s0), (a1, b1, s1) in zip(turns, turns[1:]):
        if s1 != s0:
            fto[s1].append(a1 - b0)
    # Interruptions: a floor-taking spurt that starts while the other is mid-speech
    # (other active for >0.4 s before) and the other stops within 1.5 s.
    intr = {1: 0, 2: 0}
    yield_t = {1: [], 2: []}  # how long the interrupted speaker kept talking
    for a, b, s in floor:
        o = sp[3 - s]
        if b - a > BACKCHANNEL_MAX and active_at(o, a - 0.4, a) > 0.3:
            ends = [ob for oa, ob in o if oa < a < ob]
            if ends:
                keep = ends[0] - a
                if keep < 1.5:
                    intr[s] += 1
                yield_t[3 - s].append(keep)
    t1 = sum(b - a for a, b in sp1)
    t2 = sum(b - a for a, b in sp2)
    both = sum(active_at(sp2, a, b) for a, b in sp1)
    out = {'talk_s': {1: t1, 2: t2}, 'overlap_share_of_time': both / total,
           'backchannels': {s: len(bc[s]) for s in (1, 2)},
           'backchannels_per_min_of_other_speech': {s: len(bc[s]) / (max(1e-9, (t1, t2)[2 - s]) / 60) for s in (1, 2)},
           'backchannel_spurts': {s: bc[s] for s in (1, 2)},
           'floor_transfers': {s: fto[s] for s in (1, 2)},
           'interruptions': intr,
           'kept_talking_when_interrupted_s': yield_t,
           'turns': len(turns)}
    return out


def voice(x):
    snd = parselmouth.Sound(x, sampling_frequency=FS)
    pitch = snd.to_pitch_ac(time_step=0.01, pitch_floor=60, pitch_ceiling=600)
    f0 = pitch.selected_array['frequency']
    f0 = f0[f0 > 0]
    if len(f0) < 50:
        return None
    pts = parselmouth.praat.call(snd, 'To PointProcess (periodic, cc)', 60, 600)
    jit = parselmouth.praat.call(pts, 'Get jitter (local)', 0, 0, 0.0001, 0.02, 1.3)
    shim = parselmouth.praat.call([snd, pts], 'Get shimmer (local)', 0, 0, 0.0001, 0.02, 1.3, 1.6)
    harm = snd.to_harmonicity_cc(time_step=0.01, minimum_pitch=60)
    hv = harm.values[harm.values > -200]
    p5, p50, p95 = np.percentile(f0, [5, 50, 95])
    st = 12 * np.log2(f0 / p50)
    return {'f0_median_hz': p50, 'f0_p5_hz': p5, 'f0_p95_hz': p95,
            'f0_range_semitones_p5_p95': 12 * np.log2(p95 / p5), 'f0_sd_semitones': float(np.std(st)),
            'jitter_local_pct': jit * 100, 'shimmer_local_pct': shim * 100,
            'hnr_db': float(np.mean(hv)) if len(hv) else None, 'voiced_s': len(f0) * 0.01}


def analyse(rec, audio_dir):
    raw = [load(Path(audio_dir) / f) for f in rec['files']]
    sig = [to16k(x, fs) for x, fs in raw]
    # Track 2 is the voice under test (clean model output); track 1 may carry its echo.
    sig[0], echo_removed_db = remove_crosstalk(sig[0], sig[1])
    tracks = []
    for f, (x, fs), x16 in zip(rec['files'], raw, sig):
        cut, plateau = bandwidth(x, fs)
        db = frame_db(x16)
        act, thr = vad(db)
        sp = spurts(act)
        mask = np.repeat(act, int(HOP * FS))[:len(x16)]
        speech = x16[:len(mask)][mask]
        pause_db = float(np.percentile(db[~act], 50)) if (~act).any() else None
        speech_db = float(10 * np.log10(np.mean(speech ** 2) + 1e-20)) if len(speech) else None
        tracks.append({'file': f, 'source_rate': fs, 'bandwidth_hz': cut, 'above_cutoff_rel_db': plateau,
                       'vad_threshold_db': thr, 'speech_rms_dbfs': speech_db, 'pause_median_dbfs': pause_db,
                       'spurts': sp, 'voice': voice(speech) if len(speech) > FS else None,
                       'duration_s': len(x16) / FS})
    total = max(t['duration_s'] for t in tracks)
    tt = turn_taking(tracks[0]['spurts'], tracks[1]['spurts'], total)
    return {'id': rec['id'], 'speaker1': rec['speaker1'], 'speaker2': rec['speaker2'], 'duration_s': total,
            'echo_removed_from_track1_db': echo_removed_db, 'tracks': tracks, 'turn_taking': tt}


def main():
    recs_path, audio_dir, out_path = sys.argv[1:4]
    recs = json.loads(Path(recs_path).read_text(encoding='utf8'))['recordings']
    results = []
    for r in recs:
        if not all((Path(audio_dir) / f).exists() for f in r['files']):
            continue
        res = analyse(r, audio_dir)
        tt = res['turn_taking']
        f2 = np.array(tt['floor_transfers'][2]); f1 = np.array(tt['floor_transfers'][1])
        v2 = res['tracks'][1]['voice'] or {}
        print(f"{r['id']:30s} bw={res['tracks'][0]['bandwidth_hz']:6.0f}/{res['tracks'][1]['bandwidth_hz']:6.0f}Hz "
              f"turns={tt['turns']:3d} resp2_med={np.median(f2) if len(f2) else float('nan'):+.2f}s n={len(f2):2d} "
              f"resp1_med={np.median(f1) if len(f1) else float('nan'):+.2f}s n={len(f1):2d} "
              f"bc2/min={tt['backchannels_per_min_of_other_speech'][2]:.1f} bc1/min={tt['backchannels_per_min_of_other_speech'][1]:.1f} "
              f"ovl={tt['overlap_share_of_time']*100:.1f}% f0_2={v2.get('f0_median_hz', float('nan')):.0f}Hz "
              f"echo-={res['echo_removed_from_track1_db']:.1f}dB", flush=True)
        results.append(res)
    Path(out_path).write_text(json.dumps(results, indent=1, default=float), encoding='utf8')


if __name__ == '__main__':
    main()
