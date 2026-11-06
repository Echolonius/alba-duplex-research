"""Checks tools/run_fdb.py (Full-Duplex-Bench for Alba Duplex) without any model weights: a fake duplex model that
answers with a tone a few steps after its input goes quiet runs on tiny synthetic benchmark folders (in a throwaway
folder under logs/, removed at the end), and the checks compare the files written with what the fake did: steps
fed, length and format of the outputs, the 240 ms playback offset, the folder layout the benchmark's scorers read,
a stopped or repeated run picking up where it left off, a reply cut off by the end of an input leaving nothing
behind, outputs of another model or device refused, slow inputs marked not real-time, and the same outputs however
the inputs are split between runs.

  .venv-train/Scripts/python.exe -I tools/test_run_fdb.py
"""
import contextlib
import io
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
from app.duplex import DELAY, FIRST_STEP, OUT_STEP, STEP  # noqa: E402
from run_fdb import fingerprint, run_fdb  # noqa: E402
from run_timing_duplex import run_track  # noqa: E402

TMP = ROOT / 'logs' / 'fdb-test-tmp'
TALK_S = (FIRST_STEP + 9 * STEP) / 16000      # the person talks through step 9 exactly (0.815 s)
TOTAL_S = 3.0
LOUD_UNTIL = 9                                # the last step that hears the person


class FakeDuplex:
    """AlbaDuplex's streaming interface with no model: once it has heard sound and then QUIET quiet steps in a row,
    it talks for TALK steps (a 440 Hz tone, one 80 ms frame of 24 kHz sound per step), then listens again. As in
    AlbaDuplex, its sound decoder starts afresh when a reply ends or on flush(), not on start(): a reply cut off by
    the end of an input leaves it open, and the next reply then comes 20 ms late, after the held-back sound. With
    random_talk, each reply's length is drawn from rng (set from outside, as for AlbaDuplex); slow_s makes the first
    frame of each reply take that long."""
    QUIET, TALK = 5, 10
    HELD = 480                                # what the real decoder holds back between frames: 20 ms at 24 kHz

    def __init__(self, fail_at_call=None, random_talk=False, slow_s=0.0):
        self.calls, self.closed, self.fail_at_call = 0, False, fail_at_call
        self.random_talk, self.slow_s = random_talk, slow_s
        self.rng = np.random.default_rng()    # unseeded: only the adapter's seeding makes it repeatable
        self.phase, self.flushes = 0, 0       # the sound decoder: phase 0 is a fresh one

    def start(self, **kw):
        self.calls += 1
        self.state, self.events, self.sizes = 'listen', [], []
        self.heard, self.quiet, self.left = False, 0, 0

    def step(self, pcm, listen_bias=0.0):
        k = len(self.sizes)
        self.sizes.append(len(pcm))
        if self.calls == self.fail_at_call and k == 3:
            raise RuntimeError('stopped on purpose')
        loud = np.sqrt(np.mean(np.square(pcm))) > 0.01
        self.heard |= bool(loud)
        self.quiet = 0 if loud else self.quiet + 1
        held = np.zeros(0, np.float32)
        if self.state == 'listen':
            if not (self.heard and self.quiet == self.QUIET):
                return np.zeros(0, np.float32)
            self.state, self.heard = 'speak', False
            self.left = int(self.rng.integers(2, 2 * self.TALK)) if self.random_talk else self.TALK
            self.events.append(('start', k, time.time()))
            if self.phase:                    # a decoder left open: the old reply's held-back sound comes first
                held = np.full(self.HELD, 0.9, np.float32)
            if self.slow_s:
                time.sleep(self.slow_s)
        n = np.arange(self.phase, self.phase + OUT_STEP)
        self.phase += OUT_STEP
        self.left -= 1
        if self.left == 0:                    # the reply ends: as at AlbaDuplex's stop, the decoder starts afresh
            self.state, self.phase = 'listen', 0
        return np.concatenate([held, 0.5 * np.sin(2 * np.pi * 440 * n / 24000)]).astype(np.float32)

    def flush(self):
        self.phase, self.flushes = 0, self.flushes + 1
        return np.zeros(0, np.float32)

    def close(self):
        self.closed = True


def person(sr, channels=1):
    """A caller turn: a tone for TALK_S, then quiet until TOTAL_S."""
    t = np.arange(int(round(TOTAL_S * sr))) / sr
    x = np.where(t < TALK_S, 0.3 * np.sin(2 * np.pi * 300 * t), 0.0).astype(np.float32)
    return np.stack([x] + [np.zeros_like(x)] * (channels - 1), axis=1) if channels > 1 else x


def fresh(name):
    d = TMP / name
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    return d


def bench(root):
    """A miniature benchmark download: v1.0 and v1.5 sample folders with the files they ship."""
    folders = {'v1_0/candor_pause_handling/1': ['input.wav', 'pause.json', 'transcription.json'],
               'v1_0/synthetic_user_interruption/7': ['input.wav', 'context.wav', 'interrupt.wav', 'interrupt.json'],
               'v1_5/talking_to_other/3': ['input.wav', 'clean_input.wav', 'metadata.json']}
    for folder, files in folders.items():
        (root / folder).mkdir(parents=True)
        for f in files:
            if f.endswith('.wav'):
                sf.write(root / folder / f, person(16000), 16000, subtype='PCM_16')
            else:
                (root / folder / f).write_text(json.dumps([{'timestamp': [0.0, TALK_S]}]), encoding='utf8')
    return root


def files(root):
    return {p.relative_to(root).as_posix() for p in root.rglob('*')}


def run(fake, data, log, **kw):
    """run_fdb with its progress lines kept out of the check's output."""
    with contextlib.redirect_stdout(io.StringIO()):
        return run_fdb(lambda: fake, data, log, **kw)


def rows(log):
    return [json.loads(line) for line in log.read_text(encoding='utf8').splitlines()]


def sound_at(y, at, length, tol=16):
    """Silent before `at`, sound from `at` for `length` samples, silent after (within tol samples, 1 ms)."""
    return (np.abs(y[:at - tol]).max() < 0.01 and np.abs(y[at:at + tol]).max() > 0.1 and
            np.abs(y[at + length - tol:at + length]).max() > 0.1 and np.abs(y[at + length + tol:]).max() < 0.01)


def test_timing():
    """80 ms steps fed as the live app feeds them; the reply is heard DELAY steps (240 ms) after its step began"""
    d = fresh('timing')
    x = person(16000)
    sf.write(d / 'input.wav', x, 16000, subtype='PCM_16')
    fake = FakeDuplex()
    out = run(fake, d, d.parent / 'timing.jsonl')
    n = (len(x) - FIRST_STEP) // STEP + 1
    assert fake.sizes == [FIRST_STEP] + [STEP] * (n - 1), 'not fed one 95 ms push, then 80 ms pushes'
    assert [e[1] for e in fake.events] == [LOUD_UNTIL + FakeDuplex.QUIET], fake.events
    info = sf.info(d / 'output.wav')
    assert (info.samplerate, info.channels, info.subtype, info.frames) == (16000, 1, 'PCM_16', len(x)), info
    y, _ = sf.read(d / 'output.wav', dtype='float32')
    k = fake.events[0][1]
    assert sound_at(y, (k + DELAY) * STEP, FakeDuplex.TALK * STEP), 'reply not where a listener hears it'
    # the very samples tools/run_timing_duplex.py would place, cut to the input's length
    voice24, _, _ = run_track(FakeDuplex(), x, DELAY, 0.0)
    want = signal.resample_poly(voice24, 2, 3)[:len(x)]
    assert np.abs(y - want).max() < 2 / 32768, 'differs from the timing harness'
    assert out[0]['steps'] == n and out[0]['starts'] == 1 and out[0]['sample'].endswith('timing/input.wav'), out
    assert out[0]['late_frames'] == 0 and out[0]['real_time'] is True, out


def test_delay_setting():
    """a model trained with another delay (duplex.json delay_steps) is heard that many steps later"""
    d = fresh('delay')
    sf.write(d / 'input.wav', person(16000), 16000, subtype='PCM_16')
    fake = FakeDuplex()
    run(fake, d, d.parent / 'delay.jsonl', delay=5)
    y, _ = sf.read(d / 'output.wav', dtype='float32')
    assert sound_at(y, (fake.events[0][1] + 5) * STEP, FakeDuplex.TALK * STEP)


def test_rates_and_channels():
    """a 48 kHz two-channel input is heard as 16 kHz mono; its output is as long as it, at 16 kHz"""
    d = fresh('rates')
    x = person(48000, channels=2)
    sf.write(d / 'input.wav', x, 48000, subtype='PCM_16')
    fake = FakeDuplex()
    run(fake, d, d.parent / 'rates.jsonl')
    info = sf.info(d / 'output.wav')
    assert (info.samplerate, info.channels, info.frames) == (16000, 1, round(len(x) / 3)), info
    y, _ = sf.read(d / 'output.wav', dtype='float32')
    assert len(fake.events) == 1 and sound_at(y, (fake.events[0][1] + DELAY) * STEP, FakeDuplex.TALK * STEP)


def test_short_input():
    """an input shorter than one step gets a silent output of its own length"""
    d = fresh('short')
    sf.write(d / 'input.wav', np.full(1000, 0.3, np.float32), 16000, subtype='PCM_16')
    out = run(FakeDuplex(), d, d.parent / 'short.jsonl')
    y, sr = sf.read(d / 'output.wav', dtype='float32')
    assert sr == 16000 and len(y) == 1000 and not y.any() and out[0]['steps'] == 0


def test_layout():
    """output.wav (and clean_output.wav for v1.5) next to each input, and nothing else anywhere in the data"""
    d = bench(fresh('layout') / 'fdb')
    log = d.parent / 'speed.jsonl'
    before = files(d)
    out = run(FakeDuplex(), d, log)
    added = files(d) - before
    assert added == {'v1_0/candor_pause_handling/1/output.wav', 'v1_0/synthetic_user_interruption/7/output.wav',
                     'v1_5/talking_to_other/3/output.wav', 'v1_5/talking_to_other/3/clean_output.wav'}, added
    assert files(d) >= before and len(out) == 4
    # the scorers walk task folders and expect only sample folders in them
    for task in [p for p in d.glob('*/*')]:
        assert all(q.is_dir() for q in task.iterdir()), task
    text = log.read_text(encoding='utf8')
    assert str(TMP) not in text and str(TMP.as_posix()) not in text, 'a local path in the log'
    assert {r['sample'] for r in rows(log)} == {'candor_pause_handling/1/input.wav',
                                                'synthetic_user_interruption/7/input.wav',
                                                'talking_to_other/3/input.wav', 'talking_to_other/3/clean_input.wav'}


def test_resume():
    """a repeated run does nothing and never builds the model; a removed output is made again, alone"""
    d = bench(fresh('resume') / 'fdb')
    log = d.parent / 'speed.jsonl'
    run(FakeDuplex(), d, log)
    built = []
    with contextlib.redirect_stdout(io.StringIO()):
        again = run_fdb(lambda: built.append(1) or FakeDuplex(), d, log)
    assert again == [] and built == [], 'redid finished inputs'
    (d / 'v1_5/talking_to_other/3/clean_output.wav').unlink()
    fake = FakeDuplex()
    redo = run(fake, d, log)
    assert [r['sample'] for r in redo] == ['talking_to_other/3/clean_input.wav'] and fake.calls == 1, redo
    assert (d / 'v1_5/talking_to_other/3/clean_output.wav').exists() and fake.closed
    one = run(FakeDuplex(), bench(fresh('limit') / 'fdb'), TMP / 'limit' / 'speed.jsonl', limit=1)
    assert len(one) == 1


def test_stopped_run():
    """a run stopped part way leaves no half output; the next run does only what is left"""
    d = bench(fresh('stopped') / 'fdb')
    log = d.parent / 'speed.jsonl'
    try:
        run(FakeDuplex(fail_at_call=3), d, log)
        raise AssertionError('the stop did not happen')
    except RuntimeError:
        pass
    assert len(rows(log)) == 2 and not list(d.rglob('*.tmp.wav'))
    assert len(list(d.rglob('*output.wav'))) == 2
    fake = FakeDuplex()
    rest = run(fake, d, log)
    assert len(rest) == 2 and fake.calls == 2 and len(list(d.rglob('*output.wav'))) == 4


def test_foreign_output():
    """an output this run did not make (another model's) stops the run, unless --overwrite"""
    d = fresh('foreign')
    sf.write(d / 'input.wav', person(16000), 16000, subtype='PCM_16')
    sf.write(d / 'output.wav', np.full(16000, 0.5, np.float32), 16000, subtype='PCM_16')
    try:
        run(FakeDuplex(), d, d.parent / 'foreign.jsonl')
        raise AssertionError('took over an output it did not make')
    except SystemExit as e:
        assert 'output.wav' in str(e)
    out = run(FakeDuplex(), d, d.parent / 'foreign.jsonl', overwrite=True)
    y, _ = sf.read(d / 'output.wav', dtype='float32')
    assert len(out) == 1 and len(y) == round(TOTAL_S * 16000)


def test_settings_guard():
    """a log made with one listen setting is not continued with another"""
    d = fresh('settings')
    sf.write(d / 'input.wav', person(16000), 16000, subtype='PCM_16')
    run(FakeDuplex(), d, d.parent / 'settings.jsonl', listen_bias=0.8)
    (d / 'output.wav').unlink()
    try:
        run(FakeDuplex(), d, d.parent / 'settings.jsonl', listen_bias=1.6)
        raise AssertionError('mixed two listen settings in one log')
    except SystemExit as e:
        assert 'listen_bias 0.8, now 1.6' in str(e), e


def test_model_guard():
    """outputs made by another model or on another device are not taken as done, unless --force"""
    d = fresh('model')
    sf.write(d / 'input.wav', person(16000), 16000, subtype='PCM_16')
    log = d.parent / 'model.jsonl'
    run(FakeDuplex(), d, log, model='a1', device='cuda')
    assert (rows(log)[0]['model'], rows(log)[0]['device']) == ('a1', 'cuda'), rows(log)
    for model, device in (('b2', 'cuda'), ('a1', 'cpu')):
        try:
            run(FakeDuplex(), d, log, model=model, device=device)
            raise AssertionError(f'kept the output of another model or device ({model}, {device})')
        except SystemExit as e:
            assert 'another model, device or setting' in str(e), e
    built = []
    with contextlib.redirect_stdout(io.StringIO()):
        kept = run_fdb(lambda: built.append(1) or FakeDuplex(), d, log, model='b2', device='cuda', force=True)
    assert kept == [] and built == [], 'did not keep the output with --force'
    redo = run(FakeDuplex(), d, log, model='b2', device='cuda', force=True, overwrite=True)
    assert [r['model'] for r in redo] == ['b2'], redo
    assert run(FakeDuplex(), d, log, model='b2', device='cuda') == [], 'the newer output was not taken as done'


def test_fingerprint():
    """the model's fingerprint follows its files' contents (even at the same size), not where its folder is"""
    a, b = fresh('fp-a'), fresh('fp-b')
    for m in (a, b):
        (m / 'hf').mkdir()
        (m / 'model-Q4_K_M.gguf').write_bytes(b'GGUF' + bytes(64))
        (m / 'duplex.json').write_text('{"delay_steps": 3}', encoding='utf8')
        (m / 'hf' / 'config.json').write_text('{}', encoding='utf8')
    assert fingerprint(a) == fingerprint(b), 'the same files in another folder got another fingerprint'
    (b / 'model-Q4_K_M.gguf').write_bytes(b'GGUF' + bytes(63) + b'\x01')
    assert fingerprint(a) != fingerprint(b), 'a changed GGUF of the same size kept its fingerprint'
    (b / 'model-Q4_K_M.gguf').write_bytes(b'GGUF' + bytes(64))
    (b / 'hf' / 'config.json').write_text('{"x": 1}', encoding='utf8')
    assert fingerprint(a) != fingerprint(b), 'a changed hf/ file kept its fingerprint'


def test_cut_reply():
    """a reply cut off by the end of its input leaves nothing behind: the next input's reply is on time and clean"""
    d = fresh('cut')
    k = LOUD_UNTIL + FakeDuplex.QUIET                       # the step the reply starts
    x = person(16000)[:FIRST_STEP + (k + 3) * STEP]         # the input ends 4 frames into the reply
    for name in ('1', '2'):
        (d / name).mkdir()
        sf.write(d / name / 'input.wav', x, 16000, subtype='PCM_16')
    fake = FakeDuplex()
    run(fake, d, d.parent / 'cut.jsonl')
    assert fake.calls == 2 and fake.flushes == 2, (fake.calls, fake.flushes)
    y1, _ = sf.read(d / '1' / 'output.wav', dtype='float32')
    y2, _ = sf.read(d / '2' / 'output.wav', dtype='float32')
    assert np.abs(y1[(k + DELAY) * STEP:]).max() > 0.1, 'the reply was not cut by the end of the input'
    assert np.array_equal(y1, y2), "the second input's reply was moved by the first one's leftover sound"


def test_late_frames():
    """an input too slow for real time is marked so (late frames, real_time false) and the run ends with a warning"""
    d = fresh('late')
    sf.write(d / 'input.wav', person(16000), 16000, subtype='PCM_16')
    said = io.StringIO()
    with contextlib.redirect_stdout(said):
        out = run_fdb(lambda: FakeDuplex(slow_s=0.3), d, d.parent / 'late.jsonl')
    assert out[0]['late_frames'] >= 1 and out[0]['real_time'] is False, out
    assert 'not real-time' in said.getvalue(), said.getvalue()


def test_repeatable():
    """each input draws from its own seed (one per sample folder): one run, or one input per run, gives the same
    outputs, and both inputs of a v1.5 sample share their seed"""
    a = bench(fresh('seed-a') / 'fdb')
    run(FakeDuplex(random_talk=True), a, a.parent / 'speed.jsonl')
    b = bench(fresh('seed-b') / 'fdb')
    for _ in range(4):
        run(FakeDuplex(random_talk=True), b, b.parent / 'speed.jsonl', limit=1)
    outs = sorted(p.relative_to(a) for p in a.rglob('*output.wav'))
    assert len(outs) == 4, outs
    for p in outs:
        assert np.array_equal(sf.read(a / p)[0], sf.read(b / p)[0]), f'{p} differs between the two runs'
    pair = a / 'v1_5' / 'talking_to_other' / '3'
    assert np.array_equal(sf.read(pair / 'output.wav')[0], sf.read(pair / 'clean_output.wav')[0])
    seeds = {r['sample']: r['seed'] for r in rows(a.parent / 'speed.jsonl')}
    assert seeds['talking_to_other/3/input.wav'] == seeds['talking_to_other/3/clean_input.wav'], seeds
    assert len(set(seeds.values())) == 3, seeds


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    checks = [test_timing, test_delay_setting, test_rates_and_channels, test_short_input, test_layout, test_resume,
              test_stopped_run, test_foreign_output, test_settings_guard, test_model_guard, test_fingerprint,
              test_cut_reply, test_late_frames, test_repeatable]
    failed = 0
    for check in checks:
        try:
            check()
            print(f'ok    {check.__name__}: {check.__doc__}', flush=True)
        except (Exception, SystemExit) as e:
            failed += 1
            print(f'FAIL  {check.__name__}: {type(e).__name__}: {e}', flush=True)
    shutil.rmtree(TMP, ignore_errors=True)
    print(f'{len(checks) - failed} of {len(checks)} checks passed', flush=True)
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
