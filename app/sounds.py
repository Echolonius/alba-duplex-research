"""Short sounds in the voice's own voice.

backchannel: listening sounds played while the person talks ("mm-hm", "yeah").
particle:    quick turn-openers played the moment the voice decides to answer
             ("oh,", "yeah,"), the way Dessa (44% of its turns) and people (40%)
             start a turn while the rest of the reply is still coming.

Every clip is made by the model's speech generator in its built-in female voice, the
same voice the live chat uses, then trimmed, level-matched and cached. (LFM3-Audio
ignores voice samples: the server logs "the model has no voice conditioning".)
"""
import base64
import io
import json
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf

from .dsp import Resampler

BACKCHANNELS = ['Mm-hm.', 'Yeah.', 'Right.', 'Uh-huh.', 'Mhm.', 'Okay.', 'Oh.', 'Yeah, yeah.', 'Oh, wow.', 'Hmm.']
# first word of a reply -> what the voice says to open its turn
PARTICLES = {'oh': 'Oh,', 'yeah': 'Yeah,', 'yes': 'Yeah,', 'mm': 'Mm,', 'hmm': 'Hmm,', 'okay': 'Okay,', 'ok': 'Okay,',
             'right': 'Right,', 'wow': 'Wow,', 'so': 'So,', 'well': 'Well,', 'ooh': 'Ooh,', 'aw': 'Aw,', 'ah': 'Ah,',
             'ugh': 'Ugh,', 'hey': 'Hey,', 'alright': 'Alright,', 'oh no': 'Oh no,', 'oh wow': 'Oh wow,',
             'oh yeah': 'Oh yeah,'}
# Whole first sentences that are a quick reaction: said at once from a clip instead of waiting to be
# voiced. Only ones that also work as a listening sound if the person was not finished after all.
QUICK = ('Totally!', 'Nice!', 'Exactly!', 'Definitely!', 'Absolutely!', 'Perfect!', 'Oh no!', 'Oh wow!',
         'No way!', 'Wow!', 'Same!', 'Ooh!', 'Aw!', 'Oh!', 'Yes!', 'Sure!', 'Awesome!', 'Got it!', 'Really?',
         'Sounds good!', 'Sounds awesome!', 'Sounds fun!', 'Hey there!')
OUT_RATE = 48000
VOICE = 'chat'


def tts(base, text, voice=None, seed=0, ref=None):
    """Speech from the server's /tts. `ref` (wav bytes) makes it copy that voice."""
    body = {'input': text, 'seed': seed, 'response_format': 'pcm'}
    if ref is not None:
        body['speaker_ref_b64'] = base64.b64encode(ref).decode()
    elif voice:
        body['voice'] = voice
    req = urllib.request.Request(base + '/tts', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=120) as r:
        rate = int(r.headers.get('Content-Type', 'rate=24000').split('rate=')[1].split(';')[0])
        return np.frombuffer(r.read(), dtype='<f4').copy(), rate


def trim(x, rate, floor_db=-38.0, margin_s=0.03):
    w = int(0.01 * rate)
    if len(x) < 4 * w:
        return x[:0]
    e = np.array([np.sqrt(np.mean(x[i:i + w] ** 2)) for i in range(0, len(x) - w, w)])
    ref = e.max() + 1e-9
    on = np.where(20 * np.log10(e / ref + 1e-12) > floor_db)[0]
    if not len(on):
        return x[:0]
    a = max(0, on[0] * w - int(margin_s * rate))
    b = min(len(x), (on[-1] + 1) * w + int(margin_s * rate))
    y = x[a:b].copy()
    fade = min(len(y) // 4, int(0.01 * rate))
    if fade:
        y[:fade] *= np.linspace(0, 1, fade)
        y[-fade:] *= np.linspace(1, 0, fade)
    return y


def _wav_bytes(path):
    x, fs = sf.read(path, dtype='float32')
    buf = io.BytesIO()
    sf.write(buf, x, fs, format='WAV', subtype='PCM_16')
    return buf.getvalue()


def voice_dir(cfg):
    """Each voice keeps its own clips: 'chat' for LFM3-Audio's voice, 'pocket-<name>' for Pocket TTS."""
    return f'pocket-{cfg.brain.pocket_voice}' if cfg.brain.speech == 'pocket' else VOICE


def prepare(cfg, speaker=None, base=None, status=None):
    """The listening sounds and quick openers for the voice in use, made on first use."""
    vd = voice_dir(cfg)
    out = []
    for kind in ('backchannel', 'particle'):
        only = cfg.timing.backchannel_words if kind == 'backchannel' else None
        clips = load(cfg.paths.sounds, kind, only=only, voice_dir=vd)
        if not clips:
            if status is not None:
                status('Making listening sounds (first run only)…' if kind == 'backchannel'
                       else 'Making quick reply sounds (first run only)…')
            build(base, cfg.paths.sounds, kind, speak=speaker.say if speaker is not None else None, voice_dir=vd)
            clips = load(cfg.paths.sounds, kind, only=only, voice_dir=vd)
        out.append(clips)
    return tuple(out)


def build(base, folder, kind='backchannel', voice='female', seeds=(11, 12, 13), target_rms_db=-26.0, speak=None,
          voice_dir=VOICE):
    """Makes and caches the clips. `speak(text, seed) -> (audio, rate)` makes them with another voice
    (Pocket TTS); otherwise they come from the voice server's /tts at `base`."""
    folder = Path(folder) / voice_dir / kind
    folder.mkdir(parents=True, exist_ok=True)
    items = BACKCHANNELS if kind == 'backchannel' else sorted(set(PARTICLES.values())) + list(QUICK)
    meta = []
    for word in items:
        # A turn opener must be short or the rest of the reply waits for it; a whole quick
        # reaction ("Absolutely!") is the first sentence itself, so it may be a little longer.
        longest = 0.9 if kind == 'backchannel' or ' ' in word else (1.0 if word in QUICK else 0.75)
        for seed in seeds:
            x, rate = speak(word, seed) if speak is not None else tts(base, word, voice, seed=seed)
            y = trim(x, rate)
            dur = len(y) / rate
            if not 0.12 <= dur <= longest:
                continue
            y = Resampler(rate, OUT_RATE)(y).astype(np.float32)
            y *= 10 ** (target_rms_db / 20) / (np.sqrt(np.mean(y ** 2)) + 1e-9)
            y = np.clip(y, -0.95, 0.95)
            name = f'{len(meta):02d}.wav'
            sf.write(folder / name, y, OUT_RATE)
            meta.append({'file': name, 'text': word, 'seed': seed, 'seconds': round(dur, 3)})
            break
    (folder / 'sounds.json').write_text(json.dumps(meta, indent=1), encoding='utf8')
    return meta


def load(folder, kind='backchannel', only=None, voice_dir=VOICE):
    """backchannel -> list of clips; particle -> {spoken text: clip}. `only`: keep these texts."""
    folder = Path(folder) / voice_dir / kind
    meta_path = folder / 'sounds.json'
    if not meta_path.exists():
        return [] if kind == 'backchannel' else {}
    meta = json.loads(meta_path.read_text(encoding='utf8'))
    if only:
        meta = [m for m in meta if m['text'] in only] or meta
    clips = [(m['text'], sf.read(folder / m['file'], dtype='float32')[0]) for m in meta]
    return [c for _, c in clips] if kind == 'backchannel' else dict(clips)
