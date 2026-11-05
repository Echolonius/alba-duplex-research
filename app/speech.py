"""Phone-class speaking: Kyutai's Pocket TTS on the processor.

Pocket TTS (100M parameters, CC-BY-4.0) streams speech at 24 kHz, the same rate as LFM3-Audio's
/tts. On this laptop's processor with 2 threads its first audio is ready in ~0.12 s and it runs
~3x faster than real time (tools/try_pocket_tts.py), so it needs neither the graphics card nor
much of the processor: the kind of budget a phone has.

Weights come from kyutai/pocket-tts-without-voice-cloning (pinned in the package's own config
files) into models/hf; built-in voices only, no voice cloning.
"""
import os
import threading

import numpy as np

from .config import ROOT

# Model files stay in this project (models/hf) and are read from there without going online:
# the ten voices in the call window were all downloaded by tools/try_pocket_tts.py.
os.environ.setdefault('HF_HOME', str(ROOT / 'models' / 'hf'))
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
os.environ.setdefault('HF_HUB_OFFLINE', '1')

RATE = 24000


def make(cfg):
    """The phone-class voice when the config asks for it (brain.speech = 'pocket'), else None."""
    b = cfg.brain
    if b.mode != 'split' or b.speech != 'pocket':
        return None
    return PocketVoice(b.pocket_voice, threads=b.pocket_threads)


_MODELS, _LOCKS = {}, {}
_LOAD = threading.Lock()


class PocketVoice:
    def __init__(self, voice='alba', threads=2, language='english'):
        import torch
        from pocket_tts import TTSModel
        self.torch = torch
        # one copy of the model per process, shared by every voice (a voice is only its prompt state); loading a
        # copy per voice took ~1.4 GB each and ran this PC out of memory on 5 Nov
        with _LOAD:
            if language not in _MODELS:
                _MODELS[language] = TTSModel.load_model(language=language)
                _LOCKS[language] = threading.Lock()
        self.model = _MODELS[language]
        torch.set_num_threads(threads)
        self.voice = voice
        self.state = self.model.get_state_for_audio_prompt(voice)
        self.lock = _LOCKS[language]          # one generation at a time per model (it is not thread-safe)

    def stream(self, text, stop=None, onset=0.02):
        """Speech for `text`, chunk by chunk (float32 at 24 kHz), without the near-silence that
        opens each clip (0.25-0.75 s of it: it would only delay the first word)."""
        with self.lock:
            before, talking = None, False
            for c in self.model.generate_audio_stream(self.state, text, stop=stop):
                x = c.detach().cpu().numpy().astype(np.float32, copy=False)
                if not talking:
                    if np.abs(x).max() < onset:
                        before = x
                        continue
                    talking = True
                    if before is not None:
                        yield before[-len(before) // 2:]     # a little lead-in keeps the onset soft
                yield x

    def say(self, text, seed=None):
        """The whole clip at once (for the ready-made listening and opening sounds)."""
        if seed is not None:
            self.torch.manual_seed(seed)
        parts = list(self.stream(text))
        return (np.concatenate(parts) if parts else np.zeros(0, np.float32)), RATE
