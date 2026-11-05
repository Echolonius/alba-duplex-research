"""Talk to the local voice from a console (the call window is app/call_app.py).

  python -m app.main                 laptop speakers (echo canceller on)
  python -m app.main --headphones    best results: no echo to deal with
  python -m app.main --no-record     do not keep a recording of the conversation

Everything runs on this computer; nothing is sent anywhere.
The live chat speaks in the model's one built-in voice (it ignores voice requests).
"""
import argparse
import asyncio
import signal
from datetime import datetime

from . import server, sounds, speech
from .audio import LiveAudio, Playback
from .conductor import Conductor
from .config import ROOT, Config, apply_settings
from .recorder import DiskRecorder

PERSONA = ('You are Juno, a warm, playful AI friend on a casual voice call. Talk like a real person: short, '
           'relaxed sentences, your own opinions, a little humor. One or two sentences, then let them talk. '
           'You have your own personality, tastes and opinions, and you share them freely. ')
# Chosen with tools/probe_multiturn.py: over 7-turn spoken conversations it gave no assistant-style
# phrases (the earlier wording gave 2-3) and shorter replies. A line about admitting to being an AI
# was tried in the second live call and removed: the 2B model then opened nearly every reply with
# "I'm an AI". It still says it is an AI when asked.

# A voice model trained here (paths.voice_model_dir) learned Alba's persona into its weights, with exactly
# this short prompt (tools/make_convo_text.py TRAIN_SYSTEM); it must get the same one.
TRAINED_SYSTEM = 'You are Alba, on a voice call. Respond with interleaved text and audio.'


def voice_instructions(cfg, clock=None):
    """The voice model's instructions: the trained prompt for a trained model, else the old persona.
    A trained model also gets the time, the way its training conversations carried it (one line after the
    prompt), so it can answer "what time is it"; clock overrides the real time (tests)."""
    if cfg.paths.voice_model_dir:
        from .brain import clock_note
        return TRAINED_SYSTEM + '\n' + (clock or clock_note())
    return PERSONA + cfg.voice.instructions


def main():
    ap = argparse.ArgumentParser(description='Local conversational voice')
    ap.add_argument('--headphones', action='store_true', help='skip echo cancelling (use with headphones)')
    ap.add_argument('--no-record', action='store_true', help='do not save the conversation')
    ap.add_argument('--input', type=int, help='microphone device number (see --list-devices)')
    ap.add_argument('--output', type=int, help='speaker device number')
    ap.add_argument('--list-devices', action='store_true')
    ap.add_argument('--plain', action='store_true', help="use the model's own default style instead of the chatty one")
    ap.add_argument('--q8', action='store_true', help='use the 8-bit model (slightly slower, a little closer to the original)')
    ap.add_argument('--audio-only', action='store_true',
                    help='let the voice model also choose the words (no text model; less in character)')
    ap.add_argument('--set', action='append', default=[], metavar='SECTION.NAME=VALUE',
                    help='override a setting from app/config.py, e.g. brain.speech=pocket')
    args = ap.parse_args()
    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return

    cfg = Config()
    apply_settings(cfg, args.set)
    if args.q8:
        cfg.paths.quant = 'Q8_0'
    if args.audio_only:
        cfg.brain.mode = 'audio'
    if not args.plain:
        cfg.voice.instructions = voice_instructions(cfg)
    logs = ROOT / 'logs'
    logs.mkdir(exist_ok=True)
    proc = None
    if server.needs_voice(cfg):
        print('Starting the voice model on this computer...', flush=True)
        proc = server.start(cfg, logs / 'server.log')
    text_proc = mind_proc = None
    recorder = None
    try:
        text_proc = server.start_text(cfg, logs / 'server-text.log')
        mind_proc = server.start_mind(cfg, logs / 'server-mind.log')
        speaker = speech.make(cfg)
        bank, particles = sounds.prepare(cfg, speaker, server.base_url(cfg), status=lambda s: print(s, flush=True))
        folder = cfg.paths.sessions / datetime.now().strftime('%Y%m%d-%H%M%S')
        recorder = None if args.no_record else DiskRecorder(folder)
        playback = Playback()
        conductor = Conductor(cfg, playback, bank, headphones=args.headphones,
                              echo_state=ROOT / 'models' / 'echo_path.npy',
                              events_path=None if args.no_record else folder / 'events.jsonl', particles=particles,
                              speaker=speaker)
        audio = LiveAudio(playback, conductor.on_block, recorder, args.input, args.output)
        stop = asyncio.Event()

        async def go():
            task = asyncio.create_task(conductor.run(stop))
            while conductor.reply_rs is None and not task.done():
                await asyncio.sleep(0.05)
            mic, spk = audio.start()
            print(f'Listening on "{mic}", speaking on "{spk}". Just talk. Press Ctrl+C to stop.', flush=True)
            if not args.headphones:
                print('(Using speakers: talk at a normal volume; headphones work best.)', flush=True)
            try:
                await task
            finally:
                audio.stop()

        signal.signal(signal.SIGINT, lambda *_: stop.set())
        asyncio.run(go())
    finally:
        if recorder is not None:
            recorder.close()
            print(f'Saved this conversation (on this computer only) to {recorder.folder}')
        server.stop(mind_proc)
        server.stop(text_proc)
        server.stop(proc)


if __name__ == '__main__':
    main()
