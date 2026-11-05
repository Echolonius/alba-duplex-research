"""Turns a recorded call into a timeline, measurements, and a list of the
person's spoken feedback with what the voice was doing at that moment.

  python -I analyze_call.py <session_dir>          (needs the model server for transcription)

Writes <session_dir>/analysis.md and analysis.json.
"""
import asyncio
import base64
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.dsp import EchoCanceller, Resampler, to_pcm16   # noqa: E402
import websockets                                       # noqa: E402

FEEDBACK = re.compile(r"\b(interrupt\w*|cut (me )?off|talk(ing|ed)? over|weird|uncanny|creepy|robot\w*|unnatural|natural|"
                      r"slow|delay\w*|lag\w*|wait\w*|fast|quick|laugh\w*|emotion\w*|tone|feel(s|ing)?|nuance|glitch\w*|bug\w*|"
                      r"echo|repeat\w*|didn'?t hear|can'?t hear|hear me|volume|loud|quiet|awkward|annoying|issue|problem|"
                      r"note|feedback|claude|mm-?hm|backchannel\w*|pause\w*|stopp?ed|stopping|keeps?|again)\b", re.I)


def mmss(t):
    return f'{int(t // 60):02d}:{t % 60:05.2f}'


def clean_person(person, voice, fs):
    """Removes the voice's echo from the person track (two passes so the filter has converged)."""
    p16, v16 = Resampler(fs, 16000)(person), Resampler(fs, 16000)(voice)
    n = min(len(p16), len(v16))
    p16, v16 = p16[:n], v16[:n]
    aec = EchoCanceller(mu=0.5)
    out = np.zeros(n)
    for _ in range(2):
        aec.X[:] = 0
        aec.xprev[:] = 0
        for i in range(0, n - 160, 160):
            out[i:i + 160], _ = aec(p16[i:i + 160], v16[i:i + 160])
    return out.astype(np.float32), v16.astype(np.float32)


def utterances(x, rate=16000, floor_db=-45, gap=0.6, longest=14.0):
    w = int(0.03 * rate)
    n = len(x) // w
    db = 20 * np.log10(np.sqrt(np.mean(x[:n * w].reshape(n, w) ** 2, axis=1)) + 1e-9)
    on = db > max(floor_db, np.percentile(db, 20) + 12)
    segs, start, last = [], None, None
    for k, v in enumerate(on):
        t = k * 0.03
        if v:
            if start is None:
                start = t
            last = t + 0.03
            if t - start > longest:
                segs.append((start, last))
                start = None
        elif start is not None and t - last > gap:
            if last - start > 0.25:
                segs.append((start, last))
            start = None
    if start is not None and last - start > 0.25:
        segs.append((start, last))
    return segs


PORT = int(os.environ.get('LV_ANALYSIS_PORT', '9931'))   # a separate server (e.g. 9932) leaves the call window's alone


async def transcribe(clips):
    out = []
    async with websockets.connect(f'ws://127.0.0.1:{PORT}/v1/realtime?intent=transcription', max_size=None) as ws:
        json.loads(await ws.recv())
        await ws.send(json.dumps({'type': 'session.update', 'session': {
            'type': 'transcription', 'audio': {'input': {'transcription': {'language': 'en'}}}}}))
        while json.loads(await ws.recv()).get('type') != 'session.updated':
            pass
        for clip in clips:
            # Trailing silence lets the transcriber finish the last word (see app/brain.py, commit).
            clip = np.concatenate([clip, np.zeros(int(0.3 * 16000), np.float32)])
            for i in range(0, len(clip), 1600):
                await ws.send(json.dumps({'type': 'input_audio_buffer.append',
                                          'audio': base64.b64encode(to_pcm16(clip[i:i + 1600])).decode()}))
            await ws.send(json.dumps({'type': 'input_audio_buffer.commit'}))
            text = ''
            while True:
                ev = json.loads(await ws.recv())
                if ev.get('type') == 'conversation.item.input_audio_transcription.completed':
                    text = ev.get('transcript', '').strip()
                    break
                if ev.get('type') == 'error':
                    break
            out.append(text)
    return out


def main():
    S = Path(sys.argv[1])
    meta = json.loads((S / 'session.json').read_text(encoding='utf8')) if (S / 'session.json').exists() else {'finished': False}
    person, fs = sf.read(S / 'person.flac', dtype='float32')
    voice, _ = sf.read(S / 'voice.flac', dtype='float32')
    events = [json.loads(l) for l in (S / 'events.jsonl').open(encoding='utf8') if l.strip()]
    headphones = meta.get('headphones', False)
    p16, v16 = clean_person(person, voice, fs) if not headphones else (Resampler(fs, 16000)(person).astype(np.float32),
                                                                       Resampler(fs, 16000)(voice).astype(np.float32))
    sf.write(S / 'person_clean16k.flac', p16, 16000)
    sf.write(S / 'voice16k.flac', v16, 16000)

    # Same measurements as for Dessa
    index = S / 'recordings.json'
    index.write_text(json.dumps({'recordings': [{'id': S.name, 'files': ['person_clean16k.flac', 'voice16k.flac'],
                                                 'speaker1': 'Human', 'speaker2': 'LocalVoice', 'duration': len(p16) / 16000,
                                                 'markers': [], 'captions': []}]}), encoding='utf8')
    subprocess.run([sys.executable, '-I', str(ROOT / 'tools' / 'conversation_metrics.py'), str(index), str(S), str(S / 'metrics.json')],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run([sys.executable, '-I', str(ROOT / 'tools' / 'summarize_metrics.py'), str(S / 'metrics.json'), str(index),
                    str(S / 'summary.json'), 'LocalVoice'], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    summ = json.loads((S / 'summary.json').read_text(encoding='utf8'))['test_with_people']

    # What the person said, with times
    segs = utterances(p16)
    texts = asyncio.run(transcribe([p16[int(a * 16000):int(b * 16000)] for a, b in segs]))
    said = [{'t': round(a, 2), 'end': round(b, 2), 'text': t} for (a, b), t in zip(segs, texts) if t]

    # What the voice did
    replies, last_first = [], None
    for e in events:
        if e['type'] == 'first_audio':
            last_first = e
        if e['type'] == 'reply_text':
            replies.append({'t': last_first['t'] if last_first else e['t'], 'text': e['text'],
                            'delay': last_first.get('delay_from_user_end') if last_first else None})
    kinds = {}
    for e in events:
        kinds[e['type']] = kinds.get(e['type'], 0) + 1
    delays = [e['delay_from_user_end'] for e in events if e['type'] == 'first_audio' and e.get('delay_from_user_end') is not None]
    yields = [e for e in events if e['type'] == 'yield']

    # Feedback moments and what happened around them
    flagged = []
    for s in said:
        if FEEDBACK.search(s['text']):
            near = [e for e in events if s['t'] - 12 <= e['t'] <= s['end'] + 1 and e['type'] in
                    ('respond', 'first_audio', 'yield', 'duck', 'unduck', 'backchannel', 'no_reply', 'server_error', 'empty_reply',
                     'heard', 'opener', 'dropped_sentence', 'reply_text')]
            flagged.append({**s, 'context': near})

    timeline = sorted([(s['t'], 'YOU', s['text']) for s in said] + [(r['t'], 'VOICE', r['text']) for r in replies])
    md = [f"# Call {S.name}", '',
          f"- Length: {len(p16) / 16000 / 60:.1f} min, ended cleanly: {meta.get('finished', False)}, headphones: {headphones}, "
          f"design: {meta.get('design', 'audio')}",
          f"- Voice replies: {kinds.get('reply_text', 0)}; asked for {kinds.get('respond', 0)}; cancelled before sound {sum(y['immediate'] for y in yields)}; "
          f"stopped mid-reply {sum(not y['immediate'] for y in yields)}; softened (ducked) {kinds.get('duck', 0)}",
          f"- Listening sounds: {kinds.get('backchannel', 0)}; no reply: {kinds.get('no_reply', 0)}; server errors: {kinds.get('server_error', 0)}",
          f"- First sound after you stopped: median {np.median(delays):.2f} s (25-75%: {np.percentile(delays, 25):.2f}-{np.percentile(delays, 75):.2f} s)" if delays else '- No replies with sound',
          f"- Measured like Dessa: turn start median {summ['response_offset_s']['median']:.2f} s, overlapping starts {summ['response_offset_s']['share_overlapping_start']:.0%}, "
          f"listening sounds {summ['backchannels_per_min_of_other_speech']:.1f}/min, keeps talking when interrupted {summ['kept_talking_when_talked_over_s']['median'] or 0:.2f} s",
          '', '## Your comments that sound like feedback', '']
    for f in flagged:
        md.append(f"**[{mmss(f['t'])}]** \"{f['text']}\"")
        for e in f['context']:
            md.append(f"  - {mmss(e['t'])} {e['type']} {json.dumps({k: v for k, v in e.items() if k not in ('t', 'type')})}")
        md.append('')
    md += ['## Timeline', ''] + [f"[{mmss(t)}] {who}: {text}" for t, who, text in timeline]
    (S / 'analysis.md').write_text('\n'.join(md), encoding='utf8')
    (S / 'analysis.json').write_text(json.dumps({'meta': meta, 'said': said, 'replies': replies, 'flagged': flagged,
                                                 'event_counts': kinds, 'summary': summ}, indent=1), encoding='utf8')
    print('\n'.join(md[:10]))
    print(f"\n{len(flagged)} feedback moments, {len(said)} things you said, {len(replies)} voice replies -> {S / 'analysis.md'}")


if __name__ == '__main__':
    main()
