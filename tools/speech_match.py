"""Does Alba's speech say what she planned? For a probe folder (tools/probe_duplex_val.py --save), each voice
track is transcribed by Moonshine (the app's own hearing, on the processor) and compared with everything she
planned to say in that call: word error rate (WER) of the transcript against the plan, and the share of
planned words heard at all. Planned words she never got to say (she stopped early) count as missing.

  .venv/Scripts/python.exe -I tools/speech_match.py results/tests/duplex2-probe-val
Writes <folder>/speech_match.json.
"""
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault('MOONSHINE_VOICE_CACHE', str(ROOT / 'models' / 'moonshine'))


def words(s):
    return re.findall(r"[a-z0-9]+", s.lower().replace("'", '').replace('’', ''))


def edit_distance(r, h):
    d = np.arange(len(h) + 1)
    for i, rw in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, hw in enumerate(h, 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (rw != hw))
            prev, d[j] = d[j], cur
    return int(d[len(h)])


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    folder = ROOT / sys.argv[1]
    plans = json.loads((folder / 'plans.json').read_text(encoding='utf8'))
    from moonshine_voice import Transcriber, get_model_for_language
    from moonshine_voice.moonshine_api import string_to_model_arch
    mp, arch = get_model_for_language('en', string_to_model_arch('small-streaming'))
    tr = Transcriber(model_path=mp, model_arch=arch)
    rows, err, ref_n, heard, planned = {}, 0, 0, 0, 0
    try:
        for pid, pl in sorted(plans.items()):
            x, sr = sf.read(folder / f'{pid}-voice.wav', dtype='float32')
            x = signal.resample_poly(x, 2, 3).astype(np.float32) if sr == 24000 else x
            r = tr.transcribe_without_streaming(x.tolist(), 16000)
            text = ' '.join(' '.join((l.text or '').split()) for l in r.lines if (l.text or '').strip())
            ref, hyp = words(pl['all_planned']), words(text)
            e = edit_distance(ref, hyp)
            hit = sum(1 for w in set(ref) if w in set(hyp))
            rows[pid] = {'planned': pl['all_planned'], 'heard': text, 'wer': round(e / max(1, len(ref)), 3)}
            err += e; ref_n += max(1, len(ref)); heard += hit; planned += len(set(ref))
            print(f'{pid}: wer {rows[pid]["wer"]:.2f} | planned {pl["all_planned"][:90]!r}\n          heard   {text[:90]!r}',
                  flush=True)
    finally:
        tr.close()
    out = {'wer': round(err / max(1, ref_n), 3), 'planned_words_heard': round(heard / max(1, planned), 3),
           'items': rows}
    (folder / 'speech_match.json').write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding='utf8')
    print(json.dumps({k: out[k] for k in ('wer', 'planned_words_heard')}), flush=True)


if __name__ == '__main__':
    main()
