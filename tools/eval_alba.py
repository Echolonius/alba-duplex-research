"""Step 4: hear what the audio-native Alba says, before and after training.

For each test input (tools/make_s2s_inputs.py: synthetic lines and real turns from the user's calls, which
stay on this computer) LFM3-Audio-2B hears the clip and answers in interleaved text and speech, with or
without the LoRA from tools/train_lfm3_lora.py. Saves each reply (text + 24 kHz audio) and checks it:
  assistant-talk   help-desk phrases the persona forbids ("how can I help", "as an AI"...)
  words            length of the reply
  heard back       the reply's own speech, transcribed by the same model: is the speech the words it wrote?
  speed            seconds of speech made per second (in PyTorch here; llama.cpp is much faster)

  .venv-train/Scripts/python.exe -I tools/eval_alba.py [--lora models/alba-lora] [--tag NAME] [--no-4bit]
"""
import argparse
import difflib
import json
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('NO_TORCH_COMPILE', '1')
import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_lfm3_lora import load  # noqa: E402
from make_convo_text import BAD_REPLY, TRAIN_SYSTEM  # noqa: E402


def words(s):
    return re.findall(r"[a-z0-9']+", s.lower())


@torch.inference_mode()
def reply(proc, model, wav, system, max_new=700):
    msgs = [{'role': 'system', 'content': system}, {'role': 'user', 'content': [{'type': 'audio', 'audio': wav}]}]
    inputs = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
                                      return_tensors='pt').to(model.device)
    t0 = time.perf_counter()
    out = model.generate(**inputs, max_new_tokens=max_new, audio_temperature=1.0, audio_top_k=128,
                         audio_repetition_penalty=1.1, audio_repetition_window=4096)
    dt = time.perf_counter() - t0
    n = inputs['input_ids'].shape[1]
    text = proc.decode(out.sequences[0, n:], skip_special_tokens=True).strip()
    codes = out.audio_codes[:1, :, :out.audio_lengths[0]]
    audio = proc.audio_tokenizer.decode(codes).audio_values[0].float().cpu().numpy().squeeze() if codes.shape[-1] else np.zeros(1)
    return text, audio, dt


@torch.inference_mode()
def asr(proc, model, wav24):
    import librosa
    wav = librosa.resample(np.asarray(wav24, np.float32), orig_sr=24000, target_sr=16000)
    msgs = [{'role': 'system', 'content': 'Perform ASR in English.'},
            {'role': 'user', 'content': [{'type': 'audio', 'audio': wav}]}]
    inputs = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
                                      return_tensors='pt').to(model.device)
    out = model.generate(**inputs, max_new_tokens=200)
    return proc.decode(out.sequences[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser()
    ap.add_argument('--lora', default='')
    ap.add_argument('--tag', default='')
    ap.add_argument('--no-4bit', action='store_true')
    ap.add_argument('--system', default=TRAIN_SYSTEM)
    args = ap.parse_args()
    proc, model = load(quantize=not args.no_4bit)
    if args.lora:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora)
    model.eval()
    proc.audio_tokenizer.to(model.device)
    items = json.loads((ROOT / 'results' / 's2s-inputs' / 'inputs.json').read_text(encoding='utf8'))
    out = ROOT / 'results' / 'alba-audio' / (args.tag or ('lora' if args.lora else 'base'))
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for it in items:
        wav, sr = sf.read(ROOT / 'results' / 's2s-inputs' / it['file'], dtype='float32')
        text, audio, dt = reply(proc, model, wav, args.system)
        secs = len(audio) / 24000
        heard = asr(proc, model, audio) if secs > 0.3 else ''
        match = difflib.SequenceMatcher(None, words(text), words(heard)).ratio() if text else 0.0
        sf.write(out / f"reply-{Path(it['file']).stem}.wav", audio, 24000)
        r = {'input': it['file'], 'kind': it['kind'], 'said': it['text'], 'reply': text, 'speech_s': secs,
             'gen_s': dt, 'assistant_talk': bool(BAD_REPLY.search(text)), 'words': len(words(text)),
             'heard_back': heard, 'speech_matches_text': round(match, 2)}
        rows.append(r)
        print(f"[{it['file']}] {secs:.1f}s speech in {dt:.1f}s | match {match:.2f} | {text[:140]}", flush=True)
    summary = {'lora': args.lora, 'assistant_talk': sum(r['assistant_talk'] for r in rows), 'n': len(rows),
               'words_median': float(np.median([r['words'] for r in rows])),
               'speech_matches_text_median': float(np.median([r['speech_matches_text'] for r in rows]))}
    print(json.dumps(summary), flush=True)
    (out / 'results.json').write_text(json.dumps({'summary': summary, 'replies': rows}, indent=1, ensure_ascii=False),
                                      encoding='utf8')


if __name__ == '__main__':
    main()
