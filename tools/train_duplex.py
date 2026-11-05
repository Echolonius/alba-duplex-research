"""Milestone 4: teach LFM3-Audio-2B to listen and speak at the same time ("Alba Duplex"), with QLoRA on this
laptop's 4 GB graphics card.

The whole call is one stream in 80 ms steps (12.5 per second). Each step:
  [one token of the person's voice]  (the speech encoder's output for those 80 ms + 15 ms of its sound window)
  then Alba's step:
    listening   LISTEN                                  (a spare special token: "I stay quiet")
    starting    <|mixed_start|> + up to 6 word-pieces   (her plan, as in the model's own interleaved mode)
                + a speech frame
    speaking    a speech frame (12 codec codes); every 12 frames, 6 more word-pieces of plan before it;
                after the plan's <|text_end|>, frames until the end-of-speech frame
    stopping    the end-of-speech frame (also how she gives way when cut off), then LISTEN again
Turn-taking is therefore whatever the model learned from the data: when to start, when to go "mm-hm", when to
stop for the person, when to talk through their "mm-hm".

Timing: the frame Alba makes at step t is heard DELAY steps after step t began (app/duplex.py: 95 ms to hear
the step, then time to think), so in training her speech is shifted DELAY steps earlier against what she
hears: she learns to start, stop and "mm-hm" at the real moments people did, as you will hear them.

Data (--shares): real conversations (a real person's voice and the real timing of both sides) 33%, Alba's own
synthetic calls 45%, tone of voice (tools/make_feeling_data.py) 14%, remarks meant for Claude 8%. Runs 1-2 used
the real conversations re-spoken as Alba with the other speaker's own words (their life, job, family) as context
only, so she would never learn to claim a human life (tools/make_duplex_data.py, 'seamless-train'); with a real
voice she then learned only "mm-hm"s and short reactions. Run 3 on uses the same conversations with new words
written for her in every turn ('seamless-own', tools/make_seamless_own.py), all learned. The laptop-mic sound
goes on the real conversations (the synthetic callers already have it from tools/make_convo_audio.py). Where Alba
gave way to the person, she stops 0.18-0.32 s after the person started
(people in the data take about 0.5 s; the goal asks for at most 0.4 s). What she hears gets, at random, a
laptop-microphone sound, and her own voice leaking back from the speakers at -35 to -8 dB (she must never
answer her own echo).
Losses: the model's own text loss (LISTEN / start / plan tokens; LISTEN steps are kept in the loss with
probability LISTEN_KEEP so the rare decisions count; the runtime adds log(1/LISTEN_KEEP) back to LISTEN, which
undoes exactly that re-weighting) and audio loss (frames, end-of-speech).

  .venv-train/Scripts/python.exe -I tools/train_duplex.py --data data/duplex --steps 600 [--init models/alba-lora] [--check]
Writes models/alba-duplex-lora (adapter + duplex.json with the timing settings).
"""
import argparse
import json
import math
import os
import random
import re
import sys
import time
import zlib
from pathlib import Path

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('NO_TORCH_COMPILE', '1')
import numpy as np
import soundfile as sf
import torch
from scipy import signal
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT))
from app.duplex import DELAY, HEADER, LISTEN_TOKEN, N_AUDIO, N_TEXT, SYSTEM  # noqa: E402
from build_testsets import classify, merge  # noqa: E402
from train_lfm3_lora import load  # noqa: E402


def fake_clock(rng):
    """A made-up time line, as tools/make_convo_text.py writes them (the real conversations carry none)."""
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
              'November', 'December']
    h, m = rng.randint(1, 12), rng.randint(0, 59)
    return (f"It is {rng.choice(days)}, {rng.choice(months)} {rng.randint(1, 28)}, {rng.choice([2025, 2026, 2027])}, "
            f"{h}:{m:02d} {rng.choice(['AM', 'PM'])} here.")

STEP_S = 0.08
SR_U, SR_A = 16000, 24000
LOOK = 240                                 # samples of sound window past a step (15 ms)
LISTEN_KEEP = 0.2                          # share of LISTEN steps kept in the loss
MERGE_GAP_S = 0.7      # pieces of one turn closer than this are one utterance (one plan; short pauses inside)
YIELD_S = (0.18, 0.32)
REAL_SOURCES = ('seamless-train', 'seamless-own')   # a real person's recording is what she hears


class Grammar:
    def __init__(self, proc, model):
        tok, c = proc.tokenizer, model.config
        self.tok, self.proc = tok, proc
        self.audio_in = c.audio_token_id
        self.audio_out = c.audio_out_token_id
        self.mixed_start = c.mixed_start_token_id
        self.text_end = c.text_end_token_id
        self.listen = tok.convert_tokens_to_ids(LISTEN_TOKEN)
        self.eos_code = proc.audio_eos_code
        self._heads = {}

    def head(self, clock=None, tools=None):
        """The call's opening: the system prompt (with the time as one more line when known, as live) + header;
        with tools, their list in the model's own <tools> format (as app/duplex.AlbaDuplex.start writes it)."""
        key = (clock, json.dumps(tools, sort_keys=True) if tools else None)
        if key not in self._heads:
            system = SYSTEM + (f'\n{clock}' if clock else '')
            msgs = [{'role': 'system', 'content': system}]
            text = (self.proc.apply_chat_template(msgs, tools=tools, tokenize=False) if tools else
                    self.proc.apply_chat_template(msgs, tokenize=False)) + HEADER
            self._heads[key] = self.tok(text, add_special_tokens=False)['input_ids']
        return self._heads[key]


def code_spans(meta):
    """Her utterances as encoded: pieces of a turn joined, touching ones joined (a superset of every training
    timeline, whose random give-way only shortens them), in real 80 ms frames [first, end)."""
    out = []
    for u in merge_turns(meta['utterances']):
        s, e = int(round(u['start'] / STEP_S)), int(round(u['end'] / STEP_S))
        if out and s < out[-1][1] + 3:
            out[-1][1] = max(out[-1][1], e)
        elif e > s:
            out.append([s, e])
    return out


def alba_codes(proc, cdir, meta, cache):
    """Her speech as codec codes on the call's 80 ms grid. Each utterance is encoded on its own, like the clips
    the model learned from (the codec's encoder looks at the whole clip: the same audio inside a longer stretch
    gets only ~60% of the same codes). Frames outside her utterances stay 0 and are never used."""
    out = cache / (cdir.name + '.npy')
    if out.exists():
        return np.load(out)
    x, sr = sf.read(cdir / 'alba.flac', dtype='float32')
    x = np.clip(np.nan_to_num(x), -1, 1)
    n = len(x) // 1920 + 1
    codes = np.zeros((n, 12), np.int64)
    for s, e in code_spans(meta):
        e = min(e, n - 1)
        seg = x[s * 1920:e * 1920]
        if e <= s or len(seg) < 1920:
            continue
        c = proc.encode_audio(seg).numpy().astype(np.int64)
        k = min(len(c), e - s)
        codes[s:s + k] = c[:k]
    np.save(out, codes)
    return codes


def merge_turns(utts):
    """Joins consecutive pieces of the same turn: real speech is cut at every breath by voice detection, but a
    person plans the whole sentence. A "yeah" right before the same person's turn opens that turn ("Yeah, so
    ..."), so it joins it (run 1 learned "Yeah." as a whole reply from these). Other "mm-hm"s and cut-off pieces
    stay separate."""
    out = []
    for u in sorted(utts, key=lambda u: u['start']):
        p = out[-1] if out else None
        joinable = (p and not p.get('cut_short') and not u['backchannel'] and u['start'] - p['end'] < MERGE_GAP_S)
        if joinable:
            out[-1] = {**p, 'end': max(p['end'], u['end']), 'text': (p['text'] + ' ' + u['text']).strip(),
                       'cut_short': u.get('cut_short', False), 'backchannel': False}
        else:
            out.append(dict(u))
    return out


SELF_WORDS = re.compile(r"\b(i|i'm|im|i've|ive|i'd|id|i'll|ill|me|my|mine|myself|we|we're|we've|we'd|our|ours|us)\b",
                        re.I)


def learnable_real(u):
    """A real person's words that Alba may learn to say: "mm-hm"s, and questions or short reactions (10 words or
    fewer) that say nothing about the speaker's own life (no I / my / we). Their stories stay context only."""
    if u['backchannel']:
        return True
    t = u['text'].strip()
    return not SELF_WORDS.search(t) and (t.endswith('?') or len(t.split()) <= 10)


def person_segments(meta):
    """[(start, end, is_backchannel)] of the person Alba hears."""
    if meta.get('source') in REAL_SOURCES:
        src = ROOT / 'data' / 'seamless' / 'train' / 'meta'
        fid = meta['person_file']
        vad = [json.loads(l) for l in open(src / f'{fid}.vad.jsonl', encoding='utf8') if l.strip()]
        tr = src / f'{fid}.transcript.jsonl'
        words = [w for l in open(tr, encoding='utf8') if l.strip() for w in json.loads(l).get('words', [])
                 if isinstance(w.get('start'), (int, float)) and isinstance(w.get('end'), (int, float))] if tr.exists() else []
        return classify(merge([(v['start'], v['end']) for v in vad]), words)
    return [(s, e, False) for s, e in meta.get('person_turns', [])]


def timeline(meta, persons, rng):
    """Alba's utterances as she should say them (seconds): pieces of a turn joined; where she gave way to the
    person, she stops YIELD_S after the person started; pieces that touch become one. learn_words: whether
    her plan words are trained (not for real people's own turns, see build)."""
    onsets = [s for s, e, bc in persons if not bc]
    real = meta.get('source') == 'seamless-train'
    out = []
    for u in merge_turns(meta['utterances']):
        u['learn_words'] = learnable_real(u) if real else True
        if u.get('cut_short'):
            os_ = [o for o in onsets if u['start'] + 0.3 < o <= u['end'] + 0.25]
            if os_:
                u['end'] = min(u['end'], os_[-1] + rng.uniform(*YIELD_S))
        if out and u['start'] < out[-1]['end'] + 2 * STEP_S:
            p = out[-1]
            out[-1] = {**p, 'end': max(p['end'], u['end']), 'text': (p['text'] + ' ' + u['text']).strip(),
                       'backchannel': p['backchannel'] and u['backchannel'], 'cut_short': u.get('cut_short', False),
                       'learn_words': p['learn_words'] and u['learn_words']}
        else:
            out.append(u)
    return out


def spans_of(utts, delay):
    """(first decision step, end-of-speech step, utterance): the frame made at decision step t is heard at real
    step t + delay."""
    out = []
    for u in utts:
        s, e = int(round(u['start'] / STEP_S)) - delay, int(round(u['end'] / STEP_S)) - delay
        if s >= 1 and e > s:
            out.append((s, e, u))
    return out


def build(G, spans, codes, t0, t1, rng, delay=DELAY, listen_keep=LISTEN_KEEP, clock=None, plan_chunks='count',
          tools=None):
    """Token layout for decision steps t0..t1-1 (no span may start before t0 and end after it). Returns ids,
    labels and the frame codes (one row per <|audio_out|>). plan_chunks: 'count' (exactly 6 plan pieces per 12
    frames, as the model's own interleaved mode) or 'predicted' (runs 1-2's layout). tools: the tool list in the
    opening; an utterance with a 'tool' (tools/make_tool_convos.py) starts with the call (learned) and its result
    (given, not learned), in the model's own format, then <|mixed_start|> as usual."""
    head = G.head(clock, tools)
    ids, labels, frames = list(head), [-100] * len(head), []
    speak = {s: (e, u) for s, e, u in spans if t0 <= s < t1}
    t = t0
    while t < t1:
        ids.append(G.audio_in); labels.append(-100)
        if t in speak:
            e, u = speak[t]
            text = G.tok(u['text'], add_special_tokens=False)['input_ids'] + [G.text_end]
            if u.get('tool'):
                call = G.tok(u['tool']['call'], add_special_tokens=False)['input_ids']
                resp = G.tok(u['tool']['response'], add_special_tokens=False)['input_ids']
                ids += call; labels += call                       # she writes the call
                ids += resp; labels += [-100] * len(resp)         # the result comes from the tool
            ids.append(G.mixed_start); labels.append(G.mixed_start)
            ti, n_in_block = 0, 0
            for step in range(t, e + 1):
                if step > t:
                    ids.append(G.audio_in); labels.append(-100)
                chunk_here = ti < len(text) and n_in_block == 0
                if chunk_here:
                    # 6 plan pieces per 12 frames, as in the model's own interleaved mode. 'count' (run 3 on): never
                    # more, as the model's own mode, which switches to speech by count; 'predicted' (runs 1-2): the
                    # last block before a natural end carries everything left (0.2% of chunks were longer than 6)
                    last_block = plan_chunks == 'predicted' and (e - step) <= N_AUDIO and not u.get('cut_short')
                    chunk = text[ti:] if last_block else text[ti:ti + N_TEXT]
                    ids += chunk; ti += len(chunk)
                    # words she learns to say: her own conversations and "mm-hm"s; a real person's life story,
                    # re-spoken in her voice, is context only (she must not learn to claim a human life)
                    labels += chunk if u.get('learn_words', True) else [-100] * len(chunk)
                if step == e:
                    frame = np.full(codes.shape[1], G.eos_code)           # end of speech (or given way)
                else:
                    frame = codes[min(step + delay, len(codes) - 1)]
                # 'predicted' (runs 1-2) also asked for "<|audio_out|>" after a chunk; it cannot be learned (the
                # release never trained that placeholder's row, see app/duplex.AlbaDuplex._plan), so 'count' does not
                ask = chunk_here and plan_chunks == 'predicted'
                ids.append(G.audio_out); labels.append(G.audio_out if ask else -100); frames.append(frame)
                n_in_block = (n_in_block + 1) % N_AUDIO if ti < len(text) else 1
                if step >= t1 - 1:
                    break
            t = e + 1
            continue
        ids.append(G.listen)
        labels.append(G.listen if rng.random() < listen_keep else -100)
        t += 1
    return ids, labels, np.stack(frames) if frames else np.zeros((0, codes.shape[1]), np.int64)


def pick_window(spans, n_steps, win, rng):
    t0 = rng.randint(0, max(0, n_steps - win))
    for s, e, _ in spans:              # never start inside one of her utterances
        if s < t0 <= e:
            t0 = e + 1
    return t0, min(n_steps, t0 + win)


def room_mic(x, rng):
    """What a laptop microphone in a room would hear, timing unchanged (the real conversations are studio-clean)."""
    x = np.asarray(x, np.float64)
    n = int(rng.uniform(0.12, 0.45) * SR_U)
    ir = rng.standard_normal(n) * np.exp(-6.9 * np.arange(n) / n)
    x = x + rng.uniform(0.05, 0.3) * signal.fftconvolve(x, ir / np.sqrt(np.sum(ir ** 2)))[:len(x)]
    b, a = signal.butter(2, [rng.uniform(80, 200) / 8000, rng.uniform(5500, 7800) / 8000], 'band')
    x = signal.lfilter(b, a, x)
    voiced = x[np.abs(x) > 1e-3]
    rms = np.sqrt(np.mean(voiced ** 2)) if len(voiced) else 1e-3
    x = x * 10 ** (rng.uniform(-34, -18) / 20) / rms
    return x + rng.standard_normal(len(x)) * 10 ** (rng.uniform(-66, -48) / 20)


def own_echo(alba24, utts, rng):
    """Alba's own voice as the microphone picks it up from the laptop speakers (after the echo canceller's
    leftovers): only while she really talks (her utterances as trained), delayed, reverberant, band-limited,
    a little distorted, -35 to -8 dB."""
    mask = np.zeros(len(alba24), np.float32)
    for u in utts:                     # times are relative to the window: clip (a negative index would wrap)
        a, b = int(max(0.0, u['start']) * SR_A), int(max(0.0, u['end']) * SR_A)
        if b > a:
            mask[a:b] = 1
    y = signal.resample_poly(alba24 * mask, 2, 3)
    if rng.random() < 0.5:
        y = np.tanh(rng.uniform(1, 3) * y) / rng.uniform(1, 3)
    n = int(0.25 * SR_U)
    d0 = int(rng.uniform(0.005, 0.04) * SR_U)
    ir = np.zeros(n)
    ir[d0:] = rng.standard_normal(n - d0) * np.exp(-np.arange(n - d0) / (rng.uniform(0.01, 0.06) * SR_U))
    ir[d0] += rng.uniform(1, 4)
    b, a = signal.butter(2, [rng.uniform(150, 400) / 8000, rng.uniform(4000, 7500) / 8000], 'band')
    ir = signal.lfilter(b, a, ir)
    ir /= np.sqrt(np.sum(ir ** 2))
    return signal.fftconvolve(y, ir)[:len(y)] * 10 ** (rng.uniform(-35, -8) / 20)


class Data:
    """Conversations, sampled by source share (within a source by length, capped at 10 minutes), each windowed
    at random. Sources not in the shares are not used."""

    def __init__(self, root, rng, share=None):
        self.rng = rng
        self.share = dict(share or self.SHARE)
        self.convs = sorted(p.parent for p in Path(root).rglob('meta.json'))
        self.meta = {c: json.loads((c / 'meta.json').read_text(encoding='utf8')) for c in self.convs}
        groups = {}
        for c in self.convs:
            groups.setdefault(self.meta[c].get('source', '?'), []).append(c)
        self.groups = groups
        self.persons = {}

    def describe(self):
        return {k: (len(v), round(sum(self.meta[c]['seconds'] for c in v) / 3600, 2)) for k, v in self.groups.items()}

    # run 2's shares; run 3 on (--shares): 'seamless-own' (the same real conversations with her own words,
    # tools/make_seamless_own.py) instead of 'seamless-train' (words mostly context only)
    SHARE = {'seamless-train': 0.33, 'synthetic': 0.45, 'feeling': 0.14, 'asides': 0.08}

    def pick(self):
        srcs = sorted(s for s in self.groups if self.share.get(s, 0) > 0)
        src = self.rng.choices(srcs, weights=[self.share[s] for s in srcs])[0]
        cs = self.groups[src]
        w = [min(600.0, self.meta[c]['seconds']) for c in cs]     # one very long call must not dominate
        return self.rng.choices(cs, weights=w)[0]


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=str(ROOT / 'data' / 'duplex'))
    ap.add_argument('--steps', type=int, default=600)
    ap.add_argument('--accum', type=int, default=4)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--rank', type=int, default=16)
    ap.add_argument('--window-s', type=float, default=40.0)
    ap.add_argument('--delay', type=int, default=DELAY)
    ap.add_argument('--echo', type=float, default=0.6, help='share of windows with her own voice leaking back')
    ap.add_argument('--mic', type=float, default=0.6, help='share of windows given a laptop-mic sound')
    ap.add_argument('--mic-sources', default='real', choices=['all', 'real'],
                    help="'real': only the real conversations (the synthetic callers already carry a laptop-mic sound "
                         "from tools/make_convo_audio.py); 'all': every source")
    ap.add_argument('--init', default='', help='start from an earlier LoRA (e.g. the half-duplex Alba)')
    ap.add_argument('--out', default=str(ROOT / 'models' / 'alba-duplex-lora'))
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--shares', default='{"seamless-own": 0.33, "synthetic": 0.45, "feeling": 0.14, "asides": 0.08}',
                    help='source shares as JSON; sources not named are not used (run 2: seamless-train 0.33 instead '
                         'of seamless-own)')
    ap.add_argument('--tools-head', type=float, default=0.0,
                    help='share of windows without a tool call whose opening still lists the tools (so a listed '
                         'tool is not a cue to call one); conversations with tool calls always list them')
    ap.add_argument('--plan-chunks', default='count', choices=['count', 'predicted'],
                    help="'count': 6 plan pieces per 12 frames, as the model's own mode (run 3 on)")
    args = ap.parse_args()
    torch.manual_seed(0)
    rng = random.Random(0)
    nrng = np.random.default_rng(0)
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    proc, model = load()
    G = Grammar(proc, model)
    base_cfg = model.config
    for p in model.parameters():
        p.requires_grad_(False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads()
    target = r"model\.(language_model|depth_decoder)\..*\.(q_proj|k_proj|v_proj|o_proj|in_proj|out_proj|w1|w2|w3)"
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, bias='none',
                                             target_modules=target, trainable_token_indices=[G.listen]))
    if args.init:
        from safetensors.torch import load_file
        res = set_peft_model_state_dict(model, load_file(str(Path(args.init) / 'adapter_model.safetensors')))
        print(f'started from {args.init} (keys not in it: {len(res.missing_keys)})', flush=True)
    model.print_trainable_parameters()
    data = Data(args.data, rng, share=json.loads(args.shares))
    print('data (conversations, hours):', data.describe(), flush=True)
    cache = Path(args.data) / '_codes_utt2'      # spans as merge_turns joins them (run 2: openers join their turn)
    cache.mkdir(exist_ok=True)
    # her speech -> codec codes, once, on the graphics card (the processor manages ~5x real time; this ~100x+),
    # then the codec leaves the card for training
    todo = [c for c in data.convs if not (cache / (c.name + '.npy')).exists()]
    if todo and not args.check:
        proc.audio_tokenizer.to('cuda')
        t0 = time.time()
        for k, c in enumerate(todo):
            with torch.no_grad():
                alba_codes(proc, c, data.meta[c], cache)
            if k % 200 == 0:
                print(f'codes {k + 1}/{len(todo)} | {time.time() - t0:.0f}s', flush=True)
        proc.audio_tokenizer.to('cpu')
        torch.cuda.empty_cache()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))
    warm = max(10, args.steps // 30)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps)))))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    settings = {'delay_steps': args.delay, 'listen_keep': LISTEN_KEEP, 'listen_bias': round(math.log(1 / LISTEN_KEEP), 4),
                'yield_s': YIELD_S, 'window_s': args.window_s, 'system': SYSTEM, 'header': HEADER,
                'listen_token': LISTEN_TOKEN, 'data': data.describe(), 'init': args.init, 'plan_chunks': args.plan_chunks,
                'mic': args.mic, 'mic_sources': args.mic_sources, 'shares': data.share,
                'tools': bool(args.tools_head > 0 or data.share.get('tools', 0) > 0), 'tools_head': args.tools_head}
    tool_specs = None
    if settings['tools']:
        from make_tool_convos import SPECS as tool_specs
    model.train()
    step = micro = 0
    t_start = time.time()
    hist = {'text': [], 'audio': []}
    win = int(args.window_s / STEP_S)
    while step < args.steps:
        cdir = data.pick()
        micro += 1
        meta = data.meta[cdir]
        if cdir not in data.persons:
            data.persons[cdir] = person_segments(meta)
        crng = np.random.default_rng(zlib.crc32(f'{cdir.name}-{micro}'.encode()))
        utts = timeline(meta, data.persons[cdir], crng)
        spans = spans_of(utts, args.delay)
        with torch.no_grad():
            codes = alba_codes(proc, cdir, meta, cache)
        n_steps = min(len(codes) - args.delay, (sf.info(str(cdir / 'user.flac')).frames - LOOK) // 1280)
        t0, t1 = pick_window(spans, n_steps, win, rng)
        if t1 - t0 < 25:
            continue
        clock = meta.get('clock') or (fake_clock(rng) if rng.random() < 0.3 else None)
        ids, labels, frames = build(G, spans, codes, t0, t1, rng, delay=args.delay, clock=clock,
                                    plan_chunks=args.plan_chunks,
                                    tools=meta.get('tools') or (tool_specs if tool_specs and nrng.random() < args.tools_head
                                                                else None))
        # only the window's slice of each recording is read
        n_a = (t1 - t0) * 1280 + LOOK
        a, _ = sf.read(cdir / 'user.flac', start=t0 * 1280, stop=t0 * 1280 + n_a, dtype='float64')
        a = np.pad(a, (0, n_a - len(a)))
        if (args.mic_sources == 'all' or meta.get('source') in REAL_SOURCES) and nrng.random() < args.mic:
            a = room_mic(a, nrng)
        if nrng.random() < args.echo:
            s24 = int(t0 * STEP_S * SR_A)
            seg, _ = sf.read(cdir / 'alba.flac', start=s24, stop=s24 + int(n_a * 1.5), dtype='float32')
            local = [{'start': u['start'] - t0 * STEP_S, 'end': u['end'] - t0 * STEP_S} for u in utts]
            e = own_echo(seg, local, nrng)
            a[:len(e)] += e[:len(a)]
        a = np.clip(a, -1, 1).astype(np.float32)
        feats = proc.feature_extractor(a, sampling_rate=16000, return_tensors='pt')
        x = feats['input_features'].to('cuda', torch.bfloat16)
        fmask = feats['feature_attention_mask'].to('cuda').bool()
        need = ids.count(G.audio_in)
        if micro == 1 or args.check:
            got = int(model.get_base_model().model.get_audio_features(x, fmask).shape[0])
            if got != need:
                raise SystemExit(f'encoder gives {got} tokens for {need} steps: the layout is off')
        # word scores only where a word is trained (~1 position in 10): the model's own loss, without
        # computing a 125k-wide score row for every position (that alone was ~0.8 GB on a 4 GB card)
        pos = [p for p in range(len(labels) - 1) if labels[p + 1] != -100]
        if not pos:
            continue
        batch = {'input_ids': torch.tensor([ids], device='cuda'), 'attention_mask': torch.ones(1, len(ids), dtype=torch.long, device='cuda'),
                 'input_features': x, 'feature_attention_mask': fmask,
                 'audio_out_codes': torch.from_numpy(frames).to('cuda'), 'audio_labels': torch.from_numpy(frames.copy()).to('cuda'),
                 'logits_to_keep': torch.tensor(pos, device='cuda')}
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = model(**batch)
            text_loss = F.cross_entropy(out.logits[0].float(), torch.tensor([labels[p + 1] for p in pos], device='cuda'))
        # pooled like the model's own combined_loss: each part weighted by its count and configured weight
        tw, aw = base_cfg.text_loss_weight * len(pos), base_cfg.audio_loss_weight * len(frames)
        loss = (tw * text_loss + aw * out.audio_loss) / (tw + aw)
        (loss / args.accum).backward()
        hist['text'].append(float(text_loss)); hist['audio'].append(float(out.audio_loss))
        if args.check:
            n_start = labels.count(G.mixed_start)
            print(f'check: {cdir.name} steps {t0}-{t1} | {len(ids)} tokens, {len(frames)} frames, {n_start} starts, '
                  f'{sum(1 for f in frames if f[0] == G.eos_code)} stops | text {float(text_loss):.3f} '
                  f'audio {float(out.audio_loss):.3f} | gpu peak {torch.cuda.max_memory_allocated() / 1e9:.2f} GB', flush=True)
        if micro % args.accum == 0:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % 10 == 0 or args.check:
                print(f"step {step}/{args.steps} | text {np.mean(hist['text'][-40:]):.3f} | audio {np.mean(hist['audio'][-40:]):.3f} "
                      f"| {(time.time() - t_start) / step:.1f}s/step | {len(ids)} tokens | gpu peak "
                      f"{torch.cuda.max_memory_allocated() / 1e9:.2f} GB", flush=True)
            if step % 100 == 0:
                model.save_pretrained(args.out)
                (out_dir / 'duplex.json').write_text(json.dumps({**settings, 'steps_done': step}, indent=1), encoding='utf8')
            if args.check:
                break
    model.save_pretrained(args.out)
    (out_dir / 'duplex.json').write_text(json.dumps({**settings, 'steps_done': step}, indent=1), encoding='utf8')
    print('saved', args.out, flush=True)


if __name__ == '__main__':
    main()
