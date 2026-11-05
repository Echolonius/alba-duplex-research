"""Alba Duplex runtime: one model that listens and speaks at the same time, in 80 ms steps.

Split for speed on this laptop:
  language model         llama.cpp's own library (app/llama_lib.py), 4-bit GGUF, on the graphics card
  hearing                the model's streaming speech encoder + projector (PyTorch, bf16, graphics card)
  speech frames          the model's depth decoder (MaskGIT over 12 codebooks) and codebook embeddings (PyTorch)
  sound                  the Liquid codec's streaming decoder (PyTorch), 24 kHz
Every input to the language model is a vector: a slice of your voice, one of Alba's frames, or a word-piece's
embedding row, so one llama.cpp call per step carries "what I just did" and "what I just heard".

The step grammar is the one tools/train_duplex.py teaches (see its docstring).

Timing (the same in training, the tests and live use):
  * your voice is cut into 80 ms steps; the encoder's sound window reaches 15 ms past each step, so the first
    push is FIRST_STEP samples (95 ms) and every later one STEP samples: step k's token exists 95 ms after the
    step began.
  * Alba's frame made at step k is played DELAY steps after step k began (240 ms: 95 ms of hearing, up to
    145 ms of thinking). Training shifts her speech by the same DELAY, so the timing she learned from real
    people is the timing you hear.
"""
import json
import re
import time
from importlib import import_module
from pathlib import Path

import numpy as np
import torch

from .llama_lib import LM

BASE_HF = Path(__file__).resolve().parent.parent / 'models' / 'lfm3-audio-hf'   # the untouched release

STEP = 1280               # 80 ms at 16 kHz in
FIRST_STEP = STEP + 240   # the first push also carries the encoder's 15 ms look past the step
OUT_STEP = 1920           # 80 ms at 24 kHz out
DELAY = 3                 # steps from the start of a step to when Alba's frame from it is heard
MAX_UTTERANCE_STEPS = 500  # safety rail: no single utterance longer than 40 s
N_TEXT, N_AUDIO = 6, 12
LISTEN_TOKEN = '<|img_row_1_col_1|>'
SYSTEM = 'You are Alba, in a live voice call. Listen and talk at the same time, like a person.'
HEADER = '<|im_start|>call\n'
EMPTY = np.zeros(0, np.float32)


def system_prompt(clock=None):
    """The trained opening, with the time as one more line (as in training), so she can say what time it is."""
    if clock is None:
        import datetime
        now = datetime.datetime.now()
        clock = f"It is {now.strftime('%A, %B')} {now.day}, {now.year}, {now.strftime('%I:%M %p').lstrip('0')} here."
    return SYSTEM + (f'\n{clock}' if clock else '')


TOOL_CALL = re.compile(r'<function=([A-Za-z0-9_]+)>\s*(.*?)</function>', re.S)
TOOL_PARAM = re.compile(r'<parameter=([A-Za-z0-9_]+)>\n?(.*?)\n?</parameter>', re.S)


def parse_tool_call(text):
    """The model's own tool-call text (models/lfm3-audio-hf/chat_template.jinja) -> (name, arguments), or None."""
    m = TOOL_CALL.search(text)
    if not m:
        return None
    return m.group(1), {k: v.strip() for k, v in TOOL_PARAM.findall(m.group(2))}


def tool_specs(tools):
    """What the opening lists for each tool: name, description, parameters (no addresses or bookkeeping)."""
    return [{k: t[k] for k in ('name', 'description', 'parameters') if k in t} for t in tools]


def _sample_top_p(logits, temperature, top_p, exclude, rng):
    """A word-piece from the smallest set of most likely pieces whose probability adds up to top_p."""
    lg = logits.astype(np.float64).copy()
    lg[exclude] = -np.inf
    p = np.exp((lg - lg.max()) / max(temperature, 1e-6))
    p /= p.sum()
    order = np.argsort(-p)
    keep = order[:int(np.searchsorted(np.cumsum(p[order]), top_p)) + 1]
    q = p[keep] / p[keep].sum()
    return int(rng.choice(keep, p=q))


def _sample(logits, temperature=0.8, top_k=40, allowed=None, rng=None):
    rng = rng or np.random.default_rng()
    lg = logits.astype(np.float64)
    if allowed is not None:
        mask = np.full_like(lg, -np.inf)
        mask[allowed] = lg[allowed]
        lg = mask
    if temperature <= 0:
        return int(np.argmax(lg))
    k = min(top_k, int(np.isfinite(lg).sum()))
    idx = np.argpartition(-lg, k - 1)[:k]
    p = np.exp((lg[idx] - lg[idx].max()) / temperature)
    return int(rng.choice(idx, p=p / p.sum()))


class AlbaDuplex:
    """n_ctx: the model's memory in positions (about 30 per second of call: 32768 is ~18 minutes). Before it
    fills, at a moment she is listening, the memory is rebuilt from the call's opening plus its last
    keep_tokens positions (about a minute), so a call can go on for as long as you like."""

    def __init__(self, gguf, hf_dir, device='cuda', n_ctx=32768, threads=4, seed=0, gpu_layers=-1, keep_tokens=2400,
                 eos_first=False):
        import os
        os.environ.setdefault('HF_HUB_OFFLINE', '1')
        os.environ.setdefault('NO_TORCH_COMPILE', '1')
        from transformers import AutoModelForMultimodalLM, AutoProcessor
        self.device = device
        # bf16 on the graphics card; float32 on the processor (it has no fast bf16 maths)
        self.dtype = torch.bfloat16 if str(device).startswith('cuda') else torch.float32
        if self.dtype == torch.float32:
            torch.set_num_threads(max(1, threads))
        self.rng = np.random.default_rng(seed)
        if str(device).startswith('cuda'):
            # PyTorch and llama.cpp both use CUDA 13 libraries with the same file names: let PyTorch's (the newer
            # minor version) load first, so both share them (llama.cpp's build runs on a newer minor; not the reverse)
            (torch.ones(8, 8, device=device) @ torch.ones(8, 8, device=device)).sum().item()
        self.lm = LM(gguf, n_ctx=n_ctx, threads=threads, gpu_layers=gpu_layers,
                     backends='all' if str(device).startswith('cuda') else 'cpu')
        self.n_ctx, self.keep_tokens = n_ctx, keep_tokens
        # how the model was taught to end a plan chunk (tools/train_duplex.build): 'count' (run 3 on) or
        # 'predicted' (runs 1-2, whose settings do not say)
        import json
        sp = Path(gguf).parent / 'duplex.json'
        self.plan_chunks = (json.loads(sp.read_text(encoding='utf8')).get('plan_chunks', 'predicted')
                            if sp.exists() else 'predicted')
        hf_dir = Path(hf_dir)
        trained = None
        if not (hf_dir / 'model.safetensors').exists() and (hf_dir.parent / 'trained.safetensors').exists():
            # a slim export (tools/export_alba.py): the untouched release + what training changed outside the
            # language model (which runs from the GGUF)
            trained, hf_dir = hf_dir.parent / 'trained.safetensors', BASE_HF
        self.proc = AutoProcessor.from_pretrained(hf_dir, trust_remote_code=True, local_files_only=True)
        model = AutoModelForMultimodalLM.from_pretrained(hf_dir, dtype=self.dtype, device_map='cpu',
                                                         trust_remote_code=True, local_files_only=True).eval()
        if trained is not None:
            from safetensors.torch import load_file
            sd = load_file(str(trained))
            idx, val = sd.pop('embed_rows.idx', None), sd.pop('embed_rows.val', None)
            res = model.load_state_dict({k: v.to(self.dtype) for k, v in sd.items()}, strict=False)
            if res.unexpected_keys:
                raise RuntimeError(f'trained.safetensors does not fit the model: {res.unexpected_keys[:5]}')
            if idx is not None:
                with torch.no_grad():
                    model.model.language_model.embed_tokens.weight[idx] = val.to(self.dtype)
        model.requires_grad_(False)             # inference only: no gradient bookkeeping anywhere
        m = model.model
        self.emb = m.language_model.embed_tokens.weight.detach().to(torch.bfloat16).clone()   # word-piece rows (CPU)
        m.language_model = torch.nn.Identity()                                    # llama.cpp runs the language model
        for name in ('audio_encoder', 'audio_projector', 'codebook_embedding', 'depth_decoder'):
            getattr(m, name).to(device)
        self.model, self.m = model, m
        self.codec = self.proc.audio_tokenizer.to(device).eval()
        self.codec.requires_grad_(False)
        self._cache_cls = getattr(import_module(type(self.codec).__module__), 'LiquidCodecStreamingCache')
        tok = self.proc.tokenizer
        c = model.config
        self.ids = {'listen': tok.convert_tokens_to_ids(LISTEN_TOKEN), 'mixed_start': c.mixed_start_token_id,
                    'text_end': c.text_end_token_id, 'im_end': tok.convert_tokens_to_ids('<|im_end|>'),
                    'audio_out': c.audio_out_token_id}
        self.eos_code = self.proc.audio_eos_code
        self.tok = tok
        # control tokens never sampled as plan words (the structure is decided separately, see _plan): every added
        # token (special markers, image tiles, tool and thinking tags: tok.all_special_ids alone names only 7 of
        # them) and the padding rows past the tokenizer's last id (the model has 128000 rows for 125033 ids)
        self.control = np.array(sorted(set(tok.all_special_ids) | set(tok.added_tokens_decoder) |
                                       set(range(len(tok), self.emb.shape[0])) | {self.ids[k] for k in self.ids} |
                                       {c.audio_token_id}), dtype=np.int64)
        self._control_set = set(self.control.tolist())
        # tool calls (off unless start() gets tools): the model's own format, <tool_call> ... </tool_call>
        call_ids = {k: tok.convert_tokens_to_ids(t) for k, t in (
            ('tool_call', '<tool_call>'), ('tool_call_end', '</tool_call>'), ('function', '<function='),
            ('function_end', '</function>'), ('parameter', '<parameter='), ('parameter_end', '</parameter>'))}
        self.ids.update({'tool_call': call_ids['tool_call'], 'tool_call_end': call_ids['tool_call_end']})
        self._no_call = np.array(sorted(self._control_set - set(call_ids.values())), dtype=np.int64)
        self.tools, self.tool_runner = None, None
        self.plan_log = None      # diagnostics only: a list to record each plan decision in
        self.started = False
        self.fast_enc, self._fbuf, self.fast_frames, self.fast_codec = None, None, None, None
        if str(device).startswith('cuda'):
            from .fast_audio import FastCodecDecoder, FastEncoderStream, FastFrameSampler
            self.fast_enc = FastEncoderStream(model, device=device)
            # eos_first: the stop decided by one codebook's draw (app/fast_audio.FastFrameSampler); off = the
            # model's own sampler exactly
            self.fast_frames = FastFrameSampler(model, device=device, temperature=1.0, top_k=128, repetition_penalty=1.1,
                                                eos_first=eos_first)
            self.fast_codec = FastCodecDecoder(self.codec, device=device)

    # ------------------------------------------------------------------------------------------------------
    def vec(self, token_id):
        return self.emb[token_id].float().numpy()

    def start(self, system=None, header=HEADER, tools=None, tool_runner=None):
        """A new call: clears the model's memory and every stream. system: default the trained prompt + now.
        tools: the tools this call offers (the local API's list); the opening lists them in the model's own
        <tools> format and she may call one where she would otherwise start to talk; tool_runner(name, args) ->
        result dict runs it (the call app: the local API, which runs a tool that changes something only after
        a spoken yes). None: no tools, exactly as before."""
        system = system_prompt() if system is None else system
        self.lm.rewind(0)
        self.tools = tool_specs(tools) if tools else None
        self.tool_runner = tool_runner if tools else None
        # the same tokenizer as training (tools/train_duplex.Grammar.head), so the opening is identical
        msgs = [{'role': 'system', 'content': system}]
        text = (self.proc.apply_chat_template(msgs, tools=self.tools, tokenize=False) if self.tools else
                self.proc.apply_chat_template(msgs, tokenize=False)) + header
        self.head = self.tok(text, add_special_tokens=False)['input_ids']
        self.lm.feed_tokens(self.head)
        self.fe_cache = self.enc_cache = self.codec_cache = None
        if self.fast_enc is not None:
            self.fast_enc.reset()
        if self.fast_frames is not None:
            self.fast_frames.reset()
        self.pending = None
        self.state, self.block, self.text_done, self.utt_steps = 'listen', 0, True, 0
        self.history, self.said, self.events = [], [], []
        self.trace = []           # per step: the vectors fed (float16), to rebuild the memory in long calls
        self.n_steps = 0
        self.started = True

    def _feed(self, vecs):
        vecs = np.asarray(vecs, np.float32).reshape(-1, self.lm.n_embd)
        self.trace[-1].append(vecs.astype(np.float16))
        return self.lm.feed_vectors(vecs)

    def _rebuild_if_full(self):
        """Near the end of the memory, while she listens: keep the opening and the last keep_tokens positions."""
        if self.lm.pos < self.n_ctx - 600 or self.state != 'listen':
            return False
        keep, total = [], 0
        for s in reversed(self.trace[:-1]):
            n = sum(len(v) for v in s)
            if total + n > self.keep_tokens:
                break
            keep.append(s)
            total += n
        keep.reverse()
        # cleared and replayed from the start: the model's convolution layers keep only their latest state, so
        # llama.cpp cannot rewind it to just after the opening (that failed with llama_decode -1)
        self.lm.rewind(0)
        self.lm.feed_tokens(self.head)
        if keep:
            self.lm.feed_vectors(np.concatenate([v for s in keep for v in s]).astype(np.float32))
        self.trace = keep + [self.trace[-1]]
        self.events.append(('rebuilt', self.n_steps - 1, time.time()))
        return True

    @torch.no_grad()
    def hear(self, pcm):
        """Your newest audio (16 kHz float32) -> the encoder token of the step it completes, or None."""
        if self.fe_cache is None and self.fast_enc is not None:      # a new stream
            self.fast_enc.reset()
            self._fbuf = None
        feats, self.fe_cache = self.proc.feature_extractor.stream(pcm, self.fe_cache)
        x = feats['input_features']
        if x is None or x.shape[-1] == 0:
            return None
        if self.fast_enc is not None:
            # app/fast_audio.py: the same encoder step, replayed as a captured CUDA graph (~13x faster here);
            # a token whenever 8 more frames are complete, as the library's stream
            self._fbuf = x if self._fbuf is None else torch.cat([self._fbuf, x], dim=-1)
            tok = None
            while self._fbuf.shape[-1] >= 8:
                tok = self.fast_enc.push(self._fbuf[..., :8])
                self._fbuf = self._fbuf[..., 8:]
            return None if tok is None else tok.float().cpu().numpy()
        tok, self.enc_cache = self.m.get_audio_features_stream(x.to(self.device, self.dtype), self.enc_cache)
        if tok.shape[1] == 0:
            return None
        return tok[0, -1].float().cpu().numpy()

    def _plan(self, logits, cap=24, temperature=0.7, top_p=0.9):
        """Word-pieces of plan, each fed back, until the model says the speech comes next or ends its plan
        (<|text_end|>). The structure is the model's most likely choice, the words are sampled from the nucleus
        (top-p) of its word distribution, control tokens excluded.
        A chunk ends when the model's top pick is any control token other than <|text_end|>. Training asked for
        <|audio_out|> there, but in the release that token is a placeholder (its input is always replaced by the
        frame) whose row was never trained: its row is one of the untrained, near-identical special-token rows,
        so the most the model can learn is to point at that cluster, and its top pick lands on one of them
        (measured: tools/diag_duplex_decode.py). Checking for <|audio_out|> alone missed every chunk end and
        sampled a word from the flat rest instead: run 2's stray word-pieces and the early stops after them.
        Models trained with plan_chunks 'count' (run 3 on) end every chunk after N_TEXT pieces, as the model's own
        interleaved mode does; the control-token rule stays as a guard.
        Returns the last (logits, hidden), hidden None if nothing was added."""
        hidden = None
        for n in range(cap):
            if self.plan_chunks == 'count' and n >= N_TEXT:
                break
            best = int(np.argmax(logits))
            if best != self.ids['text_end'] and best in self._control_set:
                if self.plan_log is not None:
                    self.plan_log.append(('chunk_end', self.n_steps - 1, self.utt_steps,
                                          self.tok.convert_ids_to_tokens(best)))
                break
            t = best if best == self.ids['text_end'] else _sample_top_p(logits, temperature, top_p, self.control, self.rng)
            if self.plan_log is not None:
                p = np.exp(logits.astype(np.float64) - logits.max())
                self.plan_log.append(('piece', self.n_steps - 1, self.utt_steps, self.tok.convert_ids_to_tokens(t),
                                      float(p[best] / p.sum())))
            self.said.append(t)
            logits, hidden = self._feed(self.vec(t)[None])
            if t == self.ids['text_end']:
                self.text_done = True
                break
        return logits, hidden

    def _tool_call(self, cap=120):
        """She calls a tool: the call is written greedily in the model's own format until </tool_call> (special
        markers and padding rows never), run by self.tool_runner, and the result fed back as the model's own
        <tool_response> text. The step takes longer than 80 ms; the caller's audio waits and is heard after."""
        ids = [self.ids['tool_call']]
        logits, _ = self._feed(self.vec(ids[0])[None])
        for _ in range(cap):
            lg = logits.astype(np.float64)
            lg[self._no_call] = -np.inf
            t = int(np.argmax(lg))
            ids.append(t)
            logits, _ = self._feed(self.vec(t)[None])
            if t == self.ids['tool_call_end']:
                break
        call = parse_tool_call(self.tok.decode(ids, skip_special_tokens=False))
        if call is None:
            result = {'error': 'the tool call could not be read'}
        elif call[0] not in {t['name'] for t in self.tools or []}:
            result = {'error': f'there is no tool called {call[0]}'}
        else:
            try:
                result = self.tool_runner(*call) if self.tool_runner else {'error': 'no tools in this call'}
            except Exception as e:                  # the tool's own failure, said back to her as the API would
                result = {'error': f'{type(e).__name__}: {e}'[:200]}
        self.events.append(('tool', self.n_steps - 1, time.time(), call, result))
        resp = self.tok('<tool_response>\n' + json.dumps(result, ensure_ascii=False) + '\n</tool_response>',
                        add_special_tokens=False)['input_ids']
        return self._feed(np.stack([self.vec(t) for t in resp]))

    @torch.no_grad()
    def frame(self, hidden, temperature=1.0, top_k=128, repetition_penalty=1.1):
        """One speech frame (12 codes) from the language model's state before it."""
        if self.fast_frames is not None:
            # app/fast_audio.FastFrameSampler: the model's own sampler as one captured CUDA graph (~15x faster;
            # same picks as the model's in 30/30 greedy checks at full precision, tools/test_fast_frames.py)
            return self.fast_frames.sample(hidden)
        h = torch.from_numpy(hidden).to(self.device, self.dtype)
        hist = torch.stack(self.history[-64:]) if self.history else None
        return self.model.sample_audio_frame(h, temperature=temperature, top_k=top_k,
                                             repetition_penalty=repetition_penalty, history=hist)

    @torch.no_grad()
    def sound(self, codes):
        """One frame -> its sound (the codec holds back a short overlap until the next frame or the flush)."""
        if self.fast_codec is not None:
            # app/fast_audio.FastCodecDecoder: the codec's streaming decode as a captured CUDA graph (~30x faster;
            # same sample counts; as close to full precision as the library's own 16-bit decode)
            self.codec_cache = True
            return self.fast_codec.frame(codes).float().cpu().numpy().reshape(-1)
        if self.codec_cache is None:
            self.codec_cache = self._cache_cls(self.codec.config)
        out = self.codec.decode(codes.view(1, -1, 1), streaming_cache=self.codec_cache, finalize=False,
                                return_dict=True)
        return out.audio_values[0].float().cpu().numpy().reshape(-1)

    @torch.no_grad()
    def flush(self):
        """End of an utterance: the held-back tail of the sound, and a fresh codec stream for the next one."""
        if self.codec_cache is None:
            return EMPTY
        if self.fast_codec is not None:
            self.codec_cache = None
            return self.fast_codec.flush().float().cpu().numpy().reshape(-1)
        empty = torch.zeros(1, self.codec.config.n_q, 0, dtype=torch.long, device=self.device)
        out = self.codec.decode(empty, streaming_cache=self.codec_cache, finalize=True, return_dict=True)
        self.codec_cache = None
        return out.audio_values[0].float().cpu().numpy().reshape(-1)

    def step(self, pcm, force_start=False, listen_bias=0.0):
        """One 80 ms step. pcm: your newest audio (FIRST_STEP samples on the first call of a call, then STEP).
        Returns Alba's sound made in this step (24 kHz float32): empty while she listens, about OUT_STEP
        samples while she talks, plus the codec's short tail when she stops."""
        if not self.started:
            self.start()
        u = self.hear(np.asarray(pcm, np.float32))
        if u is None:
            return EMPTY
        self.n_steps += 1
        self.trace.append([])
        self._rebuild_if_full()
        vecs = ([self.pending] if self.pending is not None else []) + [u]
        logits, hidden = self._feed(np.stack(vecs))
        self.pending = None
        if self.state == 'listen':
            lg = logits.copy()
            lg[self.ids['listen']] += listen_bias
            allowed = [self.ids['listen'], self.ids['mixed_start']] + ([self.ids['tool_call']] if self.tools else [])
            choice = self.ids['mixed_start'] if force_start else _sample(
                lg, temperature=1.0, top_k=len(allowed), allowed=allowed, rng=self.rng)
            if choice == self.ids['listen']:
                self.pending = self.vec(self.ids['listen'])
                return EMPTY
            if choice == self.ids['tool_call']:
                # as trained: the call, its result, then the utterance that uses it starts in this same step
                self._tool_call()
            self.events.append(('start', self.n_steps - 1, time.time()))
            logits, hidden = self._feed(self.vec(self.ids['mixed_start'])[None])
            self.state, self.block, self.text_done, self.utt_steps = 'speak', 0, False, 0
            logits, h2 = self._plan(logits)
            hidden = h2 if h2 is not None else hidden
        elif self.block == 0 and not self.text_done:
            logits, h2 = self._plan(logits)
            hidden = h2 if h2 is not None else hidden
        codes = self.frame(hidden)
        self.utt_steps += 1
        if self.utt_steps > MAX_UTTERANCE_STEPS:
            # safety rail (hand-written, documented): a runaway utterance is ended here
            codes = torch.full_like(codes, self.eos_code)
            self.events.append(('hard_stop', self.n_steps - 1, time.time()))
        if bool((codes == self.eos_code).any()):
            # as the model's own generator: any end-of-speech code ends the utterance, and the frame fed back is
            # the full end-of-speech frame (what training showed after a stop)
            codes = torch.full_like(codes, self.eos_code)
            self.pending = self.m.get_audio_out_embeds(codes[None]).float().cpu().numpy()[0]
            self.state = 'listen'
            self.events.append(('stop', self.n_steps - 1, time.time(), self.utt_steps, self.text_done))
            return self.flush()
        self.pending = self.m.get_audio_out_embeds(codes[None]).float().cpu().numpy()[0]
        self.history.append(codes)
        if self.fast_frames is not None:
            self.fast_frames.remember(codes)
        self.block = (self.block + 1) % N_AUDIO
        return self.sound(codes)

    def close(self):
        self.lm.close()
