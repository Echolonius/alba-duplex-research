"""A thin bridge to llama.cpp's own library (bin/llama-b12094/llama.dll, the app's build), for the full-duplex
loop: it feeds the language model a mix of text tokens and raw vectors (a slice of your voice from the speech
encoder, or one of Alba's speech frames) one step at a time, and reads back both the next-word scores and the
model's internal state (which the small frame decoder turns into speech).

Struct layouts follow include/llama.h of the same build (tools/llama-src/llama.cpp-b12094, checked by hand);
a different build needs them re-checked.
"""
import ctypes as C
import os
from pathlib import Path

# PyTorch brings Intel's OpenMP (libiomp5md) and llama.cpp's processor backend brings LLVM's (libomp); in one
# process the second one aborts at start ("OMP: Error #15") unless told the other may exist. With the language
# model on the graphics card llama.cpp's processor threads do almost nothing, so the two do not compete.
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DLL_DIR = ROOT / 'bin' / 'llama-b12094'

llama_token = C.c_int32
llama_pos = C.c_int32
llama_seq_id = C.c_int32


class Batch(C.Structure):
    _fields_ = [('n_tokens', C.c_int32), ('token', C.POINTER(llama_token)), ('embd', C.POINTER(C.c_float)),
                ('pos', C.POINTER(llama_pos)), ('n_seq_id', C.POINTER(C.c_int32)),
                ('seq_id', C.POINTER(C.POINTER(llama_seq_id))), ('logits', C.POINTER(C.c_int8))]


class ModelParams(C.Structure):
    _fields_ = [('devices', C.c_void_p), ('tensor_buft_overrides', C.c_void_p), ('n_gpu_layers', C.c_int32),
                ('split_mode', C.c_int), ('load_mode', C.c_int), ('lazy_mode', C.c_int), ('main_gpu', C.c_int32),
                ('tensor_split', C.c_void_p), ('progress_callback', C.c_void_p),
                ('progress_callback_user_data', C.c_void_p), ('kv_overrides', C.c_void_p),
                ('vocab_only', C.c_bool), ('check_tensors', C.c_bool), ('use_extra_bufts', C.c_bool),
                ('no_host', C.c_bool), ('no_alloc', C.c_bool), ('load_mtp', C.c_bool)]


class ContextParams(C.Structure):
    _fields_ = [('n_ctx', C.c_uint32), ('n_batch', C.c_uint32), ('n_ubatch', C.c_uint32), ('n_seq_max', C.c_uint32),
                ('n_rs_seq', C.c_uint32), ('n_outputs_max', C.c_uint32), ('n_outputs_max_per_seq', C.c_uint32),
                ('n_threads', C.c_int32), ('n_threads_batch', C.c_int32),
                ('ctx_type', C.c_int), ('rope_scaling_type', C.c_int), ('pooling_type', C.c_int),
                ('attention_type', C.c_int), ('flash_attn_type', C.c_int),
                ('rope_freq_base', C.c_float), ('rope_freq_scale', C.c_float), ('yarn_ext_factor', C.c_float),
                ('yarn_attn_factor', C.c_float), ('yarn_beta_fast', C.c_float), ('yarn_beta_slow', C.c_float),
                ('yarn_orig_ctx', C.c_uint32), ('defrag_thold', C.c_float),
                ('cb_eval', C.c_void_p), ('cb_eval_user_data', C.c_void_p),
                ('type_k', C.c_int), ('type_v', C.c_int), ('moe_cache_size', C.c_size_t),
                ('abort_callback', C.c_void_p), ('abort_callback_data', C.c_void_p),
                ('embeddings', C.c_bool), ('offload_kqv', C.c_bool), ('no_perf', C.c_bool), ('op_offload', C.c_bool),
                ('swa_full', C.c_bool), ('kv_unified', C.c_bool),
                ('samplers', C.c_void_p), ('n_samplers', C.c_size_t), ('ctx_other', C.c_void_p)]


_BACKENDS = []


def _load(backends='all'):
    """backends: 'all' (graphics card + processor, as llama-server) or 'cpu' (processor only, AVX2 build)."""
    os.add_dll_directory(str(DLL_DIR))
    ggml = C.CDLL(str(DLL_DIR / 'ggml.dll'))
    if not _BACKENDS:
        if backends == 'cpu':
            ggml.ggml_backend_load.argtypes, ggml.ggml_backend_load.restype = [C.c_char_p], C.c_void_p
            ggml.ggml_backend_load(str(DLL_DIR / 'ggml-cpu-haswell.dll').encode())
        else:
            # from llama.cpp's own folder (plain ggml_backend_load_all looks next to python.exe and finds nothing)
            ggml.ggml_backend_load_all_from_path.argtypes = [C.c_char_p]
            ggml.ggml_backend_load_all_from_path.restype = None
            ggml.ggml_backend_load_all_from_path(str(DLL_DIR).encode())
        _BACKENDS.append(backends)
    lib = C.CDLL(str(DLL_DIR / 'llama.dll'))
    sig = {
        'llama_backend_init': ([], None),
        'llama_model_default_params': ([], ModelParams),
        'llama_context_default_params': ([], ContextParams),
        'llama_model_load_from_file': ([C.c_char_p, ModelParams], C.c_void_p),
        'llama_init_from_model': ([C.c_void_p, ContextParams], C.c_void_p),
        'llama_model_get_vocab': ([C.c_void_p], C.c_void_p),
        'llama_model_n_embd': ([C.c_void_p], C.c_int32),
        'llama_vocab_n_tokens': ([C.c_void_p], C.c_int32),
        'llama_tokenize': ([C.c_void_p, C.c_char_p, C.c_int32, C.POINTER(llama_token), C.c_int32, C.c_bool, C.c_bool], C.c_int32),
        'llama_batch_init': ([C.c_int32, C.c_int32, C.c_int32], Batch),
        'llama_batch_free': ([Batch], None),
        'llama_decode': ([C.c_void_p, Batch], C.c_int32),
        'llama_get_logits_ith': ([C.c_void_p, C.c_int32], C.POINTER(C.c_float)),
        'llama_get_embeddings_ith': ([C.c_void_p, C.c_int32], C.POINTER(C.c_float)),
        'llama_set_embeddings': ([C.c_void_p, C.c_bool], None),
        'llama_set_n_threads': ([C.c_void_p, C.c_int32, C.c_int32], None),
        'llama_get_memory': ([C.c_void_p], C.c_void_p),
        'llama_memory_seq_rm': ([C.c_void_p, llama_seq_id, llama_pos, llama_pos], C.c_bool),
        'llama_memory_clear': ([C.c_void_p, C.c_bool], None),
        'llama_free': ([C.c_void_p], None),
        'llama_model_free': ([C.c_void_p], None),
    }
    for name, (args, res) in sig.items():
        f = getattr(lib, name)
        f.argtypes, f.restype = args, res
    lib.llama_backend_init()
    return lib


class LM:
    """One model, one sequence, fed step by step: tokens or vectors in; scores and internal state out."""

    def __init__(self, path, n_ctx=4096, gpu_layers=-1, threads=4, backends='all'):
        self.lib = lib = _load(backends)
        mp = lib.llama_model_default_params()
        mp.n_gpu_layers = gpu_layers
        self.model = lib.llama_model_load_from_file(str(path).encode(), mp)
        if not self.model:
            raise RuntimeError(f'could not load {path}')
        cp = lib.llama_context_default_params()
        cp.n_ctx, cp.n_batch, cp.n_ubatch = n_ctx, 512, 512
        cp.n_threads = cp.n_threads_batch = threads
        cp.embeddings = True
        cp.pooling_type = 0           # LLAMA_POOLING_TYPE_NONE: per-token states *and* next-word scores
        self.ctx = lib.llama_init_from_model(self.model, cp)
        if not self.ctx:
            raise RuntimeError('could not create a context')
        self.vocab = lib.llama_model_get_vocab(self.model)
        self.n_embd = lib.llama_model_n_embd(self.model)
        self.n_vocab = lib.llama_vocab_n_tokens(self.vocab)
        self.pos = 0
        self._tok = lib.llama_batch_init(512, 0, 1)
        self._emb = lib.llama_batch_init(64, self.n_embd, 1)

    def tokenize(self, text, special=True):
        b = text.encode('utf-8')
        buf = (llama_token * (len(b) + 16))()
        n = self.lib.llama_tokenize(self.vocab, b, len(b), buf, len(buf), False, special)
        if n < 0:
            raise ValueError('tokenize failed')
        return list(buf[:n])

    def _run(self, batch, n):
        batch.n_tokens = n
        for i in range(n):
            batch.pos[i] = self.pos + i
            batch.n_seq_id[i] = 1
            batch.seq_id[i][0] = 0
            batch.logits[i] = 1 if i == n - 1 else 0
        rc = self.lib.llama_decode(self.ctx, batch)
        if rc != 0:
            raise RuntimeError(f'llama_decode returned {rc}')
        self.pos += n

    def feed_tokens(self, ids):
        """Text tokens in; returns (logits, hidden) of the last one."""
        for k in range(0, len(ids), 512):
            chunk = ids[k:k + 512]
            for i, t in enumerate(chunk):
                self._tok.token[i] = t
            self._run(self._tok, len(chunk))
        return self.last()

    def feed_vectors(self, vecs):
        """Raw input vectors in ((n, n_embd) float32); returns (logits, hidden) of the last one."""
        vecs = np.ascontiguousarray(vecs, dtype=np.float32).reshape(-1, self.n_embd)
        for k in range(0, len(vecs), 64):
            chunk = vecs[k:k + 64]
            C.memmove(self._emb.embd, chunk.ctypes.data, chunk.nbytes)
            self._run(self._emb, len(chunk))
        return self.last()

    def last(self):
        lg = np.ctypeslib.as_array(self.lib.llama_get_logits_ith(self.ctx, -1), shape=(self.n_vocab,)).copy()
        hd = np.ctypeslib.as_array(self.lib.llama_get_embeddings_ith(self.ctx, -1), shape=(self.n_embd,)).copy()
        return lg, hd

    def rewind(self, to_pos):
        """Forget everything from position to_pos on (e.g. a reply cut short). Models with convolution or other
        recurrent layers (LFM2/LFM3) keep only their latest state there, so only rewind(0) works for them; a partial
        rewind is refused by llama.cpp, and is an error here rather than a silently broken memory."""
        if not self.lib.llama_memory_seq_rm(self.lib.llama_get_memory(self.ctx), 0, to_pos, -1):
            raise RuntimeError(f'the model memory cannot be rewound to position {to_pos}: rewind(0) and feed again')
        self.pos = to_pos

    def close(self):
        self.lib.llama_batch_free(self._tok)
        self.lib.llama_batch_free(self._emb)
        self.lib.llama_free(self.ctx)
        self.lib.llama_model_free(self.model)
