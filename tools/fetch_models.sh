#!/usr/bin/env bash
# Downloads the runtime pieces from their official sources and checks each
# file's SHA-256 against the fingerprint published by that source.
set -u
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
DL="$LV/models/downloads"; mkdir -p "$DL" "$LV/models/lfm3-audio" "$LV/models/vad" "$LV/bin"
get() {  # url dest sha256
  local url="$1" dest="$2" want="$3"
  if [ -f "$dest" ] && [ "$(sha256sum < "$dest" | cut -d" " -f1)" = "$want" ]; then echo "OK (already) $(basename "$dest")"; return 0; fi
  curl -sS -L --fail --retry 5 --retry-delay 3 -C - -o "$dest" "$url" || { echo "FAIL download $url"; return 1; }
  local got; got=$(sha256sum < "$dest" | cut -d" " -f1)
  if [ "$got" = "$want" ]; then echo "OK $(basename "$dest") sha256 verified"; else echo "MISMATCH $(basename "$dest") got $got want $want"; rm -f -- "${dest:?}"; return 1; fi
}
get https://github.com/ggml-org/llama.cpp/releases/download/b12094/llama-b12094-bin-win-cuda-13.4-x64.zip "$DL/llama-b12094-bin-win-cuda-13.4-x64.zip" 6881a0d29eb397c8bfcf5298cff2ee0c95e4f1dba57be73038093b020293a822
get https://github.com/ggml-org/llama.cpp/releases/download/b12094/cudart-llama-bin-win-cuda-13.4-x64.zip "$DL/cudart-llama-bin-win-cuda-13.4-x64.zip" 738f8c251ac22b70c3ae6f83a10cf222725df0395246a2cf58f32bdb85fbe668
get https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/f766f81d3cfdf7737ac64aad813d91bbfd56bf93/smart-turn-v3.2-cpu.onnx "$LV/models/vad/smart-turn-v3.2-cpu.onnx" 2bb026316b14a660486a75b1733cd3fbab8c2fd0314dc9af7be49f8cca967e4f
get https://huggingface.co/LiquidAI/LFM3-Audio-2B-GGUF/resolve/39670cb5c42e9c4bfe046519ba7ffbe47f9552ad/mmproj-LFM3-Audio-2B-Q8_0.gguf "$LV/models/lfm3-audio/mmproj-LFM3-Audio-2B-Q8_0.gguf" 2b6648b1fd48af16aa0877b0592b144d15070783673e038a379fdbc51d40e365
get https://huggingface.co/LiquidAI/LFM3-Audio-2B-GGUF/resolve/39670cb5c42e9c4bfe046519ba7ffbe47f9552ad/LFM3-Audio-2B-Q8_0.gguf "$LV/models/lfm3-audio/LFM3-Audio-2B-Q8_0.gguf" d4e36902ba8f0ee7b333596030a59d650f88ac76660b4d9ba5ff77c3c2258fc0
# 4-bit version: about 30% faster on a 4 GB card
get https://huggingface.co/LiquidAI/LFM3-Audio-2B-GGUF/resolve/39670cb5c42e9c4bfe046519ba7ffbe47f9552ad/LFM3-Audio-2B-Q4_K_M.gguf "$LV/models/lfm3-audio/LFM3-Audio-2B-Q4_K_M.gguf" 9a773289bd321ed879cd3a76545f6aca506e403851a45fb315420aee45625be2
get https://huggingface.co/LiquidAI/LFM3-Audio-2B-GGUF/resolve/39670cb5c42e9c4bfe046519ba7ffbe47f9552ad/mmproj-LFM3-Audio-2B-Q4_K_M.gguf "$LV/models/lfm3-audio/mmproj-LFM3-Audio-2B-Q4_K_M.gguf" 1a3a895160aad17c1f52117a6ee4cea8e68988651ea5a33c5b990539e8eae81d
# Text models for the split design (the voice model hears and speaks, a text model writes the words)
mkdir -p "$LV/models/text"
get https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF/resolve/8ed288026e23958ad9dfa92d53ed773a8eee7125/LFM2.5-1.2B-Instruct-Q4_K_M.gguf "$LV/models/text/LFM2.5-1.2B-Instruct-Q4_K_M.gguf" b1b3de114215d9507409a662a501a631095a479a419584e8a2ded6304b19b4f5
# The default brain since 4 November 2026 (Google Gemma 4 E2B, Apache 2.0, via the llama.cpp project):
get https://huggingface.co/ggml-org/gemma-4-E2B-it-GGUF/resolve/6938c58e698502037401a5f19f6905f2fb2f7dee/gemma-4-E2B-it-Q4_0.gguf "$LV/models/text/gemma-4-E2B-it-Q4_0.gguf" 8e30dff3ac4c8434c49a7036fa15564bdbb6044e42bf04550bf1a096ad7e6a52
# Optional: the stronger Gemma 4 E4B (needs 3.1 GB of graphics memory, so the hearing must move off the card),
# and the models compared on 4 November (results/audit/probe-call6-*.txt):
# get https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF/resolve/a69f018d7678a2b9f5e705f8903d6fb29b9b25dc/gemma-4-E4B-it-Q4_0.gguf "$LV/models/text/gemma-4-E4B-it-Q4_0.gguf" a555b900214b477d8880e7832e0b8925e139b0159640036b09fe472b6f2097f2
# get https://huggingface.co/LiquidAI/LFM2-8B-A1B-GGUF/resolve/145e98524a2ed11d35a45e0080febcee063749b6/LFM2-8B-A1B-Q4_K_M.gguf "$LV/models/text/LFM2-8B-A1B-Q4_K_M.gguf" d2185b22630fc68043dac7182f12e86e5ad14990229a90b6c9ad3f4421ddaf82
# get https://huggingface.co/LiquidAI/LFM2-2.6B-GGUF/resolve/a759abdc5955d4ca97763e5cb7ff3940589ba898/LFM2-2.6B-Q4_K_M.gguf "$LV/models/text/LFM2-2.6B-Q4_K_M.gguf" 384bc877b6c37064982f96885bef69e4475919f5969218ed4e3b9399ae0340df
# The voice (Kyutai Pocket TTS, CC BY 4.0) is fetched by the pocket-tts package on first use, pinned in its
# own config files, into models/hf: run `"$PY" -I tools/try_pocket_tts.py results/voices/pocket` once.
# Silero VAD: verify against the git blob id GitHub reports for that tag.
SIL="$LV/models/vad/silero_vad.onnx"
curl -sS -L --fail --retry 5 -o "$SIL" https://raw.githubusercontent.com/snakers4/silero-vad/v6.2.3/src/silero_vad/data/silero_vad.onnx
want=$(curl -sS "https://api.github.com/repos/snakers4/silero-vad/contents/src/silero_vad/data/silero_vad.onnx?ref=v6.2.3" | "$PY" -I -c "import json,sys; print(json.load(sys.stdin)['sha'])")
got=$(git hash-object "$SIL")
if [ "$got" = "$want" ]; then echo "OK silero_vad.onnx git-blob verified ($got)"; else echo "MISMATCH silero_vad.onnx $got vs $want"; rm -f -- "${SIL:?}"; fi
# Unpack llama.cpp into its own folder
if [ -f "$DL/llama-b12094-bin-win-cuda-13.4-x64.zip" ] && [ -f "$DL/cudart-llama-bin-win-cuda-13.4-x64.zip" ]; then
  mkdir -p "$LV/bin/llama-b12094"
  unzip -q -o "$DL/llama-b12094-bin-win-cuda-13.4-x64.zip" -d "$LV/bin/llama-b12094"
  unzip -q -o "$DL/cudart-llama-bin-win-cuda-13.4-x64.zip" -d "$LV/bin/llama-b12094"
  echo "UNPACKED llama.cpp: $(ls "$LV/bin/llama-b12094" | wc -l) files"
fi
echo ALL-DONE
