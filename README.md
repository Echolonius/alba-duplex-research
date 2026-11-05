# Alba: a full-duplex voice model on a 4 GB laptop GPU

Research code and notes from an attempt to build a private, offline, full-duplex voice assistant on one
consumer laptop (RTX 3050 Laptop, 4 GB, held at 30 W). LFM3-Audio-2B (Liquid AI) was fine-tuned with 4-bit
QLoRA into a single streaming model that hears and speaks in 80 ms steps, with turn-taking learned from audio.

**Status (5 November 2026): unfinished, paused.** The runtime is fast enough (real-time factor 0.30) and the
test bench, decision rules and local tool API are in place, but the second training run answered only 10% of
turn hand-offs in real two-person recordings (people: 66% on the same scorer). The causes were found and
fixed in code; the training run that applies them (run 3) is prepared but not trained. None of the project's
eight acceptance checks is met yet.

**Read the review paper first: [`paper/REVIEW.md`](paper/REVIEW.md).** It covers the design, data, evaluation,
results and, most usefully, the failure analysis.

## What is here

| Folder | What |
| --- | --- |
| `app/` | The runtime. `duplex.py`: the streaming duplex model (llama.cpp for the language model, PyTorch for hearing, frames and sound). `fast_audio.py`: hand-written fixed-shape streaming steps captured as CUDA graphs. `duplex_call.py`: the live call window. `api.py`: the local tool API (127.0.0.1 only). `llama_lib.py`: a ctypes bridge to llama.cpp. The rest is the earlier cascade design. |
| `tools/` | Data, training, evaluation and diagnostics. Key scripts below. |
| `integrations/` | Two example tools for the API: a to-do list and a read-only notes-folder search. |
| `data/tests/` | The frozen held-out test definitions (item texts, sandbox, README) and `HASHES.txt` with the SHA-256 of every frozen file. The audio is not included. |
| `data/tests-fresh/` | Fresh standing-test variants, never trained on (texts only). |
| `models/manifests/` | Official download sources and checksums for every model and binary used. |
| `results/` | Summary scores and diagnostics behind the paper (no audio, no per-call logs). |
| `docs/` | The full lab notebook (`lab-notebook.md`), the goal definition, the plans, and the project's own README. |
| `paper/` | The review paper. |

Key scripts:

- `tools/train_duplex.py`: the duplex training layout (`build`) and the QLoRA training loop.
- `tools/make_duplex_data.py`, `tools/make_seamless_own.py`, `tools/make_convo_text.py`,
  `tools/make_convo_audio.py`, `tools/make_feeling_data.py`, `tools/make_aside_data.py`,
  `tools/make_tool_convos.py`: training data.
- `tools/build_testsets.py`, `tools/build_tool_tests.py`: the frozen tests.
- `tools/run_timing_duplex.py` + `tools/score_timing.py`: the timing test; `tools/run_items_duplex.py`: the
  spoken item tests; `tools/run_tools_duplex.py`: the tool test; `tools/ship_gate.py`: build-to-build gate.
- `tools/calibrate_duplex.py`: the listen setting, chosen on validation data by a rule fixed in advance.
- `tools/diag_duplex_decode.py`, `tools/diag_duplex_track.py`, `tools/probe_duplex_val.py`,
  `tools/speech_match.py`, `tools/check_frame_path.py`: the diagnostics that found the failures.
- `tools/duplex_run3.sh`: the next training run, ready to go.

## What is not here, and why

- **Model weights and binaries:** downloaded from their official sources by `tools/fetch_models.sh` and
  checked against `models/manifests/`. Trained adapters were not published.
- **Datasets and generated training data:** Seamless Interaction (CC BY-NC 4.0), CREMA-D (ODbL) and RAVDESS
  (CC BY-NC-SA 4.0) are fetched from their official hosts; the generated conversations are rebuilt by the
  scripts.
- **All audio and recordings:** the owner's calls stay private, and Block's public Dessa demos were used for
  measurement only.

## Rebuilding

Windows 11, Git Bash, an NVIDIA GPU with 4 GB or more.

```bash
source tools/env.sh && uv sync && bash tools/fetch_models.sh   # app environment, models, llama.cpp b12094
```

The training environment (`.venv-train`) uses Python 3.12 and the packages in `requirements-train.txt`
(torch 2.15 with CUDA 13.2 wheels, transformers 5.22, peft 0.21, bitsandbytes 0.50). Then follow the order in
`docs/lab-notebook.md`: build the tests, make the data, train (`tools/duplex_run3.sh`), calibrate on
validation, and score on the frozen sets.

## Licenses

Code and documents in this repository: MIT (see `LICENSE`). Third-party models and data keep their own
licenses: LFM3-Audio-2B and LFM2.5 (LFM Open License v1.0), Gemma 4 (Apache 2.0), Pocket TTS
(kyutai/pocket-tts-without-voice-cloning, CC BY 4.0), Moonshine (MIT), llama.cpp (MIT), Seamless Interaction
(CC BY-NC 4.0, Meta FAIR), RAVDESS (CC BY-NC-SA 4.0), CREMA-D (ODbL). The non-commercial data licenses rule
out commercial use of models trained on them.

## How this was made

The code, tests and paper were written by Claude (Anthropic) working as a coding agent for the repository
owner, who set the goal, the rules and the acceptance checks, and tested the live builds.
