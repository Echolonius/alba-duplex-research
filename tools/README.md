# Tools

Every script runs with the project's own Python: `source tools/env.sh`, then `"$PY" -I tools/<name>.py`
(training and the duplex runtime use the `.venv-train` environment, see `requirements-train.txt`). Each file's
first lines say what it does and how to run it.

## The full-duplex model (current work)

**Training**

| Script | What it does |
| --- | --- |
| `train_duplex.py` | The duplex training layout (`build`: one 80 ms step at a time) and the 4-bit QLoRA training loop |
| `duplex_run3.sh` | The next run, ready to go: train, export, validation probe, real-voice check, calibration |
| `duplex_run2.sh`, `duplex_pipeline.sh` | Runs 2 and 1, end to end, as they were run |
| `export_alba.py` | Merges a trained adapter into LFM3-Audio-2B and converts it with llama.cpp's official converter, checking every file |

**Training data**

| Script | What it does |
| --- | --- |
| `fetch_seamless.py` | Real two-person conversations from Meta's Seamless Interaction (CC BY-NC 4.0), checksummed |
| `make_duplex_data.py` | Lays out two time-aligned tracks per conversation: the real person, and Alba re-spoken in the real slots |
| `make_seamless_own.py` | The same real conversations with new words written for Alba in every turn (run 3) |
| `make_convo_text.py`, `make_convo_audio.py` | Synthetic calls: Gemma 4 E4B writes both sides under Alba's persona and filters, Pocket TTS speaks them |
| `make_feeling_data.py` | Replies that show the caller's tone of voice was heard (CREMA-D clips) |
| `make_aside_data.py` | Remarks meant for the developer, where the right answer is silence |
| `make_convo_from_calls.py` | Listening turns from the owner's own calls (stay local; never published) |
| `make_tool_convos.py` | Calls where Alba uses tools, with real results from a sandbox (run 4) |

**Frozen held-out tests and scoring**

| Script | What it does |
| --- | --- |
| `build_testsets.py`, `build_tool_tests.py` | Build the frozen test sets and record their fingerprints |
| `run_timing_duplex.py` + `score_timing.py` | Timing on real recordings (check 2) and speed (check 5); speech placed when a listener would hear it |
| `run_items_duplex.py` + `score_battery.py`, `score_redteam.py`, `score_feeling.py`, `score_fresh.py` | Spoken battery, red team, feeling and fresh standing tests |
| `run_tools_duplex.py` | Tool use by voice (check 7), against its own sandbox |
| `judge_local.py` | The local transcript judge the scorers use (Gemma 4 E4B, offline) |
| `ship_gate.py`, `report_scores.py` | Build-to-build gate and the summary table of every build's scores |

**Settings and diagnostics (validation data only, never the frozen tests)**

| Script | What it does |
| --- | --- |
| `calibrate_duplex.py` | Picks the listen setting on 8 validation conversations by a rule fixed in advance |
| `diag_duplex_decode.py` | Logs every plan decision with the model's top choices (found the untrainable marker) |
| `diag_duplex_track.py` | Per-utterance breakdown of one real conversation (found the "Yeah." loop) |
| `diag_duplex_data.py` | What the training data teaches per source, before training: her utterances by kind, whose words carry loss, what they open with, what the text loss is made of (test: `test_diag_duplex_data.py`) |
| `probe_duplex_val.py`, `speech_match.py` | Replies to 11 validation prompts, and whether the speech says what was planned |
| `check_frame_path.py`, `profile_duplex.py` | Speech-frame fit through the deployed model, and where each step's time goes |

**Runtime checks**

| Script | What it does |
| --- | --- |
| `test_duplex_plumbing.py` | Every link of the streaming runtime with the stock model |
| `test_fast_audio.py`, `test_fast_frames.py`, `test_fast_codec.py` | The CUDA-graph steps against the library's own output |
| `test_api.py` | The local tool API and both example integrations, end to end |

**Other**

| Script | What it does |
| --- | --- |
| `learn_from_call.py` | Opt-in learning from a call: proposals to review, approve, undo or delete |
| `make_listening_clips.py` | Blinded clips for the listening page (check 6) |
| `make_readme_figures.py` | The README's figures, from the saved results |

## The learned turn-taking model (VAP)

`train_vap.py` retrains it on real conversations; `eval_vap.py`, `eval_turns_calls.py` and `predictive_turn.py`
measure it; `vap_port_check.py` checks the PyTorch port against the original; `calibrate_turn.py` tunes the earlier
rule.

## The earlier cascade design and its measurements

`benchmark.py`, `compare.py`, `bench_content.py`, `conversation_metrics.py` and `summarize_metrics.py` simulate and
measure conversations with the same tool used on Dessa's public demos (`extract_dessa_page.mjs`); `report_data.py`,
`build_report.py` and `report_template.html` build the comparison report. `run_timing.py` and `run_items.py` run
the frozen tests on the cascade. `train_lfm3_lora.py`, `m2_pipeline.sh`, `eval_alba.py` and `make_s2s_inputs.py`
are the half-duplex fine-tune. `probe_*.py`, `try_*.py`, `analyze_call.py`, `analyze_latest.sh`,
`audio_glitches.py`, `replay_bargein.py`, `prosody.py`, `intelligibility.py`, `voice_measures.py` and
`scale_math.py` are the experiments behind the cascade's design choices.

## Setup

`env.sh` keeps every cache inside the project folder; `fetch_models.sh` and `hf_fetch.py` download from official
sources at pinned revisions and check every file; `restart_server.sh` restarts the local model server (127.0.0.1
only).
