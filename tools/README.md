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
| `build_testsets.py`, `build_tool_tests.py` | Build the frozen test sets and record their fingerprints (`--public-only`: the public timing set anyone can rebuild, `data/tests-public`) |
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
| `diag_duplex_data.py` | What the training data teaches per source, before training: her utterances by kind, whose words carry loss, what they open with, and her text positions: LISTEN, starts, learned words and context-only words, which carry no loss (test: `test_diag_duplex_data.py`) |
| `probe_duplex_val.py`, `speech_match.py` | Replies to 11 validation prompts, and whether the speech says what was planned |
| `check_frame_path.py`, `profile_duplex.py` | Speech-frame fit through the deployed model, and where each step's time goes |

**Runtime checks**

| Script | What it does |
| --- | --- |
| `test_duplex_plumbing.py` | Every link of the streaming runtime with the stock model |
| `test_fast_audio.py`, `test_fast_frames.py`, `test_fast_codec.py` | The CUDA-graph steps against the library's own output |
| `test_api.py` | The local tool API and both example integrations, end to end |

**Full-Duplex-Bench**

`run_fdb.py` runs the duplex model on [Full-Duplex-Bench](https://github.com/DanielLin94144/Full-Duplex-Bench)'s
offline tasks (v1.0: pauses, backchannels, smooth turn-taking, interruptions; v1.5: overlaps). It streams each input
80 ms at a time through `run_timing_duplex.py`'s own loop (her sound placed 240 ms after the step that made it) and
writes `output.wav`, and `clean_output.wav` for v1.5, next to each input: mono, 16 kHz, exactly as long as the input.
Alba's scores then sit on the same scale as Moshi's and PersonaPlex's, with one caveat: her sound is placed as if
every step kept up with real time. A step slower than its budget makes a late frame (a gap a listener would hear),
which the run counts but does not move. So only a run without late frames, on a fast enough CUDA GPU, shows what
you would hear. Inputs with late frames are logged with `"real_time": false`, and the run ends with a warning that
says how many. With `--cpu`, expect most replies to be late: such a run checks the pipeline, not the timing.

1. **Model files.** Unzip a trained adapter from release v0.2 into `models/`, then turn it into the files the
   runtime loads: `tools/export_alba.py --lora models/<adapter folder> --out models/alba-duplex` (see "Rebuilding"
   in the main README).
2. **Data.** Download v1.0 and v1.5 from the link in the benchmark's `v1_v1.5/dataset/README.md` into `data/fdb/`,
   which is not committed. Licenses, as the benchmark gives them: its code is CC BY-NC 4.0; the synthetic v1.0 sets
   and all of v1.5 are MIT; the Candor and ICC sets of v1.0 are CC BY-NC 4.0, and the terms of the CANDOR corpus and
   ICC they come from apply too. Outputs go into the data folder, so use one copy per model or setting.
3. **Run** on a CUDA GPU (`--cpu` works, but not in real time). A stopped run continues where it left off.

       .venv-train/Scripts/python.exe -I tools/run_fdb.py --model models/alba-duplex --data data/fdb/v1_0 --tag duplex

   Each finished input gets a row in `results/fdb/<tag>/speed.jsonl`: speed, late frames, `real_time`, replies
   started, its seed, and what made it: the model's fingerprint (a hash of its files), the device and the settings.
   A log made with another model, device or setting stops the run instead of mixing outputs: use a new `--tag` with
   a fresh copy of the data, or `--force` to continue it anyway. Reruns repeat on the same machine: every input
   opens the call with the same words (no clock line) and draws its random numbers from its own seed.
4. **Transcribe** with the benchmark's `v1_v1.5/get_transcript/asr.py`, in the benchmark's own environment, once
   per task folder (`--root_dir`). It runs NVIDIA's parakeet-tdt-0.6b-v2 on a CUDA GPU, so it needs one too. For
   v1.0, add `--task user_interruption` for the interruption set (it transcribes what follows the interruption).
   v1.5 needs four runs per task folder, one for each audio file its scorers read: the default (`output.wav`), then
   `--audio_name input.wav`, `--audio_name clean_input.wav` and `--audio_name clean_output.wav`.
5. **Score** with `v1_v1.5/evaluation/evaluate.py --task <task> --root_dir <task folder>`: for v1.0 the tasks are
   `pause_handling`, `backchannel`, `smooth_turn_taking` and `user_interruption`; for v1.5, `behavior` and
   `general_before_after` (then `significance_test.py`), plus `get_timing.py` (its `evaluation/README.md` gives the
   order). Two scorers send transcripts to OpenAI and need an `OPENAI_API_KEY`: v1.0 `user_interruption`
   (GPT-4-turbo: the user's turns and Alba's reply) and v1.5 `behavior` (GPT-4o: all four transcripts of a sample).
   `get_timing.py` and `general_before_after` load Silero VAD through `torch.hub`, which downloads and runs the
   latest code of github.com/snakers4/silero-vad; `general_before_after` also downloads model weights (UTMOSv2, and
   torchaudio's SQUIM).

`test_run_fdb.py` checks the adapter with a fake model, no weights needed. The benchmark's v2 and v3 talk to a live
model (an examiner over WebRTC, tool calls over LiveKit) and are not covered.

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
