# Results

Summary scores and diagnostics behind the [review paper](../paper/REVIEW.md). No audio and no per-call logs are
included. "Validation" means data kept apart from the frozen tests and used for decisions; "frozen" means the
held-out tests in `data/tests/`.

## Duplex model

| File | What it holds |
| --- | --- |
| `calibration_alba-duplex2.json` | Run 2's listen-setting table on 8 validation conversations (28 min), the fixed rule, and its choice |
| `tests/duplex/timing_scores.json` | Run 1 on the frozen timing test (15 real tracks, 56 min) |
| `tests/duplex/speed.json` | Run 1's real-time factor, slowest step and late frames, per track |
| `tests/duplex2-probe-val/plans.json`, `speech_match.json` | Run 2's replies to 11 validation prompts, and how much of each plan the speech recognizer hears back |
| `tests/duplex2-probe-val-mic/plans.json` | The same prompts through the training's laptop-microphone sound |
| `diag_decode_alba-duplex2.json` | Every plan decision with the model's top choices, on 4 validation prompts, before the decoding fix |
| `diag_track_alba-duplex2-0.json`, `-1.json` | Per-utterance breakdown of validation conversations 0 and 1 (3 min each) |
| `diag_track_alba-duplex2-ctx2400-0.json` | The same as conversation 0, with the memory kept to about 30 s |
| `frame_path_alba-duplex2.json` | Speech-frame loss through the deployed 4-bit model, trained against stock |

## Runtime speed

| File | What it holds |
| --- | --- |
| `profile_duplex.json` | Where each step's time went before the CUDA graphs |
| `fast_audio_check.json` | The CUDA-graph speech encoder against the library: agreement and milliseconds per step |
| `fast_frames_check.json`, `fast_frames_check_fp32.json` | The CUDA-graph frame sampler against the model's own, in 16-bit and 32-bit |
| `fast_codec_check.json` | The CUDA-graph codec decoder against the library: sample counts, signal-to-noise, speed |

## References and earlier builds

| File | What it holds |
| --- | --- |
| `tests/human/timing_scores.json` | People on the frozen timing test, from the labels |
| `tests/mock-human/timing_scores.json` | People scored from their sound, the same way systems are: the fair ceiling |
| `tests/split/*_scores.json` | The cascade build on the frozen timing, battery, red-team and feeling tests |
| `tests/report.md` | Every build's frozen scores against the goal's targets |
| `dessa_metrics.json`, `dessa_summary.json` | Timing and voice measurements of Block's 21 public Dessa demo conversations |
| `fin5_summary.json` | The cascade's simulated-conversation measurements, before the frozen tests existed |
| `vap_ft.json`, `vap_vs_smartturn.json` | The learned turn model after retraining, and against Smart Turn |
| `ship_gate_split_vs_split.json` | The ship gate's self-check |
| `api_test.json` | The local tool API's 17 checks |

The README's figures are drawn from these files by `tools/make_readme_figures.py`.
