# Public timing set

The timing test (check 2) without the owner's private calls: the frozen set's 12 tracks from Meta's Seamless
Interaction test split, which anyone can download, rebuild, run and score. 12 callers, 40 min of real conversation
between strangers: 159 turn hand-offs, 326 mid-thought pauses, 67 "mm-hm"s.

| File | What it is |
| --- | --- |
| `timing/items.json` | The test itself: every track's speech, hand-offs, pauses, "mm-hm"s and onsets (no words, no audio) |
| `timing/sources.json` | The 12 conversations and the official URL, size and SHA-256 of each of their 24 original files |
| `HASHES.txt` | Fingerprints of the files here (audio included once built), the timing scorer and the builder |

## How it relates to the frozen set

The frozen timing set (`data/tests/timing`, 15 tracks, private) stays the paper's test and is unchanged. This set is
its 12 Seamless tracks exactly: the same conversations and caller side, chosen, cut and labelled by the same code
(`tools/build_testsets.py --public-only`), in the same order. Checked when it was made (lab notebook, 6 November
2026): all 24 rebuilt audio files match their fingerprints in `data/tests/HASHES.txt`, and the human reference scored
on this set is byte for byte `results/tests/human/timing_scores.json`.

So the `"seamless (strangers)"` group in any frozen `timing_scores.json` is that build's score on this set. Scores of
this set are kept apart, in `results/tests-public/`, and are never mixed with the 15-track numbers.

## What you need

| To | You need |
| --- | --- |
| Score the people reference | Only this repository and its Python environment: it reads `timing/items.json`. No data download, no graphics card |
| Rebuild the audio | Also a 0.92 GB download from Meta, with no account and no graphics card (44 MB is kept) |
| Run Alba on the set | Also release v0.2's trained adapters, made into a model with `tools/export_alba.py`, and an NVIDIA graphics card with 4 GB or more. Windows only |

Scoring and rebuilding are plain Python (numpy, scipy and soundfile) with nothing specific to Windows. Running Alba
is Windows only for now: the runtime loads llama.cpp's Windows build. The commands below are for bash (Git Bash on
Windows).

## Rebuild, run, score

```bash
source tools/env.sh && uv sync               # the Python environment, kept inside this folder
[ -x "$PY" ] || PY="$LV/.venv/bin/python"    # Linux and macOS: env.sh's $PY is the Windows path

"$PY" -I tools/score_timing.py --public human      # people reference, from the labels: no data download
"$PY" -I tools/fetch_seamless.py --pairs-from data/tests-public/timing/sources.json   # 0.92 GB download, 44 MB kept
"$PY" -I tools/build_testsets.py --public-only
git status data/tests-public        # nothing changed: the set and its fingerprints were rebuilt exactly
```

To run Alba, set up as in the main README's [Rebuilding](../../README.md#rebuilding) section, download
`alba-duplex-run1-lora.zip` from [release v0.2](https://github.com/Echolonius/alba-duplex-research/releases/tag/v0.2)
(run 1 is the one in the table below) and unzip it into `models/`. Then, on Windows:

```bash
.venv-train/Scripts/python.exe -I tools/export_alba.py --lora models/<unzipped folder> --out models/alba-duplex
.venv-train/Scripts/python.exe -I tools/run_timing_duplex.py --model models/alba-duplex --tag duplex --public
"$PY" -I tools/score_timing.py --public human duplex
```

The fetch stops if a file differs from its recorded SHA-256. Audio fingerprints were made with the versions in
`uv.lock` (soundfile 0.14.0, scipy 1.18.1, numpy 2.5.4); another FLAC encoder version can write different bytes for
the same sound, while `items.json` does not depend on it.

## Reference numbers on these 12 tracks

From the `"seamless (strangers)"` group of the frozen runs (same tracks, same scorer):

| | People (labels) | People (sound) | Cascade (split) | Duplex run 1 | Target |
| --- | --- | --- | --- | --- | --- |
| Reply starts after the caller stops (median) | 0.23 s | 0.27 s | 1.93 s | 1.70 s | <= 0.3 s |
| Hand-offs answered before the caller goes on | 81% | 66% | 42% | 4% | |
| Cuts into the caller's thinking pauses | 0.6% | 0.3% | 1.2% | 0% | < 10% |
| Stops when talked over (median) | 0.51 s | 0.49 s | 0.47 s | 0.17 s (1 case) | <= 0.4 s |
| Keeps going through the caller's "mm-hm" | 50% | 52% | 25% | - | |
| "Mm-hm"s per minute | 6.8 | 9.9 | 17.4 | 8.3 | |

People are scored two ways. **Labels:** from the speech times labelled in `items.json`, which is what
`score_timing.py --public human` gives. **Sound:** from the partners' own recorded sound, the same way a system is
scored, so it is the fair comparison for a system; the paper's 66% is this one. It comes from the frozen record
(`results/tests/mock-human`), and no script in the repository remakes it yet.

Run 2 (release v0.2) was not scored on the timing test. A fresh run of a system can differ a little from these by its
own sampling.

## License

Derived from Meta's Seamless Interaction dataset (Meta FAIR, "Seamless Interaction: Dyadic Audiovisual Motion
Modeling and Large-Scale Dataset", 2025), naturalistic test split, licensed
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/): non-commercial research only. Changes: turn timings
computed from Meta's voice-activity and word-timing files; the audio (not included) resampled to 16 kHz and cut to
the shorter of the two tracks. Test data only: never train on it or tune on it.

The repository's MIT license covers its code and documents, not this data: data keeps its own license, as the main
README's [Licenses and credits](../../README.md#licenses-and-credits) section says.
