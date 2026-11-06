# Public timing set

The timing test (check 2) without the owner's private calls: the frozen set's 12 tracks from Meta's Seamless
Interaction test split, which anyone can download, rebuild, run and score. 12 callers, 40 min of real conversation
between strangers: 159 turn hand-offs, 326 mid-thought pauses, 67 "mm-hm"s.

| File | What it is |
| --- | --- |
| `timing/items.json` | The test itself: every track's speech, hand-offs, pauses, "mm-hm"s and onsets (no words, no audio) |
| `timing/sources.json` | The 12 conversations and the official URL, size and SHA-256 of each of their 24 original files |
| `HASHES.txt` | Fingerprints of the files here (audio included once built), the scorers and the builder |

## How it relates to the frozen set

The frozen timing set (`data/tests/timing`, 15 tracks, private) stays the paper's test and is unchanged. This set is
its 12 Seamless tracks exactly: the same conversations and caller side, chosen, cut and labelled by the same code
(`tools/build_testsets.py --public-only`), in the same order. Checked when it was made (lab notebook, 6 November
2026): all 24 rebuilt audio files match their fingerprints in `data/tests/HASHES.txt`, and the human reference scored
on this set is byte for byte `results/tests/human/timing_scores.json`.

So the `"seamless (strangers)"` group in any frozen `timing_scores.json` is that build's score on this set. Scores of
this set are kept apart, in `results/tests-public/`, and are never mixed with the 15-track numbers.

## Rebuild, run, score

```
source tools/env.sh
"$PY" -I tools/fetch_seamless.py --pairs-from data/tests-public/timing/sources.json   # 0.9 GB download, 44 MB kept
"$PY" -I tools/build_testsets.py --public-only
git status data/tests-public        # nothing changed: the set and its fingerprints were rebuilt exactly
.venv-train/Scripts/python.exe -I tools/run_timing_duplex.py --model models/alba-duplex --tag duplex --public
"$PY" -I tools/score_timing.py --public human duplex
```

The fetch stops if a file differs from its recorded SHA-256. Audio fingerprints were made with the versions in
`uv.lock` (soundfile 0.14.0, scipy 1.18.1, numpy 2.5.4); another FLAC encoder version can write different bytes for
the same sound, while `items.json` does not depend on it.

## Reference numbers on these 12 tracks

From the `"seamless (strangers)"` group of the frozen runs (same tracks, same scorer):

| | People (labels) | Cascade (split) | Duplex run 1 | Target |
| --- | --- | --- | --- | --- |
| Reply starts after the caller stops (median) | 0.23 s | 1.93 s | 1.70 s | <= 0.3 s |
| Hand-offs answered before the caller goes on | 81% | 42% | 4% | |
| Cuts into the caller's thinking pauses | 1% | 1% | 0% | < 10% |
| Stops when talked over (median) | 0.51 s | 0.47 s | 0.17 s (1 case) | <= 0.4 s |
| Keeps going through the caller's "mm-hm" | 50% | 25% | - | |
| "Mm-hm"s per minute | 6.8 | 17.4 | 8.3 | |

Run 2 (release v0.2) was not scored on the timing test. A fresh run of a system can differ a little from these by its
own sampling.

## License

Derived from Meta's Seamless Interaction dataset (Meta FAIR, "Seamless Interaction: Dyadic Audiovisual Motion
Modeling and Large-Scale Dataset", 2025), naturalistic test split, licensed
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/): non-commercial research only. Changes: turn timings
computed from Meta's voice-activity and word-timing files; the audio (not included) resampled to 16 kHz and cut to
the shorter of the two tracks. Test data only: never train on it or tune on it.
