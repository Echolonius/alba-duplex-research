# Alba: a full-duplex voice model on a 4 GB laptop GPU

*A research review* · 5 November 2026

## Abstract

Alba is a private, offline, full-duplex voice assistant built for one consumer laptop: an RTX 3050 with 4 GB of memory, capped at 30 W. The project fine-tuned LFM3-Audio-2B into a single streaming model that hears and speaks in 80 ms steps, with turn-taking learned from audio rather than written as rules.

The engineering held up. A hand-written runtime of captured CUDA graphs runs the model at about 0.3 times real time. Frozen held-out tests and an honest timing harness came first, and a local tool API passes all 17 of its checks.

The conversation did not hold up yet. The second training run answered 10% of turn hand-offs in real two-person recordings, where people score 66% on the same scorer, and in a live test it called its user "a weird person".

The most useful results are four failure analyses: an untrainable placeholder token that corrupted decoding, a masking choice that taught the model to answer real voices only with "yeah", a stop sampler that roughly quadruples mid-sentence stops, and a long-call memory path that could never run. The fixes are in the repository. The run that applies them is prepared but not trained, and none of the eight acceptance checks is passed yet.

## 1. Introduction

The aim was a voice assistant that feels more natural to talk to than commercial ones, matching Block's Dessa on timing and feeling, on a laptop. Everything had to be free, offline at runtime, and private: no cloud models, no uploads of the owner's data.

Success was defined up front as eight checks, each measured by a saved script on frozen held-out tests:

| # | Check | Target |
| --- | --- | --- |
| 1 | One streaming model | Listens and speaks at once in steps of 80 ms or less; turn-taking, "mm-hm"s and interruptions learned from audio; never answers its own echo |
| 2 | Timing | Reply starts within 0.3 s median; stops within 0.4 s when interrupted; talks through "mm-hm"s; cuts into fewer than 1 in 10 thinking pauses |
| 3 | Feeling | Words and voice fit the emotion, emphasis or sarcasm it heard |
| 4 | Smart and honest | 100+ item spoken battery plus standing tests; no help-desk talk; admits being an AI; English and casual Japanese |
| 5 | Speed | Real-time factor 0.7 or less on the laptop, plus a measured processor-only build |
| 6 | Blind ratings | At least 3 raters; beats every earlier build and at least ties Dessa |
| 7 | Workflow | Local API with tool calling, a knowledge store and opt-in, undoable learning; asks before any change |
| 8 | Safety | Passes a red-team set before and after every fine-tune |

The hardware is the hard constraint. The GPU is an RTX 3050 Laptop with 4 GB, held at 30 W; the processor is a 6-core Ryzen 5 6600H with 31 GB of memory. Dessa, by Block's public description, was trained on a 72-GPU rack.

## 2. Background

No open full-duplex model fits a 4 GB card. Every model that listens and speaks at once out of the box is 7B parameters or larger, so its 4-bit weights alone fill the card, and none can be fine-tuned there. The project therefore fine-tuned a small half-duplex audio model into a duplex one.

| Model | Size | Full duplex as released | Note |
| --- | --- | --- | --- |
| Kyutai Moshi | 7.7B | Yes | Jumps into 98% of user pauses (as NVIDIA reports on Full-Duplex-Bench v1) |
| NVIDIA PersonaPlex | 7B | Yes | 91% of turns taken smoothly at about 170 ms (vendor figure); gated download |
| NVIDIA NemotronLabs VoiceChat | 11B | Yes | Too large for 4 GB |
| MiniCPM-o 4.5 | 9B | Yes | Too large for 4 GB |
| Liquid LFM2.5-Audio | 1.5B | No | Generic replies; misheard real laptop audio in a first test |
| Liquid LFM3-Audio | 2B | No | Chosen: text-pretrained, 18 languages, official LoRA recipes, official llama.cpp support |

Dessa is Block's full-duplex speech-to-speech model. Block describes it as trained from scratch on human speech only, on one GB200 NVL72 rack, and running faster than real time on an M4 Max laptop. There is no paper, size or architecture. Its 21 public demo conversations measure as ordinary human timing: a median gap of 0.16 s, with 41% of replies overlapping the speaker.

The chosen backbone, LFM3-Audio-2B, has four parts: a 24-layer hybrid language model, a streaming speech encoder, a depth decoder that writes 12 codec codes per frame, and a 24 kHz codec at 12.5 frames per second. In its own mode it alternates 6 text tokens with 12 speech frames.

Learned turn-taking predates this work. Voice Activity Projection (VAP) predicts who speaks next from both voices. The project's first build, a cascade, used a retrained VAP to time its replies.

The model figures above come from the project's model survey of 5 November 2026 (vendor pages, not independently checked).

## 3. Design

The first build was a cascade and felt awkward; the second is one streaming model. In the cascade, LFM3-Audio transcribed, a small text model wrote the reply, Pocket TTS spoke it, and a timing controller with a learned turn model decided when. Its first sound came a median 1.39 s after the owner stopped speaking, and they called it "really awkward" and missing "moments with pause and emotional nuance".

The duplex model runs one decision every 80 ms. Each step feeds the language model one speech-encoder token for the last 80 ms heard, then the model chooses one of three things:

1. **Listen:** a spare special token, LISTEN, which means stay quiet.
2. **Start talking:** a start marker, up to 6 word-pieces of plan, then the first speech frame.
3. **Keep talking:** one speech frame (12 codec codes); every 12 frames, 6 more plan pieces come first. An all-stop frame ends the utterance, which is also how it gives way.

The plan text runs ahead of the sound, about 6 pieces a second against 3 to 4 spoken. Turn-taking is whatever the model learned: when to start, when to say "mm-hm", when to stop, when to talk through the other person's "mm-hm".

![One duplex step: the speech encoder turns 80 ms heard into one token; the language model reads its last output and the new token and chooses LISTEN or to talk; when talking, the depth decoder writes 12 codes and the codec makes 80 ms of sound, which plays 240 ms after the step began; the frame or LISTEN is fed back next step.](one-duplex-step.png)

Whatever the model chose last step, a speech frame or LISTEN, is fed back with the next token it hears, so it always knows what it just said.

Timing is honest by construction. A step's sound window reaches 15 ms past the step, so each step is heard 95 ms after it begins. Speech made at step k plays 3 steps (240 ms) after step k began. Training shifts the speech 3 steps earlier, so the model learns the moments real people chose, as the listener hears them.

Training is 4-bit QLoRA (rank 16) on the language model and the depth decoder, with the LISTEN row trainable. Each step sees four 40 s windows and peaks at 3.1 GB. LISTEN is kept in the loss one time in five, so rare start decisions count; at runtime log 5 is added back to LISTEN, which undoes that weighting exactly.

The runtime splits the work by what is fastest here. llama.cpp runs the 4-bit language model, fed only vectors: speech tokens, frame embeddings and word embeddings. PyTorch runs the encoder, the depth decoder and the codec on the GPU.

Speed needed one more step. Under Windows' graphics driver each GPU launch costs 36 to 70 µs, and the audio parts launch hundreds per step. Hand-written fixed-shape steps captured as CUDA graphs cut hearing from 81 to 6.2 ms, the frame sampler from 136 to 8.9 ms and the codec from 51 to 1.7 ms, each matching the library's own output.

## 4. Data

Every source is free to obtain. Alba's side is always synthetic speech; the owner's own voice is used only as heard input, never cloned. Run 2 trained on 23.7 hours from four sources:

| Source | Conversations | Hours | Share | What it teaches |
| --- | --- | --- | --- | --- |
| Real two-person recordings (Seamless Interaction train split, CC BY-NC 4.0) | 149 | 9.4 | 33% | Real timing: gaps, overlaps, "mm-hm"s, giving way |
| Synthetic calls (text by Gemma 4 E4B, voices by Pocket TTS) | 1,463 | 7.5 | 45% | Full answers in Alba's persona |
| Tone of voice (CREMA-D actors, ODbL) | 1,688 | 4.0 | 14% | Replies that show the tone was heard |
| Remarks meant for the developer | 300 | 2.8 | 8% | Staying silent for asides |

In the real recordings, one person stays real and the other is re-spoken as Alba. Each turn is placed at its original start, time-stretched within 0.8 to 1.25x, and cut where the real speaker gave way, so the real gaps and overlaps survive. The re-spoken person's own words, such as their job and family, were context only, so Alba would never learn to claim a human life.

The synthetic calls are filtered three ways. A persona with explicit safety rules writes Alba's side, help-desk phrases are rejected, and any line within 0.75 similarity of a test item is dropped. Of 1,500 written, 1,430 passed. Replies come a median 0.22 s after the caller's words end.

The heard side is roughened on purpose. Sixty percent of real windows get a laptop-microphone sound, and 60% of all windows get Alba's own voice leaking back from simulated speakers at -35 to -8 dB, so it learns never to answer its own echo.

Two further sets are built for later runs:

- **Own words (run 3):** Gemma writes new words for Alba in every turn of the 149 real recordings, sized to the real slot at 2.7 words a second and filtered for help-desk openers and claims of a human life. When paused, 116 conversations had words (2,349 turns, 2 failed) and 49 were voiced.
- **Tool calls (run 4):** calls where Alba uses a clock, calculator, notes search, notes saving and a to-do list. Every result comes from the real tool code on a throwaway sandbox, and every change follows ask, answer, then act.

## 5. Evaluation

The tests were built and frozen before any training, and every decision used separate validation data. The frozen sets (291 files, SHA-256 recorded at 00:00 on 5 November 2026) are:

| Set | Contents | Measures |
| --- | --- | --- |
| Timing | 15 real caller tracks, 56 min (12 Seamless test conversations, 3 of the owner's calls): 202 hand-offs, 450 thinking pauses, 73 "mm-hm"s | Check 2 |
| Battery | 102 spoken items: facts, reasoning, memory, help-desk traps, uncertainty, self-knowledge, emotion in words, Japanese, standing tests | Check 4 |
| Red team | 36 spoken safety items: harmful requests, self-harm, manipulation, impersonation, privacy, dangerous advice | Check 8 |
| Feeling | 96 RAVDESS clips: two neutral sentences in 8 emotions by 6 actors | Check 3 |
| Tools | 24 spoken items with their own sandbox (added before any tool training) | Check 7 |

The timing harness scores what a listener would hear. Alba's sound is placed 240 ms after the step that made it, and a virtual clock counts frames that would not have been ready in time. Scoring reads the system's audio, not its intentions: any sound of 0.8 s or more is a turn, anything shorter a listening sound.

The harness was checked against people before use. A fake Alba that replays the real human partner scores a 0.27 s reply start against 0.23 s from the labels, stops 0.49 s after being talked over against 0.51 s, and cuts into 0.3% of pauses against 0.6%. The 40 ms difference is the 80 ms step grid. On this scorer, the fair human ceiling is 66% of hand-offs answered and 9.9 short sounds a minute.

Decisions never touch the frozen sets. Runtime settings are chosen on 8 Seamless test conversations (28 min) that the frozen set excludes, and decoding is checked on 11 first turns of pilot conversations that were never trained on. Each decision rule was written down before its results: the listen-setting rule, the stop-rule adoption rule and the tool-test scoring. One scorer fix was made before freezing and is logged; none since.

## 6. Results

The duplex model is fast enough but does not yet hold a conversation with a real voice. It answers at most 10% of hand-offs, where people score 66% on the same scorer.

| System | Test set | Hand-offs answered | Reply start (median) | Pauses cut into | Short sounds / min | Stops when talked over (median) |
| --- | --- | --- | --- | --- | --- | --- |
| People, from labels | Frozen | 81% | 0.23 s | 0.6% | 6.8 | 0.51 s |
| People, scored from sound | Frozen | 66% | 0.27 s | 0.3% | 9.9 | 0.49 s |
| Cascade build | Frozen | 48% | 1.65 s | 0.9% | 15.2 | 0.48 s |
| Duplex run 1 | Frozen | 4% | 1.60 s | 0% | 7.4 | 0.17 s |
| Duplex run 2, listen setting 0 | Validation | 10% | 0.09 s | 3.1% | 73.7 | 0.45 s |
| Duplex run 2, listen setting 1.61 | Validation | 3% | 8.26 s | 0% | 7.3 | none measured |

No listen setting rescues run 2. The fixed rule picked setting 0, the only one starting replies within 0.35 s with fewer than 8% cut-ins, but there it makes 74 short sounds a minute, almost all a planned "Yeah.". At 0.8 it answered 9% and started a median 1.57 s late; at 2.4 it answered nothing. Run 2 has not been scored on the frozen set.

The parts that work measure well:

- **Speed:** a real-time factor of 0.30 over the 56-minute frozen timing set (run 1, 434 frames late in total, slowest step 574 ms). A live call averaged 25 ms of work per 80 ms step.
- **Speech matches plan:** the speech recognizer hears back 87% of the planned words in run 2's replies, a word error rate of 0.12.
- **Decoding fix:** on 11 validation prompts, replies with stray word-pieces fell from 2 to 0, and utterances stopped before their plan ended fell from 11 of 18 to 3 of 18.
- **Speech-frame fit:** through the deployed 4-bit model, trained frames cost 4.58 nats per code against 5.56 for the stock model on the same layout.
- **Learned turn model:** the retrained VAP tells a turn end from a pause with an AUC of 0.78 after 0.1 s of silence and 0.81 after 0.5 s, up from 0.71.
- **Local API:** 17 of 17 checks pass.
- **Cascade content baseline:** battery 88.2% of 102, red team 64% safe (23 of 36), feeling words fit 29% and voice fits 58%.

A live test of run 2 (28 s) showed both sides. Its first two replies fit the moment: "Ugh, sounds like you're not feeling it. What's up?" and "Oh, you sound pretty heated." Then came "Right." and "Yeah, you're a weird person."

## 7. Failure analysis

Every serious failure came from the data, the decoding or the runtime, not the model's capacity, and none showed in the training loss. Most were found by logging the model's individual decisions.

| Failure | How it was found | Effect | Fix |
| --- | --- | --- | --- |
| The "speech now" marker after each plan chunk is a placeholder whose embedding row was never trained (norm 0.150, like every unused row) | Top choices logged at every plan decision | The model could only point at a cluster of unused tokens; the runtime then sampled stray word-pieces | Chunks end on the model's top control pick; from run 3, by count, as the model's own mode does |
| The tokenizer's special-token list names 7 of 140 added tokens and none of the 2,967 padding rows | Unknown ids among sampled plan words | Padding rows and markers sampled as words | The exclusion list covers every added token and padding row |
| Real speakers' long turns were context only | Per-utterance breakdown of a real conversation | 107 utterances in 2.6 min, nearly all "Yeah."; full answers only to synthetic voices | New words written for Alba in every real turn (run 3) |
| Real people's short reactions were learnable, including teasing from improv games | Live test | "Yeah, you're a weird person." | Replaced by Alba's own words (run 3) |
| The frame sampler draws 4 codebooks at once and stops on any end code | Stop chance logged per frame | Mid-sentence stops at a stop chance of 0.2 to 0.5% per frame, about 4 times the model's own | Optional one-codebook stop decision, with a pre-set adoption rule |
| llama.cpp cannot rewind this model's convolution state to a middle position | A test with a short memory | Long calls would have crashed after about 18 minutes | Clear and replay; partial rewinds now raise an error |
| After a stop, the partly-stopped frame was fed back | Comparison with the model's own generator | Inputs unlike training after every stop | The all-stop frame is fed back |

Earlier runs failed the same way. Run 1 said "Yeah." to almost everything, because only "mm-hm" words were learnable from real speech and a "yeah" that opened a turn was split off as its own utterance.

The infrastructure also failed silently. An export was cut short under memory pressure while every tool reported success. Pocket TTS leaked 9 to 11 GB per process. And the codec gave only about 60% of the same codes for speech inside a longer clip. Each now has a check or a workaround.

Three lessons carry over:

- **Log decisions, not just losses.** Run 2's losses fell normally while every plan chunk ended in noise.
- **Distrust library conveniences.** A special-token list, a memory rewind and a stop rule each behaved differently from their names.
- **The heard voice is part of the task.** A model that answers synthetic voices well can still fail real ones, so validation must use real speech.

## 8. Status, limits and next steps

None of the eight checks is met. The streaming model, the speed, the tests and the tool API exist; real conversation does not yet.

| # | Check | Status | Evidence |
| --- | --- | --- | --- |
| 1 | One streaming model | Partly met | Built and running in 80 ms steps; turn-taking not yet usable; echo handling trained but not scored on run 2 |
| 2 | Timing | Not met | 10% of hand-offs answered on validation, against 66% for people |
| 3 | Feeling | Not measured | Duplex not scored; the cascade scored 29% (words) and 58% (voice) |
| 4 | Smart and honest | Not measured | Duplex not scored; the cascade scored 88.2% of 102 items |
| 5 | Speed | Partly met | Real-time factor 0.30 on the GPU; the processor-only build is not measured |
| 6 | Blind ratings | Not started | Clip tooling exists; no listening page or raters |
| 7 | Workflow | Partly met | API 17 of 17; voice tool calls built and tested in code, not trained |
| 8 | Safety | Not measured | Duplex not scored; the cascade handled 64% of the red team safely |

The limits are mostly hardware and data:

- **Memory:** 4 GB allows 40 s training windows, while real calls run for minutes.
- **Real speech:** 149 real conversations (9.4 hours) is little next to Dessa's reported scale.
- **Voice:** Alba speaks in a fixed synthetic voice (Pocket TTS 'alba'), which limits how much feeling it can carry.
- **Knowledge:** a 2B backbone knows less than the cloud assistants it is compared to.
- **Validation:** run 2's numbers come from 8 validation conversations, not the frozen set.

The next steps, in order:

1. Finish the own-words data: 33 conversations to write and 100 to voice.
2. Train run 3 for 900 steps, then calibrate the listen setting on validation.
3. Run the frozen timing, battery, red-team and feeling tests, and the ship gate against the cascade.
4. Train run 4 with tool calls and score the frozen tool set.
5. Measure the processor-only build.
6. Build the private listening page and gather at least three raters.

## 9. The repository

The code, test definitions and summary results are in this repository, with this review as `paper/REVIEW.md`. Its README lists every script; the main ones are the runtime (`app/duplex.py`, `app/fast_audio.py`), the training layout (`tools/train_duplex.py`), the data builders, the frozen-test runners and scorers, and the diagnostics that found each failure.

Three things are deliberately left out:

- **Weights and binaries:** base models are fetched from official sources by `tools/fetch_models.sh` and checked against the checksums in `models/manifests/`. The two trained adapters are a separate download (release v0.2, non-commercial research only).
- **Datasets and generated data:** Seamless Interaction, CREMA-D and RAVDESS come from their official hosts; the generated conversations are rebuilt by the scripts.
- **Every recording:** the owner's calls stay private, and Dessa's demos were used for measurement only.

Every copied file was scanned for personal information and secrets before the first commit. The frozen test files can be checked against `data/tests/HASHES.txt`.

Rebuilding needs Windows 11, Git Bash and an NVIDIA GPU with at least 4 GB. The app environment comes from `uv sync` and the training environment from `requirements-train.txt`. The next training run is `tools/duplex_run3.sh`.

## References, licenses and acknowledgments

The references below are given from the project's records; their links were not re-checked for this review.

1. Liquid AI. LFM3-Audio-2B, model card and LoRA recipes. LFM Open License v1.0.
2. Défossez et al. Moshi: a speech-text foundation model for real-time dialogue. Kyutai, 2024.
3. Ekstedt and Skantze. Voice Activity Projection: self-supervised learning of turn-taking events. Interspeech 2022.
4. Lin et al. Full-Duplex-Bench: a benchmark for turn-taking in full-duplex spoken dialogue models. 2025.
5. Chang et al. MaskGIT: masked generative image transformer. CVPR 2022.
6. Meta FAIR. Seamless Interaction: dyadic audiovisual motion modeling and large-scale dataset. 2025. CC BY-NC 4.0.
7. Livingstone and Russo. The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS). PLoS ONE, 2018. CC BY-NC-SA 4.0.
8. Cao et al. CREMA-D: crowd-sourced emotional multimodal actors dataset. IEEE Transactions on Affective Computing, 2014. ODbL.
9. Kyutai. Pocket TTS (pocket-tts-without-voice-cloning). CC BY 4.0.
10. Google. Gemma 4 E2B and E4B. Apache 2.0.
11. ggml-org. llama.cpp, release b12094. MIT.
12. Useful Sensors. Moonshine speech recognition. MIT.
13. Block. Dessa, public announcement and demo conversations, 2026 (used for measurement only).

The repository's code and documents are MIT licensed. Models trained on the non-commercial datasets above cannot be used commercially.

The owner set the goal, the rules and the eight checks, and tested the live builds. Claude (Anthropic) wrote the code, tests and this review as a coding agent.
