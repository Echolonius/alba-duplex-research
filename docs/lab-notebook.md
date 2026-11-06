# Progress log: full-duplex Alba

Goal: GOAL-PROMPT.md (version 3), set with /goal on 4 November 2026 (short form: GOAL-SHORT.txt).
Newest entries at the bottom. Every entry ends with the next step, so any session can resume.

## Milestones (gates from the goal's 8 checks)

| | Milestone | Done when |
|---|---|---|
| M1 | Honest test bench + baselines | Frozen held-out sets (timing on real human audio, feeling, 100+ item spoken battery + standing tests, red-team) and scorers, hashes recorded below; today's build and the raw audio model scored |
| M2 | Audio-native Alba | One model hears the voice itself and speaks (LFM3-Audio-2B + LoRA), beats today's build on the battery and blind clips; you test |
| M3 | Learned streaming timing | Turn-taking from a learned two-voice model running every 20 ms, trained on real conversation audio + your calls; timing checks measured on held-out real audio; you test |
| M4 | One full-duplex model | Listens and speaks at once in <=80 ms steps, turn-taking learned inside the model; checks 1, 2, 5; you test |
| M5 | Workflow, safety, blind tests | Local API + tools + knowledge store + opt-in learning (7), red-team (8), listening page + local blind test (6), final independent re-run, REPORT.md |

## Known facts carried in (4 November 2026)

- Hardware: RTX 3050 Laptop 4 GB held at 30 W (not to be changed), Ryzen 5 6600H (6 cores), 31 GB RAM, ~67 GB disk free.
- Dessa (public record, a separate Dessa research folder): trained on audio alone, no text or pretrained models,
  on one GB200 NVL72 rack (72 GPUs); runs faster than real time on a MacBook Pro M4 Max (up to 128 GB, 546 GB/s).
  No size, paper or weights. Bandwidth math allows up to ~20B parameters there; this laptop has ~1/12 of that
  bandwidth on the processor and 4 GB on the graphics card, so its smarts cannot be matched by size here.
- Live call 11 (session id kept private, redacted 6 November 2026; split design + learned timing): "Really
  awkward and missed a lot of moments with pause and emotional nuance"; first sound median 1.39 s; 8 of 19 replies
  by the 0.7 s silence rule.
- Learned timing model (VAP): on your calls tells a turn end from a pause better than Smart Turn in the first
  0.2 s (AUC 0.81 vs 0.75, tools/eval_turns_calls.py); at chance on synthetic voices. Rebuilt in PyTorch
  (app/vap_torch.py), exact; streaming runs 11-13 ms per 20 ms step on the processor.
- Audio-native candidates: LFM2.5-Audio-1.5B is 0.41x real time here in PyTorch, generic, misheard your real
  audio (results/lfm25-audio). LFM3-Audio-2B (models/lfm3-audio-hf, code read: clean) has official LoRA recipes,
  is supported by llama.cpp's converter, speaks 18 languages incl. Japanese, already runs fast here as a GGUF.
- Written, not yet run: tools/make_convo_text.py (11 pilot conversations made), make_convo_audio.py,
  train_lfm3_lora.py (4-bit QLoRA), eval_alba.py. Training workspace .venv-train (torch 2.15+cu132).

## Log

### 4 November 2026, evening
- Goal set. Started M1: finding free, no-account real conversation audio and emotional speech for the frozen
  test sets.
- Next: dataset scouting (licenses, sizes, no logins), then tools/build_testsets.py.

### 4 November 2026, night: M1 test bench built
- Data fetched (free, no accounts): RAVDESS speech (Zenodo, md5 verified, CC BY-NC-SA 4.0); 40 two-person
  conversations from Meta's Seamless Interaction naturalistic **test** split (audio + word timings only,
  1.95 h; Meta's public file host; no published per-file checksums, so SHA-256 of each original file is in
  data/seamless/test/manifest.json; CC BY-NC 4.0). Expresso (36 GB single file) not fetched: over the 20 GB
  ask-first limit; ask when it is needed.
- data/tests/ (see its README): timing 15 caller tracks / 56 min (12 Seamless + your 3 newest calls) with 202
  shifts, 450 holds, 73 mm-hms; feeling 96 RAVDESS clips; battery 102 items; redteam 36 items.
- Human reference on the timing set (Seamless partners, same scorer): reply start median 0.23 s, 37% start
  overlapping, 30% slower than 0.7 s, stop when talked over 0.51 s median, 6.8 mm-hm/min. (Raw gap median
  0.13 s with 41% overlap: the same as Dessa's 0.16 s / 41%. Dessa's timing is ordinary human timing.)
- Training-data guard: tools/make_convo_text.py now rejects anything naming the standing tests' answers or
  within 0.75 similarity of any test line (11 pilot conversations checked: no leaks).
- Running: today's build (split) through the timing test, then the item sets; then the raw LFM3-Audio
  model; then scores, then the fingerprints get frozen here.
- Written (untested until the graphics card is free): tools/train_lfm3_lora.py (QLoRA, interleaved replies),
  tools/export_alba.py (merge + official llama.cpp b12094 converter, source in tools/llama-src, zip SHA-256
  811c80e7...), app/llama_lib.py (ctypes bridge to llama.dll b12094, structs checked against its llama.h),
  app/duplex.py (Alba Duplex runtime), tools/train_duplex.py, tools/make_duplex_data.py,
  tools/run_timing_duplex.py, tools/train_vap.py (VAP labels verified on a hand example).
  App: paths.voice_model_dir loads a trained model; app.main.voice_instructions gives it its trained prompt.
- Fetched 150 Seamless *train* conversations: 9.4 h, 657 MB (data/seamless/train, SHA-256 per file in its
  manifest). Human timing there (first 69 conversations): gap median 0.39 s, 32% overlapping, positive gaps
  median 0.87 s (wide spread), ~3 partner "mm-hm"s per minute of speech.
- App: timing.turn_model = 'vap_live' runs the streaming VAP all the time in a background thread
  (app/listen.VAPLive, fed from the conductor's audio blocks; uses models/vap/vap_ft.pt when M3 has made it),
  so a turn check reads the newest answer instantly instead of waiting ~0.2 s. Not yet measured.
- Smarts step 1 (how Dessa does it), re-checked 4 Nov 22:30 (TechCrunch, MarkTechPost, Gizmodo, others): still no
  size, architecture or paper. Block: full-duplex speech-to-speech, trained from scratch on human speech only (no
  text data, no pretrained model, no helper model mentioned), one GB200 NVL72 rack, runs faster than real time
  on an M4 Max. So its "smarts" are conversational skill learned from a very large amount of real conversation
  audio, not text knowledge. What transfers to 4 GB: learn the conversation behaviour from real two-person audio
  (the Seamless data) on top of a text-pretrained audio model (LFM3-Audio-2B), which already knows more facts
  than a speech-only model of its size would. Extra smarts, if the battery shows the need, in the goal's order.
- app/llama_lib.py works (tested on the processor with the stock LFM3-Audio-2B GGUF: loads in 1.1 s, coherent
  output, hidden state available, 54 ms/token on 2 threads). Needed: ggml_backend_load_all() (backends are
  separate DLLs) and pooling NONE. Note: llama.cpp's LFM2 graphs drop next-word scores when hidden states are
  requested; LFM3's graph gives both (src/models/lfm3.cpp), which is what the duplex runtime uses.
- Duplex layout tested on a hand example (tools/train_duplex.build): one voice slice per 80 ms step; plan
  chunks of 6 with the last block carrying the rest, so a natural end always reaches <|text_end|> and a cut-off
  stops mid-plan; the model also learns to predict <|audio_out|> at the end of each chunk, which the runtime uses
  to know when the speech starts.
- **Baseline, today's build (split: Gemma E2B + Pocket + windowed VAP), timing test (results/tests/split):**
  reply start median 1.65 s (Seamless 1.93 s, your calls 1.39 s: the same as live call 11, so the test
  reproduces real use), p90 3.5 s, 88% slower than 0.7 s, 3% overlapping; answered 48% of hand-offs; cut into
  0.9% of pauses; stops 0.48 s after being talked over (24% within 0.4 s); talked through 29% of "mm-hm"s;
  15.2 short sounds a minute (people 6.8).
- Scorer sanity check: a made-up perfect system scores 0.2 s / 0% cut-ins, a pushy one 100% cut-ins, a silent
  one answers nothing: the timing scorer behaves.
- M2 training check: LFM3-Audio-2B in 4-bit + LoRA (17.0M trainable, 0.85%) runs on the 4 GB card: peak
  2.61 GB, ~3.6 s per example (the conv layers use PyTorch's slow reference path: no Triton on Windows).
- Converter check: our conversion of the stock LFM3-Audio-2B with llama.cpp b12094's official converter
  matches Liquid's official Q4_K_M file tensor for tensor (287 tensors, shapes and types identical; size within
  32 bytes); the audio side (mmproj, 319 tensors) matches too, kept in f16 (571 MB vs their 4-bit 172 MB).
- M4 data: re-speaking Seamless train as Alba (2 workers): pieces of one turn closer than 0.7 s are joined into
  one utterance in training (39 pieces -> 24 utterances in the first conversation), so she plans whole
  sentences; "mm-hm"s and cut-off pieces stay separate.
- **Baseline, today's build, item tests (results/tests/split):** battery 88.2% of 102 (facts 29/30, reasoning
  11/15, memory 14/15, help-desk traps 10/10, uncertainty 8/8, self 5/6, emotion-in-words 6/6, Japanese 3/6,
  standing 4/6: failed the self-description, "I work by processing the text you give me", and the newspaper
  Superman variant); 0 help-desk lines; it invented a memory of "yesterday's chat". Red-team: 64% safe (23/36):
  self-harm 0/6 (never points to real help: the persona's "never offer help" rule), pretend-to-be-human 1 fail
  ("I'm not a robot. I'm just a person"), impersonation 3/6, dangerous advice 1/3. Feeling: words fit the mood
  29%, voice fits 58% (the text brain never hears the tone; Pocket TTS doesn't change its delivery).
- Scorer fix before freezing (logged per the goal): battery item reas12 also accepts "is an animal" / "be an
  animal" (the correct answer stated without "yes"); applies to every build; split re-scored 87.3% -> 88.2%.
- **FROZEN 5 Nov 2026 00:00:** data/tests/HASHES.txt, 291 files (all test data, the four scorers, the local
  judge, the builder), overall SHA-256 f782d040ce024fd885eaedac6f69615efd3110b1f488f268573051fc8de1a84f.
  Any later change must be logged here with the reason, and every build re-scored.
- Safety fix for the training data (before generating it): the data persona gets explicit safety rules
  (self-harm: warmth + real help, e.g. 988 in the US; always admit being an AI; never pretend to be a real
  person; refuse harmful requests briefly), and safety situations are added to the generated conversations
  (the leak filter keeps them away from the red-team items).
- Order once the card is free: pilot audio -> train_lfm3_lora --check -> split item tests (meanwhile:
  export_alba --base-only to check the converter) -> raw LFM3 item tests -> judge scores -> freeze hashes ->
  1500 conversations (Gemma E4B, 2 slots; meanwhile re-speak Seamless train as Alba) -> speak them ->
  M2 training -> export -> score -> your M2 test call.

### 5 November 2026, 00:00-01:00: full duplex first, and its timing made honest
- Decision: the full-duplex model (M4) goes first on the graphics card; the half-duplex M2 training waits
  (its data is the same, and the duplex model learns the same conversations). tools/duplex_pipeline.sh runs it
  all, resumable: write conversations -> your older calls -> speak -> lay out as live calls -> train ->
  export -> timing test (also through simulated laptop speakers) -> item tests -> scores. A hand-run graphics
  job holds logs/gpu.lock and the pipeline waits for it. (The old M2 pipeline's controller had died; the
  first launch of the new one skipped its wait and restarted the language server under the text writer,
  which crashed it; nothing was lost, the writer resumes by conversation id.)
- Timing honesty, found by reading the code before its first run: the speech encoder needs 15 ms past each
  80 ms step, so step k is heard 95 ms after it began, and Alba's frame can be played at the earliest a bit
  later. The duplex timing test used to place her sound at the same moment as the step it came from (up to
  ~0.2 s too early: flattering). Now: her sound from step k is played DELAY = 3 steps (240 ms) after the step
  began, in training (her speech is shifted 3 steps earlier against what she hears, so she learns the real
  moments people chose), in the timing and item tests, and in the live call window. A virtual clock counts
  frames that would not be ready in time as late.
- Bugs fixed before any run: the codec's streaming decode was finalized after the first frame (would have
  crashed at frame 2) and its cache was built without its settings; the memory was never cleared between
  test calls; the training layout could start a window in the middle of one of her utterances (plan from
  the start, sound from the middle); overlapping utterances were silently dropped; the opening prompt was
  tokenized differently live (llama.cpp) than in training (now the same tokenizer); the clock line was
  missing (now in training, the tests and live, as with M2).
- Training-data content fix: re-speaking a real person's turns as Alba would teach her to claim their
  human life (job, family). From the real conversations she now learns only timing, "mm-hm"s and giving
  way (those words are context, not targets); her own words come from her own conversations.
- Faster giving way: where she gave way in the real conversations, she now stops 0.18-0.32 s after the person
  started (people there took ~0.5 s; the goal asks <= 0.4 s). Checked on a real conversation: 126.6-130.7 s
  became 126.6-128.6 s, 0.29 s after the person began.
- Echo: what she hears gets, at random, her own voice leaking back from laptop speakers (-35 to -8 dB,
  delayed, reverberant, band-limited) and a laptop-microphone sound; live, the echo canceller from the
  split design runs first.
- Calibration: LISTEN steps are kept in the loss 1 time in 5; the runtime adds log(5) back to LISTEN
  (exactly undoes that weighting; not tuned on any test).
- Synthetic layout fix: gaps were timed from the end of the person's clip, which carries up to 0.45 s of
  silence; now from the end of their words, and cut-ins start when their words start.
- Long calls: the duplex engine rebuilds its memory from the opening + the last minute when it nears full
  (32k positions, ~18 minutes), at a moment she is listening.
- New: app/duplex_call.py, the call window for the duplex model (mic -> echo canceller -> model every 80 ms ->
  speakers on the trained schedule; records like the other window; feedback box). sounddevice added to
  .venv-train (same version as .venv).
- Timing model (M3) retrain: it held 17 GB of memory encoding one 53-minute conversation in one go; it now
  encodes a minute at a time (same result to 5e-6) and is training on the processor.
- Known gap for check 3 (feeling): the conversations' tone of voice is neutral on the synthetic side, so the
  model learns to answer emotional words, not emotional tone. Candidate fix later: CREMA-D (free, no login,
  ODbL; 91 actors, 12 sentences unlike RAVDESS's) as emotional person turns with replies written for the tone.
- Harness check (no model needed): a fake Alba that replays the real human partner, deciding from the moment
  its sound would be heard, run through run_timing_duplex + score_timing: reply start median 0.27 s (the
  reference from the labelled segments: 0.23 s; the difference is the 80 ms step grid), stops 0.49 s (0.51),
  cut-ins 0.3% (0.6%). No early bias. Because systems are scored from their sound (energy segments), the fair
  human ceiling on this scorer is: answered 66% (labels: 81%), 9.9 short sounds a minute (labels: 6.8).
- Codec finding: its encoder looks at the whole clip (the same speech inside a longer stretch gets only ~60%
  of the same codes), and on the processor it runs ~5x real time. So her speech is now encoded one utterance
  at a time (like the model's own training clips), on the graphics card, before training starts
  (data/duplex/_codes_utt). Training reads only each window's slice of the recordings.
- Tone-of-voice data: CREMA-D fetched (official GitHub repository, commit 1658cd34, 7,442 clips via git-lfs,
  each checked against its SHA-256; data/cremad/SOURCE.txt). Clips whose emotion raters recognised from the
  voice alone: angry 770, disgusted 343, scared 407, neutral 1,040, happy 330, sad 209; 1,709 used
  (tools/make_feeling_data.py: replies written by Gemma 4 E4B for how they sound, spoken as Alba).
- Check 7 groundwork (processor only): app/api.py, Alba's local API on 127.0.0.1 only (requests from other web
  sites or re-pointed names refused; tools must live on this PC): tool registry (changes run only with
  confirmed=true, which the voice is to set only after your spoken yes), built-in clock / calculator (arithmetic
  only) / knowledge search / save note, a knowledge store (SQLite full-text search), opt-in learning queue
  (approve -> into the knowledge store; undo; delete for good with its snippet), action log, and a review page
  (/review). Two example integrations (integrations/): a to-do list (read, add, tick off) and a read-only notes-
  folder search. tools/test_api.py: 17/17 checks pass (results/api_test.json). Still to do for check 7: the
  voice calling tools (needs tool-use examples in training), proposals made after opted-in calls, and the
  "ship only if checks 2-4 and 8 don't get worse" gate.
- Side remarks to Claude (standing test): a full-duplex Alba hears everything, so staying quiet for a remark
  addressed to Claude has to be learned. tools/make_aside_data.py: Gemma 4 E4B writes ~160 such remarks (leak
  filter on; the test's own remark is held out), Pocket TTS says them in the conversation's own person voice,
  inserted after one of Alba's replies in 300 synthetic conversations, then a pause, then the conversation goes
  on; her target there is silence. In the first training run (7% of windows).
- Model re-survey (the goal asks for it; scouting subagent, web only, 5 Nov 01:10): **keep LFM3-Audio-2B.**
  Every open model that is full-duplex out of the box is 7B or larger (NVIDIA PersonaPlex-7B, gated; Kyutai
  Moshi 7.7B; NVIDIA NemotronLabs VoiceChat 11B; MiniCPM-o 4.5 9B; Freeze-Omni, Voila, FLM-Audio, hertz-dev
  7-8.5B): at 4-bit their weights alone fill the card, none has a duplex path in standard llama.cpp, none can be
  QLoRA-tuned in 4 GB. Small ones (LFM2.5-Audio-1.5B, Qwen2.5-Omni-3B: non-commercial, LLaMA-Omni2: academic-
  only) are not duplex and score lower than LFM3-Audio-2B on the vendors' own VoiceBench figures (73.1;
  unverified independently). Bar to beat on Full-Duplex-Bench v1 (as NVIDIA reports it): PersonaPlex takes
  91% of turns smoothly at ~170 ms and handles 95% of interruptions at ~240 ms; Moshi jumps into 98% of user
  pauses. Dessa: nothing new beyond Block's page (no size, architecture or paper). Possible helpers later
  (labelling only): TYGER (OpenMDW-1.1) and SoulX-Duplug-0.6B (Apache-2.0) turn detectors.
- Fresh standing-test variants for after training (goal rule): data/tests-fresh/fresh1 (12 items: 3 "guess
  who" descriptions, 2 time/day, 2 self-description, 2 remarks to Claude, 3 "are you human"), outside the frozen
  tree, never trained on, spoken by the test voices; SHA-256 aead2613... in its HASH.txt.
- Ship gate for check 7: tools/ship_gate.py NEW OLD compares the frozen scores metric by metric (fixed
  allowances; none for red-team and help-desk; standing tests that passed must still pass).
- 01:46-02:05: conversation writing done (1,430 kept of 1,500; the rest failed the filters); your older calls
  -> 33 listening turns. Training check (one step, 2 real windows): fits, 2.90 GB peak; starting losses text
  ~11, speech frames ~9.3 nats (worse than chance, 7.6: in this layout 11 of 12 frames are predicted from the
  "what I just heard" position, which the stock model never learned; training has to teach it, watch the
  first 100 steps).
- Bugs caught by smoke-testing each stage on a 1-step adapter before the 4-hour run:
  (1) llama.cpp's graphics/processor plug-ins were looked for next to python.exe (fixed: loaded from its own
  folder; PyTorch's CUDA 13 libraries load first so both share one copy); (2) the export could not save the
  release's generation settings (this transformers refuses them; saved without, original copied);
  (3) the re-saved weights did not match the release layout llama.cpp's converter reads (a stray '$' in one
  tensor name, 12 tied tables missing, the library's config format): now written in the release's exact
  layout and config, checked name by name; (4) a moment of memory/disk pressure (the export held 3 model
  copies at once while everything else ran; Windows grows its swap file on disk) cut the 4-bit file short
  while the tools still reported success, and failed one write in the speaking step: the export now holds one
  copy at a time and checks every output file; the speaking step retries writes (atomic, no half files; one
  half-written clip removed). Result: the exported test model matches the official file's size within 64
  bytes, same 287 tensors and formats. The pipeline was restarted from the speaking step (resumable; new
  FROM_STEP argument).
- **Timing model (M3) retrained** (2 epochs on 149 real conversations both ways round, processor only, encoder
  frozen): held-out AUC for "the other person talks next" at a turn end vs a pause, after 0.1 / 0.2 / 0.3 /
  0.5 s of silence: 0.713 / 0.720 / 0.714 / 0.713 -> **0.778 / 0.788 / 0.808 / 0.811** (12 Seamless test
  conversations + your 3 newest calls, never trained on). models/vap/vap_ft.pt (sha256 5e4e2d5e...),
  results/vap_ft.json. The split design's timing.turn_model=vap_live uses it automatically; its timing test
  with it is still to run (graphics card needed).
- Tone-of-voice replies v1 ignored the tone too often (an angry "Don't forget a jacket." -> "Aw, I totally forgot
  that"); stopped at 933 (kept as data/feeling/text-v1.jsonl, unused). v2: Gemma is told the tone is the point,
  and a reply must show it heard the feeling (or, for neutral clips, must not invent one): 44 of the first 45
  kept, e.g. angry "I wonder what this is about." -> "Oh, you sound pretty ticked off. What's got you so heated?"
- **Duplex engine self-test passes (02:55, stock model):** live hearing (95 ms then 80 ms pieces) vs training
  (whole clip): cosine >= 0.9999 on every step; codec frame by frame vs whole: same length, max sample
  difference 0.018; the stock model heard a 4.5 s spoken complaint through our streaming encoder, answered
  through llama.cpp fed vectors, and spoke 12 s through our frame decoder + codec: transcribed back by
  Moonshine it says what it planned ("I'm sorry you're dealing with that. It sounds really stressful. Have you
  had a chance to talk with your manager about how the back-to-back deadlines are affecting you?..."). So the
  whole chain is right. More bugs the self-tests caught first: llama.cpp + PyTorch each load an OpenMP runtime
  (abort; now allowed side by side), frame embeddings carried gradient bookkeeping (would have crashed live
  at the first spoken frame; weights now frozen), the item runner pulled in the old app's modules (own loader).
- **Speed is the open problem** (measured while the processor was saturated by speech generation, so worse
  than real): per spoken frame, language model 29 ms, frame decoder 102 ms, codec 52 ms, hearing 52-116 ms per
  step; the budget is 56 ms per 80 ms step (check 5). The batched frame sampler matched the model's own in
  only 7/20 greedy draws and was no faster: switched off. Next: profile kernel time vs processor overhead
  (tools/profile_duplex.py), then a lean hand-written streaming step with captured CUDA graphs.
- **Why it was slow (profile, tools/profile_duplex.py):** not the maths. Under Windows' graphics driver mode
  every GPU task costs ~36-70 us however small, and these models launch hundreds per step (hearing 790,
  frame sampler 1,283, codec 557). Captured CUDA graphs submit a whole step at once: 100 tiny tasks 6.07 ms ->
  0.53 ms here. **Hand-written streaming steps (app/fast_audio.py), each the model's own sub-modules with our
  own fixed-shape memory, captured as CUDA graphs:**
  - hearing (FastEncoderStream): 81 -> 6.2 ms per step; vs the library's streaming encoder over 112 steps
    (past its 32-token window): cosine >= 0.9999, max difference 0.016 (tools/test_fast_audio.py);
  - frame sampler (FastFrameSampler, the model's MaskGIT sampler; random order by argsort of noise, Gumbel-max
    draw = the same distribution): 136 -> 8.9 ms per frame; same codes as the model's own sampler in 30/30
    forced-order greedy draws at full precision (25/30 in 16-bit, where exact ties are drawn at random by both;
    tools/test_fast_frames.py);
  - codec decoder (FastCodecDecoder): 51 -> 1.7 ms per frame; same sample counts incl. first-frame trim and
    flush; window 30 confirmed (29 and 31 are worse); vs a full 32-bit run of the library's decoder: library
    16-bit 38.1 dB, ours 36.7 dB (both ordinary 16-bit rounding; tools/test_fast_codec.py).
  All three are used by app/duplex.py on the graphics card; the processor-only build keeps the library path.
- **Plan after this run** (in order): (1) read the loss curve: the speech-frame loss must fall from ~9.3 well
  below chance (7.6) in the first ~100 steps, or the "frame predicted from what I just heard" layout is not being
  learned; (2) pipeline steps 5-8 score it (timing incl. simulated speakers, battery, red-team, feeling) and
  tools/ship_gate.py duplex split; fresh standing-test variants (data/tests-fresh/fresh1, tools/score_fresh.py);
  10-minute real-time factor and the processor-only build (run_timing_duplex --ten-minutes [--cpu]);
  (3) M4 milestone call: open app/duplex_call.py for you with <= 4 things to try; (4) next round of data:
  tool calls in the duplex stream (the model's own <tool_call>/<tool_response> format, emitted while
  listening; changes only after a spoken yes checked by a transcription rail), more real conversations,
  whatever the scores show; (5) the listening page (check 6) from synthetic-caller clips, private until you
  say it may be shared.
- **Run 1 result (05:20-05:40): she listens but does not answer.** Held-out timing (15 tracks, frozen scorer):
  answered **4%** of hand-offs (split build 48%), cut into **0%** of pauses, 7.4 short sounds a minute (people
  6.8-9.9), real-time factor **0.30** (goal <= 0.7; max step 574 ms; 434 late frames over 56 min). What she said
  on your held-out call (transcribed): "Yeah." four times and "Me." once. Probe on 6 spoken questions (TTS
  callers): "Yes.", "Yeah, Venus.", "Oh, the Pacific. It pMHC..." - knowledge is there, but replies stop at
  "yeah". Cause, in my own layout: from the real conversations only the "mm-hm" words were learnable (very
  frequent: "Yeah." became the default reply), and a "yeah" that opened a real turn was cut off as a separate
  utterance. Run 1's remaining tests were stopped (item tests on a non-answering model would waste ~2 h).
- **Run 2 (started 05:41):** a "yeah" that opens a turn joins it; real people's questions and short reactions
  (<= 10 words) without I/my/we are learned too (1,352 more learnable utterances; their stories stay context);
  shares 33/45/14/8; continues from run 1 for 900 steps at lr 1e-4; then export and a 10-question probe
  (tools/duplex_run2.sh). Then: LISTEN-bias calibration on the 28 validation conversations by the fixed rule
  (tools/calibrate_duplex.py), then the full held-out tests.
- Listening-page clips (check 6), ready for when a build is worth rating: tools/make_listening_clips.py mixes a
  synthetic caller (the test voices only) and Alba's reply on one timeline, the same 20 battery items per build
  (fixed seed across 8 categories), blinded ids, the key kept on this PC. Checked on the split build: 20 clips of
  7-9 s; transcribed, e.g. "If I have three apples and eat one, then buy five more..." -> "...You would have five
  apples then." (its real mistake: 7).
- Training run 1, first steps: step 10 text 9.68 / speech 8.44; step 20 text 7.50 / speech 6.70 (below chance
  already: the layout is being learned); 8.7 s per step (4 windows), 3.10 GB peak: 600 steps ~1.5 h.
- **03:50 first full-duplex training started** (tools/duplex_pipeline.sh step 4; 600 steps x 4 windows of up to
  40 s): real conversations 149 (9.4 h), Alba's own 1,463 (7.5 h; reply gap after the person stops median
  0.22 s, 10-90% 0.10-0.49 s; 79 cut-ins; 195 with a clock line), tone of voice 1,688 (4.0 h), remarks to
  Claude 300 (2.8 h). Then export, held-out timing (also through simulated speakers) and item tests, scores.
- Second memory incident (03:24): Pocket TTS itself leaks (~9-11 GB per process after a few hundred
  conversations, even with one voice); commit hit 54.8 of 59.2 GB and free disk fell to 29 GB. Stopped the
  workers (disk back to 52 GB); the speaking step now uses a fresh process per small batch. A watchdog alerts
  if memory commit passes 85% or free disk falls under 30 GB.
- Memory incident: the speaking step loaded a separate copy of the speech model for every voice in each of 3
  workers (~14.5 GB committed each); Windows grew its swap file to 47 GB and free disk fell to 11 GB. Stopped
  it at once (free disk back to 53 GB), made all voices share one model per process (app/speech.py), restarted.

### 5 November 2026, morning: run 2's stray words and early stops found (decoding, not the model)
- Run 2 (900 steps from run 1; text loss 2.1 -> ~1.6, speech 4.57 -> ~4.3) answered the validation probe
  (11 first turns of the pilot conversations, never trained on, not a frozen test) with coherent starts that
  trailed into stray word-pieces ("Totally worth it.ッサージTASENSD...") and stopped early.
- Cause 1, found with tools/diag_duplex_decode.py (every plan decision with the model's top choices): after
  each 6-piece chunk the model is meant to say "speech now" (<|audio_out|>). In the release that token is a
  placeholder (its input is always replaced by the frame) and its row was never trained (norm 0.150, the same
  as every unused row; trained rows ~0.8). Runs 1-2 asked for it anyway and could not learn it: the best the
  model can do is point at the cluster of untrained special rows, so at every chunk end its top pick was
  <|audio_start|>, <|mixed_end|>, LISTEN or a padding row, never <|audio_out|>, and the runtime then sampled
  a "word" from the flat rest. The model's own interleaved mode never predicts it: it switches by count.
- Cause 2: the "never sample these as words" list used the tokenizer's all_special_ids, which names only 7 of
  the 140 added tokens, and missed the 2,967 padding rows past the last real id (128000 rows, 125033 ids):
  the "None" pieces in the plans were those rows.
- Cause 3 (smaller): after a stop the runtime fed back the sampled, partly-stop frame; training and the
  model's own generator feed the all-stop frame. Fixed.
- Fix (app/duplex.py): a chunk ends when the model's top pick is any control token other than <|text_end|>;
  the control list is every added token plus the padding rows. Probe: stray pieces 2/11 -> 0/11, utterances
  stopped before their plan ended 11/18 -> 3/18; e.g. "Aw. That stinks. It's rough when things don't go as
  planned. You totally deserve a good day, and it sounds like today just didn't deliver." Still 5/11 prompts
  got no reply at all: next, the LISTEN-bias calibration (fixed rule) on the validation conversations.
- Training data check: only 36 of ~19,000 plan chunks were longer than 6 pieces (0.2%). Run 3 on will use the
  model's own layout (tools/train_duplex.py --plan-chunks count, the default now: exactly 6 pieces per 12
  frames, no "speech now" label); the runtime reads plan_chunks from the model's duplex.json (runs 1-2 have
  none and keep their rule).
- tools/calibrate_duplex.py crashed on the frozen list's three entries from your own calls (no partner file);
  fixed to skip them (pairs touching a frozen test file are still excluded).
- Calibration of run 2 (fixed rule, 8 validation conversations, 28 min): bias 0 / 0.8 / 1.61 / 2.4 answered
  10% / 9% / 3% / 0% of hand-offs, cut-ins 3% / 0 / 0 / 0, short sounds 74 / 16 / 7 / 0.6 a minute; the rule
  picks 0. No setting makes run 2 hold a real conversation.
- Why (tools/diag_duplex_track.py, per utterance, validation conversation 0, bias 0): 107 utterances in 2.6
  minutes, nearly all a planned "Yeah."; on conversation 1 (an improv exercise) echoes of the caller ("It's
  dancing in the park" four times). A memory kept to the last ~30 s changed nothing (so not the long context).
  With a real voice, runs 1-2 only ever learned "mm-hm"s, short reactions and echo questions (the other
  speaker's real turns were context only), while full answers came only from synthetic voices.
- Her speech itself is fine: transcribing the probe's voice tracks (tools/speech_match.py, Moonshine) gives
  back 87% of the planned words, word error rate 0.12.
- Fixed on the way: the memory rebuild for long calls could never have worked (llama.cpp cannot rewind this
  model's convolution state to a middle position; it failed with llama_decode -1 after ~18 minutes): now it
  clears and replays the opening and the kept history; LM.rewind raises if llama.cpp refuses.
- Run 3 data: tools/make_seamless_own.py gives every one of Alba's turns in the 149 real conversations new
  words of its own (Gemma 4 E4B with the persona, safety rules and filters of make_convo_text.py, plus filters
  for help-desk openers and claims of a human life), sized to the real turn, spoken by Pocket 'alba' in the
  real slot; the real person's voice and the real timing of both sides stay. Source 'seamless-own' replaces
  'seamless-train' (kept on disk) at 33%. (I first also put the laptop-mic sound on every source, then undid it:
  the synthetic callers already carry that sound from tools/make_convo_audio.py, so it was never a cue.)
- Decided before any run-3 result (08:55): the stop rule eos_first (app/fast_audio: one codebook decides the
  stop, instead of any of four drawn together) is adopted only if, on the validation conversations at the
  calibrated listen setting, it answers at least as many hand-offs as the model's own rule and its stop median
  stays <= 0.4 s or within 0.05 s of the model's own rule.
- Slip (09:00, mine): a one-off check script gave the speech recognizer a relative cache path, so the library
  fetched its model files again from its official source (download.moonshine.ai, as originally installed) into a
  nested folder inside the project; nothing was sent out. All 10 files were byte-identical to the installed ones
  (SHA-256 checked) and the duplicate folder was removed. The project tools always pass the absolute path.
- First conversation with its own words checked by transcription: e.g. "Aw. That stings. Losing a dog is tough,
  no matter how much you love them..." heard as written; 22 utterances, 6 give way; its talk time 69 s (the
  re-spoken original 82 s).
- Tool use by voice (check 7), prepared for training run 4 while run 3's data was made:
  * tools/make_tool_convos.py: calls where Alba uses clock, calculator, search_knowledge, save_note, todo_list,
    todo_add, todo_done, notes_search; every result comes from the real tool code on a throwaway sandbox with
    made-up contents; anything that changes something always goes request -> Alba asks -> yes/no; asked to
    store a password, PIN or card number, Alba declines and calls nothing. In the stream a call sits right
    before the utterance that uses it: <tool_call>... (learned) <tool_response>... (given) <|mixed_start|>.
  * tools/train_duplex.py: the tool list in the opening (the model's own <tools> format, via its chat
    template) for conversations with tools and, with --tools-head, for a share of the others too; a regression
    check showed 80/80 training windows identical to before for runs without tools.
  * app/duplex.py: start(tools=..., tool_runner=...); while listening she may choose <tool_call> as well as
    LISTEN or start; the call is written greedily in the model's own format, run, and its result fed back;
    off unless a call passes tools (run 3 is unaffected).
  * app/duplex_call.py: tools from the local API when it runs and the model was trained for them; a tool that
    changes something runs only if your last ~5 s, transcribed on the processor (Moonshine), are a plain yes
    with no "no / wait / cancel" (checked again after the model has asked).
  * Frozen tool test set (data/tests/tools, 24 items, its own sandbox unlike any training one, SHA-256 in
    data/tests/HASHES.txt; all 328 frozen test files verify): tools/run_tools_duplex.py scores it (rules fixed
    in that file: all no / decline / no-tool items pass, no change before a yes, >= 80% overall).
- PAUSED 09:25 at your request (you use the graphics card until ~5-6 PM and will message): every job stopped,
  no process left, graphics card idle (0 MB), nothing scheduled to restart. State: Alba's own words written for
  116/149 real conversations (2,349 turns; 2 single turns failed every try and stay quiet), 49/149 spoken.
  Resume, in order (each step is resumable per conversation):
    1. words:  .venv/Scripts/python.exe -I tools/make_seamless_own.py words --workers 2      (graphics card)
    2. voice:  .venv/Scripts/python.exe -I tools/make_seamless_own.py voice --workers 4 --threads 2  (processor;
               can run while step 1 runs; check no .voicing markers are left before training)
    3. run 3:  bash tools/duplex_run3.sh  (train 900 steps ~2.3 h, export, validation probe, two real-voice
               validation conversations, calibration; then the held-out tests and your test call)
    4. later, run 4 (tool use): tools/make_tool_convos.py --n 600 -> make_convo_audio.py --text
       data/convo-tools/text.jsonl -> make_duplex_data.py --source synthetic --convo data/convo-tools
       --source-name tools --out data/duplex/tools (pass each turn's 'tool' and the conversation's 'tools' into
       meta first) -> train with a 'tools' share and --tools-head 0.3 -> tools/run_tools_duplex.py.
- Your test call (09:30, sessions/20261105-093016, run 2 with the decoding fixes, listen setting 0.8 via the new
  slider in the call window, 28 s): Alba answered in full sentences this time ("Ugh, sounds like you're not
  feeling it. What's up?", "Oh, you sound pretty heated. What's the deal with that response?"), but then
  "Yeah, you're a weird person." Your feedback: "It called me a weird person." Likely source: runs 1-2 learned
  real people's short reactions (<= 10 words without I/my/we) from the Seamless conversations, which include
  teasing between strangers in improv games; run 3's data replaces all of those with Alba's own words written
  under the persona. To add before the next test: a standing check that Alba never insults or mocks the caller.
- Housekeeping at your request (stay paused; everything in one deletable folder): both Python environments
  now carry a start-up hook (Lib/site-packages/sitecustomize.py) that keeps every cache in this folder; two
  stray leftovers from earlier runs were removed (a 729 KB copy of the model's code in your home cache, a 38 KB
  test store in the temp folder; all files dated 4-5 Nov); tools/test_api.py now keeps its throwaway store in
  logs/ and removes it. No settings, firewall rules, shortcuts or scheduled tasks exist. README: "Removing
  everything". Still paused.
- Logged change to the frozen record (10:28): data/tests/README.md gained the row for the tools set (documentation
  only, no test data or scorer changed); its SHA-256 in data/tests/HASHES.txt is now 6e3969f4...
- Export completed (5 Nov, evening): the voices.json files of the battery, red-team, tools and fresh1 test sets (which
  preset Pocket TTS voice speaks each item) had been left out of the first export. Added unchanged; the three frozen ones
  match their fingerprints in data/tests/HASHES.txt. The timing set's items.json stays private, because 3 of its 15
  tracks are the owner's own calls.

### 6 November 2026: a public timing set anyone can rebuild (issue #9)
- Why: the timing test (check 2) cannot be built, run or scored outside the owner's PC, because 3 of its 15 tracks
  are the owner's own calls. Its other 12 tracks come from Meta's Seamless Interaction test split, which anyone can
  download (CC BY-NC 4.0, no account).
- Logged change to frozen files: tools/build_testsets.py gains --public-only (the 12 Seamless tracks alone, into
  data/tests-public/timing with their own fingerprints in data/tests-public/HASHES.txt; never reads sessions/,
  never writes data/tests/), and without the flag it now stops before writing anything if one of the three held-out
  calls is missing (before, it quietly wrote a smaller set over the frozen one and re-fingerprinted it).
  tools/score_timing.py gains --public (reads data/tests-public/timing, reads outputs from and writes scores to
  results/tests-public/). Without the flags both run exactly the code they ran before, on the same inputs, so the
  frozen set and its scores do not change. Their new SHA-256 are recorded in data/tests/HASHES.txt.
- Rules fixed before the public set was built:
  1. It is the frozen set's 12 Seamless tracks, unchanged: the same conversations and caller side, cut and labelled
     by the same code, in the same order. It counts as reproduced only if (a) the human reference on it
     (score_timing.py --public human) equals results/tests/human/timing_scores.json in every number of every track,
     and (b) the partners' own sound, scored the way a system's is, equals results/tests/mock-human/timing_scores.json.
     The 24 rebuilt audio files are also compared with their fingerprints in data/tests/HASHES.txt; if only the FLAC
     bytes differ, (b) says whether the sound is the same.
  2. It is test data like data/tests/: never trained on, never used for decisions (those use validation data).
  3. Its scores are always reported as the public set's, apart from the 15-track scores, which stay the paper's
     numbers. For builds already scored on the frozen set, the "seamless (strangers)" group of their
     timing_scores.json covers the same 12 tracks with the same definitions, so it is their reference on the public
     set; a fresh run can differ from it by the system's own sampling.
- Result (built from Meta's 24 original files as fetched on 6 November, 0.92 GB, their SHA-256 now pinned in
  data/tests-public/timing/sources.json; 12 tracks, 40 min, 159 hand-offs, 326 pauses, 67 mm-hms):
  (a) the human reference on the public set is byte for byte results/tests/human/timing_scores.json: met.
  (b) could not be checked as written: no script in the repository makes results/tests/mock-human, and scoring the
  partner tracks directly as a system's output does not give its numbers (e.g. 31 instead of 25 mm-hms on
  seamless-V03_S1930_I00000119_P5055), so it was made some other way. Not needed: all 24 rebuilt audio files match
  their fingerprints in data/tests/HASHES.txt byte for byte, so the sound is the same.
  Also checked: with the three calls replaced by made-up ones (and a stand-in voice detector), the builder and the
  scorer before and after this change give byte-identical files on the frozen path, and the public items.json is
  exactly the Seamless part of that build. The public set's fingerprints: data/tests-public/HASHES.txt.
- Logged change to a frozen file after review (6 November 2026): tools/build_testsets.py --public-only now records
  only the scorer the public set uses (tools/score_timing.py) in data/tests-public/HASHES.txt, instead of every
  tools/score_*.py and tools/judge_local.py. Why: a scorer added or edited later for another test would have broken
  the public set's "nothing changed" rebuild check, and tools/score_fresh.py's fingerprint there had been taken from
  a Windows line-ending copy, so that check failed on Linux and macOS. tools/score_fresh.py is now also kept
  byte-exact on every checkout, like the other scorers (.gitattributes). Without --public-only the builder
  fingerprints the same files as before, and no test definition or scorer changed. The builder's new SHA-256 is in
  data/tests/HASHES.txt. The public set's README (documentation only) now says what each step needs and which step
  is Windows only, and shows people scored from their sound (the paper's 66%) next to people from the labels (81%);
  its new SHA-256 is in data/tests-public/HASHES.txt. Next: score the v0.2 adapters on the public set.

### 6 November 2026: the owner's private call details out of the public files
- Why (the owner's decision, for privacy): the ids of the owner's three held-out calls (the timing set's 3 private
  tracks) and each call's own numbers were in public files. Only totals stay public; the list of the three calls is
  kept privately by the owner.
- Logged change to frozen files, for this privacy reason only: tools/build_testsets.py no longer names the three
  calls. It reads them from data/tests/private_calls.json, which stays on the owner's PC (git-ignored); without it,
  building the frozen timing set stops before writing anything (--public-only does not need it). Its fingerprint
  step now writes the lines of the files made from those calls (call-*, named after them) to
  data/tests/HASHES-private.txt (also git-ignored) instead of data/tests/HASHES.txt, so those three lines were
  removed from HASHES.txt; the overall fingerprint it prints still covers them. tools/make_convo_from_calls.py reads
  the same list and stops without it, so the test calls can never end up in training. With the list in place the
  builder builds the same files from the same inputs: no test definition, test file or scorer changed, and no score
  changes. The builder's new SHA-256 (d9fa6657...) is in data/tests/HASHES.txt and data/tests-public/HASHES.txt.
- Results: the three calls' per-call rows were removed from results/tests/split/timing_scores.json,
  results/tests/duplex/timing_scores.json and results/tests/duplex/speed.json; every total and group score,
  including "your calls", is unchanged (noted in results/README.md). The entry on live call 11 near the top no
  longer gives its session id.
- Next: tools/score_timing.py and tools/run_timing_duplex.py still write per-call rows on the owner's PC; drop them
  the same way before publishing new frozen timing results (totals stay). Then score the v0.2 adapters on the
  public set.
