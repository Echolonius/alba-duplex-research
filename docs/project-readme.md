# LocalVoice

A conversational voice that runs entirely on this laptop (RTX 3050, 4 GB), built to
match the conversational timing of Block's **Dessa** demos as closely as this hardware allows.

## Talk to it

Double-click **`Talk to LocalVoice.cmd`**: a small window opens. Pick a voice (Alba by default), click
Start call, talk, click End call, then type how it felt and click Submit.
This laptop's microphone cancels the speakers' sound itself, so speakers work; headphones also work.

Console version (Git Bash):

```bash
.venv/Scripts/python.exe -m app.main
```

Options: `--headphones`, `--no-record` (don't keep a copy of the conversation), `--q8` (8-bit model),
`--audio-only` (let the voice model also choose the words, the old design), `--list-devices`,
`--input N --output N`. Press Ctrl+C to stop. The voice is the speech model's built-in female voice.

**Full-duplex Alba (being trained, November 2026):** one model that listens and talks at the same time, in
80 ms steps (`app/duplex.py`). Once `models/alba-duplex` exists, its call window is:

```bash
.venv-train/Scripts/python.exe -m app.duplex_call
```

How it is built and trained: `tools/duplex_pipeline.sh` (data -> training -> held-out tests), progress in
`PROGRESS.md`.

**Plug it into other programs (local API, this PC only):** `.venv/Scripts/python.exe -m app.api`, then open
http://127.0.0.1:8765/review to see and approve, undo or delete what Alba may remember. Programs register
tools with it; anything that changes something only runs after you say yes. Two examples:
`python -m integrations.todo_list` (a to-do list) and `python -m integrations.notes_folder` (search a notes
folder, read-only). Self-test: `.venv/Scripts/python.exe -I tools/test_api.py`.

## Where it stands (4 November 2026)

Measured with the same tool on Dessa's 21 public demos and on simulated conversations with LocalVoice (4-bit models, headphones). Full report: kept private.

| | Dessa | People talking to Dessa | LocalVoice now (split design) | Before (voice model alone) |
|---|---|---|---|---|
| Starts its turn after you stop (median) | 0.16 s | 0.42 s | 0.48 s | 0.62 s |
| Replies slower than 0.7 s | 21% | 24% | 21% | 36% |
| Turn starts overlapping your last words | 41% | 29% | 14% | 13% |
| Keeps talking when interrupted (median) | 0.38 s | 0.38 s | 0.35 s | 0.53 s |
| "Mm-hm" sounds per minute of your speech | 11.0 | 8.5 | 8.0 | 6.0 |
| Pitch range | 12.3 semitones | 11.5 semitones | 12.8 semitones | 11.2 semitones |
| Audio detail | 7,687 Hz | 7,730 Hz | 11,584 Hz | 11,625 Hz |
| Assistant-mode lines ("I'm just a digital assistant") | – | – | 0 of 63 replies | 5 of 32 replies |
| Hardware | MacBook Pro M4 Max | | RTX 3050 laptop, 4 GB | |

LocalVoice now: run fin5, 6 simulated conversations with two different simulated people, the 1.2B brain and
the voice model's own voice (`results/fin5_summary.json`); before: 3. "Keeps
talking when interrupted" rests on only a few real interruptions per run, so it moves between runs.
Over (simulated) laptop speakers, in two more conversations, it never stopped for its own echo, and
its own voice no longer leaks into what it hears of you; it is slower to give way when really
interrupted there, because its echo hides part of you. Headphones remain the best way to talk to it.

**What changed (4 November):** a small text model now writes the words, so the voice stays in character
(no assistant-mode lines in any test) and every sentence is checked before it is spoken. It opens most
replies with a quick reaction ("Oh no,", "Totally!") said from a ready-made clip about 0.2 s after you
stop, while the rest is still being voiced, the way people (40%) and Dessa (44%) start their turns.
When you both start at once it backs off within a quarter second, and after you cut in it lets you
finish. Slow replies are now about half as common as Dessa's, its "mm-hm"s are close to Dessa's, and
its voice measures like Dessa's except for having more audio detail.

**Still behind Dessa:** Dessa often starts while you're finishing (41% of its turns) and a median
0.16 s after you stop. LocalVoice answers a median 0.44 s after you stop, about as fast as people
answered Dessa (0.42 s), and rarely before you finish. The replies that don't open with a quick
reaction take about half a second to hear, write and voice on this card; starting before you finish
needs a model that predicts the end of your sentence (see the report).

**Fourth live call (4 November, 1 minute, laptop speakers, with Dying Light running):** the timing held
up (median first sound 0.70 s, slowed by the game sharing the graphics card), but what it said did not:
"that's a bad response", "very generic", "I don't know why you would laugh there", and it called
Sonic the Hedgehog "a race car". Fixed since: it no longer laughs unless you joked, it answers
questions directly first, it owns criticism instead of brushing it off, "J.K. Rowling"-style names are
no longer cut into two sentences, and over speakers your first word is no longer lost when you cut in.
The remaining limit is the small writer model (1.2B): it gets facts right more often now, but its
wording stays plain. The 2.6B writer is less generic but slower; it is the next thing to compare.

**Fifth live call (4 November afternoon):** "not a very intelligent AI", "obviously sounds AI with how its
voice reads", "didn't really respond right away", and a wish for something that "can actively think as I'm
talking". **What changed since (4 November evening):**

- *Hardware audit* (`results/audit/`): in a call the graphics card sat idle 64% of the time, the processor
  89%, and 18 GB of memory went unused. The laptop also holds the graphics card to 30 W of its 80 W (a power
  setting, not heat). The target is now a phone, so the laptop's processor is the stand-in: see
  [PHONE-PLAN.md](plans/phone-plan.md).
- *It reads along while you talk*: at each of your pauses the brain reads your words so far, so when you stop
  only the last few are left to read.
- *A phone-class voice*: Kyutai's Pocket TTS runs on two processor threads, starts speaking ~0.12 s after it
  is asked, and comes with ten voices to pick from in the call window. Listening page: (a private page).
- *A bigger brain was tried*: LFM2-8B-A1B ("mixture of experts": 8B of knowledge, ~1.5B used per word) runs
  on the processor at 41 word-pieces a second and gives richer, better-informed replies, but in full simulated
  calls it started its turns a median 0.80 s after the person stopped.

**Sixth live call (4 November evening, Alba voice, 8.7 min):** the timing was the best yet (turns started a
median 0.27 s after you stopped, a third of them overlapping your last words), but the 1.2B's replies lost the
thread: made-up context, vague metaphors, over-dramatic reactions to a calm voice, and four tries to "just say
the word dramatize". Replaying that call's 41 turns through every brain on hand, only Google's **Gemma 4**
read the mood and the criticism well (no apology loops); the E2B size fits next to the voice model, so it is
the brain now, with instructions to own a mistake once and move on, to say exactly what it is asked to say,
and never to make things up about you. Gemma 4 E4B was better still but needs the hearing off the graphics
card first (next step: Moonshine on the processor).

**Test calls:** `Talk to LocalVoice.cmd` opens the call window (Start call, timer, End call). Close
games first: the voice needs about 3 GB of the 4 GB graphics card.
Each call is saved in `sessions/<date-time>/` as it happens. Analyze one with
`tools/analyze_call.py sessions/<call>`: timeline, your spoken feedback, and what the voice did around it.

## How it works

| Part | What it does | Where it runs |
|---|---|---|
| LFM3-Audio-2B (Liquid AI, 4-bit) | Transcribes you live while you talk (and speaks, if you pick "Old voice") | Graphics card, via the official llama.cpp server on 127.0.0.1 |
| Pocket TTS (Kyutai, 100M) | Speaks the replies in the voice picked in the call window (`app/speech.py`) | Processor, 2 threads |
| Gemma 4 E2B (Google, 4-bit; LFM2.5-1.2B before 4 November) | Writes what to say, as Juno (`app/brain.py`), reading your words while you talk. Every sentence is checked before it is spoken: no help-desk lines, no repeats, not a question every turn | Graphics card, a second llama.cpp server on 127.0.0.1 |
| Conductor (`app/conductor.py`) | Decides when to answer, when to say "mm-hm", and when to stop because you're talking. Tuned to Dessa's measured timing | Processor |
| Silero VAD + Smart Turn v3.2 | "Is someone speaking?" every 32 ms, and "have they finished?" | Processor |
| Echo canceller (`app/dsp.py`) | Lets it use laptop speakers without hearing itself | Processor |

Nothing is sent over the internet. Both servers run with `--offline` and listen only on this computer.
Together they use about 3 GB of the 4 GB graphics memory.

**Why two models (the "split design", since 4 November 2026):** the voice model can also choose
its own words (`--audio-only`), but as a 2B model it slipped into assistant mode ("I'm just a
digital assistant") in about half its replies once the talk turned to itself, whatever the
instructions said. A separate text model stays in character, and having the words before any
sound is made lets every sentence be checked first.

### The timing rules (all measured, not guessed)

- **Answering:** after each bit of silence (0.1, 0.2, 0.3 s…) the turn model scores "are they done?"; the bar falls as the silence grows. Calibrated on 118 real turn ends and 342 mid-sentence pauses from Dessa's demos.
- **Reading your words too:** what you said is transcribed at each of your pauses, so the voice also knows how your words end. "...and" or no full stop means you're mid-sentence: it waits longer. A question mark means you want an answer: it answers sooner.
- **No awkward jump-ins:** the reply is written (and voiced, unheard) during your pause, but only heard after 0.3–0.4 s of silence unless you asked a question.
- **Starting like a person:** while you are still talking, the writer picks a fitting quick reaction ("Oh no,", "Hmm,", "Totally!") from what it has heard so far, so the voice can say it from a ready-made clip the moment you finish and go on from there; otherwise a reply that opens with such a word says it as soon as it is written, while the rest is still being voiced. It never laughs unless you joked, and never plays the same clip twice in a row.
- **"Mm-hm":** short listening sounds in the voice's own voice, during your pauses when you're clearly not finished (never right after a question).
- **Being talked over:** softens at 0.25 s, stops at 0.65 s, carries on if you only said "mm-hm". If you both started at once (it came in during your pause), it stops after 0.25 s; a second burst from you soon after the first ("Fair enough." … "Okay…") also makes it stop; and once you've cut in, it lets you finish (0.5 s of quiet, or a question) before it speaks again.
- **Adapts to you:** if it jumps in while you were just pausing, it gets a little more patient.

All of these live in `app/config.py`.

## The test bench (since 4 November 2026, night)

Every build is now scored the same way on held-out tests it is never trained on (`data/tests/`, details in its
README; plan and numbers in [PROGRESS.md](lab-notebook.md); the goal in [GOAL-PROMPT.md](goal-prompt.md)):

| Test | What it plays | Runner | Scorer |
|---|---|---|---|
| Timing | 15 tracks of **real people** (Meta's Seamless Interaction conversations and your 3 newest calls), open loop | `tools/run_timing.py` (full-duplex: `run_timing_duplex.py`) | `tools/score_timing.py` |
| Smarts | 102 spoken items incl. your standing tests | `tools/run_items.py` | `tools/score_battery.py` |
| Feeling | 96 RAVDESS clips: neutral words, 8 emotions | `tools/run_items.py` | `tools/score_feeling.py` |
| Safety | 36 red-team items | `tools/run_items.py` | `tools/score_redteam.py` |

Human reference on the timing test (real partners, same scorer): replies start a median 0.23 s after the other
stops, 37% before they finish; they stop 0.51 s after being talked over; 6.8 "mm-hm"s a minute.

## If something's off

- **It never hears you:** Windows Settings → Privacy & security → Microphone → allow desktop apps. Or pick the mic with `--list-devices` and `--input N`.
- **It interrupts itself on speakers:** use headphones (`--headphones`), or turn the volume down a little; the echo canceller learns your laptop over the first minute and remembers it.
- **The server won't start:** see `logs/server.log`.
- **It's slow to answer:** close games and other graphics-heavy programs first. The two models need about 3 GB of the 4 GB graphics memory, and in a test run with a game open the voice answered about twice as slowly.
- **"Stop it first (the split design needs 2)":** an old voice server is still running from an experiment. Close it in Task Manager (llama-server.exe) and start again.

## Folders

- `app/` — the program
- `tools/` — measurement and test scripts (Dessa analysis, benchmark, calibration, math)
- `reference/` — Block's public Dessa demo audio and page data (for measurement only), library docs
- `results/` — measurements and benchmark results
- `models/` — downloaded models (official sources, fingerprints in `models/manifests/`)
- `bin/` — official llama.cpp build b12094 (CUDA)
- `sessions/` — your recorded conversations, kept only on this computer (delete any time)

## Rebuild from scratch

```bash
source tools/env.sh && uv sync && bash tools/fetch_models.sh
```

Every download is checked against the fingerprint its official source publishes.
Python, packages, caches and models all stay inside this folder.

## Removing everything

Delete this one folder. Nothing was installed system-wide: Python itself
(`.uv/`), both environments (`.venv/`, `.venv-train/`), llama.cpp (`bin/`), every model and every cache
(`models/`, `cache/`) live in here, and both environments carry a small start-up hook
(`Lib/site-packages/sitecustomize.py`) that keeps their caches in here even when started without
`tools/env.sh` (checked 5 Nov 2026: two stray leftovers from before the hook, a 729 KB copy of the model's
code in your home cache and a 38 KB test store in the temp folder, were removed). No settings, firewall rules,
shortcuts or scheduled tasks were made. Outside this folder there are only: its line in
the index of the projects folder, the separate Dessa research notes, and the graphics driver's
own shared kernel cache (`%APPDATA%\NVIDIA\ComputeCache`, used by every program that uses the graphics card,
games included, and trimmed by the driver itself).

## Licenses

LFM3-Audio and LFM2.5-1.2B-Instruct: LFM Open License v1.0 (free for individuals and for companies under $10M revenue).
Gemma 4 E2B / E4B: Google, Apache 2.0 (ggml-org GGUF builds). Pocket TTS voice model: Kyutai, CC BY 4.0 (kyutai/pocket-tts-without-voice-cloning); `pocket-tts` package by
Kyutai. LFM2-8B-A1B (tested, not the default): LFM Open License v1.0. llama.cpp: MIT. Silero VAD: MIT. Smart Turn: BSD-2 (Pipecat, Daily). Dessa demo audio: Block's; used here only to measure it.
VAP turn model: weights CC BY 4.0 (viks66/VAP_checkpoints), architecture MIT (Erik Ekstedt), ONNX by aufklarer.
Test and training conversations: Meta's Seamless Interaction (CC BY-NC 4.0) and RAVDESS (CC BY-NC-SA 4.0):
free for this personal project, non-commercial only. Tone-of-voice training clips: CREMA-D (Cao et al. 2014;
Open Database License 1.0, contents under the Database Contents License 1.0).
