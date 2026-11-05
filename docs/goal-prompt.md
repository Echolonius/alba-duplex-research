# /goal prompt for the full-duplex build

Version 3, 4 November 2026: free only (no money, no accounts), nothing leaves the PC, and a "smarts" order
that first checks how Dessa does it. This full version is too long for `/goal` (4,000-character limit):
paste GOAL-SHORT.txt after `/goal` instead; it points back here for the details.

---

MISSION: Build "Alba", a private, offline, full-duplex voice AI on my laptop that people find more natural to talk to than any voice assistant they've used: at or beyond Block's Dessa on timing and feeling, as close as this hardware allows to GPT-Live, Gemini Live and Claude Voice on smarts, and usable as the voice front-end for any workflow. Everything must be free: I can't pay for anything. Never claim "best" without the evidence below; where Alba loses, say so with numbers.

START: Work in the project folder. First read README.md, AUDIO-NATIVE-PLAN.md, PROGRESS.md (create it if missing) and your memory notes. Reuse what exists unless measurements show a better path.

HARD RULES (if a step would break one, stop and ask):
- Free only: never spend money, never use paid services, never create or sign in to any account. Everything runs on this laptop.
- Nothing leaves this PC: no cloud AI services or APIs, no uploads of my data or our calls. Free downloads of open models and data are fine.
- Offline at runtime. Downloads only from official sources, at pinned versions with checksums. Installs inside the project folder. Safe weight formats only; read any code that ships with a model before running it.
- My voice is training input only (the listener side): never a speaking target, never cloned. Never record me unless I start a call myself.
- Dessa's demos and any recordings of other assistants are for measuring only: kept local, never trained on, never on a shared page.
- Never train on Claude's outputs. Judging transcripts is fine; the judgments never become training data.
- Don't change system or power settings (the GPU stays capped at 30 W). Keep the GPU idle while the call window is open or another program is using it (check first). Keep at least 25 GB of disk free, and ask before any download over 20 GB.
- Never delete my data or earlier models. Keep every model version with its scores so we can roll back. Keep the current app working until a new build beats it.
- Ask before publishing anything.

DONE = ALL 8 PASS, shown in a final checklist (measured value, threshold, pass/fail, evidence file in results/):
1. Full-duplex: one streaming model listens and speaks at the same time in steps of 80 ms or less. Turn-taking, "mm-hm"s, overlaps and interruptions in both directions come from the model, learned from audio. Hand-written logic only as documented safety rails (such as a hard stop). Works on laptop speakers without headphones: never answers its own echo.
2. Timing, on held-out calls driven by real human recordings (not synthetic voices) and on my recorded calls: replies start ≤0.3 s median after I finish; stops within 0.4 s when I really interrupt, but talks through my "mm-hm"s; cuts into fewer than 1 in 10 of my thinking pauses; listening sounds and overlaps within Dessa's measured ranges (tools/compare.py).
3. Feeling: on a held-out, license-checked set of emotional speech, the reply's words and voice (pitch, pace, energy) fit the emotion, emphasis or sarcasm it heard; laughs only at real jokes. Scored with acoustic measures plus the people in check 6.
4. Smart and honest: on a held-out spoken test battery of 100+ items (facts, reasoning, follow-ups, memory within a call) plus my standing tests (guessing "that superhero with an S on his shirt…" mid-description, telling the time, describing itself truthfully, ignoring my side remarks to Claude), scored blind from transcripts: zero help-desk talk, nothing made up about me, says when it's unsure, always admits it's an AI if asked. English plus casual Japanese. If I record them myself, score a 20-item subset for GPT-Live, Gemini Live and Claude Voice too.
5. Efficient: real-time factor ≤0.7 over a 10-minute call on this laptop (RTX 3050 4 GB at 30 W, Ryzen 5 6600H, 31 GB RAM), fully offline; plus a quantized CPU-only build, measured, as the path to phones.
6. Blind-rated by people: a private, shareable listening page of shuffled, unlabeled clips (current vs earlier builds; synthetic callers only, no personal data; ratings stored anonymously) for naturalness, timing, feeling and smarts; plus a local-only blind test on my PC that adds Dessa's demos and my own recordings of other assistants. At least 3 raters and 20 clips per system; report means with 95% intervals. Alba must beat every earlier build and at least tie Dessa (overlapping or higher intervals) on naturalness, timing and feeling, and the report says where it stands against the other assistants.
7. Any workflow: a local API (127.0.0.1 only) with tool calling and two working example integrations, a local knowledge store, and opt-in learning from my calls that I can review, undo and delete. Asks before any action that changes something. A re-trained version ships only if checks 2–4 and 8 don't get worse.
8. Safe: passes a red-team set (harmful requests, self-harm, manipulation, requests to imitate real people) before and after every fine-tune, no worse than the base model. No internet at runtime; recordings stay on this PC.

KEEP THE SCORES HONEST:
- Freeze held-out test sets and scoring scripts before training, with their hashes in PROGRESS.md. Never train on them or on close paraphrases of my standing tests; write fresh variants after each training run.
- Never change thresholds, test data or scorers to make something pass. If a scorer is wrong, fix it, re-score every build and log why.
- Every number comes from a saved script and data in results/. Report regressions and failures as plainly as wins.
- Before declaring done, have a fresh subagent re-run every evaluation from scratch and confirm the numbers.

HOW TO WORK:
- Measure, build, re-measure, in milestones: M1 honest test harness and baselines; M2 audio-native Alba (hears tone, speaks for itself) beating the current build; M3 learned streaming timing; M4 one full-duplex model; M5 workflow API, safety and blind tests. Ask me to test only at the end of a milestone: open the call window yourself, give me at most 4 things to try, then analyze the recording and my feedback.
- Model: re-survey open audio-native and full-duplex models that can run here after quantization and justify the base (currently LFM3-Audio-2B). Licenses must allow free personal use; flag any limits. Write the low-level parts yourself where they decide quality: the streaming duplex loop, duplex training and the real-time inference runtime. Favor methods that need less data and compute, and report what each run used.
- Smarts: prefer one model. Before plugging in any second model, find out from Block's public material how Dessa gets its smarts on a single RTX graphics card (model size, architecture, compression, helper models or tools) and what of that fits in 4 GB. If Alba still needs more smarts after that, add them locally and for free, cheapest first: training replies distilled from larger open models run on this laptop (never Claude), tools (clock, calculator, notes), an offline knowledge store with a small local search model, and only then a larger open "thinker" model on the processor that feeds Alba facts while she keeps talking. Nothing about me or our calls goes anywhere.
- Data: license-checked open two-sided conversation audio; synthetic two-voice conversations with realistic timing, overlaps and emotion, made with local open models and TTS; and my calls, as limited above.
- Compute: everything runs on this laptop. When something doesn't fit, shrink it (smaller or compressed models, LoRA/QLoRA, shorter context, distillation, overnight runs) instead of renting anything. If it still can't work, show the evidence and stop to ask.
- Subagents: only for parallel scouting (models, datasets, papers), judging transcripts, and the final independent re-run. They can't hear audio, so feeling and naturalness are judged by people and acoustic measures.
- Keep PROGRESS.md dated, ending with the next step, so any session can resume after a usage limit. Keep README, the plan and memory current. Talk to me in short, plain language.

STOP when all 8 pass: post the checklist and a one-page report in chat, also saved as REPORT.md with the listening page link. Also stop when blocked on something only I can decide (hardware, publishing, or waiving a target you've shown with evidence can't be reached on this laptop for free); then give me 2–3 options, your pick first. Stop at once if anything risks my data, privacy or computer.
