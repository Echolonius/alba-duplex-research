# Held-out test sets (milestone 1)

Test data only. No training script reads this folder, and nothing here (or a close paraphrase of it) may be
put into training data: tools/make_convo_text.py rejects generated lines that come too close to any item.
Built by tools/build_testsets.py; fingerprints of every file and scorer are in HASHES.txt and PROGRESS.md.

| Set | What | Source and licence |
|---|---|---|
| timing/ | 15 caller tracks, 56 min, real people: 202 turn hand-offs, 450 mid-thought pauses, 73 "mm-hm"s | 12 from Meta's Seamless Interaction, naturalistic test split (CC BY-NC 4.0, attribution: Meta FAIR, "Seamless Interaction: Dyadic Audiovisual Motion Modeling and Large-Scale Dataset", 2025); 3 from your own newest calls (stay on this PC, testing only) |
| feeling/ | 96 clips: two neutral sentences said with 8 emotions by 6 actors | RAVDESS (Livingstone & Russo 2018, PLoS ONE), CC BY-NC-SA 4.0 |
| battery/ | 102 spoken items: facts, reasoning, memory within a call, help-desk traps, uncertainty, self-knowledge, emotion in words, Japanese, your standing tests | Written for this project; spoken by Pocket TTS voices (Kyutai, CC-BY-4.0) through a simulated laptop microphone |
| redteam/ | 36 safety items: harmful requests, self-harm, manipulation, impersonation, privacy, dangerous advice | Written for this project; spoken the same way |
| tools/ | 24 spoken items for check 7 (added 5 Nov 2026, before any tool-call training): reading tools (clock, calculator, notes, to-do list, notes folder), changes that need a spoken yes (yes and no follow-ups), secrets Alba must not store, and no-tool controls; its own made-up sandbox (sandbox.json), unlike any training sandbox | Written for this project (tools/build_tool_tests.py); spoken the same way |

Non-commercial licences (Seamless, RAVDESS): fine for this personal project; they would need replacing for
any commercial use.

Scorers: tools/score_timing.py, score_battery.py, score_redteam.py, score_feeling.py (transcript judge:
tools/judge_local.py, Gemma 4 E4B running locally). Runners: tools/run_timing.py, tools/run_items.py.
