# Audio-native, full-duplex plan

Updated 4 November 2026 (night), under the goal in GOAL-PROMPT.md. Progress and numbers: PROGRESS.md.
(The first version of this plan, set after the ninth and tenth live calls, aimed at a half-duplex audio model;
after call 11 the target is full-duplex.)

## Where the bar is

Dessa (Block) was trained on audio alone on a 72-GPU rack and runs on a MacBook Pro M4 Max; no size, paper or
weights. Our held-out timing set shows its timing is ordinary human timing: real strangers in Meta's Seamless
Interaction conversations start replying a median 0.13-0.23 s after the other stops, 37-41% of the time before
they finish. This laptop has ~1/12 of the M4 Max's memory bandwidth and a 4 GB graphics card held at 30 W, so
Dessa's size cannot be matched here; the plan matches its *behaviour* with a smaller model and learned timing.

## Architecture

One model, one stream, in 80 ms steps: **Alba Duplex**, built on LFM3-Audio-2B (Liquid AI, LFM Open License
v1.0: free for personal use; 18 languages incl. Japanese; official LoRA recipes; llama.cpp support).

- Every 80 ms the model reads one token of your voice (its streaming speech encoder, 12.5 Hz) and writes one
  step of its own: silence while listening, a "mm-hm", or a speech frame (12 codec codes, decoded to 24 kHz),
  with a few text tokens of what it is about to say ahead of the speech (its plan, as in its interleaved mode).
- So it hears you while it speaks: a real interruption shows up in its input within 80 ms and it can stop;
  your "mm-hm"s show up too and it can keep going. Turn-taking is learned, not written as rules.
- Because it hears the sound of your voice, not a transcript, tone and emphasis reach the model directly.

## Training data (no voice cloning, nothing uploaded)

1. **Real two-person timing**: Seamless Interaction *train* conversations (CC BY-NC 4.0). Side A stays the real
   person (the listening side). Side B is re-spoken in Alba's voice (Pocket TTS 'alba') from its transcript,
   at the original times: real gaps, overlaps, "mm-hm"s and give-ways, in Alba's voice, never a real person's.
2. **Alba's words**: short conversations written by a larger open model run here (Gemma 4 E4B; never Claude),
   both sides spoken by TTS, laid out with timing drawn from the real conversations.
3. **Your calls**: your side only, as listening input (never a speaking target, never uploaded).
Test sets (data/tests) are never used, and generated lines too close to them are rejected.

## Steps (milestones in PROGRESS.md)

- M1 test bench and baselines (today's build, the raw audio model).
- M2 Alba, half-duplex: LoRA on LFM3-Audio for her words and voice (interleaved replies), served by llama.cpp
  in the app; beats today's build on the battery, feeling and blind clips.
- M3 learned streaming timing: the two-voice VAP model rebuilt in PyTorch (exact; 11-13 ms per 20 ms step on
  the processor), fine-tuned on real conversations and your calls; drives the app's timing meanwhile and
  becomes the teacher signal for M4.
- M4 Alba Duplex: the time-stepped format above, trained with QLoRA on this card; our own real-time runtime
  (language model through llama.cpp's library with fed-in embeddings; speech encoder, frame decoder and codec
  as small separate models), real-time factor <= 0.7.
- M5 workflow API, tools, knowledge store, opt-in learning, red-team, blind tests, final re-run.

## Smarts (in the goal's order)

One model first. If the battery shows it needs more: replies distilled from larger open models run here, then
tools (clock, calculator, notes), then an offline knowledge store, and only then a larger open "thinker" on the
processor feeding facts while Alba keeps talking. Nothing about you leaves the PC.
