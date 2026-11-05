# Phone plan

Goal: the same conversation quality on a phone, offline. Set 4 November 2026, after the hardware audit.

## Why a phone is possible

Writing each word means reading the whole active part of the model from memory once, so speed is set by
memory speed, not raw computing power. A recent flagship phone reads memory at roughly 60–80 GB/s. This
laptop's processor and memory measured about 44 GB/s effective, so **the laptop's processor is the
phone stand-in**: if a part runs well there on 2–4 threads, it is phone-class. The laptop's graphics card
(176–192 GB/s, but held to 30 W by the laptop's power mode) is not used for phone-class tests.

## Budget

| | Flagship phone (12–16 GB) | Any recent phone (6–8 GB) |
|---|---|---|
| All models together | ≤ 6 GB | ≤ 1.5 GB |
| Brain speed needed | ≥ 30 word-pieces/s | ≥ 30 word-pieces/s |
| First sound after you stop | ≤ 0.3 s | ≤ 0.3 s |

## Parts

| Part | Today (laptop) | Phone-class choice | Measured on this laptop's processor |
|---|---|---|---|
| Hearing | LFM3-Audio 2B on the graphics card (51% busy while you talk) | Moonshine Voice streaming (MIT, tiny/small/medium) | not yet |
| Brain | LFM2.5-1.2B on the graphics card | LFM2-8B-A1B "experts" model, 4-bit (5.0 GB, ~1.5B active) on flagships; LFM2.5-1.2B (0.7 GB) elsewhere | 8B: 41 pieces/s writing, 124–147 reading (4–6 threads). 1.2B: 62 writing, 258 reading (4 threads) |
| Speaking | LFM3-Audio 2B `/tts` | Kyutai Pocket TTS (100M, CC-BY-4.0, ports for Android via sherpa-onnx and LiteRT) | first sound 0.12 s, 3.2x real time on 2 threads |
| Turn-taking | Silero VAD + Smart Turn v3.2 | same (already tiny, processor) | |

## Software tricks, used and not yet used

Used: 4-bit weights; a hybrid model (LFM2: short convolutions instead of most attention layers); prompt caching;
experts kept in normal memory; **reading along while you talk** (`brain._read_along`): the brain reads your
words at every pause, so when you stop only the last few are left (0.06 s instead of 0.42 s for the 8B).

Not yet: hearing and speaking on the processor; speculative decoding (a tiny model guesses several words, the
brain checks them in one pass); training the brain on good casual conversation so a small model talks like a
bigger one; the phone's own AI chip (NPU).

## What the first tests showed (4 November)

- **8B brain on the processor in full simulated calls** (results/bench fin6, 6 conversations): smarter
  replies, but it starts its turn a median **0.80 s** after you stop (1.2B on the graphics card: 0.48 s;
  people: 0.42 s), and 59% of its turns take over 0.7 s. Even a one-word reply ("Exactly.") took 0.55 s
  from the decision to answer: the processor reads new text at only ~130 pieces a second, and the reading,
  the quick-reaction picks and the reply all queue for it. Phones read new text on their AI chip, which is
  much faster at that part, so this test is harsher than a phone; on this laptop the 8B needs more of its
  work moved into the time you are talking before it can be the default.
- In memory the 8B server took ~9 GB here, not 5. Not yet checked why; the likely cause is a second,
  rearranged copy of the weights that the processor version makes for speed. A phone build must fit in 5.

## Steps

1. Pick the voice by ear (on a private comparison page), then make Pocket TTS the speaking part.
2. Make Moonshine Voice the hearing part. After that nothing needs the graphics card.
3. Switch the brain to LFM2-8B-A1B on the processor if it wins the conversation tests, with shorter replies.
4. Live call on the laptop with everything on the processor; then a phone build.
