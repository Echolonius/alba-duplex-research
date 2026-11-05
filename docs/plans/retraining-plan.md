# Retraining plan: teach the voice to talk like a person

> **Update, 4 November 2026: no longer needed for the main problem.** The app now uses a
> split design (`app/brain.py`): the voice model still hears you and speaks, but a small text
> model (LFM2.5-1.2B-Instruct) writes the words, and every sentence is checked before it is
> spoken. In the benchmark conversations that took assistant-mode lines from 5 in 32 replies
> to 0, with no retraining. What follows is kept for reference. If the wording ever needs to
> be better than the small text model manages, the cheaper route is now a text-only LoRA of
> that text model (Liquid's tools support text fine-tuning officially, and the model is built
> into the standard libraries, so no downloaded model code would run). A 1.2B text LoRA is
> small enough to train on this laptop's 4 GB card, so it needs no cloud account.

## Why

Three live test calls (3 November 2026) and the tests in `tools/probe_loops.py` show
the limit of the 2-billion-parameter model (LFM3-Audio-2B), not of the timing code:

- When the talk turns to the voice itself ("your voice sounds weird"), about half its
  replies fall into assistant mode ("I'm just a digital assistant", "I don't have
  feelings"). No instruction wording changed that (4.5–5 of 10 replies for every wording tried).
- It falls into stock lines ("What would you like to talk about?"). The app now catches
  many repeats before they are heard, but cannot stop the model from reaching for them.
- The server's built-in anti-repetition settings did not help and made the speech less
  clear (word error 4.2% → 6.3%), because they also affect the model's sound output.

Timing is no longer the main problem: in the live calls the voice started its turn a
median 0.41–0.46 s after the person stopped, close to people talking to Dessa (0.42 s).

## What would fix it

A small adjustment of the model (a "LoRA" fine-tune, which trains about 1% of the
weights) on examples of the replies we want: short, casual, opinionated, never
assistant-speak, never the same line twice. The voice stays the same because the
example replies are spoken by the model's own voice (its speech generator), so only
*what* it says changes, not how it sounds.

1. **Write ~10,000 example exchanges.** A larger open model running on the rented
   machine writes casual replies in the Juno style to varied things people say,
   including the awkward moments from the live calls (complaints, "are you an AI?",
   "why do you keep saying that?"). No personal data is used.
2. **Turn them into speech** with LFM3-Audio's own speech generator (same voice).
3. **Fine-tune** on one rented GPU. Liquid AI has no official audio fine-tuning recipe yet:
   its open-source LEAP Finetune stack covers LoRA for text and vision models, not audio.
   So the training loop would be built on Liquid's official model code from Hugging Face,
   which needs that repository's own code to run. It would run only on the rented
   machine, never on this PC. Building and testing that recipe is the main work (about a
   day of mine), not the GPU time.
4. **Export** the result to the same GGUF format and bring it back to this laptop. It
   runs exactly like today's model (same size, same speed).
5. **Check** with the same tests: `probe_loops.py` (assistant mode, repeats), the
   benchmark, and a live call.

## What it costs

| | Estimate |
|---|---|
| Rented GPU (one A100/H100 class) | 3–6 hours at about $2–3/hour: **$6–18** |
| Data | Written on the same machine by an open model; no paid services |
| Your part | Create the temporary cloud account and pay (I can't create accounts or enter payment details); I do the rest from a script |
| Risk | No official recipe yet, so the first run may need a second try (another $6–18) |

Sources: [Liquid AI fine-tuning docs](https://docs.liquid.ai/lfm/fine-tuning/overview),
[LFM3-Audio-2B model card](https://huggingface.co/LiquidAI/LFM3-Audio-2B).
| What leaves this PC | Nothing personal: only the training scripts. Your call recordings stay here |

## Not covered by this

- **Overlapping, Dessa-style timing** (starting before you finish, 41% of Dessa's turns):
  that needs a model that predicts the end of your sentence from sound. A separate small
  training run (also a few dollars) on open conversation recordings.
- **A bigger brain:** a 12–16 GB graphics card would run 7B full-duplex models
  (Moshi, NVIDIA PersonaPlex) that listen and talk at the same time, like Dessa.
