# Related work on GitHub

Open projects on full-duplex spoken dialogue and turn-taking, as found on 5 November 2026 (stars at that date).
They are larger, better funded or narrower than Alba; each is listed for what it can teach this project.

## Full-duplex models

| Project | Stars | What it is | Why it matters here |
| --- | --- | --- | --- |
| [kyutai-labs/moshi](https://github.com/kyutai-labs/moshi) | 11.4k | Speech-text foundation model and full-duplex dialogue framework (7.7B, Apache 2.0) | The reference design: parallel user and model audio streams, with text that runs ahead of speech, like Alba's plan words |
| [NVIDIA/personaplex](https://github.com/NVIDIA/personaplex) | 12.1k | Full-duplex speech-to-speech model (7B, MIT code) | The strongest open turn-taking reported on Full-Duplex-Bench; too large for 4 GB |
| [VITA-MLLM/Freeze-Omni](https://github.com/VITA-MLLM/Freeze-Omni) | 402 | Low-latency speech-to-speech built around a frozen language model | Another way to keep an LLM's knowledge while adding speech |
| [OpenBMB/MiniCPM-V](https://github.com/OpenBMB/MiniCPM-V) | 27.1k | Home of MiniCPM-o, an omni model with live full-duplex streaming (9B) | Full duplex in a general multimodal model |
| [nu-dialogue/j-moshi](https://github.com/nu-dialogue/j-moshi), [llm-jp/llm-jp-moshi](https://github.com/llm-jp/llm-jp-moshi) | 327, 25 | Japanese full-duplex models built on Moshi | Alba aims at casual Japanese too |
| [walln/parley](https://github.com/walln/parley) | 1 | A small full-duplex speech model for experimentation | Closest in spirit: small and hackable |

## Pieces of the problem

| Project | Stars | What it is | Why it matters here |
| --- | --- | --- | --- |
| [ErikEkstedt/VoiceActivityProjection](https://github.com/ErikEkstedt/VoiceActivityProjection) | 127 | Voice Activity Projection: self-supervised turn-taking prediction (MIT) | The turn model Alba's cascade used, retrained here |
| [Soul-AILab/SoulX-Duplug](https://github.com/Soul-AILab/SoulX-Duplug) | 332 | Streaming semantic voice-activity detection for full-duplex systems (Apache 2.0) | A candidate helper for labelling turn ends |
| [ASLP-lab/Easy-Turn](https://github.com/ASLP-lab/Easy-Turn) | 153 | Turn-taking detection model and dataset (Apache 2.0) | Labels and data for turn ends |
| [FreedomIntelligence/LoopSpeech](https://github.com/FreedomIntelligence/LoopSpeech) | 8 | "What Did I Just Say? Self-Listening for Full-Duplex Speech Models" | Alba feeds back what it just said every step; this studies exactly that |
| [lazybutai/personaplex-native-tool-use](https://github.com/lazybutai/personaplex-native-tool-use) | 0 | Native tool calling for PersonaPlex | Tool calls inside a duplex stream, as Alba's run 4 plans |
| [FireRedTeam/FireRedChat](https://github.com/FireRedTeam/FireRedChat) | 600 | A fully self-hosted full-duplex voice system (Apache 2.0) | A complete self-hosted system to compare against |

## Benchmarks, surveys and lists

| Project | Stars | What it is |
| --- | --- | --- |
| [DanielLin94144/Full-Duplex-Bench](https://github.com/DanielLin94144/Full-Duplex-Bench) | 339 | Benchmark for turn-taking and overlap handling in full-duplex dialogue |
| [MM-Speech/DuplexSurvey](https://github.com/MM-Speech/DuplexSurvey) | 111 | "Speaking While Listening": a survey and empirical audit of full-duplex models (EMNLP 2026) |
| [Ruiqi-Yan/Awesome-Full-Duplex-SDM](https://github.com/Ruiqi-Yan/Awesome-Full-Duplex-SDM) | 291 | A curated list of full-duplex spoken dialogue models and benchmarks |

Running Alba's runtime against Full-Duplex-Bench, next to the project's own timing harness, would make its numbers
comparable with these models.
