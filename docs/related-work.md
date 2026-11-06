# Related work on GitHub

Open projects on full-duplex spoken dialogue and turn-taking, as found on 5 November 2026, with 2026 additions found
on 6 November (stars on the day each was added).
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
| [MaikeZuefle/f-actor](https://github.com/MaikeZuefle/f-actor) | 33 | F-Actor: a 1B full-duplex model on Llama 3.2, trained on 2,000 hours with the audio encoder frozen; told by instruction how often to backchannel or interrupt | At 1B it should fit a 4 GB card, and its recipe (hearing frozen, only the language model trained) is close to Alba's |
| [penguinfish1688/D2-dialogue-to-duplex](https://github.com/penguinfish1688/D2-dialogue-to-duplex) | 14 | D2: turns half-duplex speech models (Qwen3-Omni, LLaMA-Omni 2) into full-duplex ones by distilling a streaming audio encoder, then fine-tuning on dialogue | Alba's job on bigger models; it reports 99% turn-taking success at the same 12.5 Hz (80 ms) step Alba uses, though the same model waits through only about 21% of a caller's pauses |
| [avera-labs/AwakenFullDuplex](https://github.com/avera-labs/AwakenFullDuplex) | 0 | Awaken: Fun-Audio-Chat-8B adapted from half to full duplex with little dialogue data (Apache 2.0) | Two fixes for problems Alba has too: blending fine-tuned and base weights 50/50 to keep the base model's skills, and a short GRPO stage that raised good backchannels from 1% to 63% |
| [NVIDIA-NeMo/Speech](https://github.com/NVIDIA-NeMo/Speech/tree/nemotron-labs-voicechat) | 18.7k | Home of NemotronLabs VoiceChat, an 11B full-duplex model with tool calling (OpenMDW license) | Tool calls go out on a separate channel while a set "on hold" line is spoken: a pattern for Alba's run 4 |
| [krafton-ai/Raon-Speech](https://github.com/krafton-ai/Raon-Speech) | 125 | Raon-SpeechChat, a 9B English and Korean full-duplex model (weights CC BY-NC 4.0) | Trained in three stages (streaming encoder, duplex pre-training on about 120,000 hours, persona fine-tuning) where Alba has one stage on 24 hours |
| [kyutai-labs/moshi-rag](https://github.com/kyutai-labs/moshi-rag) | 196 | MoshiRAG: a compact full-duplex model that looks facts up in the background while it starts to answer (ICML 2026) | Alba's 2B backbone knows little; retrieval timed into the gap before the key words of an answer works around that |
| [sbintuitions/DuplexCascade](https://github.com/sbintuitions/DuplexCascade) | 50 | DuplexCascade: a cascade without voice-activity detection that talks in short micro-turns steered by control tokens | A way to make Alba's first build, the cascade, full duplex without training audio models |

## Pieces of the problem

| Project | Stars | What it is | Why it matters here |
| --- | --- | --- | --- |
| [ErikEkstedt/VoiceActivityProjection](https://github.com/ErikEkstedt/VoiceActivityProjection) | 127 | Voice Activity Projection: self-supervised turn-taking prediction (MIT) | The turn model Alba's cascade used, retrained here |
| [Soul-AILab/SoulX-Duplug](https://github.com/Soul-AILab/SoulX-Duplug) | 332 | Streaming semantic voice-activity detection for full-duplex systems (Apache 2.0) | A candidate helper for labelling turn ends |
| [ASLP-lab/Easy-Turn](https://github.com/ASLP-lab/Easy-Turn) | 153 | Turn-taking detection model and dataset (Apache 2.0) | Labels and data for turn ends |
| [FreedomIntelligence/LoopSpeech](https://github.com/FreedomIntelligence/LoopSpeech) | 8 | "What Did I Just Say? Self-Listening for Full-Duplex Speech Models" | Alba feeds back what it just said every step; this studies exactly that |
| [lazybutai/personaplex-native-tool-use](https://github.com/lazybutai/personaplex-native-tool-use) | 0 | Native tool calling for PersonaPlex | Tool calls inside a duplex stream, as Alba's run 4 plans |
| [FireRedTeam/FireRedChat](https://github.com/FireRedTeam/FireRedChat) | 600 | A fully self-hosted full-duplex voice system (Apache 2.0) | A complete self-hosted system to compare against |
| [Interactivity Alignment](https://huggingface.co/collections/kyutai/interactivity-alignment) (Kyutai, on Hugging Face) | n/a | Reinforcement learning that retunes Moshi and PersonaPlex on pause handling, turn-taking, backchannels and interruptions (EMNLP 2026; weights only) | Supervised training alone left those models too quiet or badly timed, as it left Alba; rewards built from clips of human conversation target the hand-offs Alba misses |
| [KentoNishi/icassp27-spurious-onsets](https://github.com/KentoNishi/icassp27-spurious-onsets) | 1 | Why Moshi and PersonaPlex start talking into silence, and a filter for it at run time | Its test (would the model still say this if the caller were muted?) might help with Alba's reflex "Yeah."s without retraining, though it targets talk into silence, not replies to a real voice, and needs a second model pass each step (under 61 ms on the authors' 96 GB card; Alba's budget is about 56 ms a step) |
| [Liyht/def-fsm](https://github.com/Liyht/def-fsm) | 1 | Learns turn-taking from real two-person audio and content from written dialogues, with a loss that weights each source (EMNLP 2026) | The same split as Alba's run 3: timing from real people, words written for the assistant |
| [petrichor20211/Speculative-Listening](https://github.com/petrichor20211/Speculative-Listening) | 0 | Uses idle listening time to decode the next chunk as if the caller stays silent, then checks it when the sound arrives (code announced, not yet out) | Alba is mostly listening; drafting ahead could free time for the processor-only build |
| [KyuDan1/MuDuSi](https://github.com/KyuDan1/MuDuSi) | 4 | A simulator that seats several full-duplex agents in one virtual room and records what each one hears (MIT) | Repeatable tests of Alba talking with Moshi or PersonaPlex, before any real-room test |
| [BertilBraun/Voice-Light](https://github.com/BertilBraun/Voice-Light) | 11 | A full-duplex cascaded voice agent with a learned turn-end model and tool calls | An honest result like Alba's: its learned turn-end model caught 12.5% of turn ends against 95.6% for a simple timing rule, so it ships a hybrid |

## Benchmarks, surveys and lists

| Project | Stars | What it is |
| --- | --- | --- |
| [DanielLin94144/Full-Duplex-Bench](https://github.com/DanielLin94144/Full-Duplex-Bench) | 339 | Benchmark for turn-taking and overlap handling in full-duplex dialogue |
| [MM-Speech/DuplexSurvey](https://github.com/MM-Speech/DuplexSurvey) | 111 | "Speaking While Listening": a survey and empirical audit of full-duplex models (EMNLP 2026) |
| [Ruiqi-Yan/Awesome-Full-Duplex-SDM](https://github.com/Ruiqi-Yan/Awesome-Full-Duplex-SDM) | 291 | A curated list of full-duplex spoken dialogue models and benchmarks |
| [avera-labs/InteractionBench](https://github.com/avera-labs/InteractionBench) | 0 | Timing, spoken-task accuracy, whispering and grounding over 20+ turns, scored from the same recorded callers for Moshi, PersonaPlex, Freeze-Omni and cloud systems |
| [nttcslab-sp/m3-duplexbench](https://github.com/nttcslab-sp/m3-duplexbench) | 1 | Multi-turn benchmark in English and Japanese that scores timing and content together (NTT) |
| [sierra-research/tau2-bench](https://github.com/sierra-research/tau2-bench) | 2.4k | Includes τ-Voice: full-duplex voice agents doing airline, retail and telecom tasks with tools, for noisy and accented callers |

Running Alba's runtime against Full-Duplex-Bench, next to the project's own timing harness, would make its numbers
comparable with these models. M3-DuplexBench would do the same for casual Japanese, and τ-Voice for tool calls once
run 4 is trained.
