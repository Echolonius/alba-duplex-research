# Tools

Run each with the project's Python: `source tools/env.sh` then `"$PY" -I tools/<name>.py ...`.

| Script | What it does |
|---|---|
| `fetch_models.sh` | Downloads llama.cpp, LFM3-Audio and the listening models from their official sources and checks each fingerprint |
| `extract_dessa_page.mjs` | Reads the saved Dessa page as plain data: subtitles, speaker roles, chapter markers, audio file names (`node`) |
| `conversation_metrics.py` | Measures a two-track conversation: bandwidth, timing, backchannels, interruptions, yielding, pitch, voice quality |
| `summarize_metrics.py` | Pools those measurements into one comparison table |
| `calibrate_turn.py` | Tunes the "has the person finished?" rule on Dessa's subtitled conversations |
| `benchmark.py` | Real-time conversations between LocalVoice and a simulated person (`--speakers` simulates laptop echo, `--brain split\|audio` picks the design) |
| `compare.py` | Measures a benchmark run with the Dessa tool and prints both side by side |
| `bench_content.py` | What the voice said in benchmark runs: assistant-mode and stock lines, repeats, reply length, timing, how often it gave way |
| `probe_text_brain.py` | The split design's text side alone: plays the benchmark conversations as text through `app/brain.py`'s real reply logic |
| `probe_loops.py`, `probe_modes.py`, `probe_multiturn.py` | Does the voice model loop, slip into assistant mode, or drift over many turns (old all-audio design) |
| `probe_cascade_speed.py` | How fast the split design can be: transcript, first sentence, first sound |
| `analyze_call.py` | Turns a recorded live call into a timeline with your spoken feedback and what the voice did around it |
| `restart_server.sh` | Restarts the voice server for experiments (`NP=2` for the split design) |
| `probe_server.py`, `probe_persona.py`, `probe_style.py` | Speed and personality checks against the running model server |
| `scale_math.py` | Training-cost and real-time arithmetic for Dessa-style models |
| `report_data.py`, `build_report.py`, `report_template.html` | Build the comparison report page from the result files |
