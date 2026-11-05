#!/usr/bin/env bash
# Milestone 4 end to end: write the training conversations, speak them, lay them out as live calls, train
# Alba Duplex, export, run the held-out timing and item tests, score. Each step is resumable (finished work is
# skipped), so the whole script can simply be started again; logs in logs/.
#   bash tools/duplex_pipeline.sh [TRAIN_STEPS] [N_CONVERSATIONS] [FROM_STEP]
# The graphics card: a job started by hand holds logs/gpu.lock while it runs; this script waits for the lock
# to go before it uses the card (and holds it itself while it does). FROM_STEP skips the steps before it.
set -uo pipefail
source "$(dirname "$0")/env.sh"
cd "$LV"
STEPS=${1:-600}
N=${2:-1500}
FROM=${3:-0}
PY="$LV/.venv/Scripts/python.exe"
PYT="$LV/.venv-train/Scripts/python.exe"
step() { echo "== $(date '+%d %b %H:%M') $*"; }
wait_gpu() { while [ -e logs/gpu.lock ] && [ "$(cat logs/gpu.lock 2>/dev/null)" != "pipeline" ]; do sleep 30; done; }
take_gpu() { wait_gpu; echo pipeline > logs/gpu.lock; }
free_gpu() { if [ "$(cat logs/gpu.lock 2>/dev/null)" = "pipeline" ]; then rm -f logs/gpu.lock; fi; true; }
trap free_gpu EXIT
set -e

if [ "$FROM" -le 0 ]; then
step "0 training conversations (Gemma 4 E4B writes both sides, 2 at a time; resumes)"
take_gpu
"$PY" -I tools/make_convo_text.py --n "$N" --out data/convo/text.jsonl --workers 2 >> logs/m2-text.log 2>&1
free_gpu
fi
if [ "$FROM" -le 1 ]; then
step "1 your older calls as listening input (newest 3 stay held out)"
if [ ! -s data/convo/calls.jsonl ]; then
  take_gpu; "$PY" -I tools/make_convo_from_calls.py > logs/m2-calls.log 2>&1; free_gpu
fi
fi
step "2 speak the conversations (Pocket TTS; Alba = 'alba'; processor only)"
"$PY" -I tools/make_convo_audio.py --text data/convo/text.jsonl --workers 3 --threads 2 > logs/m2-audio.log 2>&1
cat data/convo/calls.jsonl >> data/convo/audio.jsonl
step "3 lay them out as live calls (human-like gaps, cut-ins, mm-hms)"
"$PY" -I tools/make_duplex_data.py --source synthetic --out data/duplex/synthetic > logs/duplex-synthetic.log 2>&1
step "4 train Alba Duplex ($STEPS steps, QLoRA on the graphics card)"
take_gpu
"$PYT" -I tools/train_duplex.py --data data/duplex --steps "$STEPS" --out models/alba-duplex-lora > logs/duplex-train.log 2>&1
step "5 export (merge + official llama.cpp converter + Q4_K_M)"
"$PYT" -I tools/export_alba.py --lora models/alba-duplex-lora --out models/alba-duplex > logs/duplex-export.log 2>&1
step "6 timing test on held-out real callers (and again through laptop speakers)"
"$PYT" -I tools/run_timing_duplex.py --model models/alba-duplex --tag duplex > logs/duplex-timing.log 2>&1
"$PYT" -I tools/run_timing_duplex.py --model models/alba-duplex --tag duplex --speakers > logs/duplex-timing-speakers.log 2>&1
step "7 held-out item tests (battery, red-team, feeling)"
"$PYT" -I tools/run_items_duplex.py --model models/alba-duplex --tag duplex --sets battery,redteam,feeling > logs/duplex-items.log 2>&1
step "8 scores"
"$PY" -I tools/score_timing.py duplex > logs/duplex-scores.log 2>&1
"$PY" -I tools/score_timing.py duplex-speakers >> logs/duplex-scores.log 2>&1
"$PY" -I tools/score_battery.py duplex >> logs/duplex-scores.log 2>&1
"$PY" -I tools/score_redteam.py duplex >> logs/duplex-scores.log 2>&1
"$PY" -I tools/score_feeling.py duplex >> logs/duplex-scores.log 2>&1
free_gpu
step "done"
