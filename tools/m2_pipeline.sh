#!/usr/bin/env bash
# Milestone 2 end to end: Alba's words and voice trained into LFM3-Audio-2B (half-duplex), then exported and
# scored on the held-out tests. Every step is resumable; logs in logs/m2-*.log. Run from Git Bash:
#   bash tools/m2_pipeline.sh [N_CONVERSATIONS] [TRAIN_STEPS]
# Keep the graphics card free (no call window, no games) while it runs.
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$LV"
N=${1:-1500}
STEPS=${2:-600}
PY="$LV/.venv/Scripts/python.exe"
PYT="$LV/.venv-train/Scripts/python.exe"
step() { echo "== $(date +%H:%M) $*"; }

step "1/7 conversations (Gemma 4 E4B writes both sides, 2 at a time)"
"$PY" -I tools/make_convo_text.py --n "$N" --out data/convo/text.jsonl --workers 2 > logs/m2-text.log 2>&1
step "2/7 your older calls as listening input (newest 3 stay held out)"
"$PY" -I tools/make_convo_from_calls.py > logs/m2-calls.log 2>&1
step "3/7 speak them (Pocket TTS voices; Alba = 'alba')"
"$PY" -I tools/make_convo_audio.py --text data/convo/text.jsonl --workers 3 --threads 2 > logs/m2-audio.log 2>&1
cat data/convo/calls.jsonl >> data/convo/audio.jsonl
step "4/7 train (QLoRA on the 4 GB card)"
"$PYT" -I tools/train_lfm3_lora.py --steps "$STEPS" --out models/alba-lora > logs/m2-train.log 2>&1
step "5/7 export (merge + official llama.cpp converter + Q4_K_M)"
"$PYT" -I tools/export_alba.py --lora models/alba-lora --out models/alba-m2 > logs/m2-export.log 2>&1
step "6/7 held-out item tests through the app (voice model does everything, trained prompt)"
"$PY" -I tools/run_items.py --tag m2 --sets battery,redteam,feeling --set brain.mode=audio \
    --set paths.voice_model_dir=models/alba-m2 --set timing.turn_model=vap_live > logs/m2-items.log 2>&1
step "7/7 timing test through the app"
"$PY" -I tools/run_timing.py --tag m2 --set brain.mode=audio --set paths.voice_model_dir=models/alba-m2 \
    --set timing.turn_model=vap_live > logs/m2-timing.log 2>&1
"$PY" -I tools/score_battery.py m2
"$PY" -I tools/score_timing.py m2
"$PY" -I tools/score_redteam.py m2
"$PY" -I tools/score_feeling.py m2
step "done"
