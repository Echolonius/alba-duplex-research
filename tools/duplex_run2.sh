#!/usr/bin/env bash
# Duplex training run 2 (5 Nov 2026): continues run 1 (models/alba-duplex-lora) with the run-2 layout (a "yeah"
# that opens a turn joins it; real people's questions and short reactions without I/my/we are learned), then
# exports and probes 10 battery questions. Calibration and the full held-out tests follow by hand.
set -uo pipefail
source "$(dirname "$0")/env.sh"
cd "$LV"
PYT="$LV/.venv-train/Scripts/python.exe"
step() { echo "== $(date '+%d %b %H:%M') $*"; }
set -e
echo pipeline > logs/gpu.lock
trap 'rm -f logs/gpu.lock' EXIT
step "1 train run 2 (900 steps from run 1, lr 1e-4)"
"$PYT" -I tools/train_duplex.py --data data/duplex --init models/alba-duplex-lora --steps 900 --lr 1e-4 \
    --out models/alba-duplex2-lora > logs/duplex2-train.log 2>&1
step "2 export"
"$PYT" -I tools/export_alba.py --lora models/alba-duplex2-lora --out models/alba-duplex2 > logs/duplex2-export.log 2>&1
step "3 probe: 10 battery questions"
rm -f results/tests/duplex2-probe/battery.jsonl
"$PYT" -I tools/run_items_duplex.py --model models/alba-duplex2 --tag duplex2-probe --sets battery --limit 10 \
    > logs/duplex2-probe.log 2>&1
step "done"
