#!/usr/bin/env bash
# Duplex training run 3 (5 Nov 2026): continues run 2 (models/alba-duplex2-lora) with
#   * the model's own plan layout (--plan-chunks count: exactly 6 plan pieces per 12 frames and no "speech now"
#     label, which the release cannot learn),
#   * the real conversations with her own words in every turn (data/duplex/seamless-own, from
#     tools/make_seamless_own.py) instead of the other speaker's words as context only: with real voices run 2
#     had learned only "yeah",
# then exports, probes the VALIDATION prompts (never the frozen tests) and checks her speech against her plan on
# the processor. Calibration and the held-out tests follow by hand. PROGRESS.md, 5 Nov morning.
set -uo pipefail
source "$(dirname "$0")/env.sh"
cd "$LV"
PYT="$LV/.venv-train/Scripts/python.exe"
PYA="$LV/.venv/Scripts/python.exe"
STEPS="${STEPS:-900}"
step() { echo "== $(date '+%d %b %H:%M') $*"; }
set -e
echo pipeline > logs/gpu.lock
trap 'rm -f logs/gpu.lock' EXIT
step "1 train run 3 ($STEPS steps from run 2, lr 1e-4, plan chunks by count, real voices with her own words)"
"$PYT" -I tools/train_duplex.py --data data/duplex --init models/alba-duplex2-lora --steps "$STEPS" --lr 1e-4 \
    --plan-chunks count --mic-sources real \
    --shares '{"seamless-own": 0.33, "synthetic": 0.45, "feeling": 0.14, "asides": 0.08}' \
    --out models/alba-duplex3-lora > logs/duplex3-train.log 2>&1
step "2 export"
"$PYT" -I tools/export_alba.py --lora models/alba-duplex3-lora --out models/alba-duplex3 > logs/duplex3-export.log 2>&1
step "3 validation probe"
"$PYT" -I tools/probe_duplex_val.py --model models/alba-duplex3 --save results/tests/duplex3-probe-val \
    > logs/duplex3-probe.log 2>&1
step "4 real voices: two validation conversations, 3 minutes each (run 2 said 'Yeah.' over and over here)"
for k in 0 1; do
    "$PYT" -I tools/diag_duplex_track.py --model models/alba-duplex3 --k "$k" --minutes 3 >> logs/duplex3-probe.log 2>&1
done
step "5 listen-setting calibration on the validation conversations (fixed rule)"
"$PYT" -I tools/calibrate_duplex.py --model models/alba-duplex3 > logs/calibrate-duplex3.log 2>&1
rm -f logs/gpu.lock
step "6 speech vs plan (processor)"
"$PYA" -I tools/speech_match.py results/tests/duplex3-probe-val >> logs/duplex3-probe.log 2>&1
step "done"
