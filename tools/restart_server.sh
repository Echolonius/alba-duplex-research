#!/usr/bin/env bash
# Restarts the local model server (127.0.0.1 only) for experiments.
#   bash tools/restart_server.sh [extra llama-server flags...]
#   QUANT=Q8_0 bash tools/restart_server.sh                 (8-bit LFM3-Audio)
#   NP=2 bash tools/restart_server.sh                       (two slots: transcribe and speak at once)
#   MODEL=lfm3-omni/LFM3-Omni-2B-Q4_K_M.gguf MMPROJ=lfm3-omni/mmproj-LFM3-Omni-2B-Q4_K_M.gguf bash tools/restart_server.sh --reasoning off
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
MODEL="${MODEL:-lfm3-audio/LFM3-Audio-2B-${QUANT:-Q4_K_M}.gguf}"
MMPROJ="${MMPROJ:-lfm3-audio/mmproj-LFM3-Audio-2B-${QUANT:-Q4_K_M}.gguf}"
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"name='llama-server.exe'\" | Where-Object { \$_.CommandLine -match 'LocalVoice' -and \$_.CommandLine -match 'port 9931' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force -Confirm:\$false }"
sleep 1
cd "$LV/bin/llama-b12094" && nohup ./llama-server.exe -m "$LV/models/$MODEL" --mmproj "$LV/models/$MMPROJ" --host 127.0.0.1 --port 9931 -ngl 99 -c 8192 -np "${NP:-1}" --no-webui --offline "$@" > "$LV/logs/server-experiment.log" 2>&1 &
curl -sS --retry 60 --retry-delay 1 --retry-connrefused --retry-all-errors --max-time 5 http://127.0.0.1:9931/health >/dev/null; echo "server up: $MODEL $*"
