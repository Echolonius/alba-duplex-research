#!/usr/bin/env bash
# Analyzes the newest call in sessions/ without touching the call window's own servers:
# its own transcription server on port 9932 (127.0.0.1 only), stopped again afterwards.
#   bash tools/analyze_latest.sh [session_folder_name]
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$LV" || exit 1
S="${1:-$(ls sessions | sort | tail -1)}"
M="$LV/models/lfm3-audio"
(cd "$LV/bin/llama-b12094" && exec ./llama-server.exe -m "$M/LFM3-Audio-2B-Q4_K_M.gguf" --mmproj "$M/mmproj-LFM3-Audio-2B-Q4_K_M.gguf" \
  --host 127.0.0.1 --port 9932 -ngl 99 -c 4096 -np 1 --no-webui --offline > "$LV/logs/server-analysis.log" 2>&1) &
for i in $(seq 1 90); do curl -sS --max-time 2 http://127.0.0.1:9932/health 2>/dev/null | grep -q ok && break; sleep 1; done
LV_ANALYSIS_PORT=9932 "$PY" -I tools/analyze_call.py "sessions/$S"
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"name='llama-server.exe'\" | Where-Object { \$_.CommandLine -match 'LocalVoice' -and \$_.CommandLine -match 'port 9932' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force -Confirm:\$false }"
