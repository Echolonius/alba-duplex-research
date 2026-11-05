# Keeps every tool, cache and model download inside this project folder.
# Usage (Git Bash):  source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
export LV="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export UV_PYTHON_INSTALL_DIR="$LV/.uv/python"
export UV_CACHE_DIR="$LV/.uv/cache"
export UV_PYTHON_BIN_DIR="$LV/.uv/bin"
export UV_PROJECT_ENVIRONMENT="$LV/.venv"
export UV_PYTHON_PREFERENCE=only-managed
export HF_HOME="$LV/models/hf"
export HF_HUB_DISABLE_TELEMETRY=1
export TORCH_HOME="$LV/models/torch"
export PYTHONUTF8=1
export PY="$LV/.venv/Scripts/python.exe"
