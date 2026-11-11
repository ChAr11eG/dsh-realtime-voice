#!/usr/bin/env bash
# Creates .venv for the speech engine and installs engine/requirements.txt into it.
# BASE_PYTHON=/path/to/python: build the venv on that interpreter with --system-site-packages,
# so packages it already has (torch, qwen_asr) are reused instead of downloaded.
# The Breeze worker is not installed here: it runs in Breeze's own venv (settings: Breeze Python).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [ -n "${BASE_PYTHON:-}" ]; then
  "$BASE_PYTHON" -m venv --system-site-packages "$ROOT/.venv"
else
  python3 -m venv "$ROOT/.venv"
fi
"$ROOT/.venv/bin/python" -m pip install -r "$ROOT/engine/requirements.txt"
echo "Engine Python: $ROOT/.venv/bin/python"
