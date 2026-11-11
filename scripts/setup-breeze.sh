#!/usr/bin/env bash
# Optional: installs Breeze TTS 2 (local voice cloning, NVIDIA GPU, ~12 GiB VRAM) under .breeze/.
# Breeze pins torch 2.9.1 / transformers 4.57.3, so it gets its own venv, separate from .venv.
# Weights are licensed for research and non-commercial use.
# Afterwards set in 实时语音 → 语音合成 → Breeze: Python = .breeze/venv/bin/python,
# 代码目录 = .breeze/breeze-tts, 模型目录 = .breeze/model.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIR="$ROOT/.breeze"
CODE_COMMIT=6eba901d1845353802652c708a4740fdfc5baa75
MODEL_REVISION=3e28c5151381a722f1d8661b4118c298caa77aa4
mkdir -p "$DIR"
if [ ! -d "$DIR/breeze-tts" ]; then
  git clone --no-checkout --filter=blob:none https://github.com/breezeblue-ai/breeze-tts.git "$DIR/breeze-tts"
  git -C "$DIR/breeze-tts" checkout "$CODE_COMMIT"
fi
[ -x "$DIR/venv/bin/python" ] || python3 -m venv "$DIR/venv"
"$DIR/venv/bin/python" -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 \
  torch==2.9.1 torchaudio==2.9.1 transformers==4.57.3 accelerate==1.12.0 huggingface-hub==0.36.2 \
  numpy==2.3.2 soundfile==0.13.1 librosa==0.11.0 einops==0.8.1 onnxruntime==1.23.2 \
  fastapi==0.141.1 uvicorn==0.53.0 python-multipart==0.0.20 sox==1.5.0
"$DIR/venv/bin/python" -m pip install --no-deps qwen-tts==0.1.1  # without --no-deps it pulls in gradio
"$DIR/venv/bin/python" -c "from huggingface_hub import snapshot_download as s; s('BreezeBlue/Breeze-TTS-2', revision='$MODEL_REVISION', local_dir='$DIR/model')"
echo "Breeze Python: $DIR/venv/bin/python"
echo "Breeze code:   $DIR/breeze-tts"
echo "Breeze model:  $DIR/model"
