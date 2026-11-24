#!/usr/bin/env bash
# Bootstrap a fresh CUDA box (RunPod, Lambda, or any Ubuntu + NVIDIA driver) for training.
# Run from the repo root on the box:  bash scripts/gpu_setup.sh
set -euo pipefail

# RunPod images run as root and often lack these; elsewhere, use sudo.
SUDO=""; [[ $EUID -ne 0 ]] && SUDO="sudo"
if ! command -v tmux >/dev/null || ! command -v rsync >/dev/null; then
  $SUDO apt-get update -qq && $SUDO apt-get install -y -qq tmux rsync
fi

if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

# On RunPod only /workspace survives a pod stop (~ is wiped), so keep the HF cache, uv cache and
# uv's Python there: the venv in /workspace links to that Python. Copy mode avoids cross-device
# hard-link warnings. After a pod restart, just rerun this script: everything is cached.
if [[ -d /workspace && -z "${HF_HOME:-}" ]]; then
  export HF_HOME=/workspace/hf UV_CACHE_DIR=/workspace/.uv-cache \
    UV_PYTHON_INSTALL_DIR=/workspace/.uv-python UV_LINK_MODE=copy
  {
    echo "export HF_HOME=$HF_HOME UV_CACHE_DIR=$UV_CACHE_DIR UV_PYTHON_INSTALL_DIR=$UV_PYTHON_INSTALL_DIR"
    echo 'export UV_LINK_MODE=copy PATH="$HOME/.local/bin:$PATH"'
  } >> ~/.bashrc
fi

uv sync --all-packages --locked
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
uv run python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; \
print('torch', torch.__version__, '|', torch.cuda.get_device_name(), '| bf16:', torch.cuda.is_bf16_supported())"

# Download weights and data now, so the first training step doesn't stall on the network.
uv run python - <<'EOF'
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "Qwen/Qwen2.5-0.5B-Instruct"
AutoTokenizer.from_pretrained(name)
AutoModelForCausalLM.from_pretrained(name)
load_dataset("openai/gsm8k", "main")
print("model + dataset cached")
EOF

uv run pytest -q
echo "ready. next: pilot run (see README 'Running on a GPU')"
