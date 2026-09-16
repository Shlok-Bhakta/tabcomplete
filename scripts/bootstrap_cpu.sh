#!/usr/bin/env bash
# Local CPU-only bootstrap. Never run this in Colab (Colab uses its CUDA torch).
# Installs a CPU PyTorch build into the local uv venv WITHOUT adding it to
# pyproject.toml / uv.lock, so the lockfile never forces a CPU build on CUDA machines.
set -euo pipefail
uv pip install --index-url https://download.pytorch.org/whl/cpu torch --python "$(pwd)/.venv/bin/python" 2>/dev/null \
  || uv pip install --index-url https://download.pytorch.org/whl/cpu torch
python3 -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available())" 2>/dev/null \
  || uv run python -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available())"
