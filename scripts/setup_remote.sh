#!/usr/bin/env bash
# One-command bootstrap for vocos_rt fine-tuning on a fresh H100/A100 rental.
#
# Tested target environments: Vast.ai PyTorch image, RunPod PyTorch 2.1+,
# Lambda Labs Ubuntu+CUDA. Requires Python 3.12 (or 3.11) + nvidia-smi present.
#
# Usage on remote:
#   git clone <this repo>           # OR scp the directory
#   cd vocos_rt
#   bash scripts/setup_remote.sh    # ~15 min: installs deps, downloads model+data
#   bash scripts/train_remote.sh    # ~30 min on H100 / ~60 min on A100

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "============================================================"
echo " vocos_rt remote setup -- $(date)"
echo " Repo: $REPO_ROOT"
echo "============================================================"

# --- 1. Detect / create venv ---
if [ ! -x .venv/bin/python ]; then
    PY=$(command -v python3.12 || command -v python3.11 || command -v python3)
    "$PY" -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools

# --- 2. Detect CUDA + install matched torch ---
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    # H100/A100 default to CUDA 12.x; cu121 wheels work for both
    pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1 torchaudio==2.5.1
else
    echo "ERROR: no nvidia-smi" >&2
    exit 1
fi

# --- 3. Install everything else ---
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e .

# --- 4. Download upstream Vocos checkpoint ---
python scripts/download_checkpoints.py

# --- 5. Download LibriTTS train-clean-100 + dev-clean (~7.5 GB) ---
if [ ! -d datasets/libritts/LibriTTS/train-clean-100 ]; then
    python scripts/download_corpora.py --subsets train-clean-100 dev-clean
fi

echo
echo "============================================================"
echo " Setup complete."
echo " GPU memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader)"
echo " Disk free:  $(df -h . | tail -1 | awk '{print $4}')"
echo " Next step:  bash scripts/train_remote.sh"
echo "============================================================"
