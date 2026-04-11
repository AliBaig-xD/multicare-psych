#!/usr/bin/env bash
# setup.sh
# Run this once on a fresh Ubuntu 22.04 VPS after cloning the repo.
# Usage:  bash setup.sh

set -e  # exit on first error

echo "=========================================="
echo " MultiCaRe Psych Atlas — VPS Bootstrap"
echo "=========================================="

# ── 1. System packages ────────────────────────────────────────────────────────
echo ""
echo "[1/4] Installing system packages..."
sudo apt update -y
sudo apt upgrade -y
sudo apt install -y \
  build-essential \
  curl wget git unzip bzip2 \
  python3 python3-pip python3-venv \
  tmux

# ── 2. Project directories ────────────────────────────────────────────────────
echo ""
echo "[2/4] Creating project directories..."
mkdir -p data results logs notebooks medical_datasets

# ── 3. Python virtualenv ──────────────────────────────────────────────────────
echo ""
echo "[3/4] Creating Python virtual environment at env/py ..."
python3 -m venv env/py
source env/py/bin/activate
pip install --upgrade pip

# PyTorch CPU build (no CUDA needed; swap the index URL for GPU if desired)
echo "      Installing PyTorch (CPU)..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo "      Installing remaining requirements..."
pip install -r requirements.txt

# ── 4. Smoke test ─────────────────────────────────────────────────────────────
echo ""
echo "[4/4] Smoke-testing key imports..."
python3 - <<'EOF'
import torch, transformers, pandas, umap, hdbscan, streamlit
print("  torch        :", torch.__version__)
print("  transformers :", transformers.__version__)
print("  pandas       :", pandas.__version__)
print("  All imports OK")
EOF

echo ""
echo "=========================================="
echo " Setup complete!"
echo " Next step:"
echo "   tmux new -s multicare"
echo "   source env/py/bin/activate"
echo "   python scripts/01_download_multicare.py"
echo "=========================================="
