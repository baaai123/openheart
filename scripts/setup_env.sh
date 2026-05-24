#!/usr/bin/env bash
# Environment setup script for OpenHeart v4.5.0
# Usage: bash scripts/setup_env.sh

set -euo pipefail

ENV_NAME="openheart"
PYTHON_VERSION="3.11"

echo "=== OpenHeart Environment Setup ==="
echo "Creating conda environment: ${ENV_NAME} (Python ${PYTHON_VERSION})"

# Step 1: Create conda environment
conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y

# Step 2: Activate environment
# shellcheck source=/dev/null
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

# Step 3: Upgrade pip
pip install --upgrade pip

# Step 4: Install project in editable mode
pip install -e ".[dev]"

echo ""
echo "=== Installation Complete ==="
echo "Activate with: conda activate ${ENV_NAME}"
echo "Run tests with: pytest tests/contracts/ -v"

# Optional: Download spaCy Chinese model (uncomment if needed)
# python -m spacy download zh_core_web_sm

# CUDA Upgrade Guide (optional)
cat << 'EOF'

=== CUDA Upgrade Guide (Optional) ===
If you have an NVIDIA GPU and want CUDA support:

1. Check CUDA version:
   nvidia-smi

2. Install CUDA-enabled PyTorch (example for CUDA 12.1):
   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

3. Verify GPU access:
   python -c "import torch; print(torch.cuda.is_available())"

For other CUDA versions, see: https://pytorch.org/get-started/locally/
EOF
