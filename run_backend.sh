#!/bin/bash
set -e

# Source .env if it exists
if [ -f "$(dirname "$0")/.env" ]; then
    export $(grep -v '^#' "$(dirname "$0")/.env" | xargs)
fi

# Activate conda
source /home/baaai/miniforge3/etc/profile.d/conda.sh
conda activate cv311

# Run
cd /home/baaai/projects/openheart
python scripts/demo_full.py "$@"
