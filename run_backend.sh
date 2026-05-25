#!/bin/bash
source /home/baaai/miniforge3/etc/profile.d/conda.sh
conda activate cv311
cd /home/baaai/projects/openheart
python scripts/demo_full.py "$@"
