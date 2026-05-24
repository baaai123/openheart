#!/bin/bash
source $(conda info --base)/etc/profile.d/conda.sh
conda activate sovits
cd /home/baaai/projects/openheart/deps/GPT-SoVITS
python api_v2.py -a 127.0.0.1 -p 9781 -c GPT_SoVITS/configs/feixiao_tts_infer.yaml
