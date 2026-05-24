#!/bin/bash
# OpenHeart 一次性预热脚本
# 运行一次后，后续启动秒级就绪

set -e
echo "=== OpenHeart Setup ==="

# 1. Docker
echo "[1/4] Docker..."
docker compose ps &>/dev/null && echo "  ✅ Docker running" || echo "  ⚠️ Docker not found — Redis will be degraded"

# 2. Models
echo "[2/4] Models..."
python -c "
from pathlib import Path
models={'yolo_world':'yolo_world_nano','yolov11n':'yolov11n.pt','faster_whisper':'faster_whisper_large_v3','qwen_3b':'qwen2.5-3b-gptq','qwen_0.5b':'qwen2.5-0.5b','cosyvoice':'cosyvoice-300m','bge_small':'bge-small-zh-v1.5'}
for k,v in models.items():
    p=Path('models')/v
    print(f'  {\"✅\" if p.exists() else \"⚠️\"} {k}: {v}')
" 2>/dev/null

# 3. CUDA kernel (this takes ~2min first time)
echo "[3/4] CUDA Marlin kernel (first time ~2min)..."
python -c "
import torch
from transformers import AutoModelForCausalLM
print('  Loading Qwen 3B to compile kernel...')
m=AutoModelForCausalLM.from_pretrained('models/qwen2.5-3b-gptq',device_map='auto',torch_dtype='auto',trust_remote_code=True)
print('  ✅ Kernel compiled')
del m;torch.cuda.empty_cache()
" 2>/dev/null
echo "  ✅ Done"

# 4. Tests
echo "[4/4] Smoke test..."
python -m pytest tests/contracts/test_message_envelope_contract.py -q 2>/dev/null
echo "  ✅ Core tests pass"

echo ""
echo "=== Setup complete ==="
echo "Run: python scripts/run_demo.py"
