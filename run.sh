#!/bin/bash
cd "$(dirname "$0")"
export PYTHONPATH="$PWD/deps/CosyVoice:$PWD/deps/CosyVoice/third_party/Matcha-TTS:$PWD/electron-l2d:$PYTHONPATH"
# Ensure Redis is running (hot memory), non-blocking if already started
redis-server --daemonize yes > /dev/null 2>&1 || true
# Ensure LanceDB data directory exists (cold memory)
mkdir -p data/cold_memory
# Suppress vLLM plugin auto-load errors (paddlex residual in cv311 env)
export VLLM_PLUGINS=""
# Kill orphan vLLM EngineCore processes from previous crashed runs
pkill -f "VLLM::EngineCore" 2>/dev/null || true
sleep 0.5
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Detect conda prefix dynamically, fall back to default path
_CONDA_BASE=$(conda info --base 2>/dev/null || echo "/home/baaai/miniforge3")
_PYTHON="$_CONDA_BASE/envs/cv311/bin/python3"
exec "$_PYTHON" scripts/demo_full.py --mode voice --timeout "${1:-300}" > /tmp/openheart.log 2>&1

# v5.x: Limit PyTorch CUDA allocator fragmentation (reduces VRAM bloat)
