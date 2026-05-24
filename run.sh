#!/bin/bash
cd "$(dirname "$0")"
export PYTHONPATH="$PWD/deps/CosyVoice:$PWD/deps/CosyVoice/third_party/Matcha-TTS:$PWD/electron-l2d:$PYTHONPATH"
# Ensure Redis is running (hot memory), non-blocking if already started
redis-server --daemonize yes 2>/dev/null || true
# Ensure LanceDB data directory exists (cold memory)
mkdir -p data/cold_memory
# Suppress vLLM plugin auto-load errors (paddlex residual in cv311 env)
export VLLM_PLUGINS=""
# Kill orphan vLLM EngineCore processes from previous crashed runs
pkill -f "VLLM::EngineCore" 2>/dev/null || true
sleep 0.5
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
exec /home/baaai/miniforge3/envs/cv311/bin/python3 scripts/demo_full.py --mode voice --timeout "${1:-300}"

# v5.x: Limit PyTorch CUDA allocator fragmentation (reduces VRAM bloat)
