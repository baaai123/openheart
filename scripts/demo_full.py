#!/usr/bin/env python3
"""OpenHeart Voice Demo — thin wrapper around src.runtime_loop.run_voice_loop.

v4.5.0 §1.4 — Uses the extracted runtime loop for the full
ASR → DeepSeek → CosyVoice3-0.5B TTS pipeline.
"""
import argparse
import asyncio
import json
import logging
import os
import sys
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("NO_PROXY", "api.modelbest.cn,localhost,127.0.0.1")

# v5.x: Reduce PyTorch CUDA fragmentation (~1GB savings)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.runtime import RuntimeConfig
from src.runtime_loop import run_voice_loop  # v4.5.0 §0.6

logging.basicConfig(level=logging.WARNING, format="%(name)s: %(message)s")
logging.getLogger("cosyvoice").setLevel(logging.ERROR)
logging.getLogger("funasr").setLevel(logging.ERROR)
_log = logging.getLogger("demo_full")


def _load_char_name() -> str:
    """Load character name from 雪奈.json.  v4.5.0 §5.4."""
    default = "雪奈"
    try:
        _persona_path = os.path.join(
            os.path.dirname(__file__), "..", "雪奈.json"
        )
        with open(_persona_path, encoding="utf-8") as _f:
            _persona = json.load(_f)
            return _persona.get("basic_info", {}).get("name", default)
    except Exception:
        return default


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OpenHeart Voice Demo — thin wrapper for runtime loop",
    )
    parser.add_argument(
        "--mode", choices=["voice"], default="voice",
        help="Demo mode (only 'voice' supported)",
    )
    parser.add_argument(
        "--timeout", type=float, default=0.0,
        help="Max runtime in seconds (0 = run until Ctrl+C)",
    )
    args = parser.parse_args()

    if args.mode != "voice":
        print("Only --mode voice is supported in this demo.")
        sys.exit(1)

    config = RuntimeConfig.from_environ()
    char_name = _load_char_name()
    asyncio.run(run_voice_loop(config=config, char_name=char_name, timeout=args.timeout))


# v5.x: Limit vLLM GPU memory — two vLLM instances (TTS+VLM) share single GPU
os.environ.setdefault("VLLM_GPU_MEMORY_UTILIZATION", "0.25")

if __name__ == "__main__":
    main()
# Each vLLM instance ≤ 35% GPU → 5.6GB per instance on 16GB card
