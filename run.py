#!/usr/bin/env python3
"""
OpenHeart — multi-modal virtual companion system entry point. v4.5.0

Usage:
    python run.py --mode=mock
    python run.py --mode=real --vram-tier=auto
    python run.py --mode=real --vram-tier=high

Flags:
    --mode       mock | real   (default: mock)
    --vram-tier  auto | high | low  (default: auto)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from src.config.runtime import RuntimeConfig, VRAMTier, SystemRequirementError
from src.orchestrator import BootReport, Orchestrator

# Ensure project root is on sys.path for imports.
_project_root: Path = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

logger = logging.getLogger(__name__)


def _build_config(mode: str, vram_tier_override: str | None) -> RuntimeConfig:
    """Build a RuntimeConfig for the given mode and VRAM tier."""
    if mode == "mock":
        tier: VRAMTier = {
            "auto": VRAMTier.HIGH,
            "high": VRAMTier.HIGH,
            "low": VRAMTier.LOW,
        }.get(vram_tier_override or "auto", VRAMTier.HIGH)

        return RuntimeConfig(
            vram_tier=tier,
            vram_total_gb=16.0 if tier == VRAMTier.HIGH else 8.0,
            low_vram=(tier == VRAMTier.LOW),
            performance_mode=False,
            enable_shadow=False,
            show_transcript=False,
            redis_host="localhost",
            redis_port=6379,
            redis_db=0,
            redis_password=None,
            redis_aof=True,
            context_limit=2048,
        )

    # Real mode — detect from environment
    config: RuntimeConfig = RuntimeConfig.from_environ()
    if vram_tier_override and vram_tier_override != "auto":
        tier_override = VRAMTier(vram_tier_override)
        config = RuntimeConfig(
            vram_tier=tier_override,
            vram_total_gb=config.vram_total_gb,
            low_vram=(tier_override == VRAMTier.LOW),
            performance_mode=config.performance_mode,
            enable_shadow=False,
            show_transcript=config.show_transcript,
            redis_host=config.redis_host,
            redis_port=config.redis_port,
            redis_db=config.redis_db,
            redis_password=config.redis_password,
            redis_aof=config.redis_aof,
            context_limit=2048,
        )
    return config


async def _run_mock(config: RuntimeConfig, verbose: bool = False) -> int:
    """Mock mode: boot all layers, print summary, and exit."""
    orchestrator: Orchestrator = Orchestrator(config=config, mock=True, verbose=verbose)
    report = await orchestrator.boot()
    print()
    print(report.summary())
    print()
    print("All layers initialized")
    return 0 if report.all_pass else 1


async def _run_real(config: RuntimeConfig, verbose: bool = False) -> int:
    """Real mode: boot all layers and enter voice runtime loop."""
    orchestrator: Orchestrator = Orchestrator(
        config=config, mock=False, voice_mode=True
    )
    print()
    report: BootReport = await orchestrator.start()
    print()
    print(report.summary())
    print()
    return 0 if report.all_pass else 1


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="OpenHeart v4.5.0 — multi-modal virtual companion",
    )
    parser.add_argument(
        "--mode", choices=("mock", "real"), default="mock",
        help="Run mode: mock (no real GPU/models) or real",
    )
    parser.add_argument(
        "--vram-tier", choices=("auto", "high", "low"), default="auto",
        help="VRAM tier override (default: auto-detect)",
    )
    parser.add_argument(
        "--verbose", action="store_true", default=False,
        help="Enable INFO-level logging (default: WARNING)",
    )
    args = parser.parse_args()

    # Print boot banner
    print("=" * 60)
    print(" OpenHeart  v4.5.0")
    print("  Multi-modal virtual companion system")
    print("=" * 60)

    try:
        config: RuntimeConfig = _build_config(args.mode, args.vram_tier)
    except SystemRequirementError as exc:
        print(f"Fatal: System requirements not met — {exc}")
        return 1

    print(f"  Mode:       {args.mode}")
    print(f"  VRAM tier:  {config.vram_tier.value}")
    print(f"  VRAM total: {config.vram_total_gb:.1f} GB")
    print(f"  Context:    {config.context_limit} tokens")
    print()

    try:
        if args.mode == "mock":
            return asyncio.run(_run_mock(config, verbose=args.verbose))
        else:
            return asyncio.run(_run_real(config, verbose=args.verbose))
    except SystemRequirementError as exc:
        print(f"Fatal: {exc}")
        return 1
    except KeyboardInterrupt:
        print()
        print("Shutting down...")
        return 0


if __name__ == "__main__":
    sys.exit(main())
