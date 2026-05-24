"""
Contract tests for VisualOrchestrator — v5.x extracted coordinator.

Validates:
  - .available property returns bool (True if VisualPipeline usable)
  - .available property returns bool (False before first poll cycle)
  - .start / .stop lifecycle (mock config, no GPU required)

v5.x §VisualOrchestrator
"""
from __future__ import annotations

import pytest


# ===================================================================
# Test 1 — VisualOrchestrator.available property
# ===================================================================


def test_available_returns_bool() -> None:
    """VisualOrchestrator.available returns bool even when GPU is absent.

    The constructor gracefully degrades if VisualPipeline init fails;
    available returns False in that case. The contract is that it
    always returns a bool, never raises.
    """
    try:
        from src.config.runtime import RuntimeConfig, VRAMTier
        from src.perception.visual.visual_orchestrator import VisualOrchestrator
    except ImportError as e:
        pytest.skip(f"VisualOrchestrator requires GPU/ML libraries: {e}")

    config = RuntimeConfig(
        vram_tier=VRAMTier.LOW,
        vram_total_gb=8.0,
        low_vram=True,
        performance_mode=False,
        enable_shadow=False,
        show_transcript=True,
        redis_host="localhost",
        redis_port=6379,
        redis_db=0,
        redis_password=None,
        redis_aof=True,
        deepseek_api_key="test-key",
        deepseek_base_url="https://api.deepseek.com",
        deepseek_model="deepseek-v4-flash",
        deepseek_max_tokens=200,
        deepseek_temperature=0.8,
        context_limit=2048,
    )

    orchestrator = VisualOrchestrator(config, log_callback=lambda msg: None)
    result = orchestrator.available
    assert isinstance(result, bool)


# ===================================================================
# Test 2 — VisualOrchestrator.vlm_ready property
# ===================================================================


def test_vlm_ready_returns_bool() -> None:
    """VisualOrchestrator.available returns bool.

    v5.x: vlm_ready removed — VLM is now in src/insight/prompt_learner.py.
    Verify basic orchestrator availability before any poller cycle.
    """
    try:
        from src.perception.visual.visual_orchestrator import VisualOrchestrator
        from unittest.mock import Mock
    except ImportError as e:
        pytest.skip(f"VisualOrchestrator unavailable: {e}")

    config = Mock()
    config.vram_tier = "high"

    orchestrator = VisualOrchestrator(config, log_callback=lambda msg: None)
    result = orchestrator.available
    assert isinstance(result, bool)


# ===================================================================
# Test 3 — VisualOrchestrator start/stop lifecycle
# ===================================================================


@pytest.mark.asyncio
async def test_start_stop_lifecycle() -> None:
    """VisualOrchestrator.start/stop runs without error.

    start() creates a background poller task; stop() cancels it
    gracefully. Should be idempotent and not raise.
    """
    try:
        from src.config.runtime import RuntimeConfig, VRAMTier
        from src.perception.visual.visual_orchestrator import VisualOrchestrator
    except ImportError as e:
        pytest.skip(f"VisualOrchestrator requires GPU/ML libraries: {e}")

    config = RuntimeConfig(
        vram_tier=VRAMTier.LOW,
        vram_total_gb=8.0,
        low_vram=True,
        performance_mode=False,
        enable_shadow=False,
        show_transcript=True,
        redis_host="localhost",
        redis_port=6379,
        redis_db=0,
        redis_password=None,
        redis_aof=True,
        deepseek_api_key="test-key",
        deepseek_base_url="https://api.deepseek.com",
        deepseek_model="deepseek-v4-flash",
        deepseek_max_tokens=200,
        deepseek_temperature=0.8,
        context_limit=2048,
    )

    orchestrator = VisualOrchestrator(config, log_callback=lambda msg: None)
    # Start — should not raise
    await orchestrator.start()

    # Double start — should be idempotent, no error
    await orchestrator.start()

    # Stop — should not raise
    await orchestrator.stop()

    # Double stop — should be idempotent, no error
    await orchestrator.stop()

    # After stop the poller task should be None
    assert orchestrator._poller_task is None
