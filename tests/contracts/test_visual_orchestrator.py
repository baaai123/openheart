"""
Contract tests for visual orchestrator behavior.

Anchors the current visual pipeline behavior BEFORE extraction
into a new VisualOrchestrator module (v5.x). All three tests use
no-GPU paths: dataclass instantiation, early-return from
process_frame(None, ...), and constructor property checks.
These stay green throughout the refactor.
"""
from __future__ import annotations

import pytest


# ===================================================================
# Test 1 — WindowAttentionPipeline.process_frame(None, ...) early return
# ===================================================================


@pytest.mark.asyncio
async def test_window_attention_pipeline_output() -> None:
    """process_frame(None, (0,0)) returns empty dict — early exit path.

    When screenshot is None, the pipeline returns immediately without
    touching window enumeration, CLIP, or any GPU code path.
    v5.x: WindowAttentionPipeline.process_frame() contract.
    """
    from src.perception.visual.window_attention import WindowAttentionPipeline

    pipeline = WindowAttentionPipeline()
    result = await pipeline.process_frame(None, (0, 0))

    assert isinstance(result, dict)
    assert "windows" in result
    assert "top_windows" in result
    assert "l2d_crop" in result
    assert result["windows"] == []
    assert result["top_windows"] == []
    assert result["l2d_crop"] is None


# ===================================================================
# Test 2 — VisualPipeline initialization with skip_vlm=True
# ===================================================================


@pytest.mark.asyncio
async def test_visual_pipeline_initialization() -> None:
    """VisualPipeline(skip_vlm=True) constructs and exposes properties.

    Uses a minimal RuntimeConfig mock — no GPU models are loaded,
    only the constructor path and property access are validated.
    v4.5.0 §1.3: VisualPipeline interface contract.
    """
    try:
        from src.config.runtime import RuntimeConfig, VRAMTier
        from src.perception.visual.visual_pipeline import VisualPipeline
    except ImportError as e:
        pytest.skip(f"VisualPipeline requires GPU/ML libraries: {e}")

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
        deepseek_base_url="https://api.deepseek.com/v1",
        deepseek_model="deepseek-v4-flash",
        deepseek_max_tokens=200,
        deepseek_temperature=0.8,
        context_limit=2048,
    )

    pipeline = VisualPipeline(config, skip_vlm=True)

    assert isinstance(pipeline.lane5_available, bool)
    assert isinstance(pipeline.lane2_available, bool)
    assert isinstance(pipeline.lane3_available, bool)
    assert isinstance(pipeline.lane4_available, bool)
    assert isinstance(pipeline.any_degraded, bool)

    preload_result = pipeline.preload()
    assert isinstance(preload_result, dict)


# ===================================================================
# Test 3 — WindowAttentionSnapshot dataclass roundtrip
# ===================================================================


def test_snapshot_type_roundtrip() -> None:
    """WindowAttentionSnapshot defaults: top_windows=[], l2d_crop=None.

    Pure Python dataclass instantiation — no GPU, no imports beyond
    the snapshot_types module and its local type dependencies.
    v5.x: WindowAttentionSnapshot contract.
    """
    from src.perception.visual.snapshot_types import (
        WindowAttentionSnapshot,
        WindowMeta,
        LLMContext,
    )

    snapshot = WindowAttentionSnapshot()
    assert snapshot.top_windows == []
    assert snapshot.l2d_crop is None
    assert snapshot.top_window is None
    assert snapshot.scene_label == ""
    assert snapshot.llm_context is None
    assert snapshot.timestamp == 0.0
    assert snapshot.cycle == 0

    wm = WindowMeta()
    assert wm.title == ""
    assert wm.attention_score == 0.0
    assert wm.change_score == 0.0
    assert wm.tags == []
    assert wm.bounds is None

    llm_ctx = LLMContext()
    assert llm_ctx.text == ""
    assert llm_ctx.scene == ""
    assert llm_ctx.position == ""
    assert llm_ctx.vlm_description == ""
    assert llm_ctx.matched == []
