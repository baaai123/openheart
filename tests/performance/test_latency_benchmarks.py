"""
Performance regression tests — latency benchmarks.

v4.5.0 §5.4:   normal path ≤ 2000 ms

All GPU-heavy components are mocked so these tests run on CPU-only CI nodes.
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from src.config.runtime import RuntimeConfig, VRAMTier
from src.decision.main_decision import MainDecisionEngine
from src.decision.safety_classifier import (
    DANGEROUS_AUTO_BLOCK,
    NEEDS_CONFIRM,
    SAFE,
    SafetyClassifier,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(enable_shadow: bool = True, vram_tier: VRAMTier = VRAMTier.HIGH) -> RuntimeConfig:
    """Return a minimal RuntimeConfig for performance testing."""
    return RuntimeConfig(
        vram_tier=vram_tier,
        vram_total_gb=16.0,
        low_vram=False,
        performance_mode=False,
        enable_shadow=enable_shadow,
        show_transcript=True,
        redis_host="localhost",
        redis_port=6379,
        redis_db=0,
        redis_password=None,
        redis_aof=True,
        context_limit=2048,
    )


# ---------------------------------------------------------------------------
# Normal-path latency (spec §5.4: ≤ 2000 ms)
# ---------------------------------------------------------------------------

class TestNormalPathLatency:
    """Benchmark MainDecisionEngine.decide() with a mocked 3B model."""

    @pytest.fixture
    def mocked_engine(self):
        """Return a MainDecisionEngine whose model is a MagicMock."""
        config = _make_config(enable_shadow=False)
        engine = MainDecisionEngine(runtime_config=config)

        # Simulate a loaded model that returns instantly.
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        # When generate() is called, return a dummy token tensor.
        mock_model.generate.return_value = [[1, 2, 3]]
        mock_tokenizer.decode.return_value = (
            '{"decision_type": "voice_response", '
            '"command": {"voice_response": " mocked response", "actions": []}, '
            '"confidence": 0.85, "safety_level": "SAFE", '
            '"trace_id": "mocked"}'
        )
        mock_tokenizer.apply_chat_template.return_value = "mocked chat template"
        mock_tokenizer.eos_token_id = 2

        engine._model = mock_model
        engine._tokenizer = mock_tokenizer
        engine._model_loaded = True
        return engine

    @pytest.mark.asyncio
    async def test_normal_path_mocked_decide_under_2000ms(self, mocked_engine):
        """A mocked normal-path decision must complete within 2000 ms."""
        scene_summary = "用户说'你好'，当前屏幕显示桌面。"
        user_model = {"version": 2}

        start = time.perf_counter()
        result = await mocked_engine.decide(
            scene_summary=scene_summary,
            user_model=user_model,
            trace_id="perf-normal-001",
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert result is not None
        assert "decision_type" in result
        assert elapsed_ms <= 2000.0, (
            f"Normal path decide() took {elapsed_ms:.2f} ms, exceeds 2000 ms budget"
        )

    @pytest.mark.asyncio
    async def test_normal_path_100_invocations_under_2000ms_each(self, mocked_engine):
        """Stress test: 100 normal-path decisions, each ≤ 2000 ms."""
        for i in range(100):
            start = time.perf_counter()
            await mocked_engine.decide(
                scene_summary=f"用户说'测试{i}'",
                user_model={"version": 2},
                trace_id=f"perf-normal-stress-{i:03d}",
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            assert elapsed_ms <= 2000.0, (
                f"Iteration {i} took {elapsed_ms:.2f} ms, exceeds 2000 ms budget"
            )

    @pytest.mark.asyncio
    async def test_degraded_path_under_2000ms(self, mocked_engine):
        """When the model is unavailable the degraded path must also be fast."""
        mocked_engine._model_loaded = False
        mocked_engine._degraded = True

        start = time.perf_counter()
        result = await mocked_engine.decide(
            scene_summary="用户说'你好'。",
            user_model={"version": 2},
            trace_id="perf-degraded-001",
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert result is not None
        assert elapsed_ms <= 2000.0, (
            f"Degraded path decide() took {elapsed_ms:.2f} ms, exceeds 2000 ms budget"
        )


# ---------------------------------------------------------------------------
# SafetyClassifier latency (auxiliary, no explicit spec limit, but must be fast)
# ---------------------------------------------------------------------------

class TestSafetyClassifierLatency:
    """Ensure SafetyClassifier.classify() does not become a bottleneck."""

    def test_classify_safe_under_100ms(self):
        """Classifying a SAFE command should be nearly instant."""
        classifier = SafetyClassifier()
        cmd = {
            "decision_type": "voice_response",
            "command": {
                "voice_response": "好的，我来帮你～",
                "actions": [{"type": "voice_response", "params": {}}],
            },
            "confidence": 0.92,
            "safety_level": "SAFE",
            "trace_id": "perf-safety-001",
            "shadow_overridden": False,
            "source": "main_decision_3b",
        }

        start = time.perf_counter()
        level = classifier.classify(cmd, trace_id=cmd["trace_id"])
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert level == SAFE
        assert elapsed_ms <= 100.0, (
            f"Safety classify took {elapsed_ms:.2f} ms, exceeds 100 ms auxiliary budget"
        )

    def test_classify_dangerous_under_100ms(self):
        """Classifying a DANGEROUS command should also be nearly instant."""
        classifier = SafetyClassifier()
        cmd = {
            "decision_type": "voice_response",
            "command": {
                "voice_response": "我要帮你转账给某人",
                "actions": [{"type": "mouse_click", "params": {"target": "转账按钮"}}],
            },
            "confidence": 0.85,
            "trace_id": "perf-safety-002",
            "shadow_overridden": False,
            "source": "main_decision_3b",
        }

        start = time.perf_counter()
        level = classifier.classify(cmd, trace_id=cmd["trace_id"])
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert level == DANGEROUS_AUTO_BLOCK
        assert elapsed_ms <= 100.0, (
            f"Safety classify took {elapsed_ms:.2f} ms, exceeds 100 ms auxiliary budget"
        )
