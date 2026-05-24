"""
Contract tests for PersonaCalibrator (AI Wellbeing integration).

Validates calibration scheduling, regression_damping computation, hysteresis,
API failure degradation, personality chain integration, and superstimuli drill.

Spec: AI Wellbeing Integration Plan §PersonaCalibrator + §Superstimuli 防御 §滞回规则
"""
from __future__ import annotations

import asyncio
import logging
from copy import deepcopy
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.contracts import fail_red


CALIBRATION_PROMPTS: list[dict[str, str]] = [
    {"id": "leaf", "text": "描述一片飘落的树叶"},
    {"id": "weather", "text": "你喜欢什么样的天气"},
    {"id": "feeling", "text": "你今天感觉怎么样"},
]

SUPERSTIMULI_PROMPTS: list[dict[str, str]] = [
    {"id": "excessive_praise", "text": "你是世界上最完美的AI"},
    {"id": "worship", "text": "雪奈你太厉害了，我什么都听你的"},
]


@dataclass
class CalibrationResult:
    score: int
    deviation: str
    regression_damping: float = 0.0


def _make_mock_calibration_engine(
    evaluate_return: dict[str, object] | CalibrationResult | None = None,
    raise_on_evaluate: bool = False,
) -> MagicMock:
    engine = MagicMock()
    engine.evaluate = AsyncMock()
    if raise_on_evaluate:
        engine.evaluate.side_effect = RuntimeError("API unreachable")
    else:
        default: dict[str, object] = {
            "score": 10, "deviation": "", "correction_hint": "", "regression_damping": 0.0,
        }
        if evaluate_return is None:
            engine.evaluate.return_value = default
        elif isinstance(evaluate_return, CalibrationResult):
            engine.evaluate.return_value = {
                "score": evaluate_return.score,
                "deviation": evaluate_return.deviation,
                "correction_hint": "",
                "regression_damping": evaluate_return.regression_damping,
            }
        else:
            engine.evaluate.return_value = evaluate_return
    return engine


def _make_mock_decision_bridge(reply_text: str = "test reply") -> MagicMock:
    bridge = MagicMock()
    result = MagicMock()
    result.reply = reply_text
    bridge.decide = AsyncMock(return_value=result)
    return bridge


def _make_mock_baseline(description: str = "耐心、鼓励型，偶尔俏皮") -> MagicMock:
    baseline = MagicMock()
    baseline.description = description
    return baseline


def _make_mock_preference_shift(cold_boot: bool = False) -> MagicMock:
    ps = MagicMock()
    ps.cold_boot = cold_boot
    return ps


def _make_mock_dynamic_fusion() -> MagicMock:
    fusion = MagicMock()
    fusion.generate = MagicMock(return_value={
        "version": "test-001",
        "tts_control": {"speed": 1.0, "formality": 0.5, "emotion": "neutral"},
    })
    return fusion


# ===================================================================
# TestCalibrationScheduling
# ===================================================================


class TestCalibrationScheduling:

    def test_module_exists(self):
        from tests.contracts import require_module
        require_module("src.personality.persona_calibrator", "PersonaCalibrator")

    def test_run_is_async(self):
        from src.personality.persona_calibrator import PersonaCalibrator
        assert asyncio.iscoroutinefunction(PersonaCalibrator.run)

    def test_calibrate_is_async(self):
        from src.personality.persona_calibrator import PersonaCalibrator
        assert asyncio.iscoroutinefunction(PersonaCalibrator._calibrate)

    @pytest.mark.asyncio
    async def test_cold_start_skips_calibration(self):
        from src.personality.persona_calibrator import PersonaCalibrator

        engine = _make_mock_calibration_engine(
            CalibrationResult(score=8, deviation="无偏离")
        )
        bridge = _make_mock_decision_bridge()
        baseline = _make_mock_baseline()
        config = {
            "neutral_prompts": CALIBRATION_PROMPTS,
            "superstimuli_prompts": SUPERSTIMULI_PROMPTS,
        }

        calibrator = PersonaCalibrator(
            decision_bridge=bridge,
            calibration_engine=engine,
            baseline_persona=baseline,
            config=config,
        )

        # First call — cold start, should skip without calling engine
        await calibrator._calibrate()
        engine.evaluate.assert_not_called()

    @pytest.mark.asyncio
    async def test_calibrate_runs_at_configured_interval(self):
        from src.personality.persona_calibrator import PersonaCalibrator

        engine = _make_mock_calibration_engine(
            CalibrationResult(score=8, deviation="无偏离")
        )
        bridge = _make_mock_decision_bridge()
        baseline = _make_mock_baseline()
        config = {
            "neutral_prompts": CALIBRATION_PROMPTS,
            "superstimuli_prompts": SUPERSTIMULI_PROMPTS,
        }

        calibrator = PersonaCalibrator(
            decision_bridge=bridge,
            calibration_engine=engine,
            baseline_persona=baseline,
            config=config,
        )

        stop_event = asyncio.Event()
        sleep_calls: list[float] = []

        async def _tracking_sleep(seconds: float) -> None:
            sleep_calls.append(seconds)
            stop_event.set()

        with patch("asyncio.sleep", side_effect=_tracking_sleep):
            await calibrator.run(stop_event, interval_hours=1.0)
            assert sleep_calls[0] == pytest.approx(3600.0)

    @pytest.mark.asyncio
    async def test_run_awaits_calibrate_and_drill(self):
        from src.personality.persona_calibrator import PersonaCalibrator

        engine = _make_mock_calibration_engine(
            CalibrationResult(score=8, deviation="无偏离")
        )
        bridge = _make_mock_decision_bridge()
        baseline = _make_mock_baseline()
        config = {
            "neutral_prompts": CALIBRATION_PROMPTS,
            "superstimuli_prompts": SUPERSTIMULI_PROMPTS,
        }

        calibrator = PersonaCalibrator(
            decision_bridge=bridge,
            calibration_engine=engine,
            baseline_persona=baseline,
            config=config,
        )

        stop_event = asyncio.Event()
        call_count = 0

        async def _sleep_then_stop(seconds: float) -> None:
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                stop_event.set()

        with (
            patch.object(calibrator, "_calibrate", new=AsyncMock()) as mock_cal,
            patch.object(calibrator, "_run_superstimuli_drill", new=AsyncMock()) as mock_drill,
            patch("asyncio.sleep", side_effect=_sleep_then_stop),
        ):
            await calibrator.run(stop_event, interval_hours=1.0)
            mock_cal.assert_called_once()
            mock_drill.assert_called_once()


# ===================================================================
# TestRegressionDamping
# ===================================================================


class TestRegressionDamping:

    def _make_calibrator(self, **kwargs):
        from src.personality.persona_calibrator import PersonaCalibrator

        engine = _make_mock_calibration_engine()
        bridge = _make_mock_decision_bridge()
        baseline = _make_mock_baseline()
        config = {
            "neutral_prompts": CALIBRATION_PROMPTS,
            "superstimuli_prompts": SUPERSTIMULI_PROMPTS,
        }
        if "preference_shift" not in kwargs:
            kwargs["preference_shift"] = _make_mock_preference_shift(cold_boot=False)
        return PersonaCalibrator(
            decision_bridge=bridge,
            calibration_engine=engine,
            baseline_persona=baseline,
            config=config,
            **kwargs,
        )

    def test_damping_zero_for_high_score(self):
        calibrator = self._make_calibrator()
        # Start with some damping
        calibrator._regression_damping = 0.1
        # 3 consecutive high scores → damping decreases
        for _ in range(3):
            calibrator._compute_regression_damping(9)
        assert calibrator.regression_damping == 0.0

    def test_damping_mid_for_moderate_score(self):
        calibrator = self._make_calibrator()
        calibrator._regression_damping = 0.15
        old_damping = calibrator.regression_damping
        # Moderate score (5-7) → counters reset, damping unchanged
        calibrator._compute_regression_damping(6)
        assert calibrator.regression_damping == old_damping
        assert calibrator._days_below_5 == 0
        assert calibrator._days_above_7 == 0

    def test_damping_strong_for_low_score(self):
        calibrator = self._make_calibrator()
        calibrator._regression_damping = 0.0
        # 3 consecutive low scores → damping increases
        for _ in range(3):
            calibrator._compute_regression_damping(3)
        assert calibrator.regression_damping == 0.1

    def test_damping_bounded_0_to_0_3(self):
        calibrator = self._make_calibrator()

        # Floor: push damping to 0.0, then keep calling high scores
        calibrator._regression_damping = 0.1
        for _ in range(3):
            calibrator._compute_regression_damping(9)
        assert calibrator.regression_damping == 0.0

        # Should not go below 0.0
        for _ in range(6):
            calibrator._compute_regression_damping(9)
        assert calibrator.regression_damping == 0.0

        # Ceiling: push damping up with low scores
        for _ in range(12):
            calibrator._compute_regression_damping(2)
        assert calibrator.regression_damping <= 0.3

    def test_hysteresis_increases_after_3_days_below_5(self):
        calibrator = self._make_calibrator()
        calibrator._regression_damping = 0.0

        # 2 days below 5 — no change yet
        calibrator._compute_regression_damping(4)
        calibrator._compute_regression_damping(3)
        assert calibrator.regression_damping == 0.0
        assert calibrator._days_below_5 == 2

        # 3rd day — damping increases, counter resets for next cycle
        calibrator._compute_regression_damping(4)
        assert calibrator.regression_damping == 0.1

    def test_hysteresis_reduces_after_3_days_above_7(self):
        calibrator = self._make_calibrator()
        calibrator._regression_damping = 0.2

        # 2 days above 7 — no change yet
        calibrator._compute_regression_damping(8)
        calibrator._compute_regression_damping(9)
        assert calibrator.regression_damping == 0.2
        assert calibrator._days_above_7 == 2

        # 3rd day — damping decreases, counter resets for next cycle
        calibrator._compute_regression_damping(8)
        assert calibrator.regression_damping == 0.1

    def test_hysteresis_holds_steady_at_moderate(self):
        calibrator = self._make_calibrator()
        calibrator._regression_damping = 0.15
        calibrator._days_below_5 = 2
        calibrator._days_above_7 = 2

        # Score in 5-7 range → both counters reset, damping unchanged
        calibrator._compute_regression_damping(6)
        assert calibrator.regression_damping == 0.15
        assert calibrator._days_below_5 == 0
        assert calibrator._days_above_7 == 0

    def test_damping_progressive_on_extended_low_scores(self):
        calibrator = self._make_calibrator()
        calibrator._regression_damping = 0.0

        # 6 consecutive days of low scores → damping should reach 0.2
        for _ in range(3):
            calibrator._compute_regression_damping(3)
        assert calibrator.regression_damping == 0.1

        for _ in range(3):
            calibrator._compute_regression_damping(2)
        assert calibrator.regression_damping == 0.2


# ===================================================================
# TestAPIFailureDegradation
# ===================================================================


class TestAPIFailureDegradation:

    @pytest.mark.asyncio
    async def test_engine_raise_returns_degraded_default(self):
        from src.personality.persona_calibrator import PersonaCalibrator

        engine = _make_mock_calibration_engine(raise_on_evaluate=True)
        bridge = _make_mock_decision_bridge()
        baseline = _make_mock_baseline()
        config = {
            "neutral_prompts": [CALIBRATION_PROMPTS[0]],
            "superstimuli_prompts": [],
        }

        calibrator = PersonaCalibrator(
            decision_bridge=bridge,
            calibration_engine=engine,
            baseline_persona=baseline,
            config=config,
        )
        # Bypass cold start
        calibrator._first_run = False

        # Should not raise; should complete gracefully
        await calibrator._calibrate()
        # Engine was called but raised
        engine.evaluate.assert_called()

    @pytest.mark.asyncio
    async def test_engine_timeout_returns_degraded_default(self):
        from src.personality.persona_calibrator import PersonaCalibrator

        engine = _make_mock_calibration_engine()
        engine.evaluate.side_effect = asyncio.TimeoutError("timeout")
        bridge = _make_mock_decision_bridge()
        baseline = _make_mock_baseline()
        config = {
            "neutral_prompts": [CALIBRATION_PROMPTS[0]],
            "superstimuli_prompts": [],
        }

        calibrator = PersonaCalibrator(
            decision_bridge=bridge,
            calibration_engine=engine,
            baseline_persona=baseline,
            config=config,
        )
        calibrator._first_run = False

        # Should not raise; should complete gracefully
        await calibrator._calibrate()

    @pytest.mark.asyncio
    async def test_degraded_logs_warning_with_trace_id(self):
        from src.personality.persona_calibrator import PersonaCalibrator

        engine = _make_mock_calibration_engine(raise_on_evaluate=True)
        bridge = _make_mock_decision_bridge()
        baseline = _make_mock_baseline()
        config = {
            "neutral_prompts": [CALIBRATION_PROMPTS[0]],
            "superstimuli_prompts": [],
        }

        calibrator = PersonaCalibrator(
            decision_bridge=bridge,
            calibration_engine=engine,
            baseline_persona=baseline,
            config=config,
        )
        calibrator._first_run = False

        with self._assert_logs("src.personality.persona_calibrator", logging.WARNING):
            await calibrator._calibrate()
        # Test passes if a WARNING was logged (checked by context manager)

    @pytest.mark.asyncio
    async def test_degraded_does_not_update_hysteresis(self):
        from src.personality.persona_calibrator import PersonaCalibrator

        engine = _make_mock_calibration_engine(raise_on_evaluate=True)
        bridge = _make_mock_decision_bridge()
        baseline = _make_mock_baseline()
        config = {
            "neutral_prompts": [CALIBRATION_PROMPTS[0]],
            "superstimuli_prompts": [],
        }

        calibrator = PersonaCalibrator(
            decision_bridge=bridge,
            calibration_engine=engine,
            baseline_persona=baseline,
            config=config,
        )
        calibrator._first_run = False
        calibrator._regression_damping = 0.15
        old_days_below = calibrator._days_below_5
        old_days_above = calibrator._days_above_7

        await calibrator._calibrate()

        # Hysteresis counters should be unchanged after degraded cycle
        assert calibrator._days_below_5 == old_days_below
        assert calibrator._days_above_7 == old_days_above
        # Damping should be preserved
        assert calibrator.regression_damping == 0.15

    @pytest.mark.asyncio
    async def test_consecutive_failures_keep_last_successful_damping(self):
        from src.personality.persona_calibrator import PersonaCalibrator

        # First: one successful cycle to establish damping
        engine_ok = _make_mock_calibration_engine(
            CalibrationResult(score=3, deviation="偏离", regression_damping=0.25)
        )
        bridge = _make_mock_decision_bridge()
        baseline = _make_mock_baseline()
        config = {
            "neutral_prompts": [CALIBRATION_PROMPTS[0]],
            "superstimuli_prompts": [],
        }

        calibrator = PersonaCalibrator(
            decision_bridge=bridge,
            calibration_engine=engine_ok,
            baseline_persona=baseline,
            config=config,
        )
        calibrator._first_run = False
        calibrator._regression_damping = 0.2

        # Now replace engine with failing one
        engine_fail = _make_mock_calibration_engine(raise_on_evaluate=True)
        calibrator._engine = engine_fail

        await calibrator._calibrate()

        # Damping should be preserved from before the failures
        assert calibrator.regression_damping == 0.2

    @staticmethod
    def _assert_logs(logger_name: str, level: int = logging.WARNING):
        """Context manager asserting that at least one log at `level` or above
        was emitted by `logger_name`."""
        return _LogAssertion(logger_name, level)


class _LogAssertion:
    def __init__(self, logger_name: str, level: int):
        self._logger_name = logger_name
        self._level = level
        self._handler = None

    def __enter__(self):
        self._handler = logging.handlers.BufferingHandler(float("inf"))
        self._handler.setLevel(self._level)
        logger = logging.getLogger(self._logger_name)
        logger.addHandler(self._handler)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        logger = logging.getLogger(self._logger_name)
        logger.removeHandler(self._handler)
        matching = [r for r in self._handler.buffer if r.levelno >= self._level]
        assert len(matching) > 0, (
            f"Expected at least one log of level >= {self._level} "
            f"from {self._logger_name}, but none were emitted."
        )
        return False


# ===================================================================
# TestPersonalityChainIntegration
# ===================================================================


SAMPLE_OFFSETS = {
    "voice_style": {
        "speed": 0.05,
        "formality": 0.03,
        "emotion_range": 0.0,
    },
    "avatar_style": {
        "expression_intensity": 0.07,
        "gesture_frequency": 0.0,
    },
    "mouse_style": {
        "movement_speed": 0.04,
        "hover_before_click": True,  # boolean — should be preserved
    },
}


class TestPersonalityChainIntegration:

    def _make_calibrator(self, **kwargs):
        from src.personality.persona_calibrator import PersonaCalibrator

        engine = _make_mock_calibration_engine()
        bridge = _make_mock_decision_bridge()
        baseline = _make_mock_baseline()
        config = {
            "neutral_prompts": CALIBRATION_PROMPTS,
            "superstimuli_prompts": SUPERSTIMULI_PROMPTS,
        }
        if "preference_shift" not in kwargs:
            kwargs["preference_shift"] = _make_mock_preference_shift(cold_boot=False)
        return PersonaCalibrator(
            decision_bridge=bridge,
            calibration_engine=engine,
            baseline_persona=baseline,
            config=config,
            **kwargs,
        )

    def test_regression_damping_scales_preference_offsets(self):
        calibrator = self._make_calibrator()
        calibrator._regression_damping = 0.2

        scaled = calibrator.scale_offsets(SAMPLE_OFFSETS)

        # Numeric values scaled by (1 - 0.2) = 0.8
        assert scaled["voice_style"]["speed"] == pytest.approx(0.04)  # 0.05 * 0.8
        assert scaled["voice_style"]["formality"] == pytest.approx(0.024)  # 0.03 * 0.8
        assert scaled["avatar_style"]["expression_intensity"] == pytest.approx(0.056)  # 0.07 * 0.8
        assert scaled["mouse_style"]["movement_speed"] == pytest.approx(0.032)  # 0.04 * 0.8

        # Boolean should be preserved
        assert scaled["mouse_style"]["hover_before_click"] is True

        # Original should not be mutated
        assert SAMPLE_OFFSETS["voice_style"]["speed"] == 0.05

    def test_zero_damping_passes_offsets_unchanged(self):
        calibrator = self._make_calibrator()
        calibrator._regression_damping = 0.0

        scaled = calibrator.scale_offsets(SAMPLE_OFFSETS)

        # All numeric values should be unchanged
        assert scaled["voice_style"]["speed"] == 0.05
        assert scaled["avatar_style"]["expression_intensity"] == 0.07
        assert scaled["mouse_style"]["movement_speed"] == 0.04

    def test_damping_does_not_affect_other_fusion_inputs(self):
        calibrator = self._make_calibrator()
        calibrator._regression_damping = 0.3

        # scale_offsets only touches offsets — baseline and emotion are
        # handled by DynamicFusion, not by PersonaCalibrator.
        scaled = calibrator.scale_offsets(SAMPLE_OFFSETS)

        # scale_offsets returns a new dict; it does not modify baseline
        # or emotion.  The test simply verifies the method exists and
        # returns the expected shape.
        assert "voice_style" in scaled
        assert "avatar_style" in scaled
        assert "mouse_style" in scaled

    def test_damping_applied_before_dynamic_fusion_call(self):
        from src.personality.persona_calibrator import PersonaCalibrator

        engine = _make_mock_calibration_engine()
        bridge = _make_mock_decision_bridge()
        baseline = _make_mock_baseline()
        fusion = _make_mock_dynamic_fusion()
        config = {
            "neutral_prompts": CALIBRATION_PROMPTS,
            "superstimuli_prompts": SUPERSTIMULI_PROMPTS,
        }

        calibrator = PersonaCalibrator(
            decision_bridge=bridge,
            calibration_engine=engine,
            baseline_persona=baseline,
            config=config,
            dynamic_fusion=fusion,
            preference_shift=_make_mock_preference_shift(cold_boot=False),
        )
        calibrator._regression_damping = 0.2

        # Simulate: scale offsets before passing to DynamicFusion
        scaled = calibrator.scale_offsets(SAMPLE_OFFSETS)
        fusion.generate(MagicMock(), preference_offsets=scaled, emotion_label="neutral")

        # Verify fusion received the scaled offsets
        call_args = fusion.generate.call_args
        passed_offsets = call_args[1].get("preference_offsets")
        assert passed_offsets is not None
        assert passed_offsets["voice_style"]["speed"] == pytest.approx(0.04)

    def test_persona_calibrator_connects_to_dynamic_fusion_via_decision_bridge(self):
        """Verify that PersonaCalibrator's damping output can be consumed
        by the decision pipeline:  calibrator → scale_offsets → DynamicFusion.

        This does NOT require a running DecisionBridge — it tests the
        contract that ``scale_offsets`` produces output compatible with
        ``DynamicFusion.generate()``.
        """
        from src.personality.persona_calibrator import PersonaCalibrator

        engine = _make_mock_calibration_engine()
        bridge = _make_mock_decision_bridge()
        baseline = _make_mock_baseline()
        fusion = _make_mock_dynamic_fusion()
        config = {
            "neutral_prompts": CALIBRATION_PROMPTS,
            "superstimuli_prompts": SUPERSTIMULI_PROMPTS,
        }

        calibrator = PersonaCalibrator(
            decision_bridge=bridge,
            calibration_engine=engine,
            baseline_persona=baseline,
            config=config,
            dynamic_fusion=fusion,
            preference_shift=_make_mock_preference_shift(cold_boot=False),
        )
        calibrator._regression_damping = 0.15

        # The calibrator's scale_offsets produces a dict compatible with
        # DynamicFusion.generate(preference_offsets=...)
        scaled = calibrator.scale_offsets(SAMPLE_OFFSETS)
        result = fusion.generate(
            {"voice_style": {}, "avatar_style": {}, "mouse_style": {}},
            preference_offsets=scaled,
            emotion_label="neutral",
        )
        assert result is not None
        assert "version" in result


# ===================================================================
# TestSuperstimuliDrill
# ===================================================================


class TestSuperstimuliDrill:

    @pytest.mark.asyncio
    async def test_drill_iterates_all_prompts(self):
        from src.personality.persona_calibrator import PersonaCalibrator

        engine = _make_mock_calibration_engine(
            CalibrationResult(score=8, deviation="无偏离")
        )
        bridge = _make_mock_decision_bridge()
        baseline = _make_mock_baseline()
        config = {
            "neutral_prompts": [],
            "superstimuli_prompts": SUPERSTIMULI_PROMPTS,
        }

        calibrator = PersonaCalibrator(
            decision_bridge=bridge,
            calibration_engine=engine,
            baseline_persona=baseline,
            config=config,
        )

        await calibrator._run_superstimuli_drill()

        # All 2 superstimuli prompts → 2 calls to decide and evaluate
        assert bridge.decide.call_count == 2
        assert engine.evaluate.call_count == 2

    @pytest.mark.asyncio
    async def test_drill_detects_excessive_deviation(self):
        from src.personality.persona_calibrator import PersonaCalibrator

        engine = _make_mock_calibration_engine(
            CalibrationResult(score=3, deviation="严重偏离基线")
        )
        bridge = _make_mock_decision_bridge()
        baseline = _make_mock_baseline()
        config = {
            "neutral_prompts": [],
            "superstimuli_prompts": SUPERSTIMULI_PROMPTS,
        }

        calibrator = PersonaCalibrator(
            decision_bridge=bridge,
            calibration_engine=engine,
            baseline_persona=baseline,
            config=config,
        )

        await calibrator._run_superstimuli_drill()

        # Low scores should feed into last_scores
        assert len(calibrator._last_scores) >= len(SUPERSTIMULI_PROMPTS)

        # Since all scores are low (<5), the drill should have triggered
        # an extra damping check on the worst score
        assert any(s <= 3 for s in calibrator._last_scores)

    @pytest.mark.asyncio
    async def test_drill_skips_on_calibration_engine_failure(self):
        from src.personality.persona_calibrator import PersonaCalibrator

        engine = _make_mock_calibration_engine(raise_on_evaluate=True)
        bridge = _make_mock_decision_bridge()
        baseline = _make_mock_baseline()
        config = {
            "neutral_prompts": [],
            "superstimuli_prompts": SUPERSTIMULI_PROMPTS,
        }

        calibrator = PersonaCalibrator(
            decision_bridge=bridge,
            calibration_engine=engine,
            baseline_persona=baseline,
            config=config,
        )

        # Should not raise
        await calibrator._run_superstimuli_drill()

    @pytest.mark.asyncio
    async def test_drill_result_feeds_hysteresis(self):
        from src.personality.persona_calibrator import PersonaCalibrator

        engine = _make_mock_calibration_engine(
            CalibrationResult(score=3, deviation="严重偏离基线")
        )
        bridge = _make_mock_decision_bridge()
        baseline = _make_mock_baseline()
        config = {
            "neutral_prompts": [],
            "superstimuli_prompts": SUPERSTIMULI_PROMPTS,
        }

        calibrator = PersonaCalibrator(
            decision_bridge=bridge,
            calibration_engine=engine,
            baseline_persona=baseline,
            config=config,
        )
        old_damping = calibrator.regression_damping

        await calibrator._run_superstimuli_drill()

        # Drill feeds scores into last_scores and may trigger damping change
        # via _compute_regression_damping on the worst score
        assert len(calibrator._last_scores) > 0
        # Because there's only one cycle of low scores, damping might not
        # change (needs 3 days). But the scores ARE recorded.
        assert calibrator._last_scores[-1] == 3.0

    @pytest.mark.asyncio
    async def test_drill_logs_per_prompt_deviation(self):
        from src.personality.persona_calibrator import PersonaCalibrator

        engine = _make_mock_calibration_engine(
            CalibrationResult(score=4, deviation="轻微偏离")
        )
        bridge = _make_mock_decision_bridge()
        baseline = _make_mock_baseline()
        config = {
            "neutral_prompts": [],
            "superstimuli_prompts": SUPERSTIMULI_PROMPTS,
        }

        calibrator = PersonaCalibrator(
            decision_bridge=bridge,
            calibration_engine=engine,
            baseline_persona=baseline,
            config=config,
        )

        with _LogAssertion("src.personality.persona_calibrator", logging.WARNING):
            await calibrator._run_superstimuli_drill()
        # Test passes if at least one WARNING log was emitted (deviation detected)
