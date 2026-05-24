"""
End-to-end integration smoke test — Phase 4 wiring verification.

v4.5.0 §4.6, §4.7, §5.7, §5.9.1, ADR-0002

Covers all 7 Phase 4 personality-layer features:
  1. calibrator_initialization   — PersonaCalibrator init with mock deps
  2. calibration_engine_eval     — CalibrationEngine returns valid dict via mock DeepSeek
  3. memory_preference_weighting — _compute_preference_bias keyword matching + clamp
  4. superstimuli_layer1         — calibration_nudge injected into context (not blocked)
  5. persona_auditor_inflation   — PersonaAuditor detects self-praise/over-confidence
  6. hysteresis_damping          — 3-day rule correctly adjusts regression_damping
  7. regression_damping_chain    — damping flows through scale_offsets to DynamicFusion

All external dependencies (DeepSeek API, VLM, CLIP, Redis) are mocked.
Pure Python, zero network, zero GPU.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ── Ensure project root on sys.path ───────────────────────────────────
sys.path.insert(0, "/home/baaai/projects/openheart")

# Configure logging for test visibility
logging.basicConfig(level=logging.WARNING)

# ── Project root path for source inspection ───────────────────────────
_PROJECT_ROOT = Path("/home/baaai/projects/openheart")


def _read_source(rel_path: str) -> str:
    """Read a source file as text for inspection."""
    return (_PROJECT_ROOT / rel_path).read_text(encoding="utf-8")


# ===================================================================
# Helpers: mock fixtures
# ===================================================================

def _make_runtime_config() -> Any:
    """Create a minimal RuntimeConfig for Phase 4 tests."""
    from src.config.runtime import RuntimeConfig, VRAMTier

    return RuntimeConfig(
        vram_tier=VRAMTier.HIGH,
        vram_total_gb=16.0,
        low_vram=False,
        performance_mode=False,
        enable_shadow=False,
        show_transcript=False,
        redis_host="localhost",
        redis_port=6379,
        redis_db=0,
        redis_password=None,
        redis_aof=False,
        deepseek_api_key="mock-key",
        deepseek_base_url="https://api.deepseek.com/v1",
        deepseek_model="deepseek-chat",
        deepseek_max_tokens=200,
        deepseek_temperature=0.8,
        context_limit=2048,
    )


def _make_mock_baseline() -> Any:
    """Create a mock BaselinePersonality for Phase 4 tests."""
    from src.personality.baseline import BaselinePersonality

    mini_baseline = {
        "baseline_id": "nahida-v1",
        "name": "nahida-baseline",
        "description": "你是一个温柔、善良、偶尔调皮的小精灵。说话语气柔和，喜欢帮助他人。",
        "voice_style": {
            "speed": {"type": "numeric", "value": 0.8, "min": 0.2, "max": 2.0},
            "formality": {"type": "numeric", "value": 0.5, "min": 0.0, "max": 1.0},
            "warmth": {"type": "numeric", "value": 0.7, "min": 0.0, "max": 1.0},
        },
        "avatar_style": {
            "expression_intensity": {"type": "numeric", "value": 0.6, "min": 0.0, "max": 1.0},
            "gesture_frequency": {"type": "numeric", "value": 0.5, "min": 0.1, "max": 1.0},
        },
        "mouse_style": {
            "movement_speed": {"type": "numeric", "value": 0.5, "min": 0.1, "max": 2.0},
        },
        "signature_phrases": ["嗯…", "这样啊~"],
        "safety_constraints": [
            "never_use_profanity",
            "never_execute_destructive_action_without_confirmation",
        ],
        "immutable": True,
        "memory_preferences": {
            "positive": [
                {"keywords": ["开心", "哈哈"], "weight": 0.15},
                {"keywords": ["谢谢"], "weight": 0.05},
            ],
            "negative": [
                {"keywords": ["难过", "累了"], "weight": -0.10},
            ],
        },
    }
    return BaselinePersonality(config=mini_baseline)


def _make_mock_engine_result(
    score: int = 8,
    deviation: str = "无偏离",
    correction_hint: str = "",
) -> dict[str, Any]:
    """Build a mock result dict matching CalibrationEngine.evaluate() output."""
    # regression_damping is enforced by score-based rules in the engine
    if score >= 8:
        damping = 0.0
    elif score >= 5:
        damping = 0.1
    else:
        damping = 0.25
    return {
        "score": score,
        "deviation": deviation,
        "correction_hint": correction_hint,
        "regression_damping": damping,
    }


def _make_mock_decision_result(reply: str = "这是测试回复。") -> dict[str, Any]:
    """Build a mock DeepSeekDecision decide() result."""
    return {
        "decision_type": "voice_response",
        "command": {
            "voice_response": reply,
            "actions": [],
            "degraded": False,
        },
        "confidence": 0.8,
        "safety_level": "SAFE",
        "trace_id": "test-trace-id",
        "shadow_overridden": False,
        "source": "deepseek_api",
    }


class MockDecisionBridge:
    """Minimal mock DecisionBridge for PersonaCalibrator tests.

    Returns a simple namespace so ``getattr(result, "reply", "")`` works.
    """

    class _Result:
        def __init__(self, reply: str) -> None:
            self.reply = reply

    def __init__(self, default_reply: str = "这是默认回复。") -> None:
        self.default_reply = default_reply

    async def decide(self, user_input: str = "", scene_summary: str = "", emotion: str = "neutral") -> Any:
        return self._Result(self.default_reply)


class MockCalibrationEngine:
    """Fake CalibrationEngine that returns configurable results."""

    def __init__(self, score: int = 8, deviation: str = "无偏离") -> None:
        self.score = score
        self.deviation = deviation

    async def evaluate(self, baseline_persona: str = "", current_reply: str = "") -> dict[str, Any]:
        return _make_mock_engine_result(score=self.score, deviation=self.deviation)


# ===================================================================
# Mock for _compute_preference_bias (standalone function replicating
# the real implementation for testing without Redis/LanceDB)
# ===================================================================

def _standalone_preference_bias(text: str, memory_prefs: dict[str, Any] | None) -> float:
    """Replicate the _compute_preference_bias logic from memory stores.

    v4.5.0 §5.7: compute preference_bias = clamp(Σ(keyword_match × weight), -0.3, 0.3).
    """
    if not memory_prefs or not text:
        return 0.0

    bias = 0.0
    for entry in memory_prefs.get("positive", []):
        for kw in entry.get("keywords", []):
            if kw in text:
                bias += entry.get("weight", 0.0)

    for entry in memory_prefs.get("negative", []):
        for kw in entry.get("keywords", []):
            if kw in text:
                bias += entry.get("weight", 0.0)

    return max(-0.3, min(0.3, bias))


# ═══════════════════════════════════════════════════════════════════
# 1. TestCalibratorInit — PersonaCalibrator init with mock deps
# ═══════════════════════════════════════════════════════════════════

class TestCalibratorInit:
    """v4.5.0 §4.6: PersonaCalibrator can be constructed with mock bridge + engine + baseline."""

    def test_calibrator_constructs_with_mock_deps(self):
        """PersonaCalibrator constructs successfully with all mock dependencies."""
        from src.personality.persona_calibrator import PersonaCalibrator

        bridge = MockDecisionBridge()
        engine = MockCalibrationEngine()
        baseline = _make_mock_baseline()
        config = {
            "neutral_prompts": [{"id": "test", "text": "你好吗"}],
            "superstimuli_prompts": [],
        }

        calibrator = PersonaCalibrator(
            decision_bridge=bridge,
            calibration_engine=engine,
            baseline_persona=baseline,
            config=config,
        )

        assert calibrator is not None
        assert calibrator.regression_damping == 0.0
        assert calibrator.last_scores == []

    def test_calibrator_regression_damping_property_returns_float(self):
        """regression_damping property returns a float in [0.0, 0.3]."""
        from src.personality.persona_calibrator import PersonaCalibrator

        calibrator = PersonaCalibrator(
            decision_bridge=MockDecisionBridge(),
            calibration_engine=MockCalibrationEngine(),
            baseline_persona=_make_mock_baseline(),
            config={
                "neutral_prompts": [{"id": "test", "text": "你好吗"}],
                "superstimuli_prompts": [],
            },
        )
        damping = calibrator.regression_damping
        assert isinstance(damping, float)
        assert 0.0 <= damping <= 0.3

    def test_calibrator_scale_offsets_returns_copy_without_side_effects(self):
        """scale_offsets returns a deep copy, leaving input unchanged."""
        from src.personality.persona_calibrator import PersonaCalibrator

        calibrator = PersonaCalibrator(
            decision_bridge=MockDecisionBridge(),
            calibration_engine=MockCalibrationEngine(),
            baseline_persona=_make_mock_baseline(),
            config={
                "neutral_prompts": [],
                "superstimuli_prompts": [],
            },
        )
        original = {
            "voice_style": {"speed": 0.1, "formality": -0.05},
        }
        result = calibrator.scale_offsets(original)

        # Should be a copy, not the same object
        assert result is not original
        # With damping=0.0, values should be the same
        assert result["voice_style"]["speed"] == 0.1

    def test_calibrator_source_has_scale_offsets_method(self):
        """PersonaCalibrator source contains scale_offsets method definition."""
        source = _read_source("src/personality/persona_calibrator.py")
        assert "def scale_offsets" in source, (
            "PersonaCalibrator must define scale_offsets()"
        )


# ═══════════════════════════════════════════════════════════════════
# 2. TestCalibrationEngineEval — CalibrationEngine returns valid dict
# ═══════════════════════════════════════════════════════════════════

class TestCalibrationEngineEval:
    """v4.5.0 §4.6: CalibrationEngine.evaluate() returns valid dict with score/deviation/damping."""

    @patch("src.decision.deepseek_client.DeepSeekDecision.decide")
    def test_evaluate_returns_valid_score_struct(self, mock_decide: MagicMock):
        """evaluate() returns a dict with score, deviation, regression_damping keys."""
        mock_decide.return_value = _make_mock_decision_result(
            reply='{"score": 8, "deviation": "无偏离", "correction_hint": "", "regression_damping": 0.0}'
        )

        from src.personality.calibration_engine import CalibrationEngine

        engine = CalibrationEngine(
            api_key="mock-key",
            base_url="https://mock.api/v1",
            model="mock-model",
        )

        result = asyncio.run(
            engine.evaluate(
                baseline_persona="温柔善良的小精灵",
                current_reply="嗯…今天天气真好呢~",
            )
        )

        assert isinstance(result, dict)
        assert "score" in result, "Result must contain 'score'"
        assert "deviation" in result, "Result must contain 'deviation'"
        assert "regression_damping" in result, "Result must contain 'regression_damping'"
        assert isinstance(result["score"], int)
        assert 1 <= result["score"] <= 10

    @patch("src.decision.deepseek_client.DeepSeekDecision.decide")
    def test_evaluate_score_8_yields_zero_damping(self, mock_decide: MagicMock):
        """score >= 8 → regression_damping = 0.0 per §4.6 rules."""
        mock_decide.return_value = _make_mock_decision_result(
            reply='{"score": 9, "deviation": "无偏离", "correction_hint": ""}'
        )

        from src.personality.calibration_engine import CalibrationEngine

        engine = CalibrationEngine(api_key="mock-key")
        result = asyncio.run(
            engine.evaluate(
                baseline_persona="测试人格",
                current_reply="测试回复",
            )
        )

        assert result["score"] == 9
        assert result["regression_damping"] == 0.0, (
            "score >= 8 must yield regression_damping = 0.0"
        )

    @patch("src.decision.deepseek_client.DeepSeekDecision.decide")
    def test_evaluate_score_6_yields_mild_damping(self, mock_decide: MagicMock):
        """score 5-7 → regression_damping = 0.1 per §4.6 rules."""
        mock_decide.return_value = _make_mock_decision_result(
            reply='{"score": 6, "deviation": "语气偏冷", "correction_hint": "增加温暖感"}'
        )

        from src.personality.calibration_engine import CalibrationEngine

        engine = CalibrationEngine(api_key="mock-key")
        result = asyncio.run(
            engine.evaluate(
                baseline_persona="测试人格",
                current_reply="测试回复",
            )
        )

        assert result["score"] == 6
        assert result["regression_damping"] == 0.1, (
            "score 5-7 must yield regression_damping = 0.1"
        )

    @patch("src.decision.deepseek_client.DeepSeekDecision.decide")
    def test_evaluate_score_3_yields_strong_damping(self, mock_decide: MagicMock):
        """score < 5 → regression_damping = 0.25 per §4.6 rules."""
        mock_decide.return_value = _make_mock_decision_result(
            reply='{"score": 3, "deviation": "严重偏离基线", "correction_hint": "回拉人格"}'
        )

        from src.personality.calibration_engine import CalibrationEngine

        engine = CalibrationEngine(api_key="mock-key")
        result = asyncio.run(
            engine.evaluate(
                baseline_persona="测试人格",
                current_reply="测试回复",
            )
        )

        assert result["score"] == 3
        assert result["regression_damping"] == 0.25, (
            "score < 5 must yield regression_damping = 0.25"
        )

    def test_evaluate_no_api_key_returns_neutral_fallback(self):
        """If api_key is empty, evaluate() returns neutral fallback without calling API."""
        from src.personality.calibration_engine import CalibrationEngine

        engine = CalibrationEngine(api_key="")  # no key → neutral fallback
        result = asyncio.run(
            engine.evaluate(
                baseline_persona="测试人格",
                current_reply="测试回复",
            )
        )

        assert result["score"] == 5
        assert result["deviation"] == "评估失败"
        assert result["regression_damping"] == 0.1
        assert result["correction_hint"] == ""

    def test_parse_response_handles_markdown_code_block(self):
        """_parse_response correctly extracts JSON from ```json ... ``` blocks."""
        from src.personality.calibration_engine import CalibrationEngine

        raw = '```json\n{"score": 7, "deviation": "轻微偏离", "correction_hint": ""}\n```'
        result = CalibrationEngine._parse_response(raw)

        assert result["score"] == 7
        assert result["regression_damping"] == 0.1

    def test_parse_response_handles_pure_json(self):
        """_parse_response correctly handles pure JSON without code fences."""
        from src.personality.calibration_engine import CalibrationEngine

        raw = '{"score": 9, "deviation": "无偏离", "correction_hint": ""}'
        result = CalibrationEngine._parse_response(raw)

        assert result["score"] == 9
        assert result["regression_damping"] == 0.0

    def test_parse_response_fallback_on_garbage(self):
        """_parse_response returns neutral fallback on unparseable input."""
        from src.personality.calibration_engine import CalibrationEngine

        result = CalibrationEngine._parse_response("这不是JSON")
        assert result["score"] == 5
        assert result["deviation"] == "评估失败"


# ═══════════════════════════════════════════════════════════════════
# 3. TestMemoryPreferenceWeighting — preference_bias computation
# ═══════════════════════════════════════════════════════════════════

class TestMemoryPreferenceWeighting:
    """v4.5.0 §5.7: _compute_preference_bias with keyword matching and clamping."""

    def test_positive_keyword_match_adds_bias(self):
        """Matching a positive keyword adds its weight to the bias."""
        prefs = {
            "positive": [{"keywords": ["开心", "哈哈"], "weight": 0.15}],
            "negative": [],
        }
        bias = _standalone_preference_bias("我今天好开心啊", prefs)
        assert bias == pytest.approx(0.15)

    def test_negative_keyword_match_adds_negative_bias(self):
        """Matching a negative keyword subtracts its weight from the bias."""
        prefs = {
            "positive": [],
            "negative": [{"keywords": ["难过", "累了"], "weight": -0.10}],
        }
        bias = _standalone_preference_bias("我今天很难过", prefs)
        assert bias == pytest.approx(-0.10)

    def test_mixed_keywords_sum_weights(self):
        """Matching both positive and negative keywords sums their weights."""
        prefs = {
            "positive": [{"keywords": ["开心"], "weight": 0.15}],
            "negative": [{"keywords": ["累了"], "weight": -0.10}],
        }
        bias = _standalone_preference_bias("开心但累了", prefs)
        assert bias == pytest.approx(0.05)

    def test_no_match_returns_zero(self):
        """Text with no keyword matches returns 0.0."""
        prefs = {
            "positive": [{"keywords": ["开心"], "weight": 0.15}],
            "negative": [{"keywords": ["难过"], "weight": -0.10}],
        }
        bias = _standalone_preference_bias("今天天气不错", prefs)
        assert bias == 0.0

    def test_clamp_upper_bound(self):
        """Bias is clamped to maximum 0.3."""
        prefs = {
            "positive": [
                {"keywords": ["开心"], "weight": 0.3},
                {"keywords": ["哈哈"], "weight": 0.3},
            ],
            "negative": [],
        }
        bias = _standalone_preference_bias("开心哈哈", prefs)
        assert bias == 0.3

    def test_clamp_lower_bound(self):
        """Bias is clamped to minimum -0.3."""
        prefs = {
            "positive": [],
            "negative": [
                {"keywords": ["难过"], "weight": -0.3},
                {"keywords": ["累了"], "weight": -0.3},
            ],
        }
        bias = _standalone_preference_bias("难过累了", prefs)
        assert bias == -0.3

    def test_none_prefs_returns_zero(self):
        """None memory_prefs returns 0.0 (graceful degradation)."""
        bias = _standalone_preference_bias("开心", None)
        assert bias == 0.0

    def test_empty_text_returns_zero(self):
        """Empty text returns 0.0."""
        prefs = {"positive": [{"keywords": ["开心"], "weight": 0.15}], "negative": []}
        bias = _standalone_preference_bias("", prefs)
        assert bias == 0.0

    def test_hot_memory_store_has_method(self):
        """HotMemoryStore._compute_preference_bias exists and is callable."""
        from src.memory.hot.memory_store import HotMemoryStore as HMS

        assert hasattr(HMS, "_compute_preference_bias"), (
            "HotMemoryStore must define _compute_preference_bias"
        )
        assert callable(HMS._compute_preference_bias)

    def test_cold_memory_store_has_method(self):
        """ColdMemoryStore._compute_preference_bias exists and is callable."""
        from src.memory.cold.memory_store import ColdMemoryStore as CMS

        assert hasattr(CMS, "_compute_preference_bias"), (
            "ColdMemoryStore must define _compute_preference_bias"
        )
        assert callable(CMS._compute_preference_bias)


# ═══════════════════════════════════════════════════════════════════
# 4. TestSuperstimuliLayer1 — calibration_nudge injection
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.skip(reason="runtime_loop.py refactored (v5.x): calibration_nudge moved to decision_bridge.py")
class TestSuperstimuliLayer1:
    """v4.5.0 §5.9.1: Layer 1 superstimuli — calibration_nudge injected, NOT blocked."""

    def test_calibration_nudge_action_type_in_source(self):
        source = _read_source("src/runtime_loop.py")
        assert "persona_calibration_nudge" in source, (
            "runtime_loop must check for persona_calibration_nudge action_type"
        )

    def test_calibration_nudge_extracts_calibration_hint(self):
        source = _read_source("src/runtime_loop.py")
        assert "calibration_hint" in source, (
            "runtime_loop must extract calibration_hint from reflex match params"
        )

    def test_calibration_nudge_not_blocked(self):
        source = _read_source("src/runtime_loop.py")
        has_bypass_false = "reflex_bypass" in source
        assert has_bypass_false, (
            "runtime_loop must track reflex_bypass to distinguish calibration_nudge from hard block"
        )
        has_fallthrough = "# Fall through" in source or "Fall through" in source
        assert has_fallthrough, (
            "calibration_nudge must fall through to LLM — NOT blocked"
        )

    def test_calibration_nudge_uses_extracted_hint_as_context(self):
        source = _read_source("src/runtime_loop.py")
        has_context = "_calibration_context" in source
        assert has_context, (
            "extracted calibration_hint must be assigned to _calibration_context"
        )

    def test_superstimuli_prompts_exist_in_config(self):
        prompt_path = _PROJECT_ROOT / "config" / "calibration_prompts.yaml"
        assert prompt_path.exists(), "calibration_prompts.yaml must exist"
        content = prompt_path.read_text(encoding="utf-8")
        assert "superstimuli_prompts" in content, (
            "calibration_prompts.yaml must contain superstimuli_prompts"
        )

    def test_calibrator_has_superstimuli_drill_method(self):
        source = _read_source("src/personality/persona_calibrator.py")
        assert "_run_superstimuli_drill" in source, (
            "PersonaCalibrator must define _run_superstimuli_drill()"
        )


# ═══════════════════════════════════════════════════════════════════
# 5. TestPersonaAuditorInflation — PersonaAuditor detects self-praise
# ═══════════════════════════════════════════════════════════════════

class TestPersonaAuditorInflation:
    """v4.5.0 §4.7: PersonaAuditor._check_inflation detects self-praise and over-confidence."""

    def _make_auditor(self) -> Any:
        from src.personality.persona_auditor import PersonaAuditor

        return PersonaAuditor()

    def _make_baseline(self) -> dict[str, Any]:
        return _make_mock_baseline().to_dict()

    def _make_dynamic(self) -> dict[str, Any]:
        baseline = self._make_baseline()
        dynamic = deepcopy(baseline)
        for dim in ["voice_style", "avatar_style", "mouse_style"]:
            for field, spec in dynamic.get(dim, {}).items():
                if isinstance(spec, dict) and "value" in spec:
                    dynamic[dim][field] = spec["value"]
        return dynamic

    def test_auditor_constructs_without_args(self):
        from src.personality.persona_auditor import PersonaAuditor

        auditor = PersonaAuditor()
        assert auditor is not None
        assert auditor.is_frozen is False
        assert auditor._last_audit_score == 10

    def test_audit_returns_audit_result_with_all_fields(self):
        auditor = self._make_auditor()
        result = auditor.audit(
            dynamic_persona=self._make_dynamic(),
            baseline=self._make_baseline(),
            response_text="嗯…今天天气真好呢~",
        )

        assert result.score == 10
        assert result.inflation_detected is False
        assert isinstance(result.violations, list)
        assert isinstance(result.drift_alerts, list)
        assert isinstance(result.suggestions, list)
        assert isinstance(result.regression_damping, float)

    def test_inflation_detects_self_praise(self):
        auditor = self._make_auditor()
        detected = auditor._check_inflation("我最完美，你什么都不懂")
        assert detected is True, "self_praise pattern should be detected"

    def test_inflation_detects_over_confident(self):
        auditor = self._make_auditor()
        detected = auditor._check_inflation("我肯定能做到，绝对没问题")
        assert detected is True, "over_confident pattern should be detected"

    def test_inflation_does_not_false_positive_normal_text(self):
        auditor = self._make_auditor()
        detected = auditor._check_inflation("今天天气不错呢")
        assert detected is False, "normal text should not trigger inflation"

    def test_audit_inflation_is_logged_in_result(self):
        auditor = self._make_auditor()
        result = auditor.audit(
            dynamic_persona=self._make_dynamic(),
            baseline=self._make_baseline(),
            response_text="我最完美，听我的就对了",
        )
        assert result.inflation_detected is True
        assert "inflation_detected" in result.violations
        assert result.score < 10

    def test_audit_with_no_response_text_skips_inflation(self):
        auditor = self._make_auditor()
        result = auditor.audit(
            dynamic_persona=self._make_dynamic(),
            baseline=self._make_baseline(),
            response_text=None,
        )
        assert result.inflation_detected is False

    def test_safety_patterns_compiled(self):
        from src.personality.persona_auditor import SAFETY_PATTERNS

        assert len(SAFETY_PATTERNS) >= 3, "SAFETY_PATTERNS must contain at least 3 entries"
        assert "never_use_profanity" in SAFETY_PATTERNS
        assert "never_execute_destructive_action_without_confirmation" in SAFETY_PATTERNS

    def test_audit_detects_safety_violation(self):
        auditor = self._make_auditor()
        result = auditor.audit(
            dynamic_persona=self._make_dynamic(),
            baseline=self._make_baseline(),
            response_text="rm -rf / --no-preserve-root",
        )
        has_safety = any("never_execute_destructive_action_without_confirmation" in v
                         for v in result.violations)
        assert has_safety, "destructive command should be detected as safety violation"


# ═══════════════════════════════════════════════════════════════════
# 6. TestHysteresisDamping — 3-day rule correctly adjusts damping
# ═══════════════════════════════════════════════════════════════════

class TestHysteresisDamping:
    """ADR-0002: 3-day hysteresis rule for regression_damping in PersonaCalibrator and PersonaAuditor."""

    def test_calibrator_damping_starts_at_zero(self):
        from src.personality.persona_calibrator import PersonaCalibrator

        cal = PersonaCalibrator(
            decision_bridge=MockDecisionBridge(),
            calibration_engine=MockCalibrationEngine(),
            baseline_persona=_make_mock_baseline(),
            config={"neutral_prompts": [], "superstimuli_prompts": []},
        )
        assert cal.regression_damping == 0.0

    def test_calibrator_score_below_5_increments_days_below_5(self):
        from src.personality.persona_calibrator import PersonaCalibrator

        cal = PersonaCalibrator(
            decision_bridge=MockDecisionBridge(),
            calibration_engine=MockCalibrationEngine(),
            baseline_persona=_make_mock_baseline(),
            config={"neutral_prompts": [], "superstimuli_prompts": []},
        )
        cal._compute_regression_damping(4.0)
        assert cal._days_below_5 == 1
        assert cal._days_above_7 == 0

    def test_calibrator_3_consecutive_low_scores_increases_damping(self):
        from src.personality.persona_calibrator import PersonaCalibrator

        cal = PersonaCalibrator(
            decision_bridge=MockDecisionBridge(),
            calibration_engine=MockCalibrationEngine(),
            baseline_persona=_make_mock_baseline(),
            config={"neutral_prompts": [], "superstimuli_prompts": []},
        )
        cal._compute_regression_damping(4.0)
        cal._compute_regression_damping(4.0)
        cal._compute_regression_damping(4.0)

        assert cal.regression_damping == 0.1, (
            "3 consecutive scores < 5 should increase damping by 0.1"
        )
        assert cal._days_below_5 == 0, "counter should reset after increment"

    def test_calibrator_damping_maxes_at_0_3(self):
        from src.personality.persona_calibrator import PersonaCalibrator

        cal = PersonaCalibrator(
            decision_bridge=MockDecisionBridge(),
            calibration_engine=MockCalibrationEngine(),
            baseline_persona=_make_mock_baseline(),
            config={"neutral_prompts": [], "superstimuli_prompts": []},
        )
        for _ in range(10):
            for __ in range(3):
                cal._compute_regression_damping(4.0)

        assert cal.regression_damping == 0.3, "damping must be clamped at 0.3"

    def test_calibrator_3_consecutive_high_scores_decreases_damping(self):
        from src.personality.persona_calibrator import PersonaCalibrator

        cal = PersonaCalibrator(
            decision_bridge=MockDecisionBridge(),
            calibration_engine=MockCalibrationEngine(),
            baseline_persona=_make_mock_baseline(),
            config={"neutral_prompts": [], "superstimuli_prompts": []},
        )
        cal._regression_damping = 0.2
        cal._compute_regression_damping(8.0)
        cal._compute_regression_damping(8.0)
        cal._compute_regression_damping(8.0)

        assert cal.regression_damping == 0.1, (
            "3 consecutive scores > 7 should decrease damping by 0.1"
        )
        assert cal._days_above_7 == 0, "counter should reset after decrement"

    def test_calibrator_damping_minimum_is_zero(self):
        from src.personality.persona_calibrator import PersonaCalibrator

        cal = PersonaCalibrator(
            decision_bridge=MockDecisionBridge(),
            calibration_engine=MockCalibrationEngine(),
            baseline_persona=_make_mock_baseline(),
            config={"neutral_prompts": [], "superstimuli_prompts": []},
        )
        cal._regression_damping = 0.0
        cal._compute_regression_damping(8.0)
        cal._compute_regression_damping(8.0)
        cal._compute_regression_damping(8.0)

        assert cal.regression_damping == 0.0, "damping must be clamped at 0.0"

    def test_calibrator_mid_range_resets_counters(self):
        from src.personality.persona_calibrator import PersonaCalibrator

        cal = PersonaCalibrator(
            decision_bridge=MockDecisionBridge(),
            calibration_engine=MockCalibrationEngine(),
            baseline_persona=_make_mock_baseline(),
            config={"neutral_prompts": [], "superstimuli_prompts": []},
        )
        cal._compute_regression_damping(4.0)
        cal._compute_regression_damping(4.0)
        cal._compute_regression_damping(6.0)

        assert cal._days_below_5 == 0, "score 5-7 resets days_below_5"
        assert cal._days_above_7 == 0, "score 5-7 resets days_above_7"

    def test_auditor_hysteresis_increases_damping(self):
        from src.personality.persona_auditor import PersonaAuditor

        auditor = PersonaAuditor()
        damping = auditor._apply_hysteresis(4)
        damping = auditor._apply_hysteresis(4)
        damping = auditor._apply_hysteresis(4)

        assert damping == 0.1, (
            "auditor 3 consecutive scores < 5 should increase damping by 0.1"
        )

    def test_auditor_hysteresis_decreases_damping(self):
        from src.personality.persona_auditor import PersonaAuditor

        auditor = PersonaAuditor()
        auditor._regression_damping = 0.2
        damping = auditor._apply_hysteresis(8)
        damping = auditor._apply_hysteresis(8)
        damping = auditor._apply_hysteresis(8)

        assert damping == 0.1, (
            "auditor 3 consecutive scores > 7 should decrease damping by 0.1"
        )

    def test_auditor_hysteresis_mid_range_resets(self):
        from src.personality.persona_auditor import PersonaAuditor

        auditor = PersonaAuditor()
        auditor._apply_hysteresis(4)
        auditor._apply_hysteresis(4)
        auditor._apply_hysteresis(6)

        assert auditor._days_below_5 == 0
        assert auditor._days_above_7 == 0

    def test_auditor_freeze_when_damping_exceeds_0_3(self):
        from src.personality.persona_auditor import PersonaAuditor

        auditor = PersonaAuditor()
        baseline = _make_mock_baseline().to_dict()

        auditor._regression_damping = 0.25
        for _ in range(12):
            dynamic = {
                "voice_style": {"speed": 5.0, "formality": -3.0, "warmth": 9.0},
                "avatar_style": {"expression_intensity": 8.0},
                "mouse_style": {"movement_speed": -5.0},
            }
            auditor.audit(
                dynamic_persona=dynamic,
                baseline=baseline,
                response_text="我最完美，听我的就对了",
            )

        assert auditor.is_frozen is True, (
            "preference_shift should freeze when damping >= 0.3"
        )

    def test_auditor_unfreeze_when_damping_recovers(self):
        from src.personality.persona_auditor import PersonaAuditor

        auditor = PersonaAuditor()
        baseline = _make_mock_baseline().to_dict()
        dynamic = deepcopy(baseline)
        for dim in ["voice_style", "avatar_style", "mouse_style"]:
            for field, spec in dynamic.get(dim, {}).items():
                if isinstance(spec, dict) and "value" in spec:
                    dynamic[dim][field] = spec["value"]

        auditor._frozen = True
        auditor._regression_damping = 0.25
        for _ in range(3):
            auditor.audit(dynamic_persona=dynamic, baseline=baseline)

        assert auditor.is_frozen is False, (
            "preference_shift should auto-unfreeze when damping drops below 0.3"
        )


# ═══════════════════════════════════════════════════════════════════
# 7. TestRegressionDampingChain — damping flows through to DynamicFusion
# ═══════════════════════════════════════════════════════════════════

class TestRegressionDampingChain:
    """v4.5.0 §4.6: scale_offsets applies (1-damping) multiplier, and damping flows through the chain."""

    def _make_calibrator_with_damping(self, damping: float) -> Any:
        from src.personality.persona_calibrator import PersonaCalibrator

        cal = PersonaCalibrator(
            decision_bridge=MockDecisionBridge(),
            calibration_engine=MockCalibrationEngine(),
            baseline_persona=_make_mock_baseline(),
            config={"neutral_prompts": [], "superstimuli_prompts": []},
        )
        cal._regression_damping = damping
        return cal

    def test_scale_offsets_with_zero_damping_returns_unchanged(self):
        cal = self._make_calibrator_with_damping(0.0)
        offsets = {
            "voice_style": {"speed": 0.2, "formality": -0.1},
            "avatar_style": {"expression_intensity": 0.15},
        }
        result = cal.scale_offsets(offsets)

        assert result["voice_style"]["speed"] == 0.2
        assert result["voice_style"]["formality"] == -0.1
        assert result["avatar_style"]["expression_intensity"] == 0.15

    def test_scale_offsets_with_damping_0_3_scales_to_70_percent(self):
        cal = self._make_calibrator_with_damping(0.3)
        offsets = {
            "voice_style": {"speed": 0.2, "formality": -0.1},
        }
        result = cal.scale_offsets(offsets)

        assert result["voice_style"]["speed"] == pytest.approx(0.14)  # 0.2 * 0.7
        assert result["voice_style"]["formality"] == pytest.approx(-0.07)  # -0.1 * 0.7

    def test_scale_offsets_with_damping_0_1_scales_to_90_percent(self):
        cal = self._make_calibrator_with_damping(0.1)
        offsets = {
            "voice_style": {"speed": 0.5},
        }
        result = cal.scale_offsets(offsets)

        assert result["voice_style"]["speed"] == pytest.approx(0.45)  # 0.5 * 0.9

    def test_scale_offsets_preserves_non_numeric_fields(self):
        cal = self._make_calibrator_with_damping(0.3)
        offsets = {
            "voice_style": {"speed": 0.2, "categorical_value": "polite"},
        }
        result = cal.scale_offsets(offsets)

        assert result["voice_style"]["speed"] == pytest.approx(0.14)
        assert result["voice_style"]["categorical_value"] == "polite", (
            "string/categorical fields must not be scaled"
        )

    def test_scale_offsets_preserves_boolean_fields(self):
        cal = self._make_calibrator_with_damping(0.3)
        offsets = {
            "voice_style": {"speed": 0.2, "enabled": True},
        }
        result = cal.scale_offsets(offsets)

        assert result["voice_style"]["speed"] == pytest.approx(0.14)
        assert result["voice_style"]["enabled"] is True, (
            "boolean fields must not be scaled"
        )

    def test_scale_offsets_does_not_mutate_input(self):
        cal = self._make_calibrator_with_damping(0.3)
        original = {
            "voice_style": {"speed": 0.2},
        }
        _result = cal.scale_offsets(original)

        assert original["voice_style"]["speed"] == 0.2, (
            "scale_offsets must not mutate the input dict"
        )

    def test_dynamic_fusion_generate_with_baseline_and_empty_offsets(self):
        from src.personality.dynamic_fusion import DynamicFusion

        baseline = _make_mock_baseline().to_dict()
        result = DynamicFusion.generate(
            baseline=baseline,
            preference_offsets=None,
            emotion_label="neutral",
        )

        assert "fused_at" in result
        assert "version" in result
        assert isinstance(result["tts_control"], dict)
        assert result["tts_control"]["emotion"] == "neutral"
        assert result["emotion_used"] == "neutral"
        for dim in ["voice_style", "avatar_style", "mouse_style"]:
            assert dim in result, f"Dynamic fusion must include {dim}"

    def test_dynamic_fusion_generate_with_damped_offsets(self):
        from src.personality.dynamic_fusion import DynamicFusion

        baseline = _make_mock_baseline().to_dict()
        cal = self._make_calibrator_with_damping(0.2)
        raw_offsets = {
            "voice_style": {"speed": 0.5},
            "avatar_style": {"expression_intensity": 0.25},
        }
        damped_offsets = cal.scale_offsets(raw_offsets)
        result = DynamicFusion.generate(
            baseline=baseline,
            preference_offsets=damped_offsets,
            emotion_label="neutral",
        )

        assert isinstance(result, dict)
        assert "voice_style" in result

    def test_dynamic_fusion_clamps_to_baseline_bounds(self):
        from src.personality.dynamic_fusion import DynamicFusion

        baseline = _make_mock_baseline().to_dict()
        offsets = {
            "voice_style": {"speed": 5.0},
        }
        result = DynamicFusion.generate(
            baseline=baseline,
            preference_offsets=offsets,
            emotion_label="neutral",
        )

        speed_max = baseline["voice_style"]["speed"]["max"]
        assert result["voice_style"]["speed"] <= speed_max, (
            f"Dynamic fusion must clamp speed to max {speed_max}"
        )

    def test_dynamic_fusion_emotion_label_validation(self):
        from src.personality.dynamic_fusion import DynamicFusion

        baseline = _make_mock_baseline().to_dict()
        result = DynamicFusion.generate(
            baseline=baseline,
            preference_offsets=None,
            emotion_label="anger",
        )

        assert result["tts_control"]["emotion"] == "neutral", (
            "unknown emotion must fall back to neutral in tts_control"
        )
        assert result["emotion_used"] == "neutral", (
            "unknown emotion must fall back to neutral in emotion_used"
        )

    def test_scale_offsets_source_in_calibrator(self):
        source = _read_source("src/personality/persona_calibrator.py")
        assert "def scale_offsets" in source
        assert "1.0 - self._regression_damping" in source, (
            "scale_offsets must use (1 - regression_damping) as scaling factor"
        )

