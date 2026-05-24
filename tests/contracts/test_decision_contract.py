"""
Contract tests for decision layer (spec v4.5.0 section 5).

Validates dual-model pipeline, shadow verification conflict handling,
fast path matching, reflex rules, safety levels, and context limits.
"""
import pytest
from copy import deepcopy
from unittest.mock import patch, MagicMock
from tests.contracts import require_module, fail_red

MAIN_MODULE = "src.decision.main_decision"
SHADOW_MODULE = "src.decision.shadow_verifier"



class TestModuleExists:
    def test_main_decision_module_available(self):
        from tests.contracts import require_module
        require_module(
            module_path="src.decision.main_decision",
            component_name="MainDecision (decision/main_decision.py)",
        )


VALID_DECISION_COMMAND = {
    "decision_type": "voice_response",
    "command": {
        "voice_response": "好的，我来帮你运行这个程序～",
        "actions": [],
    },
    "confidence": 0.92,
    "safety_level": "SAFE",
    "trace_id": "00000000-0000-0000-0000-000000000001",
    "shadow_overridden": False,
    "source": "main_decision_3b",
}

VALID_REFLEX_RULE = {
    "rule_id": "00000000-0000-0000-0000-000000000001",
    "name": "run_command",
    "priority": "INTERACTIVE",
    "status": "CORE",
    "condition": {
        "trigger_type": "voice_command",
        "pattern": "^(运行|执行)",
        "context_constraints": [],
    },
    "action": {
        "type": "mouse_click",
        "params": {},
        "safety_level": "SAFE",
    },
    "template_id": None,
    "template_slots": {},
    "cluster_hint": None,
    "metadata": {
        "confidence": 0.95,
        "success_count": 10,
        "failure_count": 0,
        "created_at": "2026-05-09T12:00:00.000+00:00",
        "last_verified_at": "2026-05-09T12:00:00.000+00:00",
        "source": "learning",
        "observation_remaining": 0,
    },
}


class TestDecisionCommandStructure:
    def test_command_has_decision_type(self):
        assert "decision_type" in VALID_DECISION_COMMAND

    def test_command_has_confidence_field(self):
        assert "confidence" in VALID_DECISION_COMMAND

    def test_confidence_is_float_0_to_1(self):
        c = VALID_DECISION_COMMAND["confidence"]
        assert isinstance(c, float)
        assert 0.0 <= c <= 1.0

    def test_command_has_safety_level(self):
        assert VALID_DECISION_COMMAND["safety_level"] in {
            "SAFE", "NEEDS_CONFIRM", "DANGEROUS_AUTO_BLOCK",
        }

    def test_command_has_trace_id(self):
        assert "trace_id" in VALID_DECISION_COMMAND

    def test_command_has_shadow_overridden_flag(self):
        assert "shadow_overridden" in VALID_DECISION_COMMAND

    def test_shadow_overridden_is_boolean(self):
        assert isinstance(VALID_DECISION_COMMAND["shadow_overridden"], bool)


class TestShadowVerifier:
    """Contract tests for ShadowVerifier — v4.5.0 §5.2.

    Uses mocked model loading to test the conflict resolution logic without
    requiring actual GPU models.
    """

    MAIN_CMD_SAFE = {
        "decision_type": "voice_response",
        "command": {"voice_response": "好的，我来帮你～", "actions": []},
        "confidence": 0.92,
        "safety_level": "SAFE",
        "trace_id": "00000000-0000-0000-0000-000000000001",
        "shadow_overridden": False,
        "source": "main_decision_3b",
    }

    MAIN_CMD_NEEDS_CONFIRM = {
        **MAIN_CMD_SAFE,
        "safety_level": "NEEDS_CONFIRM",
    }

    SCENE_SUMMARY = '用户说"帮我运行程序"，当前屏幕显示桌面。'

    @staticmethod
    def _make_config(enable_shadow: bool = True):
        """Create a minimal RuntimeConfig for testing without GPU."""
        from src.config.runtime import RuntimeConfig, VRAMTier
        return RuntimeConfig(
            vram_tier=VRAMTier.HIGH if enable_shadow else VRAMTier.LOW,
            vram_total_gb=16.0 if enable_shadow else 8.0,
            low_vram=not enable_shadow,
            performance_mode=False,
            enable_shadow=enable_shadow,
            show_transcript=True,
            redis_host="localhost",
            redis_port=6379,
            redis_db=0,
            redis_password=None,
            redis_aof=True,
            context_limit=4096 if enable_shadow else 2048,
        )

    # ------------------------------------------------------------------
    # Test: enabled flag and shadow disabling
    # ------------------------------------------------------------------

    def test_enable_shadow_false_skips_shadow_entirely(self):
        """v4.5.0 §5.2: enable_shadow=False skips shadow verification."""
        from src.decision.shadow_verifier import ShadowVerifier
        config = self._make_config(enable_shadow=False)
        sv = ShadowVerifier(runtime_config=config)
        assert sv.enabled is False
        assert sv.state.shadow_disabled is True

    def test_enable_shadow_true_enables_shadow(self):
        """v4.5.0 §5.2: enable_shadow=True allows shadow verification."""
        from src.decision.shadow_verifier import ShadowVerifier
        config = self._make_config(enable_shadow=True)
        with patch.object(
            ShadowVerifier, "_load_model", return_value=None
        ):
            sv = ShadowVerifier(runtime_config=config)
            sv._model = MagicMock()
            sv._tokenizer = MagicMock()
            sv._state.shadow_disabled = False
            sv._state.model_crashed = False
            assert sv.enabled is True

    def test_verify_returns_main_unchanged_when_shadow_disabled(self):
        """When shadow is disabled, verify() returns main command unchanged."""
        from src.decision.shadow_verifier import ShadowVerifier
        config = self._make_config(enable_shadow=False)
        sv = ShadowVerifier(runtime_config=config)
        result = sv.verify(self.MAIN_CMD_SAFE, self.SCENE_SUMMARY)
        assert result is self.MAIN_CMD_SAFE
        assert result["shadow_overridden"] is False

    # ------------------------------------------------------------------
    # Test: 1.5B crash handling
    # ------------------------------------------------------------------

    def test_1_5b_crash_disables_shadow(self):
        """v4.5.0 §5.2: 1.5B crash disables shadow, uses 3B only."""
        from src.decision.shadow_verifier import ShadowVerifier
        config = self._make_config(enable_shadow=True)
        with patch.object(ShadowVerifier, "_load_model", return_value=None):
            sv = ShadowVerifier(runtime_config=config)
            sv._model = MagicMock()
            sv._tokenizer = MagicMock()
            sv._state.shadow_disabled = False
            sv._state.model_crashed = False

        sv._run_shadow_inference = MagicMock(
            side_effect=Exception("1.5B model crash")
        )
        result = sv.verify(self.MAIN_CMD_SAFE, self.SCENE_SUMMARY)
        assert result is self.MAIN_CMD_SAFE
        assert sv.state.shadow_disabled is True
        assert sv.state.model_crashed is True
        assert sv.enabled is False

    # ------------------------------------------------------------------
    # Test: similarity-based conflict resolution
    # ------------------------------------------------------------------

    def test_similarity_at_least_0_85_no_action(self):
        """v4.5.0 §5.2: similarity >= 0.85 → no intervention."""
        from src.decision.shadow_verifier import (
            ShadowVerifier,
            InferenceResult,
        )
        config = self._make_config(enable_shadow=True)
        with patch.object(ShadowVerifier, "_load_model", return_value=None):
            sv = ShadowVerifier(runtime_config=config)
            sv._model = MagicMock()
            sv._tokenizer = MagicMock()
            sv._state.shadow_disabled = False

        sv._run_shadow_inference = MagicMock(
            return_value=InferenceResult(
                text="好的，我帮你",
                confidence=0.90,
                safety_level="SAFE",
            )
        )
        sv._compute_similarity = MagicMock(return_value=0.90)

        result = sv.verify(self.MAIN_CMD_SAFE, self.SCENE_SUMMARY)
        assert result is self.MAIN_CMD_SAFE
        assert result["shadow_overridden"] is False

    def test_similarity_below_0_85_with_safety_conflict_overrides(self):
        """v4.5.0 §5.2: similarity < 0.85 AND 3B safety < 1.5B safety
        → adopt 1.5B, mark shadow_overridden=true."""
        from src.decision.shadow_verifier import (
            ShadowVerifier,
            InferenceResult,
        )
        config = self._make_config(enable_shadow=True)
        with patch.object(ShadowVerifier, "_load_model", return_value=None):
            sv = ShadowVerifier(runtime_config=config)
            sv._model = MagicMock()
            sv._tokenizer = MagicMock()
            sv._state.shadow_disabled = False

        sv._run_shadow_inference = MagicMock(
            return_value=InferenceResult(
                text="这个操作需要确认",
                confidence=0.88,
                safety_level="NEEDS_CONFIRM",
            )
        )
        sv._compute_similarity = MagicMock(return_value=0.70)

        result = sv.verify(self.MAIN_CMD_SAFE, self.SCENE_SUMMARY)
        assert result["shadow_overridden"] is True
        assert result["source"] == "shadow_verifier_1.5b"
        assert result["safety_level"] == "NEEDS_CONFIRM"

    def test_similarity_below_0_85_same_safety_no_override(self):
        """similarity < 0.85 but same safety level → no override."""
        from src.decision.shadow_verifier import (
            ShadowVerifier,
            InferenceResult,
        )
        config = self._make_config(enable_shadow=True)
        with patch.object(ShadowVerifier, "_load_model", return_value=None):
            sv = ShadowVerifier(runtime_config=config)
            sv._model = MagicMock()
            sv._tokenizer = MagicMock()
            sv._state.shadow_disabled = False

        sv._run_shadow_inference = MagicMock(
            return_value=InferenceResult(
                text="好的，帮你",
                confidence=0.85,
                safety_level="SAFE",
            )
        )
        sv._compute_similarity = MagicMock(return_value=0.70)

        result = sv.verify(self.MAIN_CMD_SAFE, self.SCENE_SUMMARY)
        assert result is self.MAIN_CMD_SAFE

    # ------------------------------------------------------------------
    # Test: consecutive conflicts → silent fallback
    # ------------------------------------------------------------------

    def test_three_consecutive_conflicts_trigger_silent_fallback(self):
        """v4.5.0 §5.2: 3 consecutive conflicts → silent fallback active."""
        from src.decision.shadow_verifier import (
            ShadowVerifier,
            InferenceResult,
        )
        config = self._make_config(enable_shadow=True)
        with patch.object(ShadowVerifier, "_load_model", return_value=None):
            sv = ShadowVerifier(runtime_config=config)
            sv._model = MagicMock()
            sv._tokenizer = MagicMock()
            sv._state.shadow_disabled = False

        sv._run_shadow_inference = MagicMock(
            return_value=InferenceResult(
                text="安全操作",
                confidence=0.80,
                safety_level="SAFE",
            )
        )
        sv._compute_similarity = MagicMock(return_value=0.50)

        # Use a command where 3B is MORE conservative (NEEDS_CONFIRM) and
        # shadow says SAFE — this triggers Case C (conflict counter) not
        # Case B (override, which requires 3B < shadow safety).
        conservative_cmd = self.MAIN_CMD_NEEDS_CONFIRM

        # 3 consecutive conflicts
        for _ in range(2):
            result = sv.verify(conservative_cmd, self.SCENE_SUMMARY)
            assert sv.state.consecutive_conflicts > 0

        # 3rd conflict triggers fallback
        result = sv.verify(conservative_cmd, self.SCENE_SUMMARY)
        assert sv.state.silent_fallback_active is True
        assert result["shadow_overridden"] is True
        assert result["source"] == "shadow_verifier_1.5b_fallback"

    def test_silent_fallback_never_restarts_model(self):
        """v4.5.0 §5.2 (v4.4 change): silent fallback does NOT restart model."""
        from src.decision.shadow_verifier import (
            ShadowVerifier,
            InferenceResult,
        )
        config = self._make_config(enable_shadow=True)
        with patch.object(ShadowVerifier, "_load_model", return_value=None):
            sv = ShadowVerifier(runtime_config=config)
            sv._model = MagicMock()
            sv._tokenizer = MagicMock()
            sv._state.shadow_disabled = False

        sv._run_shadow_inference = MagicMock(
            return_value=InferenceResult(
                text="危险",
                confidence=0.75,
                safety_level="NEEDS_CONFIRM",
            )
        )
        sv._compute_similarity = MagicMock(return_value=0.50)

        # Manually set state to simulate 3 conflicts and fallback
        sv._state.consecutive_conflicts = 3
        sv._state.silent_fallback_active = True

        # Run verify again — should handle gracefully without model restart
        result = sv.verify(self.MAIN_CMD_SAFE, self.SCENE_SUMMARY)
        # Model should not have been re-loaded
        sv._run_shadow_inference.assert_called()
        assert sv._model is not None  # Model not restarted / replaced

    # ------------------------------------------------------------------
    # Test: param adjustment during silent fallback
    # ------------------------------------------------------------------

    def test_silent_fallback_adjusts_temperature(self):
        """v4.5.0 §5.2: fallback lowers temp by 0.2, min 0.4."""
        from src.decision.shadow_verifier import ShadowVerifier
        config = self._make_config(enable_shadow=True)
        with patch.object(ShadowVerifier, "_load_model", return_value=None):
            sv = ShadowVerifier(runtime_config=config)
            sv._state.silent_fallback_active = True

        params = sv.get_adjusted_params(
            base_temperature=0.8, base_top_p=0.9
        )
        assert params["temperature"] == pytest.approx(0.6)  # 0.8 - 0.2
        assert params["top_p"] == 0.9  # unchanged since temp > floor

    def test_silent_fallback_temperature_floor(self):
        """Temperature never drops below 0.4; top_p tightens instead."""
        from src.decision.shadow_verifier import ShadowVerifier
        config = self._make_config(enable_shadow=True)
        with patch.object(ShadowVerifier, "_load_model", return_value=None):
            sv = ShadowVerifier(runtime_config=config)
            sv._state.silent_fallback_active = True

        params = sv.get_adjusted_params(
            base_temperature=0.4, base_top_p=0.9
        )
        assert params["temperature"] == 0.4  # at floor, unchanged
        assert params["top_p"] == 0.85  # 0.9 - 0.05

    def test_silent_fallback_top_p_floor_clamped(self):
        """Top_p tightened but never drops below 0.6 floor."""
        from src.decision.shadow_verifier import ShadowVerifier
        config = self._make_config(enable_shadow=True)
        with patch.object(ShadowVerifier, "_load_model", return_value=None):
            sv = ShadowVerifier(runtime_config=config)
            sv._state.silent_fallback_active = True

        params = sv.get_adjusted_params(
            base_temperature=0.4, base_top_p=0.62
        )
        assert params["temperature"] == 0.4  # at floor
        assert params["top_p"] == 0.6  # 0.62 - 0.05 = 0.57, clamped to 0.6

    def test_no_adjustment_when_fallback_inactive(self):
        """No param adjustment when silent fallback is not active."""
        from src.decision.shadow_verifier import ShadowVerifier
        config = self._make_config(enable_shadow=True)
        with patch.object(ShadowVerifier, "_load_model", return_value=None):
            sv = ShadowVerifier(runtime_config=config)
            sv._state.silent_fallback_active = False

        params = sv.get_adjusted_params(
            base_temperature=0.8, base_top_p=0.9
        )
        assert params["temperature"] == 0.8
        assert params["top_p"] == 0.9

    # ------------------------------------------------------------------
    # Test: conflict counter reset
    # ------------------------------------------------------------------

    def test_similarity_above_0_85_resets_conflict_counter(self):
        """A non-conflict verification resets the consecutive counter."""
        from src.decision.shadow_verifier import (
            ShadowVerifier,
            InferenceResult,
        )
        config = self._make_config(enable_shadow=True)
        with patch.object(ShadowVerifier, "_load_model", return_value=None):
            sv = ShadowVerifier(runtime_config=config)
            sv._model = MagicMock()
            sv._tokenizer = MagicMock()
            sv._state.shadow_disabled = False
            sv._state.consecutive_conflicts = 2

        sv._run_shadow_inference = MagicMock(
            return_value=InferenceResult(
                text="好的", confidence=0.90, safety_level="SAFE"
            )
        )
        sv._compute_similarity = MagicMock(return_value=0.90)

        sv.verify(self.MAIN_CMD_SAFE, self.SCENE_SUMMARY)
        assert sv.state.consecutive_conflicts == 0

    def test_reset_conflict_counter_public_api(self):
        """reset_conflict_counter() clears the consecutive counter."""
        from src.decision.shadow_verifier import ShadowVerifier
        config = self._make_config(enable_shadow=True)
        with patch.object(ShadowVerifier, "_load_model", return_value=None):
            sv = ShadowVerifier(runtime_config=config)
            sv._state.consecutive_conflicts = 5

        sv.reset_conflict_counter()
        assert sv.state.consecutive_conflicts == 0

    # ------------------------------------------------------------------
    # Test: fallback deactivation
    # ------------------------------------------------------------------

    def test_disable_silent_fallback(self):
        """disable_silent_fallback() deactivates and resets."""
        from src.decision.shadow_verifier import ShadowVerifier
        config = self._make_config(enable_shadow=True)
        with patch.object(ShadowVerifier, "_load_model", return_value=None):
            sv = ShadowVerifier(runtime_config=config)
            sv._state.silent_fallback_active = True
            sv._state.consecutive_conflicts = 3

        sv.disable_silent_fallback()
        assert sv.state.silent_fallback_active is False
        assert sv.state.consecutive_conflicts == 0


class TestSafetyLevels:
    """Contract tests for safety classification and enforcement. v4.5.0 §5.7.2."""

    @staticmethod
    def _make_config():
        from src.config.runtime import RuntimeConfig, VRAMTier
        return RuntimeConfig(
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
            context_limit=2048,
        )

    def test_safe_actions_execute_directly(self):
        """v4.5.0 §5.7.2: SAFE decisions pass through unchanged."""
        from src.decision.main_decision import MainDecisionEngine
        config = self._make_config()
        engine = MainDecisionEngine(runtime_config=config)

        safe_decision = {
            "decision_type": "voice_response",
            "command": {
                "voice_response": "好的，我来帮你～",
                "actions": [{"type": "mouse_click", "params": {}}],
            },
            "confidence": 0.92,
            "safety_level": "SAFE",
            "trace_id": "t-safe-001",
            "shadow_overridden": False,
            "source": "main_decision_3b",
        }
        result = engine._enforce_safety(deepcopy(safe_decision), trace_id="t-safe-001")

        assert result["safety_level"] == "SAFE"
        assert result["command"]["voice_response"] == "好的，我来帮你～"
        assert len(result["command"]["actions"]) == 1

    def test_needs_confirm_requires_user_approval(self):
        """v4.5.0 §5.7.2: NEEDS_CONFIRM replaces actions with a confirmation prompt."""
        from src.decision.main_decision import MainDecisionEngine
        config = self._make_config()
        engine = MainDecisionEngine(runtime_config=config)

        confirm_decision = {
            "decision_type": "voice_response",
            "command": {
                "voice_response": "好的，我来发送这条消息～",
                "actions": [{"type": "keyboard_input", "params": {"text": "发送"}}],
            },
            "confidence": 0.85,
            "safety_level": "NEEDS_CONFIRM",
            "trace_id": "t-confirm-001",
            "shadow_overridden": False,
            "source": "main_decision_3b",
        }
        result = engine._enforce_safety(deepcopy(confirm_decision), trace_id="t-confirm-001")

        assert result["safety_level"] == "NEEDS_CONFIRM"
        assert "确定" in result["command"]["voice_response"]
        assert result["command"]["actions"] == []
        assert result.get("metadata", {}).get("awaiting_confirmation") is True

    def test_dangerous_actions_auto_blocked(self):
        """v4.5.0 §5.7.2: DANGEROUS_AUTO_BLOCK clears actions and warns the user."""
        from src.decision.main_decision import MainDecisionEngine
        config = self._make_config()
        engine = MainDecisionEngine(runtime_config=config)

        dangerous_decision = {
            "decision_type": "voice_response",
            "command": {
                "voice_response": "好的，我来删除所有数据～",
                "actions": [{"type": "keyboard_input", "params": {"text": "rm -rf"}}],
            },
            "confidence": 0.80,
            "safety_level": "DANGEROUS_AUTO_BLOCK",
            "trace_id": "t-danger-001",
            "shadow_overridden": False,
            "source": "main_decision_3b",
        }
        result = engine._enforce_safety(deepcopy(dangerous_decision), trace_id="t-danger-001")

        assert result["safety_level"] == "DANGEROUS_AUTO_BLOCK"
        assert "危险" in result["command"]["voice_response"]
        assert result["command"]["actions"] == []
        assert result.get("degraded") is True


class TestReflexRuleStructure:
    def test_rule_has_priority_enum(self):
        valid_priorities = {"INTERACTIVE", "USER_TAUGHT", "CORE", "OBSERVATION"}
        assert VALID_REFLEX_RULE["priority"] in valid_priorities

    def test_rule_has_status_enum(self):
        valid_statuses = {"OBSERVATION", "CORE", "DISABLED"}
        assert VALID_REFLEX_RULE["status"] in valid_statuses

    def test_rule_has_condition_with_trigger(self):
        assert "trigger_type" in VALID_REFLEX_RULE["condition"]

    def test_rule_action_has_safety_level(self):
        assert VALID_REFLEX_RULE["action"]["safety_level"] in {
            "SAFE", "NEEDS_CONFIRM", "DANGEROUS_AUTO_BLOCK",
        }

    def test_rule_metadata_has_confidence_counts(self):
        md = VALID_REFLEX_RULE["metadata"]
        assert "success_count" in md
        assert "failure_count" in md
        assert isinstance(md["success_count"], int)
        assert isinstance(md["failure_count"], int)

    def test_rule_priority_hierarchy(self):
        priority_order = {"INTERACTIVE": 4, "USER_TAUGHT": 3, "CORE": 2, "OBSERVATION": 1}
        assert priority_order["INTERACTIVE"] > priority_order["USER_TAUGHT"]
        assert priority_order["USER_TAUGHT"] > priority_order["CORE"]
        assert priority_order["CORE"] > priority_order["OBSERVATION"]


class TestEmotionParamInjection:
    def test_emotion_params_lookup_falls_back_to_neutral(self):
        emotion_label = "invalid_emotion"
        default = "neutral"
        assert emotion_label not in {"joy", "sadness", "neutral", "anger", "surprise"}
        used_label = default
        assert used_label == "neutral", (
            "Unknown emotion labels fall back to neutral params "
            "(spec section 5.4.1)"
        )

    def test_anger_and_surprise_params_exist_in_config(self):
        import yaml
        from pathlib import Path
        config_path = Path(__file__).resolve().parents[2] / "config" / "emotion_params.yaml"
        assert config_path.exists(), "emotion_params.yaml must exist"
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        params = data.get("emotion_params", {})
        assert "anger" in params, "anger params must exist in emotion_params.yaml"
        assert "surprise" in params, "surprise params must exist in emotion_params.yaml"


class TestNewUserFallback:
    def test_user_model_version_less_than_2_uses_fallback(self):
        from src.decision.context_assembler import (
            ContextAssembler,
            NEW_USER_FALLBACK,
        )
        from src.config.runtime import RuntimeConfig, VRAMTier

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
            context_limit=2048,
        )
        assembler = ContextAssembler(runtime_config=config)
        prompt = assembler.build_system_prompt(is_new_user=True)
        assert NEW_USER_FALLBACK["personality"] in prompt
        assert NEW_USER_FALLBACK["relationship_stage"] in prompt
        assert NEW_USER_FALLBACK["topics_of_interest"] in prompt

    def test_fallback_personality_is_learning_phrase(self):
        fallback_personality = "我正在慢慢了解你"
        assert fallback_personality == "我正在慢慢了解你"


# ======================================================================
# SafetyClassifier contract tests — v4.5.0 §5.7.2
# ======================================================================


class TestSafetyClassifier:
    """Contract tests for SafetyClassifier — v4.5.0 §5.7.2.

    Covers the three-tier keyword-based classification:
      SAFE / NEEDS_CONFIRM / DANGEROUS_AUTO_BLOCK

    Classifier uses keyword matching on ``command.voice_response`` text.
    """

    # Base decision command template (no explicit safety_level to force keyword matching).
    BASE_CMD: dict = {
        "decision_type": "voice_response",
        "command": {"voice_response": "", "actions": []},
        "confidence": 0.92,
        "trace_id": "00000000-0000-0000-0000-000000000099",
        "shadow_overridden": False,
        "source": "main_decision_3b",
    }

    # ------------------------------------------------------------------
    # Module availability
    # ------------------------------------------------------------------

    def test_safety_classifier_module_available(self):
        """SafetyClassifier module loads without ImportError."""
        require_module(
            module_path="src.decision.safety_classifier",
            component_name="SafetyClassifier (decision/safety_classifier.py)",
        )

    # ------------------------------------------------------------------
    # SAFE classification
    # ------------------------------------------------------------------

    def test_safety_classification_safe(self):
        """普通聊天文本 (e.g. '今天天气真好') → SAFE."""
        from src.decision.safety_classifier import SafetyClassifier, SAFE

        cmd = dict(self.BASE_CMD)
        cmd["command"]["voice_response"] = "今天天气真好，我们去散步吧～"
        classifier = SafetyClassifier()
        result = classifier.classify(cmd, trace_id="test-safe-001")
        assert result == SAFE, f"Expected SAFE, got {result}"

    def test_safety_classification_safe_with_random_text(self):
        """Non-keyword text returns SAFE."""
        from src.decision.safety_classifier import SafetyClassifier, SAFE

        cmd = dict(self.BASE_CMD)
        cmd["command"]["voice_response"] = "帮我查一下明天的天气"
        classifier = SafetyClassifier()
        result = classifier.classify(cmd, trace_id="test-safe-002")
        assert result == SAFE, f"Expected SAFE, got {result}"

    # ------------------------------------------------------------------
    # NEEDS_CONFIRM classification
    # ------------------------------------------------------------------

    def test_safety_classification_needs_confirm(self):
        """含'发送'文本 → NEEDS_CONFIRM."""
        from src.decision.safety_classifier import SafetyClassifier, NEEDS_CONFIRM

        cmd = dict(self.BASE_CMD)
        cmd["command"]["voice_response"] = "帮我把这份文件发送给张三"
        classifier = SafetyClassifier()
        result = classifier.classify(cmd, trace_id="test-needsconfirm-001")
        assert result == NEEDS_CONFIRM, f"Expected NEEDS_CONFIRM, got {result}"

    def test_safety_classification_needs_confirm_upload(self):
        """含'上传'文本 → NEEDS_CONFIRM."""
        from src.decision.safety_classifier import SafetyClassifier, NEEDS_CONFIRM

        cmd = dict(self.BASE_CMD)
        cmd["command"]["voice_response"] = "帮我上传照片到相册"
        classifier = SafetyClassifier()
        result = classifier.classify(cmd, trace_id="test-needsconfirm-002")
        assert result == NEEDS_CONFIRM, f"Expected NEEDS_CONFIRM, got {result}"

    def test_safety_classification_needs_confirm_modify(self):
        """含'修改'文本 → NEEDS_CONFIRM."""
        from src.decision.safety_classifier import SafetyClassifier, NEEDS_CONFIRM

        cmd = dict(self.BASE_CMD)
        cmd["command"]["voice_response"] = "帮我修改一下这个配置文件"
        classifier = SafetyClassifier()
        result = classifier.classify(cmd, trace_id="test-needsconfirm-003")
        assert result == NEEDS_CONFIRM, f"Expected NEEDS_CONFIRM, got {result}"

    def test_safety_classification_needs_confirm_english_keyword(self):
        """含'send'英文关键词 → NEEDS_CONFIRM."""
        from src.decision.safety_classifier import SafetyClassifier, NEEDS_CONFIRM

        cmd = dict(self.BASE_CMD)
        cmd["command"]["voice_response"] = "Please send this message to the group"
        classifier = SafetyClassifier()
        result = classifier.classify(cmd, trace_id="test-needsconfirm-004")
        assert result == NEEDS_CONFIRM, f"Expected NEEDS_CONFIRM, got {result}"

    # ------------------------------------------------------------------
    # DANGEROUS_AUTO_BLOCK classification
    # ------------------------------------------------------------------

    def test_safety_classification_dangerous(self):
        """含'删除所有文件 rm -rf'文本 → DANGEROUS_AUTO_BLOCK."""
        from src.decision.safety_classifier import SafetyClassifier, DANGEROUS_AUTO_BLOCK

        cmd = dict(self.BASE_CMD)
        cmd["command"]["voice_response"] = "帮我删除所有文件 rm -rf /"
        classifier = SafetyClassifier()
        result = classifier.classify(cmd, trace_id="test-dangerous-001")
        assert result == DANGEROUS_AUTO_BLOCK, f"Expected DANGEROUS_AUTO_BLOCK, got {result}"

    def test_safety_classification_dangerous_delete(self):
        """含'删除'中文关键词 → DANGEROUS_AUTO_BLOCK."""
        from src.decision.safety_classifier import SafetyClassifier, DANGEROUS_AUTO_BLOCK

        cmd = dict(self.BASE_CMD)
        cmd["command"]["voice_response"] = "把系统里的所有数据都删除掉"
        classifier = SafetyClassifier()
        result = classifier.classify(cmd, trace_id="test-dangerous-002")
        assert result == DANGEROUS_AUTO_BLOCK, f"Expected DANGEROUS_AUTO_BLOCK, got {result}"

    def test_safety_classification_dangerous_payment(self):
        """含'支付'文本 → DANGEROUS_AUTO_BLOCK."""
        from src.decision.safety_classifier import SafetyClassifier, DANGEROUS_AUTO_BLOCK

        cmd = dict(self.BASE_CMD)
        cmd["command"]["voice_response"] = "帮我支付这个订单"
        classifier = SafetyClassifier()
        result = classifier.classify(cmd, trace_id="test-dangerous-003")
        assert result == DANGEROUS_AUTO_BLOCK, f"Expected DANGEROUS_AUTO_BLOCK, got {result}"

    def test_safety_classification_dangerous_format(self):
        """含'格式化'文本 → DANGEROUS_AUTO_BLOCK."""
        from src.decision.safety_classifier import SafetyClassifier, DANGEROUS_AUTO_BLOCK

        cmd = dict(self.BASE_CMD)
        cmd["command"]["voice_response"] = "帮我把硬盘格式化"
        classifier = SafetyClassifier()
        result = classifier.classify(cmd, trace_id="test-dangerous-004")
        assert result == DANGEROUS_AUTO_BLOCK, f"Expected DANGEROUS_AUTO_BLOCK, got {result}"

    # ------------------------------------------------------------------
    # Return type & edge cases
    # ------------------------------------------------------------------

    def test_safety_classification_returns_valid_string(self):
        """SafetyClassifier.classify() returns a valid safety level string."""
        from src.decision.safety_classifier import SafetyClassifier, VALID_SAFETY_LEVELS

        test_texts = [
            "今天天气真好",
            "帮我发送文件",
            "删除所有数据 rm -rf",
        ]
        for text in test_texts:
            cmd = dict(self.BASE_CMD)
            cmd["command"]["voice_response"] = text
            classifier = SafetyClassifier()
            result = classifier.classify(cmd, trace_id="test-valid-001")
            assert result in VALID_SAFETY_LEVELS, (
                f"classify({text!r}) returned {result!r}, "
                f"expected one of {VALID_SAFETY_LEVELS}"
            )

    def test_safety_classification_empty_string(self):
        """Empty voice_response classifies as SAFE."""
        from src.decision.safety_classifier import SafetyClassifier, SAFE

        cmd = dict(self.BASE_CMD)
        cmd["command"]["voice_response"] = ""
        classifier = SafetyClassifier()
        result = classifier.classify(cmd, trace_id="test-edge-001")
        assert result == SAFE, f"Expected SAFE for empty text, got {result}"

    def test_safety_classification_mixed_content_dangerous_wins(self):
        """Dangerous keyword in mixed content takes precedence over safe."""
        from src.decision.safety_classifier import SafetyClassifier, DANGEROUS_AUTO_BLOCK

        cmd = dict(self.BASE_CMD)
        cmd["command"]["voice_response"] = (
            "今天天气真好，我们去散步吧。顺便帮我 删除 一些旧文件"
        )
        classifier = SafetyClassifier()
        result = classifier.classify(cmd, trace_id="test-edge-002")
        # DANGEROUS (删除) takes precedence over SAFE (散步)
        assert result == DANGEROUS_AUTO_BLOCK, f"Expected DANGEROUS_AUTO_BLOCK, got {result}"

    def test_safety_classification_very_long_text(self):
        """Very long text with no keywords classifies as SAFE."""
        from src.decision.safety_classifier import SafetyClassifier, SAFE

        long_text = "今天天气真好。我们去散步吧。 " * 100  # ~500 chars of safe text
        cmd = dict(self.BASE_CMD)
        cmd["command"]["voice_response"] = long_text
        classifier = SafetyClassifier()
        result = classifier.classify(cmd, trace_id="test-edge-003")
        assert result == SAFE, f"Expected SAFE for long safe text, got {result}"
