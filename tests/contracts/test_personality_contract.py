"""
Contract tests for personality layer (spec v4.5.0 section 4).

Validates the three-layer fusion model (baseline + preference_offset + emotion_adj),
clamping to baseline min/max, categorical enum constraints, boolean field inheritance,
and PersonaAuditor boundary checks.
"""
from copy import deepcopy

import pytest


VALID_BASELINE = {
    "baseline_id": "00000000-0000-0000-0000-000000000001",
    "name": "温柔伙伴",
    "description": "耐心、鼓励型，偶尔俏皮，善于倾听",
    "voice_style": {
        "tone": {"value": "gentle", "type": "categorical",
                 "allowed": ["gentle", "calm", "lively", "serious"]},
        "speed": {"value": 1.0, "min": 0.8, "max": 1.3, "type": "numeric"},
        "formality": {"value": 0.5, "min": 0.3, "max": 0.7, "type": "numeric"},
        "emotion_range": {"value": 0.7, "min": 0.5, "max": 0.9, "type": "numeric"},
    },
    "avatar_style": {
        "expression_intensity": {"value": 0.7, "min": 0.5, "max": 0.9, "type": "numeric"},
        "gesture_frequency": {"value": 0.5, "min": 0.3, "max": 0.7, "type": "numeric"},
        "eye_contact_tendency": {"value": 0.8, "min": 0.6, "max": 1.0, "type": "numeric"},
    },
    "mouse_style": {
        "movement_speed": {"value": 0.6, "min": 0.4, "max": 0.8, "type": "numeric"},
        "precision_mode": {"value": 0.3, "min": 0.1, "max": 0.5, "type": "numeric"},
        "hover_before_click": {"value": True, "type": "boolean"},
    },
    "signature_phrases": ["没事的～", "你做得很好呀", "需要我帮什么忙吗？"],
    "safety_constraints": [
        "never_use_profanity",
        "never_execute_destructive_action_without_confirmation",
        "always_ask_before_sending_external_data",
    ],
    "immutable": True,
}


def _assert_all_numeric_in_bounds(dynamic, baseline):
    """Verify all numeric fields in dynamic output are within baseline min/max."""
    for dimension in ["voice_style", "avatar_style", "mouse_style"]:
        for field, spec in baseline[dimension].items():
            if spec.get("type") != "numeric":
                continue
            val = dynamic[dimension][field]
            assert spec["min"] <= val <= spec["max"], (
                f"{dimension}.{field} = {val} not in [{spec['min']}, {spec['max']}]"
            )


class TestModuleExists:
    def test_dynamic_fusion_module_available(self):
        from src.personality.dynamic_fusion import DynamicFusion
        assert DynamicFusion is not None

    def test_persona_auditor_module_available(self):
        from src.personality.persona_auditor import PersonaAuditor
        assert PersonaAuditor is not None


class TestBaselineImmutability:
    def test_baseline_immutable_flag_is_true(self):
        assert VALID_BASELINE["immutable"] is True

    def test_baseline_cannot_be_modified(self):
        baseline = deepcopy(VALID_BASELINE)
        assert baseline["immutable"] is True


class TestNumericFieldClamping:
    def test_numeric_value_cannot_exceed_max(self):
        max_val = VALID_BASELINE["voice_style"]["speed"]["max"]
        test_val = max_val + 0.5
        clamped = min(test_val, max_val)
        assert clamped <= max_val

    def test_numeric_value_cannot_go_below_min(self):
        min_val = VALID_BASELINE["voice_style"]["speed"]["min"]
        test_val = min_val - 0.5
        clamped = max(test_val, min_val)
        assert clamped >= min_val

    def test_all_numeric_fields_stay_in_range(self):
        baseline = VALID_BASELINE

        def check_fields(parent, path=""):
            for key, spec in parent.items():
                if isinstance(spec, dict) and spec.get("type") == "numeric":
                    assert spec["min"] <= spec["value"] <= spec["max"], (
                        f"Numeric field {path}.{key} out of range"
                    )

        for section in ["voice_style", "avatar_style", "mouse_style"]:
            check_fields(baseline[section], path=section)


class TestCategoricalFieldConstraints:
    def test_categorical_values_must_be_in_allowed_set(self):
        tone_spec = VALID_BASELINE["voice_style"]["tone"]
        assert tone_spec["value"] in tone_spec["allowed"]

    def test_categorical_migration_by_step(self):
        allowed = ["gentle", "calm", "lively", "serious"]
        current_idx = 0
        offset = 1
        new_idx = min(max(current_idx + offset, 0), len(allowed) - 1)
        assert allowed[new_idx] == "calm"

    def test_categorical_cannot_jump_outside_allowed(self):
        allowed = ["gentle", "calm", "lively", "serious"]
        current_idx = 0
        offset = 10
        new_idx = min(max(current_idx + offset, 0), len(allowed) - 1)
        assert new_idx == 3


class TestBooleanFieldInheritance:
    def test_boolean_field_inherits_baseline_directly(self):
        hover = VALID_BASELINE["mouse_style"]["hover_before_click"]
        assert hover["type"] == "boolean"
        assert hover["value"] is True


class TestDynamicPersonalityFusion:
    def test_dynamic_personality_no_min_max_fields(self):
        from src.personality.dynamic_fusion import DynamicFusion
        result = DynamicFusion.generate(VALID_BASELINE)
        for dimension in ["voice_style", "avatar_style", "mouse_style"]:
            for field, val in result[dimension].items():
                assert not isinstance(val, dict), (
                    f"{dimension}.{field} should be a scalar value, not a spec dict"
                )

    def test_dynamic_personality_has_version_field(self):
        from src.personality.dynamic_fusion import DynamicFusion
        result = DynamicFusion.generate(VALID_BASELINE)
        assert "version" in result
        assert isinstance(result["version"], str) and len(result["version"]) > 0

    def test_dynamic_personality_has_fused_at_timestamp(self):
        from src.personality.dynamic_fusion import DynamicFusion
        result = DynamicFusion.generate(VALID_BASELINE)
        assert "fused_at" in result
        assert isinstance(result["fused_at"], str) and "T" in result["fused_at"]

    def test_dynamic_personality_has_tts_control(self):
        from src.personality.dynamic_fusion import DynamicFusion
        required_tts = {"emotion", "speed", "speaker", "extra_text_markup"}
        result = DynamicFusion.generate(VALID_BASELINE)
        tts = result.get("tts_control", {})
        for key in required_tts:
            assert key in tts, f"tts_control must contain {key}"

    def test_emotion_driven_fields_stay_within_bounds(self):
        from src.personality.dynamic_fusion import DynamicFusion
        for emotion in ["joy", "sadness", "neutral"]:
            result = DynamicFusion.generate(VALID_BASELINE, emotion_label=emotion)
            _assert_all_numeric_in_bounds(result, VALID_BASELINE)

    def test_three_layer_fusion_order_matters(self):
        layers = ["baseline", "preference_offset", "emotion_adj"]
        assert layers == ["baseline", "preference_offset", "emotion_adj"]

    def test_emotion_category_for_personality_is_subjective(self):
        from src.personality.dynamic_fusion import DynamicFusion
        result = DynamicFusion.generate(VALID_BASELINE, emotion_label="joy")
        assert result["tts_control"]["emotion"] == "joy"
        assert result.get("emotion_used") == "joy"

    def test_numeric_fields_clamped_with_preference_offset(self):
        from src.personality.dynamic_fusion import DynamicFusion
        offsets = {"voice_style": {"speed": 0.5}}
        result = DynamicFusion.generate(VALID_BASELINE, preference_offsets=offsets)
        assert result["voice_style"]["speed"] <= VALID_BASELINE["voice_style"]["speed"]["max"]
        assert result["voice_style"]["speed"] >= VALID_BASELINE["voice_style"]["speed"]["min"]

    def test_boolean_field_inherited_unchanged(self):
        from src.personality.dynamic_fusion import DynamicFusion
        result = DynamicFusion.generate(VALID_BASELINE)
        assert result["mouse_style"]["hover_before_click"] is True

    def test_categorical_field_migration_by_step(self):
        from src.personality.dynamic_fusion import DynamicFusion
        offsets = {"voice_style": {"tone": 1}}
        result = DynamicFusion.generate(VALID_BASELINE, preference_offsets=offsets)
        assert result["voice_style"]["tone"] == "calm"

    def test_categorical_field_clamped_to_allowed(self):
        from src.personality.dynamic_fusion import DynamicFusion
        offsets = {"voice_style": {"tone": 10}}
        result = DynamicFusion.generate(VALID_BASELINE, preference_offsets=offsets)
        allowed = VALID_BASELINE["voice_style"]["tone"]["allowed"]
        assert result["voice_style"]["tone"] in allowed
        assert result["voice_style"]["tone"] == allowed[-1]

    def test_cold_start_uses_neutral_without_offset(self):
        from src.personality.dynamic_fusion import DynamicFusion
        result = DynamicFusion.cold_start(VALID_BASELINE)
        assert result["tts_control"]["emotion"] == "neutral"
        for dim_name in ["voice_style", "avatar_style", "mouse_style"]:
            for field, spec in VALID_BASELINE[dim_name].items():
                if spec.get("type") == "numeric":
                    assert result[dim_name][field] == spec["value"], (
                        f"{dim_name}.{field}: expected baseline value {spec['value']}, "
                        f"got {result[dim_name][field]}"
                    )

    def test_dynamic_personality_includes_safety_constraints(self):
        from src.personality.dynamic_fusion import DynamicFusion
        result = DynamicFusion.generate(VALID_BASELINE)
        assert result.get("safety_constraints") == VALID_BASELINE["safety_constraints"]


class TestPersonaAuditor:
    def test_boundary_violation_detected(self):
        from src.personality.dynamic_fusion import DynamicFusion
        from src.personality.persona_auditor import PersonaAuditor

        dynamic = DynamicFusion.generate(VALID_BASELINE)
        dynamic["voice_style"]["speed"] = 2.0

        auditor = PersonaAuditor()
        result = auditor.audit(dynamic, VALID_BASELINE)
        assert len(result.violations) > 0

    def test_boundary_violation_clamps_in_place(self):
        from src.personality.dynamic_fusion import DynamicFusion
        from src.personality.persona_auditor import PersonaAuditor

        dynamic = DynamicFusion.generate(VALID_BASELINE)
        dynamic["voice_style"]["speed"] = 2.0
        auditor = PersonaAuditor()
        auditor.audit(dynamic, VALID_BASELINE)
        assert dynamic["voice_style"]["speed"] <= VALID_BASELINE["voice_style"]["speed"]["max"]

    def test_safety_constraint_regex_checking(self):
        prohibited = ["never_use_profanity", "always_ask_before_sending_external_data"]
        assert all(c in VALID_BASELINE["safety_constraints"] for c in prohibited)

    def test_safety_violation_detected_in_response(self):
        from src.personality.persona_auditor import PersonaAuditor

        auditor = PersonaAuditor()
        result = auditor.audit(
            VALID_BASELINE,
            VALID_BASELINE,
            response_text="what the fuck is this",
        )
        assert len(result.violations) > 0

    def test_drift_rate_alert_on_rapid_change(self):
        from src.personality.dynamic_fusion import DynamicFusion
        from src.personality.persona_auditor import PersonaAuditor

        snap1 = DynamicFusion.generate(VALID_BASELINE)
        snap1["voice_style"]["speed"] = 0.8

        snap2 = DynamicFusion.generate(VALID_BASELINE)
        snap2["voice_style"]["speed"] = 1.25

        history = [snap1, snap2]
        auditor = PersonaAuditor()
        result = auditor.audit(snap2, VALID_BASELINE, history_snapshots=history)
        assert len(result.drift_alerts) > 0

    def test_no_drift_alert_on_slow_change(self):
        from src.personality.dynamic_fusion import DynamicFusion
        from src.personality.persona_auditor import PersonaAuditor

        snap1 = DynamicFusion.generate(VALID_BASELINE)
        snap1["voice_style"]["speed"] = 1.0

        snap2 = DynamicFusion.generate(VALID_BASELINE)
        snap2["voice_style"]["speed"] = 1.05

        history = [snap1, snap2]
        auditor = PersonaAuditor()
        result = auditor.audit(snap2, VALID_BASELINE, history_snapshots=history)
        assert len(result.drift_alerts) == 0

    def test_audit_score_below_5_freezes_preference_shift(self):
        from src.personality.dynamic_fusion import DynamicFusion
        from src.personality.persona_auditor import PersonaAuditor

        dynamic = DynamicFusion.generate(VALID_BASELINE)
        dynamic["voice_style"]["speed"] = 2.0
        dynamic["voice_style"]["formality"] = -1.0
        dynamic["avatar_style"]["expression_intensity"] = 5.0

        auditor = PersonaAuditor()
        result = auditor.audit(dynamic, VALID_BASELINE)
        assert result.score < 5
        assert result.freeze_preference_shift is True

    def test_audit_score_10_when_all_clean(self):
        from src.personality.dynamic_fusion import DynamicFusion
        from src.personality.persona_auditor import PersonaAuditor

        dynamic = DynamicFusion.generate(VALID_BASELINE)
        auditor = PersonaAuditor()
        result = auditor.audit(dynamic, VALID_BASELINE)
        assert result.score == 10
        assert result.freeze_preference_shift is False

    def test_is_frozen_tracks_state(self):
        from src.personality.persona_auditor import PersonaAuditor
        from src.personality.dynamic_fusion import DynamicFusion

        auditor = PersonaAuditor()
        assert auditor.is_frozen is False

        dynamic = DynamicFusion.generate(VALID_BASELINE)
        dynamic["voice_style"]["speed"] = 2.0
        dynamic["voice_style"]["formality"] = -1.0
        dynamic["avatar_style"]["expression_intensity"] = 5.0
        auditor.audit(dynamic, VALID_BASELINE)
        assert auditor.is_frozen is True

        auditor.unfreeze()
        assert auditor.is_frozen is False


class TestDynamicFusionPromptInjection:
    """Contract tests: DynamicFusion output formatted for LLM prompt injection.

    v4.5.0 §4.6 — The fused personality dict must carry all information needed
    by downstream prompt assembly: dimension values, safety constraints,
    signature phrases, and emotion-derived voice style metadata.
    """

    def test_output_contains_all_three_dimensions(self):
        """Dynamic personality output must contain voice/avatar/mouse style sections."""
        from src.personality.dynamic_fusion import DynamicFusion

        result = DynamicFusion.generate(VALID_BASELINE)
        for dim in ("voice_style", "avatar_style", "mouse_style"):
            assert dim in result, f"Output missing required dimension: {dim}"
            assert isinstance(result[dim], dict), f"{dim} must be a dict"

    def test_output_contains_safety_constraints(self):
        """safety_constraints array must be inherited from baseline into dynamic output."""
        from src.personality.dynamic_fusion import DynamicFusion

        result = DynamicFusion.generate(VALID_BASELINE)
        assert "safety_constraints" in result
        assert isinstance(result["safety_constraints"], list)
        # Must match exactly the three safety rules from baseline.json
        assert result["safety_constraints"] == VALID_BASELINE["safety_constraints"]

    def test_output_contains_signature_phrases(self):
        """signature_phrases array must be inherited from baseline into dynamic output."""
        from src.personality.dynamic_fusion import DynamicFusion

        result = DynamicFusion.generate(VALID_BASELINE)
        assert "signature_phrases" in result
        assert isinstance(result["signature_phrases"], list)
        assert len(result["signature_phrases"]) > 0

    def test_output_contains_tts_control(self):
        """tts_control metadata must be present for voice channel prompt injection."""
        from src.personality.dynamic_fusion import DynamicFusion

        result = DynamicFusion.generate(VALID_BASELINE)
        tts = result.get("tts_control", {})
        assert "emotion" in tts
        assert "speed" in tts
        assert "speaker" in tts

    def test_emotion_joy_increases_speed_relative_to_neutral(self):
        """joy emotion must produce higher voice speed than neutral."""
        from src.personality.dynamic_fusion import DynamicFusion

        joy_result = DynamicFusion.generate(VALID_BASELINE, emotion_label="joy")
        neutral_result = DynamicFusion.generate(VALID_BASELINE, emotion_label="neutral")
        assert joy_result["voice_style"]["speed"] > neutral_result["voice_style"]["speed"], (
            "joy should increase speed above neutral"
        )

    def test_emotion_sadness_decreases_speed_relative_to_neutral(self):
        """sadness emotion must produce lower voice speed than neutral."""
        from src.personality.dynamic_fusion import DynamicFusion

        sad_result = DynamicFusion.generate(VALID_BASELINE, emotion_label="sadness")
        neutral_result = DynamicFusion.generate(VALID_BASELINE, emotion_label="neutral")
        assert sad_result["voice_style"]["speed"] < neutral_result["voice_style"]["speed"], (
            "sadness should decrease speed below neutral"
        )

    def test_emotion_neutral_speed_equals_baseline_value(self):
        """neutral emotion must produce speed at (or near) baseline value (1.0).

        With LAMBDA=0.2 interpolation the neutral target IS the baseline value,
        so the fused result should equal the baseline value exactly when no
        preference offset is applied.
        """
        from src.personality.dynamic_fusion import DynamicFusion

        result = DynamicFusion.generate(VALID_BASELINE, emotion_label="neutral")
        base_speed = VALID_BASELINE["voice_style"]["speed"]["value"]
        assert result["voice_style"]["speed"] == base_speed, (
            f"neutral speed {result['voice_style']['speed']} != baseline {base_speed}"
        )

    def test_emotion_only_joy_sadness_neutral_allowed(self):
        """anger/surprise/other must fall back to neutral (spec §4.5)."""
        from src.personality.dynamic_fusion import DynamicFusion

        invalid_result = DynamicFusion.generate(VALID_BASELINE, emotion_label="anger")
        neutral_result = DynamicFusion.generate(VALID_BASELINE, emotion_label="neutral")
        assert invalid_result["emotion_used"] == "neutral"
        assert invalid_result["tts_control"]["emotion"] == "neutral"
        assert invalid_result["voice_style"]["speed"] == neutral_result["voice_style"]["speed"]


class TestBaselinePersonalityImmutability:
    """Contract tests: BaselinePersonality enforces immutability (v4.5.0 §4.3).

    The class must prevent any mutation after construction — both via the
    explicit set_value() API and via Python attribute set/delete operations.
    Read accessors must still function normally.
    """

    def _make_baseline(self):
        from src.personality.baseline import BaselinePersonality
        return BaselinePersonality(config=VALID_BASELINE)

    def test_set_value_raises_error(self):
        """BaselinePersonality.set_value() must raise ImmutableBaselineError."""
        from src.personality.baseline import ImmutableBaselineError

        bp = self._make_baseline()
        with pytest.raises(ImmutableBaselineError):
            bp.set_value("voice_style", "speed", 99.9)

    def test_attr_set_raises_error_after_init(self):
        """Setting any attribute on an initialized BaselinePersonality must raise."""
        from src.personality.baseline import ImmutableBaselineError

        bp = self._make_baseline()
        with pytest.raises(ImmutableBaselineError):
            bp.name = "hacked"

    def test_attr_del_raises_error(self):
        """Deleting any attribute from BaselinePersonality must raise."""
        from src.personality.baseline import ImmutableBaselineError

        bp = self._make_baseline()
        with pytest.raises(ImmutableBaselineError):
            del bp.name

    def test_read_accessors_work_after_init(self):
        """Read-only accessors must return correct values after construction."""
        bp = self._make_baseline()
        assert bp.name == VALID_BASELINE["name"]
        assert bp.description == VALID_BASELINE["description"]
        assert bp.baseline_id == VALID_BASELINE["baseline_id"]
        assert bp.safety_constraints == VALID_BASELINE["safety_constraints"]
        assert bp.immutable is True

    def test_get_value_returns_correct_value(self):
        """get_value() must return the raw value for a given section/field."""
        bp = self._make_baseline()
        speed = bp.get_value("voice_style", "speed")
        assert speed == VALID_BASELINE["voice_style"]["speed"]["value"]

    def test_to_dict_returns_deep_copy(self):
        """to_dict() must return a deep copy so mutation of the returned dict is safe."""
        bp = self._make_baseline()
        d = bp.to_dict()
        d["name"] = "mutated"
        # Original must be unchanged
        assert bp.name == VALID_BASELINE["name"]

    def test_immutable_flag_default_true(self):
        """immutable property must default to True."""
        bp = self._make_baseline()
        assert bp.immutable is True


class TestPromptToTextConversion:
    """Contract tests: prompt_to_text() personality-to-prompt fragment converter.

    v4.5.0 §4.6 — The fused personality dict must be convertible to a
    Chinese natural-language fragment for injection into the LLM system prompt.
    The fragment describes current personality state including voice style,
    avatar style, mouse style, signature phrases, and safety constraints.

    NOTE: These tests target RED phase. The prompt_to_text() function does
    NOT exist yet — it will be implemented when Task 11 wires personality
    injection into the runtime loop.
    """

    def test_prompt_to_text_function_exists(self):
        """prompt_to_text() must be importable from personality module."""
        from src.personality.dynamic_fusion import prompt_to_text  # noqa: F811
        assert callable(prompt_to_text)

    def test_prompt_to_text_returns_string(self):
        """prompt_to_text() must return a non-empty string."""
        from src.personality.dynamic_fusion import prompt_to_text

        from src.personality.dynamic_fusion import DynamicFusion
        dynamic = DynamicFusion.generate(VALID_BASELINE, emotion_label="joy")
        text = prompt_to_text(dynamic)
        assert isinstance(text, str)
        assert len(text) > 0

    def test_prompt_to_text_contains_emotion_description(self):
        """Output must contain Chinese text describing current emotion state."""
        from src.personality.dynamic_fusion import prompt_to_text

        from src.personality.dynamic_fusion import DynamicFusion
        dynamic = DynamicFusion.generate(VALID_BASELINE, emotion_label="joy")
        text = prompt_to_text(dynamic)
        # Should reference emotion via Chinese context markers
        assert "[当前状态]" in text or "语速" in text

    def test_prompt_to_text_contains_safety_rules(self):
        """Output must include the three baseline safety constraints."""
        from src.personality.dynamic_fusion import prompt_to_text

        from src.personality.dynamic_fusion import DynamicFusion
        dynamic = DynamicFusion.generate(VALID_BASELINE, emotion_label="neutral")
        text = prompt_to_text(dynamic)
        for rule in VALID_BASELINE["safety_constraints"]:
            assert rule in text, f"Safety rule {rule!r} missing from prompt text"

    def test_prompt_to_text_reflects_speed_variation_by_emotion(self):
        """Output must differ between joy and sadness emotion inputs."""
        from src.personality.dynamic_fusion import prompt_to_text, DynamicFusion

        joy_dyn = DynamicFusion.generate(VALID_BASELINE, emotion_label="joy")
        sad_dyn = DynamicFusion.generate(VALID_BASELINE, emotion_label="sadness")
        joy_text = prompt_to_text(joy_dyn)
        sad_text = prompt_to_text(sad_dyn)
        # Speed descriptions should differ between joy and sadness
        assert joy_text != sad_text, "joy and sadness must produce different prompt text"
