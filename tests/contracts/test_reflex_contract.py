"""
Contract tests for Reflex Rule Engine (spec v4.5.0 section 5.3).

RED phase: RuleEngine at src/decision/reflex/rule_engine.py is NOT yet implemented.
All tests should FAIL with ImportError until implementation.

Validates:
  - Rule loading from JSON dicts/filesystem
  - 3-tier priority resolution (INTERACTIVE > USER_TAUGHT > CORE)
  - Regex pattern matching with context_constraints filtering
  - USER_TAUGHT rule state machine: OBSERVATION -> CORE after threshold
  - Matched rule output passes SafetyClassifier
"""
from __future__ import annotations

import uuid
import pytest
from unittest.mock import MagicMock, patch
from typing import Any

from tests.contracts import require_module, fail_red


# ---------------------------------------------------------------------------
# Constants — spec v4.5.0 §5.3.1
# ---------------------------------------------------------------------------

# Priority values (spec §5.3.1: "priority": "INTERACTIVE=4 | USER_TAUGHT=3 | CORE=2 | OBSERVATION=1")
PRIORITY_INTERACTIVE: int = 4
PRIORITY_USER_TAUGHT: int = 3
PRIORITY_CORE: int = 2
PRIORITY_OBSERVATION: int = 1

# Status values (spec §5.3.1: "status": "OBSERVATION | CORE | DISABLED")
STATUS_OBSERVATION: str = "OBSERVATION"
STATUS_CORE: str = "CORE"
STATUS_DISABLED: str = "DISABLED"

# Threshold: number of observation hits before OBSERVATION -> CORE transition
# (spec §5.6.1: "同一类 Scene + 同一类决策结果出现 ≥ 3 次")
OBSERVATION_THRESHOLD: int = 3

# Valid action types (spec §5.3.1)
VALID_ACTION_TYPES: frozenset[str] = frozenset({
    "mouse_click", "mouse_move", "keyboard_input",
    "voice_response", "composite", "animation_trigger",
})

# Valid safety levels (spec §5.3.1 / §5.7.2)
VALID_SAFETY_LEVELS: frozenset[str] = frozenset({"SAFE", "NEEDS_CONFIRM", "DANGEROUS_AUTO_BLOCK"})


# ---------------------------------------------------------------------------
# Factory helpers — produce spec-compliant rule dicts
# ---------------------------------------------------------------------------

def make_rule(
    *,
    name: str = "test_rule",
    priority: int = PRIORITY_CORE,
    status: str = STATUS_CORE,
    trigger_type: str = "voice_command",
    pattern: str = ".*",
    context_constraints: list[str] | None = None,
    action_type: str = "voice_response",
    action_params: dict[str, Any] | None = None,
    safety_level: str = "SAFE",
    confidence: float = 0.92,
    source: str = "system_default",
    observation_remaining: int = 0,
) -> dict[str, Any]:
    """Return a rule dict matching spec v4.5.0 §5.3.1."""
    return {
        "rule_id": str(uuid.uuid4()),
        "name": name,
        "priority": priority,
        "status": status,
        "condition": {
            "trigger_type": trigger_type,
            "pattern": pattern,
            "context_constraints": context_constraints or [],
        },
        "action": {
            "type": action_type,
            "params": action_params or {},
            "safety_level": safety_level,
        },
        "template_id": None,
        "template_slots": {},
        "cluster_hint": None,
        "metadata": {
            "confidence": confidence,
            "success_count": 0,
            "failure_count": 0,
            "created_at": "2026-05-14T00:00:00.000+00:00",
            "last_verified_at": "2026-05-14T00:00:00.000+00:00",
            "source": source,
            "observation_remaining": observation_remaining,
        },
    }


def make_scene_context(
    *,
    entities: list[str] | None = None,
    scene_type: str | None = None,
    emotion: str | None = None,
) -> dict[str, Any]:
    """Return a minimal scene-context dict for constraint matching."""
    return {
        "entities": {e: {"name": e} for e in (entities or [])},
        "scene_type": scene_type or "desktop",
        "emotion": emotion or "neutral",
    }


# ---------------------------------------------------------------------------
# Test Classes
# ---------------------------------------------------------------------------


class TestRuleLoading:
    """RuleEngine loads rules from JSON and validates format."""

    RULES_JSON = [
        make_rule(name="greeting", pattern=r"你好", priority=PRIORITY_CORE),
        make_rule(
            name="click_save",
            pattern=r"保存",
            priority=PRIORITY_USER_TAUGHT,
            action_type="mouse_click",
            safety_level="SAFE",
        ),
    ]

    def test_module_exists(self):
        """src.decision.reflex.rule_engine must exist."""
        require_module(
            module_path="src.decision.reflex.rule_engine",
            component_name="RuleEngine (decision/reflex/rule_engine.py)",
        )

    def test_load_from_dicts(self):
        """RuleEngine can be constructed with inline rule dicts."""
        from src.decision.reflex.rule_engine import RuleEngine
        engine = RuleEngine(rules=self.RULES_JSON)
        assert len(engine.get_rules()) == 2  # type: ignore[attr-defined]

    def test_load_rules_returns_rule_objects(self):
        """Loaded rules preserve all spec §5.3.1 fields."""
        from src.decision.reflex.rule_engine import RuleEngine
        engine = RuleEngine(rules=self.RULES_JSON)
        for rule in engine.get_rules():  # type: ignore[attr-defined]
            assert "rule_id" in rule
            assert "name" in rule
            assert "priority" in rule
            assert "status" in rule
            assert "condition" in rule
            assert "action" in rule
            assert "metadata" in rule

    def test_load_with_empty_rules(self):
        """Empty rule list should not crash."""
        from src.decision.reflex.rule_engine import RuleEngine
        engine = RuleEngine(rules=[])
        assert len(engine.get_rules()) == 0  # type: ignore[attr-defined]

    def test_load_from_json_path(self):
        """RuleEngine loads from a JSON file path if rules not given inline."""
        from src.decision.reflex.rule_engine import RuleEngine
        engine = RuleEngine(rules_path="rules/core_rules.json")
        loaded = engine.get_rules()  # type: ignore[attr-defined]
        assert isinstance(loaded, list)

    def test_skip_invalid_rule_entries(self):
        """Malformed entries are skipped with a warning, not raising."""
        from src.decision.reflex.rule_engine import RuleEngine
        bad_rules = [
            make_rule(name="good"),
            {"not_a_rule": True},  # missing required fields
            make_rule(name="also_good", pattern=r"hello"),
        ]
        engine = RuleEngine(rules=bad_rules)
        valid = engine.get_rules()  # type: ignore[attr-defined]
        # At least the valid ones should be loaded
        assert len(valid) >= 2

    def test_rule_has_required_action_fields(self):
        """Action dict must have type, params, safety_level."""
        from src.decision.reflex.rule_engine import RuleEngine
        engine = RuleEngine(rules=[make_rule()])
        rule = engine.get_rules()[0]  # type: ignore[attr-defined]
        action = rule["action"]
        assert action["type"] in VALID_ACTION_TYPES
        assert isinstance(action["params"], dict)
        assert action["safety_level"] in VALID_SAFETY_LEVELS


class TestPriorityResolution:
    """Multiple matching rules -> highest priority wins.  v4.5.0 §5.3.1/§5.6.2."""

    # Rules that all match "帮我保存" but at different priority tiers
    GREETING_MATCH_ALL = make_rule(
        name="greeting",
        pattern=r".*",
        priority=PRIORITY_CORE,
        action_params={"voice_response": "你好"},
    )
    SAVE_INTERACTIVE = make_rule(
        name="save_interactive",
        pattern=r"保存",
        priority=PRIORITY_INTERACTIVE,
        action_type="mouse_click",
        action_params={"target": "save_button"},
    )
    SAVE_USER_TAUGHT = make_rule(
        name="save_taught",
        pattern=r"保存",
        priority=PRIORITY_USER_TAUGHT,
        action_type="mouse_click",
    )
    SAVE_CORE = make_rule(
        name="save_core",
        pattern=r"保存",
        priority=PRIORITY_CORE,
        action_type="voice_response",
    )

    def test_interactive_over_user_taught(self):
        """INTERACTIVE(4) beats USER_TAUGHT(3)."""
        from src.decision.reflex.rule_engine import RuleEngine
        engine = RuleEngine(rules=[self.SAVE_USER_TAUGHT, self.SAVE_INTERACTIVE])
        result = engine.match("帮我保存", trace_id="t1")  # type: ignore[attr-defined]
        assert result is not None
        assert result["rule"]["name"] == "save_interactive"

    def test_user_taught_over_core(self):
        """USER_TAUGHT(3) beats CORE(2)."""
        from src.decision.reflex.rule_engine import RuleEngine
        engine = RuleEngine(rules=[self.SAVE_CORE, self.SAVE_USER_TAUGHT])
        result = engine.match("帮我保存", trace_id="t2")  # type: ignore[attr-defined]
        assert result is not None
        assert result["rule"]["name"] == "save_taught"

    def test_interactive_over_core(self):
        """INTERACTIVE(4) beats CORE(2)."""
        from src.decision.reflex.rule_engine import RuleEngine
        engine = RuleEngine(rules=[self.SAVE_CORE, self.SAVE_INTERACTIVE])
        result = engine.match("帮我保存", trace_id="t3")  # type: ignore[attr-defined]
        assert result is not None
        assert result["rule"]["name"] == "save_interactive"

    def test_same_priority_higher_confidence_wins(self):
        """At equal priority, higher metadata.confidence wins.  v4.5.0 §5.6.2."""
        from src.decision.reflex.rule_engine import RuleEngine
        low_conf = make_rule(
            name="low_conf",
            pattern=r"保存",
            priority=PRIORITY_CORE,
            confidence=0.6,
        )
        high_conf = make_rule(
            name="high_conf",
            pattern=r"保存",
            priority=PRIORITY_CORE,
            confidence=0.95,
        )
        engine = RuleEngine(rules=[low_conf, high_conf])
        result = engine.match("保存", trace_id="t4")  # type: ignore[attr-defined]
        assert result is not None
        assert result["rule"]["name"] == "high_conf"

    def test_observation_lowest_priority(self):
        """OBSERVATION(1) is lowest and loses to CORE(2)."""
        from src.decision.reflex.rule_engine import RuleEngine
        core = make_rule(
            name="core_rule",
            pattern=r".*",
            priority=PRIORITY_CORE,
            status=STATUS_CORE,
        )
        obs = make_rule(
            name="obs_rule",
            pattern=r".*",
            priority=PRIORITY_OBSERVATION,
            status=STATUS_OBSERVATION,
            observation_remaining=5,
        )
        engine = RuleEngine(rules=[obs, core])
        result = engine.match("hello", trace_id="t5")  # type: ignore[attr-defined]
        assert result is not None
        assert result["rule"]["name"] == "core_rule"


class TestPatternMatching:
    """Regex patterns match correct inputs, reject non-matching."""

    def test_exact_match(self):
        """Simple regex matches input."""
        from src.decision.reflex.rule_engine import RuleEngine
        rule = make_rule(name="greet", pattern=r"^你好$", priority=PRIORITY_CORE)
        engine = RuleEngine(rules=[rule])
        result = engine.match("你好", trace_id="t1")  # type: ignore[attr-defined]
        assert result is not None

    def test_non_match_returns_none(self):
        """Non-matching input returns None."""
        from src.decision.reflex.rule_engine import RuleEngine
        rule = make_rule(name="greet", pattern=r"^你好$", priority=PRIORITY_CORE)
        engine = RuleEngine(rules=[rule])
        result = engine.match("Hello", trace_id="t2")  # type: ignore[attr-defined]
        assert result is None

    def test_partial_match(self):
        """Regex can match substrings (no ^/$ anchor)."""
        from src.decision.reflex.rule_engine import RuleEngine
        rule = make_rule(name="save_pattern", pattern=r"保存", priority=PRIORITY_CORE)
        engine = RuleEngine(rules=[rule])
        result = engine.match("帮我保存这个文件", trace_id="t3")  # type: ignore[attr-defined]
        assert result is not None

    def test_multiple_rules_only_matched_one(self):
        """Only matching rules are candidates; non-matching are ignored."""
        from src.decision.reflex.rule_engine import RuleEngine
        r1 = make_rule(name="english", pattern=r"hello", priority=PRIORITY_CORE)
        r2 = make_rule(name="chinese", pattern=r"你好", priority=PRIORITY_CORE)
        engine = RuleEngine(rules=[r1, r2])
        result = engine.match("hello", trace_id="t4")  # type: ignore[attr-defined]
        assert result is not None
        assert result["rule"]["name"] == "english"

    def test_chinese_regex_match(self):
        """Unicode (Chinese) patterns work correctly with re module."""
        from src.decision.reflex.rule_engine import RuleEngine
        rule = make_rule(name="bom_scan", pattern=r"扫描|识别|检测", priority=PRIORITY_CORE)
        engine = RuleEngine(rules=[rule])
        assert engine.match("扫描二维码", trace_id="t5") is not None  # type: ignore[attr-defined]
        assert engine.match("识别物体", trace_id="t6") is not None  # type: ignore[attr-defined]
        assert engine.match("检测温度", trace_id="t7") is not None  # type: ignore[attr-defined]
        assert engine.match("你好世界", trace_id="t8") is None  # type: ignore[attr-defined]

    def test_disabled_rule_skipped(self):
        """Status=DISABLED rules are not matched."""
        from src.decision.reflex.rule_engine import RuleEngine
        disabled = make_rule(
            name="disabled_greet",
            pattern=r"你好",
            priority=PRIORITY_CORE,
            status=STATUS_DISABLED,
        )
        enabled = make_rule(name="enabled_greet", pattern=r"你好", priority=PRIORITY_CORE)
        engine = RuleEngine(rules=[disabled, enabled])
        result = engine.match("你好", trace_id="t9")  # type: ignore[attr-defined]
        assert result is not None
        assert result["rule"]["name"] == "enabled_greet"


class TestContextConstraints:
    """Match only when context_constraints are satisfied.  v4.5.0 §5.3.1."""

    def test_no_constraints_matches_always(self):
        """Empty context_constraints = always match."""
        from src.decision.reflex.rule_engine import RuleEngine
        rule = make_rule(name="always", pattern=r".*", priority=PRIORITY_CORE)
        engine = RuleEngine(rules=[rule])
        ctx = make_scene_context()
        result = engine.match("anything", scene_context=ctx, trace_id="t1")  # type: ignore[attr-defined]
        assert result is not None

    def test_entity_exists_satisfied(self):
        """entity_exists:X constraints pass when X is in scene entities."""
        from src.decision.reflex.rule_engine import RuleEngine
        rule = make_rule(
            name="check_icon",
            pattern=r"点那个",
            priority=PRIORITY_CORE,
            context_constraints=["entity_exists:save_button"],
        )
        engine = RuleEngine(rules=[rule])
        ctx = make_scene_context(entities=["save_button"])
        result = engine.match("点那个", scene_context=ctx, trace_id="t2")  # type: ignore[attr-defined]
        assert result is not None

    def test_entity_exists_not_satisfied(self):
        """entity_exists:X constraints fail when X not in scene."""
        from src.decision.reflex.rule_engine import RuleEngine
        rule = make_rule(
            name="check_icon",
            pattern=r"点那个",
            priority=PRIORITY_CORE,
            context_constraints=["entity_exists:save_button"],
        )
        engine = RuleEngine(rules=[rule])
        ctx = make_scene_context(entities=[])
        result = engine.match("点那个", scene_context=ctx, trace_id="t3")  # type: ignore[attr-defined]
        assert result is None

    def test_scene_type_satisfied(self):
        """scene_type:Y constraints pass when scene matches."""
        from src.decision.reflex.rule_engine import RuleEngine
        rule = make_rule(
            name="desktop_only",
            pattern=r"打开",
            priority=PRIORITY_CORE,
            context_constraints=["scene_type:desktop"],
        )
        engine = RuleEngine(rules=[rule])
        ctx = make_scene_context(scene_type="desktop")
        result = engine.match("打开文件", scene_context=ctx, trace_id="t4")  # type: ignore[attr-defined]
        assert result is not None

    def test_scene_type_not_satisfied(self):
        """scene_type:Y constraints fail when scene mismatches."""
        from src.decision.reflex.rule_engine import RuleEngine
        rule = make_rule(
            name="desktop_only",
            pattern=r"打开",
            priority=PRIORITY_CORE,
            context_constraints=["scene_type:desktop"],
        )
        engine = RuleEngine(rules=[rule])
        ctx = make_scene_context(scene_type="browser")
        result = engine.match("打开文件", scene_context=ctx, trace_id="t5")  # type: ignore[attr-defined]
        assert result is None

    def test_emotion_constraint_satisfied(self):
        """emotion:Z constraints pass when user emotion matches."""
        from src.decision.reflex.rule_engine import RuleEngine
        rule = make_rule(
            name="comfort",
            pattern=r"难过",
            priority=PRIORITY_CORE,
            context_constraints=["emotion:sadness"],
        )
        engine = RuleEngine(rules=[rule])
        ctx = make_scene_context(emotion="sadness")
        result = engine.match("我很难过", scene_context=ctx, trace_id="t6")  # type: ignore[attr-defined]
        assert result is not None

    def test_emotion_constraint_not_satisfied(self):
        """emotion:Z constraints fail when emotion mismatches."""
        from src.decision.reflex.rule_engine import RuleEngine
        rule = make_rule(
            name="comfort",
            pattern=r"难过",
            priority=PRIORITY_CORE,
            context_constraints=["emotion:sadness"],
        )
        engine = RuleEngine(rules=[rule])
        ctx = make_scene_context(emotion="joy")
        result = engine.match("我很难过", scene_context=ctx, trace_id="t7")  # type: ignore[attr-defined]
        assert result is None

    def test_multiple_constraints_all_must_pass(self):
        """All context_constraints must be satisfied for a match."""
        from src.decision.reflex.rule_engine import RuleEngine
        rule = make_rule(
            name="specific",
            pattern=r"点保存",
            priority=PRIORITY_CORE,
            context_constraints=[
                "entity_exists:save_button",
                "scene_type:editor",
                "emotion:neutral",
            ],
        )
        engine = RuleEngine(rules=[rule])
        # All three satisfied
        ctx_all = make_scene_context(
            entities=["save_button"], scene_type="editor", emotion="neutral",
        )
        assert engine.match("点保存", scene_context=ctx_all, trace_id="t8") is not None  # type: ignore[attr-defined]
        # One fails
        ctx_missing_entity = make_scene_context(
            entities=[], scene_type="editor", emotion="neutral",
        )
        assert engine.match("点保存", scene_context=ctx_missing_entity, trace_id="t9") is None  # type: ignore[attr-defined]


class TestUserTaughtStateMachine:
    """OBSERVATION -> CORE transition after observation_threshold hits.

    Spec v4.5.0 §5.6.1: 部署 -> status=OBSERVATION, observation_remaining=N.
    Each successful match decrements observation_remaining.
    When observation_remaining reaches 0, status -> CORE.
    The spec-default threshold is 5 (§5.6.1 部署), but the task specifies 3 observations.
    """

    def test_initial_state_is_observation(self):
        """Newly taught rules start at status=OBSERVATION."""
        from src.decision.reflex.rule_engine import RuleEngine
        rule = make_rule(
            name="taught_rule",
            pattern=r"保存",
            priority=PRIORITY_USER_TAUGHT,
            status=STATUS_OBSERVATION,
            observation_remaining=OBSERVATION_THRESHOLD,
            source="user_teaching",
        )
        engine = RuleEngine(rules=[rule])
        assert engine.get_rules()[0]["status"] == STATUS_OBSERVATION  # type: ignore[attr-defined]

    def test_match_decrements_remaining(self):
        """Matching an OBSERVATION rule decrements observation_remaining."""
        from src.decision.reflex.rule_engine import RuleEngine
        rule = make_rule(
            name="taught_rule",
            pattern=r"保存",
            priority=PRIORITY_USER_TAUGHT,
            status=STATUS_OBSERVATION,
            observation_remaining=OBSERVATION_THRESHOLD,
            source="user_teaching",
        )
        engine = RuleEngine(rules=[rule])
        ctx = make_scene_context()
        engine.match("帮我保存", scene_context=ctx, trace_id="t1")  # type: ignore[attr-defined]
        remaining = engine.get_rules()[0]["metadata"]["observation_remaining"]  # type: ignore[attr-defined]
        assert remaining == OBSERVATION_THRESHOLD - 1

    def test_observation_remaining_zero_becomes_core(self):
        """After OBSERVATION_THRESHOLD matches, status transitions to CORE."""
        from src.decision.reflex.rule_engine import RuleEngine
        rule = make_rule(
            name="taught_rule",
            pattern=r"保存",
            priority=PRIORITY_USER_TAUGHT,
            status=STATUS_OBSERVATION,
            observation_remaining=OBSERVATION_THRESHOLD,
            source="user_teaching",
        )
        engine = RuleEngine(rules=[rule])
        ctx = make_scene_context()
        for i in range(OBSERVATION_THRESHOLD):
            engine.match("帮我保存", scene_context=ctx, trace_id=f"obs_{i}")  # type: ignore[attr-defined]
        updated = engine.get_rules()[0]  # type: ignore[attr-defined]
        assert updated["status"] == STATUS_CORE
        assert updated["metadata"]["observation_remaining"] == 0

    def test_non_match_does_not_decrement(self):
        """Non-matching input does not decrement observation_remaining."""
        from src.decision.reflex.rule_engine import RuleEngine
        rule = make_rule(
            name="taught_rule",
            pattern=r"保存",
            priority=PRIORITY_USER_TAUGHT,
            status=STATUS_OBSERVATION,
            observation_remaining=OBSERVATION_THRESHOLD,
            source="user_teaching",
        )
        engine = RuleEngine(rules=[rule])
        engine.match("你好", trace_id="t2")  # type: ignore[attr-defined]  # non-match
        remaining = engine.get_rules()[0]["metadata"]["observation_remaining"]  # type: ignore[attr-defined]
        assert remaining == OBSERVATION_THRESHOLD

    def test_core_rule_does_not_decrement(self):
        """Already-CORE rules do not decrement (no observation_remaining tracking)."""
        from src.decision.reflex.rule_engine import RuleEngine
        rule = make_rule(
            name="core_rule",
            pattern=r"保存",
            priority=PRIORITY_CORE,
            status=STATUS_CORE,
            observation_remaining=0,
        )
        engine = RuleEngine(rules=[rule])
        engine.match("保存", trace_id="t3")  # type: ignore[attr-defined]
        remaining = engine.get_rules()[0]["metadata"]["observation_remaining"]  # type: ignore[attr-defined]
        # Should stay at 0 (not go negative)
        assert remaining == 0


class TestRuleSafety:
    """Matched rule output passes through SafetyClassifier.  v4.5.0 §5.7.2."""

    def test_rule_result_has_safety_level(self):
        """RuleEngine.match() returns a result with safety_level field."""
        from src.decision.reflex.rule_engine import RuleEngine
        rule = make_rule(name="safe_rule", pattern=r"保存", safety_level="SAFE")
        engine = RuleEngine(rules=[rule])
        result = engine.match("保存", trace_id="t1")  # type: ignore[attr-defined]
        assert result is not None
        assert "safety_level" in result
        assert result["safety_level"] in VALID_SAFETY_LEVELS

    def test_safe_rule_passes_safety_classifier(self):
        """SafetyClassifier.classify() returns SAFE for a SAFE rule action."""
        from src.decision.reflex.rule_engine import RuleEngine
        from src.decision.safety_classifier import SafetyClassifier
        rule = make_rule(
            name="safe_rule",
            pattern=r"你好",
            safety_level="SAFE",
            action_type="voice_response",
        )
        engine = RuleEngine(rules=[rule])
        result = engine.match("你好", trace_id="t2")  # type: ignore[attr-defined]
        assert result is not None
        safety = SafetyClassifier()
        decision_command = {
            "decision_type": "reflex",
            "command": {
                "voice_response": result.get("response", ""),
                "actions": [{"type": rule["action"]["type"]}],
            },
            "safety_level": result["safety_level"],
            "trace_id": "t2_reflex",
        }
        assert safety.classify(decision_command) == "SAFE"

    def test_dangerous_rule_blocked_by_classifier(self):
        """SafetyClassifier blocks DANGEROUS_AUTO_BLOCK rules."""
        from src.decision.reflex.rule_engine import RuleEngine
        from src.decision.safety_classifier import SafetyClassifier
        rule = make_rule(
            name="dangerous_rule",
            pattern=r"删除所有",
            safety_level="DANGEROUS_AUTO_BLOCK",
            action_type="mouse_click",
        )
        engine = RuleEngine(rules=[rule])
        result = engine.match("删除所有文件", trace_id="t3")  # type: ignore[attr-defined]
        assert result is not None
        safety = SafetyClassifier()
        decision_command = {
            "decision_type": "reflex",
            "command": {
                "voice_response": result.get("response", ""),
                "actions": [{"type": rule["action"]["type"]}],
            },
            "safety_level": result["safety_level"],
            "trace_id": "t3_reflex",
        }
        assert safety.classify(decision_command) == "DANGEROUS_AUTO_BLOCK"

    def test_result_contains_rule_ref(self):
        """Match result includes a reference to the matched rule."""
        from src.decision.reflex.rule_engine import RuleEngine
        rule = make_rule(name="ref_rule", pattern=r"测试", priority=PRIORITY_CORE)
        engine = RuleEngine(rules=[rule])
        result = engine.match("测试一下", trace_id="t4")  # type: ignore[attr-defined]
        assert result is not None
        assert "rule" in result or "rule_id" in result

    def test_result_contains_trace_id_propagation(self):
        """Match result includes the trace_id for logging."""
        from src.decision.reflex.rule_engine import RuleEngine
        rule = make_rule(name="trace_test", pattern=r".*", priority=PRIORITY_CORE)
        engine = RuleEngine(rules=[rule])
        result = engine.match("anything", trace_id="trace-xyz-789")  # type: ignore[attr-defined]
        assert result is not None
        # The result should propagate the trace_id
        assert "trace_id" in result
