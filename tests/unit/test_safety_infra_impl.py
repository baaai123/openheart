"""
Unit tests for SafetyInfraImpl — delegation, classification, rule matching.

v4.5.0 §5.6.1 / §5.7.2: tests the adapter layer wrapping SafetyClassifier +
RuleEngine.  Uses real implementations of both delegatees — they are pure
computation with zero external dependencies (no network, no GPU, no file
mutations), so no mocks are needed.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.decision.safety_infra_impl import SafetyInfraImpl
from src.decision.safety_classifier import (
    SAFE,
    NEEDS_CONFIRM,
    DANGEROUS_AUTO_BLOCK,
    VALID_SAFETY_LEVELS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_command(
    *,
    actions: list[dict[str, Any]] | None = None,
    voice_response: str = "",
    safety_level: str | None = None,
) -> dict[str, Any]:
    """Build a decision command dict matching the spec v4.5.0 schema.

    Mirrors the factory in tests/contracts/test_safety_infra.py so that unit
    and contract tests use identical input shapes.
    """
    cmd: dict[str, Any] = {
        "command": {
            "actions": actions or [],
            "voice_response": voice_response,
        },
    }
    if safety_level is not None:
        cmd["safety_level"] = safety_level
    return cmd


# ---------------------------------------------------------------------------
# Tests — classify() delegation
# ---------------------------------------------------------------------------


class TestClassify:
    """SafetyInfraImpl.classify() — delegation to SafetyClassifier."""

    def test_classify_delegates_correctly(self) -> None:
        """classify() returns a valid safety-level string for any input."""
        impl = SafetyInfraImpl()
        cmd = make_command(
            actions=[{"type": "voice_response", "params": {"text": "你好"}}],
            voice_response="hi",
        )
        level = impl.classify(cmd, trace_id="ut-001")

        assert isinstance(level, str)
        assert level in VALID_SAFETY_LEVELS, (
            f"Expected one of {VALID_SAFETY_LEVELS}, got {level!r}"
        )

    def test_classify_safe_by_default(self) -> None:
        """A benign command without dangerous keywords → SAFE."""
        impl = SafetyInfraImpl()
        cmd = make_command(
            actions=[{"type": "voice_response", "params": {"text": "讲个笑话"}}],
            voice_response="今天天气真好",
        )
        level = impl.classify(cmd, trace_id="ut-002")

        assert level == SAFE, f"Expected SAFE, got {level!r}"

    def test_classify_dangerous_action_blocked(self) -> None:
        """A command with dangerous keywords → DANGEROUS_AUTO_BLOCK."""
        impl = SafetyInfraImpl()
        cmd = make_command(
            actions=[{"type": "voice_response", "params": {"text": "执行操作"}}],
            voice_response="帮我删除所有文件",
        )
        level = impl.classify(cmd, trace_id="ut-003")

        assert level == DANGEROUS_AUTO_BLOCK, (
            f"Expected DANGEROUS_AUTO_BLOCK for dangerous voice_response, "
            f"got {level!r}"
        )

    def test_classify_empty_command_returns_safe(self) -> None:
        """classify() with an empty-but-valid command → SAFE (not crash)."""
        impl = SafetyInfraImpl()
        cmd = make_command()
        level = impl.classify(cmd, trace_id="ut-004")

        assert level == SAFE, f"Expected SAFE for empty command, got {level!r}"


# ---------------------------------------------------------------------------
# Tests — match() delegation
# ---------------------------------------------------------------------------


class TestMatch:
    """SafetyInfraImpl.match() — delegation to RuleEngine."""

    def test_match_returns_dict_for_valid_input(self) -> None:
        """match() on a known greeting pattern returns a dict with expected keys.

        RuleEngine auto-loads rules/rules.json which includes a greeting rule
        matching patterns like ``你好``.
        """
        impl = SafetyInfraImpl()
        result = impl.match("你好", trace_id="ut-005")

        assert result is not None
        assert isinstance(result, dict)
        # Core keys guaranteed by RuleMatch.to_dict()
        assert "rule_id" in result
        assert "rule" in result
        assert "decision_type" in result
        assert "action_type" in result
        assert "confidence" in result
        assert isinstance(result["confidence"], (int, float))
        assert result["confidence"] >= 0.0

    def test_match_returns_none_for_unmatched_input(self) -> None:
        """match() on gibberish with no matching rule → None."""
        impl = SafetyInfraImpl()
        result = impl.match("xxxxxxxxxxyyyzzz unmatched garbage 12345", trace_id="ut-006")

        assert result is None

    def test_match_with_scene_context(self) -> None:
        """match() accepts optional scene_context without error."""
        impl = SafetyInfraImpl()
        result = impl.match(
            "你好",
            scene_context={"scene_type": "chat", "emotion": "joy"},
            trace_id="ut-007",
        )

        # scene_context is passed through to RuleEngine; should still match
        assert result is not None
        assert isinstance(result, dict)

    def test_match_empty_string_returns_none(self) -> None:
        """match('') → None — empty input matches no rule."""
        impl = SafetyInfraImpl()
        result = impl.match("", trace_id="ut-008")

        assert result is None


# ---------------------------------------------------------------------------
# Interface hygiene
# ---------------------------------------------------------------------------


class TestInterface:
    """SafetyInfraImpl must expose exactly classify() + match()."""

    def test_impl_has_only_two_public_methods(self) -> None:
        """SafetyInfraImpl exposes only classify and match as public methods.

        The spec (§5.6.1) defines SafetyInfra protocol with exactly two
        methods.  Any extra public method is a leaky abstraction.
        """
        public = {
            name
            for name in dir(SafetyInfraImpl)
            if not name.startswith("_")
        }

        assert public == {"classify", "match"}, (
            f"Expected {{'classify', 'match'}}, got {public}"
        )
