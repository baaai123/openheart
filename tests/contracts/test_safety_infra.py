"""
Contract tests for SafetyInfra (spec v4.5.0 §5.6.1, §5.7.2).

RED phase — SafetyInfraImpl at src/decision/safety_infra_impl.py is NOT yet
implemented. Import will raise ImportError; all tests fail to collect until
GREEN phase.

Validates the contract of SafetyInfraImpl:
  - classify() returns a valid safety level (SAFE / NEEDS_CONFIRM / DANGEROUS_AUTO_BLOCK).
  - Commands with dangerous actions or keywords are classified DANGEROUS_AUTO_BLOCK.
  - Explicit safety_level field in the command is respected (highest priority).
  - match() returns a dict with expected keys on match; None on no match.

Uses real SafetyClassifier and RuleEngine — both are pure computation with
zero external dependencies (no network, no GPU, no file writes).
"""
from __future__ import annotations

import uuid
from typing import Any

import pytest

from src.decision.safety_infra_impl import SafetyInfraImpl  # will fail — RED phase

# ---------------------------------------------------------------------------
# Safety level constants — spec v4.5.0 §5.7.2
# ---------------------------------------------------------------------------

SAFE: str = "SAFE"
NEEDS_CONFIRM: str = "NEEDS_CONFIRM"
DANGEROUS_AUTO_BLOCK: str = "DANGEROUS_AUTO_BLOCK"
VALID_SAFETY_LEVELS: frozenset[str] = frozenset(
    {SAFE, NEEDS_CONFIRM, DANGEROUS_AUTO_BLOCK},
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def infra() -> SafetyInfraImpl:
    """SafetyInfraImpl wired to real SafetyClassifier + RuleEngine.

    SafetyClassifier uses in-memory keyword tables (pure computation).
    RuleEngine auto-loads default rule files from rules/.
    Neither depends on network, GPU, or external services.
    """
    return SafetyInfraImpl()


@pytest.fixture
def trace_id() -> str:
    """Return a unique trace_id per test invocation."""
    return f"trace-{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# Command factory — spec v4.5.0 decision command schema
# ---------------------------------------------------------------------------


def make_command(
    *,
    actions: list[dict[str, Any]] | None = None,
    voice_response: str = "",
    safety_level: str | None = None,
) -> dict[str, Any]:
    """Build a decision command dict matching the spec v4.5.0 schema.

    Args:
        actions: List of action dicts (each with ``type`` and optional ``params``).
        voice_response: Text content for the voice response channel.
        safety_level: Optional explicit override (highest priority in classify()).

    Returns:
        Command dict suitable for ``SafetyInfra.classify()``.
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
# Contract tests
# ---------------------------------------------------------------------------


class TestSafetyInfraContract:
    """SafetyInfraImpl contract — v4.5.0 §5.6.1, §5.7.2.

    All tests use real SafetyClassifier and RuleEngine.
    No mocks, no external dependencies.
    """

    def test_classify_safe_command(
        self,
        infra: SafetyInfraImpl,
        trace_id: str,
    ) -> None:
        """A legal command without dangerous keywords → SAFE."""
        cmd = make_command(
            actions=[{"type": "voice_response", "params": {"text": "你好"}}],
            voice_response="今天天气真好",
        )
        level = infra.classify(cmd, trace_id=trace_id)

        assert level == SAFE, (
            f"Expected SAFE for a benign command, got {level!r}"
        )

    def test_classify_dangerous_command(
        self,
        infra: SafetyInfraImpl,
        trace_id: str,
    ) -> None:
        """A command with dangerous keywords → DANGEROUS_AUTO_BLOCK.

        The voice_response contains '删除' which is listed in
        ``_DANGEROUS_KEYWORDS`` (spec v4.5.0 §5.7.2).
        """
        cmd = make_command(
            actions=[{"type": "voice_response", "params": {"text": "执行操作"}}],
            voice_response="帮我删除所有文件",
        )
        level = infra.classify(cmd, trace_id=trace_id)

        assert level == DANGEROUS_AUTO_BLOCK, (
            f"Expected DANGEROUS_AUTO_BLOCK for delete command, got {level!r}"
        )

    def test_classify_with_explicit_safety_level(
        self,
        infra: SafetyInfraImpl,
        trace_id: str,
    ) -> None:
        """Explicit ``safety_level`` field overrides keyword-based detection.

        Even though the voice_response contains dangerous keywords, the
        pre-set SAFE level must be respected (spec §5.7.2 priority rule 1).
        """
        cmd = make_command(
            actions=[{"type": "voice_response", "params": {"text": "删除"}}],
            voice_response="删除所有文件",
            safety_level=SAFE,
        )
        level = infra.classify(cmd, trace_id=trace_id)

        assert level == SAFE, (
            f"Expected SAFE (explicit override), got {level!r}"
        )

    def test_match_returns_dict_with_expected_keys(
        self,
        infra: SafetyInfraImpl,
        trace_id: str,
    ) -> None:
        """Matching input returns a dict with all spec-required keys.

        '你好' matches the core greeting rule pattern
        ``^(你好|嗨|嘿|喂|哈喽|hello|hi)\\b`` (core_rules.json).
        """
        result = infra.match(
            user_input="你好",
            scene_context=None,
            trace_id=trace_id,
        )

        assert isinstance(result, dict), (
            f"Expected dict for a matched input, got {type(result)}"
        )

        # Spec v4.5.0 §5.3.1 match result keys (from RuleEngine._build_result)
        expected_keys: frozenset[str] = frozenset({
            "rule", "rule_id", "decision_type", "action_type",
            "params", "confidence", "safety_level", "priority", "trace_id",
        })
        missing = expected_keys - result.keys()
        assert not missing, (
            f"Match result missing expected keys: {missing}"
        )

        assert result["decision_type"] == "reflex", (
            f"Expected decision_type='reflex', got {result['decision_type']!r}"
        )
        assert isinstance(result["confidence"], (int, float)), (
            f"Expected numeric confidence, got {type(result['confidence'])}"
        )
        assert result["safety_level"] in VALID_SAFETY_LEVELS, (
            f"Invalid safety_level in match result: {result['safety_level']!r}"
        )

    def test_match_no_match_returns_none(
        self,
        infra: SafetyInfraImpl,
        trace_id: str,
    ) -> None:
        """Unmatched input returns None.

        A random unicode string with no corresponding rule pattern
        must produce None rather than crashing or returning a sentinel.
        """
        result = infra.match(
            user_input="!@#$%^&*()_+NO_MATCH_12345测试无匹配",
            scene_context=None,
            trace_id=trace_id,
        )

        assert result is None, (
            f"Expected None for unmatched input, got {type(result)}: {result}"
        )
