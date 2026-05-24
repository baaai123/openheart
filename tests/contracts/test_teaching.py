"""
Contract tests for TeachingModule (spec v4.5.0 §5.7, v5.x slim architecture).

Validates the slim TeachingModule contract:
  - teach(rule) processes SAFE rules → {"action": "learned"}
  - teach(rule) blocks DANGEROUS rules → {"action": "blocked"}
  - teach(rule) stores NEEDS_CONFIRM rules → {"action": "needs_confirmation", "pending_id"}
  - confirm_rule(rule_id) returns True for existing pending rule
  - confirm_rule(rule_id) returns False for non-existent/expired rule

RED phase — TeachingModule not yet slimmed. Tests may fail to run if
confirm_rule() does not exist or teach() signature differs from v5.x.
"""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from src.decision.learning.learner import (
    Rule,
    RuleAction,
    RuleCondition,
    RuleMetadata,
    RulePriority,
    RuleSource,
    RuleStatus,
    SafetyLevel,
)
from src.decision.teaching import TeachingModule  # may fail — RED phase


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_rule(safety: SafetyLevel, rule_id: str = "") -> dict:
    """Build a rule dict with the given safety level for teach().

    v5.x: Includes flat top-level keys (condition_pattern, action_type,
    safety_level, trigger_phrase) consumed by TeachingModule.teach(),
    alongside nested condition/action/metadata for Rule.from_dict().
    """
    rid = rule_id or f"rule-{safety.value.lower()}"
    pattern: str = f"trigger_{safety.value.lower()}"
    return {
        # v5.x flat keys for TeachingModule.teach()
        "condition_pattern": pattern,
        "action_type": "voice_response",
        "safety_level": safety.value,
        "trigger_phrase": f"test phrase {safety.value.lower()}",
        # Legacy nested keys for Rule.from_dict()
        "rule_id": rid,
        "name": f"test_{safety.value.lower()}",
        "priority": RulePriority.USER_TAUGHT.name,
        "status": RuleStatus.OBSERVATION.value,
        "condition": {
            "trigger_type": "voice_command",
            "pattern": pattern,
            "context_constraints": [],
        },
        "action": {
            "type": "voice_response",
            "params": {},
            "safety_level": safety.value,
        },
        "template_id": None,
        "template_slots": {},
        "cluster_hint": None,
        "metadata": {
            "confidence": 0.5,
            "success_count": 0,
            "failure_count": 0,
            "created_at": "",
            "last_verified_at": "",
            "source": RuleSource.USER_TEACHING.value,
            "observation_remaining": 3,
        },
    }


@pytest.fixture
def mock_learner() -> AsyncMock:
    """Return a RuleLearner-like AsyncMock.

    The mock provides the minimal interface TeachingModule depends on:
    - learn_from_interaction()  — called for SAFE rules that get learned directly
    - add_rule()               — called when a pending rule is confirmed
    """
    learner = AsyncMock()

    async def fake_learn(*args, **kwargs):
        return Rule(
            rule_id="rule-learned",
            name="learned_rule",
            priority=RulePriority.USER_TAUGHT.name,
            status=RuleStatus.OBSERVATION.value,
            condition=RuleCondition(trigger_type="voice_command", pattern="trigger"),
            action=RuleAction(type="voice_response", params={}, safety_level=SafetyLevel.SAFE.value),
            metadata=RuleMetadata(confidence=0.5, source=RuleSource.USER_TEACHING.value),
        )

    learner.learn_from_interaction = AsyncMock(side_effect=fake_learn)
    learner.add_rule = AsyncMock()
    learner._detect_adjacent_risk = lambda trigger, action_type: False
    return learner


@pytest.fixture
def teaching(mock_learner: AsyncMock) -> TeachingModule:
    """Return a TeachingModule wired to mock learner + in-memory pending storage.

    redis_client=None forces TeachingModule to use _pending_local dict,
    avoiding any real Redis dependency.
    """
    return TeachingModule(rule_learner=mock_learner, redis_client=None)


# ---------------------------------------------------------------------------
# Contract tests
# ---------------------------------------------------------------------------

class TestTeachingContract:
    """TeachingModule teach() + confirm_rule() contract — v5.x slim architecture.

    All tests avoid real Redis — in-memory pending storage is used.
    The RuleLearner is mocked so tests focus on TeachingModule's orchestration.
    """

    @pytest.mark.asyncio
    async def test_teach_safe_rule(
        self, teaching: TeachingModule, mock_learner: AsyncMock,
    ) -> None:
        """SAFE rule → teach() returns {'action': 'learned'}."""
        rule_dict: dict = _make_rule(SafetyLevel.SAFE)
        result: dict = await teaching.teach(rule_dict, trace_id="trace-safe")
        assert result["action"] == "learned", (
            f"Expected action='learned', got {result['action']!r}"
        )

    @pytest.mark.asyncio
    async def test_teach_dangerous_rule(
        self, teaching: TeachingModule,
    ) -> None:
        """DANGEROUS_AUTO_BLOCK rule → teach() returns {'action': 'blocked'}."""
        rule_dict: dict = _make_rule(SafetyLevel.DANGEROUS_AUTO_BLOCK)
        result: dict = await teaching.teach(rule_dict, trace_id="trace-danger")
        assert result["action"] == "blocked", (
            f"Expected action='blocked', got {result['action']!r}"
        )

    @pytest.mark.asyncio
    async def test_teach_needs_confirmation(
        self, teaching: TeachingModule,
    ) -> None:
        """NEEDS_CONFIRM rule → teach() returns {'action': 'needs_confirmation', 'pending_id'}."""
        rule_dict: dict = _make_rule(SafetyLevel.NEEDS_CONFIRM)
        result: dict = await teaching.teach(rule_dict, trace_id="trace-confirm")
        assert result["action"] == "needs_confirmation", (
            f"Expected action='needs_confirmation', got {result['action']!r}"
        )
        assert "pending_id" in result, (
            f"Expected 'pending_id' in result, got keys: {list(result.keys())}"
        )
        assert isinstance(result["pending_id"], str), (
            f"Expected pending_id to be str, got {type(result['pending_id'])}"
        )

    @pytest.mark.asyncio
    async def test_confirm_rule_success(
        self, teaching: TeachingModule, mock_learner: AsyncMock,
    ) -> None:
        """confirm_rule() with existing pending rule → returns True."""
        # Arrange: teach a NEEDS_CONFIRM rule first to create a pending entry.
        rule_dict: dict = _make_rule(SafetyLevel.NEEDS_CONFIRM, rule_id="rule-pending-1")
        teach_result: dict = await teaching.teach(rule_dict, trace_id="trace-confirm-success")
        pending_id: str = teach_result["pending_id"]

        # Act: confirm the pending rule.
        confirmed: bool = await teaching.confirm_rule(pending_id)

        # Assert: should be True, and learner.add_rule must have been called.
        assert confirmed is True, (
            f"Expected confirm_rule('{pending_id}') to return True, got {confirmed}"
        )
        mock_learner.add_rule.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_confirm_nonexistent_rule(
        self, teaching: TeachingModule,
    ) -> None:
        """confirm_rule() with non-existent rule_id → returns False."""
        confirmed: bool = await teaching.confirm_rule("nonexistent-rule-id")
        assert confirmed is False, (
            f"Expected confirm_rule('nonexistent') to return False, got {confirmed}"
        )
