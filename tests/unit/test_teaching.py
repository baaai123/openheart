"""
Unit tests for TeachingModule — rule creation, safety classification, confirmation.

v4.5.0 §5.7: Tests the slim TeachingModule with a mocked RuleLearner
and in-memory pending storage (no Redis). All methods are async.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.decision.teaching import TeachingModule, PendingRule
from src.decision.learning.learner import (
    Rule,
    RuleStatus,
    RulePriority,
    RuleSource,
    SafetyLevel,
    RuleCondition,
    RuleAction,
    RuleMetadata,
)


def _make_rule(
    rule_id: str = "test-rule-id",
    pattern: str = "test pattern",
    action_type: str = "voice_response",
    safety_level: str = "SAFE",
) -> Rule:
    """Helper to build a Rule dataclass for test assertions."""
    return Rule(
        rule_id=rule_id,
        name=f"test_{pattern[:30]}",
        priority=RulePriority.USER_TAUGHT.name,
        status=RuleStatus.OBSERVATION.value,
        condition=RuleCondition(
            trigger_type="voice_command",
            pattern=pattern,
        ),
        action=RuleAction(
            type=action_type,
            params={},
            safety_level=safety_level,
        ),
        metadata=RuleMetadata(
            confidence=0.5,
            source=RuleSource.USER_TEACHING.value,
            created_at="2025-01-01T00:00:00.000+00:00",
        ),
    )


class TestTeachSafe:
    """TeachingModule.teach() — SAFE rules learned immediately."""

    @pytest.mark.asyncio
    async def test_teach_safe_rule_learns_direct(self) -> None:
        """SAFE rule with pre-classified safety_level → learn_from_interaction called, returns learned."""
        # Arrange
        mock_learner = MagicMock()
        mock_learner.learn_from_interaction = AsyncMock(
            return_value=_make_rule(safety_level=SafetyLevel.SAFE.value),
        )
        mock_learner._detect_adjacent_risk = MagicMock(return_value=False)

        module = TeachingModule(rule_learner=mock_learner, redis_client=None)

        rule_input: dict[str, str] = {
            "condition_pattern": "开灯",
            "action_type": "voice_response",
            "safety_level": SafetyLevel.SAFE.value,
            "trigger_phrase": "记住：说开灯就打开灯光",
        }

        # Act
        result = await module.teach(rule=rule_input, trace_id="trace-safe-001")

        # Assert
        assert result["action"] == "learned"
        assert result["pending_id"] is None
        assert result["rule"] is not None
        mock_learner.learn_from_interaction.assert_awaited_once()
        mock_learner._detect_adjacent_risk.assert_called_once()

    @pytest.mark.asyncio
    async def test_teach_safe_upgraded_by_adjacent_risk(self) -> None:
        """SAFE rule → adjacent risk detected → upgraded to NEEDS_CONFIRM."""
        # Arrange
        mock_learner = MagicMock()
        # Adjacent risk returns True → SAFE becomes NEEDS_CONFIRM
        mock_learner._detect_adjacent_risk = MagicMock(return_value=True)
        # teach() won't call learn_from_interaction for the upgraded path
        mock_learner.learn_from_interaction = AsyncMock()

        module = TeachingModule(rule_learner=mock_learner, redis_client=None)

        rule_input: dict[str, str] = {
            "condition_pattern": "开灯",
            "action_type": "voice_response",
            "safety_level": SafetyLevel.SAFE.value,
            "trigger_phrase": "记住：说开灯就打开灯光",
        }

        # Act
        result = await module.teach(rule=rule_input, trace_id="trace-adj-001")

        # Assert
        assert result["action"] == "needs_confirmation"
        assert result["pending_id"] is not None
        # learn_from_interaction should NOT have been called
        mock_learner.learn_from_interaction.assert_not_awaited()


class TestTeachNeedsConfirm:
    """TeachingModule.teach() — NEEDS_CONFIRM rules stored as pending."""

    @pytest.mark.asyncio
    async def test_teach_needs_confirmation_stores_pending(self) -> None:
        """NEEDS_CONFIRM rule → stored in _pending_local, returns pending_id."""
        # Arrange
        mock_learner = MagicMock()

        module = TeachingModule(rule_learner=mock_learner, redis_client=None)

        rule_input: dict[str, str] = {
            "condition_pattern": "发送消息",
            "action_type": "voice_response",
            "safety_level": SafetyLevel.NEEDS_CONFIRM.value,
            "trigger_phrase": "记住：说发送消息就发微信",
        }

        # Act
        result = await module.teach(rule=rule_input, trace_id="trace-conf-001")

        # Assert
        assert result["action"] == "needs_confirmation"
        assert result["pending_id"] is not None
        assert result["rule"] is not None

        # Verify rule is stored in local pending storage
        pending_id: str = result["pending_id"]
        assert pending_id in module._pending_local
        stored: PendingRule = module._pending_local[pending_id]
        assert stored.rule.rule_id == pending_id
        assert stored.trace_id == "trace-conf-001"
        assert not stored.expired


class TestTeachDangerous:
    """TeachingModule.teach() — DANGEROUS_AUTO_BLOCK rules rejected."""

    @pytest.mark.asyncio
    async def test_teach_dangerous_rule_blocked(self) -> None:
        """DANGEROUS_AUTO_BLOCK rule → blocked, no learner interaction."""
        # Arrange
        mock_learner = MagicMock()
        mock_learner.learn_from_interaction = AsyncMock()

        module = TeachingModule(rule_learner=mock_learner, redis_client=None)

        rule_input: dict[str, str] = {
            "condition_pattern": "删除文件",
            "action_type": "voice_response",
            "safety_level": SafetyLevel.DANGEROUS_AUTO_BLOCK.value,
            "trigger_phrase": "记住：说删除文件就清空文档",
        }

        # Act
        result = await module.teach(rule=rule_input, trace_id="trace-danger-001")

        # Assert
        assert result["action"] == "blocked"
        assert result["rule"] is None
        assert result["pending_id"] is None
        mock_learner.learn_from_interaction.assert_not_awaited()


class TestConfirmRule:
    """TeachingModule.confirm_rule() — promotes pending to observation."""

    @pytest.mark.asyncio
    async def test_confirm_rule_promotes_to_learner(self) -> None:
        """confirm_rule calls learner.add_rule and removes pending."""
        # Arrange
        mock_learner = MagicMock()
        mock_learner.add_rule = AsyncMock()

        module = TeachingModule(rule_learner=mock_learner, redis_client=None)

        # First, teach a NEEDS_CONFIRM rule to create a pending entry
        rule_input: dict[str, str] = {
            "condition_pattern": "发送消息",
            "action_type": "voice_response",
            "safety_level": SafetyLevel.NEEDS_CONFIRM.value,
            "trigger_phrase": "记住：说发送消息就发微信",
        }
        teach_result = await module.teach(rule=rule_input, trace_id="trace-confirm-001")
        pending_id: str = teach_result["pending_id"]

        # Sanity: pending exists
        assert pending_id in module._pending_local

        # Act
        confirmed: bool = await module.confirm_rule(rule_id=pending_id)

        # Assert
        assert confirmed is True
        mock_learner.add_rule.assert_awaited_once()
        # Pending entry should be removed
        assert pending_id not in module._pending_local

    @pytest.mark.asyncio
    async def test_confirm_rule_nonexistent_returns_false(self) -> None:
        """confirm_rule with unknown rule_id returns False."""
        # Arrange
        mock_learner = MagicMock()
        mock_learner.add_rule = AsyncMock()

        module = TeachingModule(rule_learner=mock_learner, redis_client=None)

        # Act
        result: bool = await module.confirm_rule(rule_id="nonexistent-rule")

        # Assert
        assert result is False
        mock_learner.add_rule.assert_not_awaited()


class TestClassifySafety:
    """TeachingModule._classify_safety() — keyword-based classification."""

    def test_dangerous_keyword_blocked(self) -> None:
        """Dangerous keyword (付款) → DANGEROUS_AUTO_BLOCK."""
        mock_learner = MagicMock()
        module = TeachingModule(rule_learner=mock_learner, redis_client=None)

        level = module._classify_safety("记住：说付款就转账")
        assert level == SafetyLevel.DANGEROUS_AUTO_BLOCK.value

    def test_needs_confirm_keyword(self) -> None:
        """Needs-confirm keyword (发送) → NEEDS_CONFIRM."""
        mock_learner = MagicMock()
        module = TeachingModule(rule_learner=mock_learner, redis_client=None)

        level = module._classify_safety("记住：说发送消息就通知")
        assert level == SafetyLevel.NEEDS_CONFIRM.value

    def test_safe_keyword(self) -> None:
        """Benign keyword → SAFE."""
        mock_learner = MagicMock()
        module = TeachingModule(rule_learner=mock_learner, redis_client=None)

        level = module._classify_safety("记住：说开灯就打开灯光")
        assert level == SafetyLevel.SAFE.value

    @pytest.mark.asyncio
    async def test_trigger_phrase_passed_to_classification(self) -> None:
        """When safety_level is empty, _classify_safety is called with trigger_phrase."""
        mock_learner = MagicMock()
        # learn_from_interaction not called for DANGEROUS classification
        mock_learner.learn_from_interaction = AsyncMock()

        module = TeachingModule(rule_learner=mock_learner, redis_client=None)

        rule_input: dict[str, str] = {
            "condition_pattern": "test",
            "action_type": "voice_response",
            "safety_level": "",  # Empty → triggers _classify_safety
            "trigger_phrase": "记住：说付款就转账",
        }

        # We can't easily spy on _classify_safety, but we can
        # verify that DANGEROUS is detected through the public path
        result = await module.teach(rule=rule_input, trace_id="trace-classify-001")

        assert result["action"] == "blocked"
