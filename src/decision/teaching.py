"""
Teaching module — pure rule creation/confirmation engine. v5.x

Teaching intent detection is LLM-driven (handled by Orchestrator). This module
is the rule engine that validates safety, stores rules, and manages the async
confirmation flow.

v5.x Flow (LLM-driven):
  1. Orchestrator's LLM detects teaching intent → produces a `rule` dict
  2. TeachingModule.teach(rule) → validates safety
  3. SAFE rules: learn immediately via RuleLearner
  4. NEEDS_CONFIRM rules: store in pending_rules:{rule_id} (Redis, TTL 120s)
  5. DANGEROUS_AUTO_BLOCK rules: auto-reject with explanation
  6. Orchestrator calls confirm_rule(rule_id) on user's affirmative response
  7. confirm_rule: promotes pending rule to OBSERVATION via learner

Safety classification (v4.5.0 §5.7.2):
  SAFE: no delete/send/payment/system ops
  NEEDS_CONFIRM: involves sending messages, modifying files
  DANGEROUS_AUTO_BLOCK: involves payment, data deletion, system settings
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from src.decision.learning.learner import (
    Rule,
    RuleLearner,
    RuleStatus,
    RulePriority,
    RuleSource,
    SafetyLevel,
    RuleCondition,
    RuleAction,
    RuleMetadata,
)

logger = logging.getLogger(__name__)

# v4.5.0 §5.7.3: pending rules TTL
_PENDING_TTL_SECONDS: int = 120

# Dangerous action patterns — v4.5.0 §5.7.2
_DANGEROUS_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(支付|付款|转账|汇钱)"),
    re.compile(r"(删除|永久删除|清空|格式化)"),
    re.compile(r"(系统设置|控制面板|注册表|sudo|root)"),
]

# Needs-confirm patterns — v4.5.0 §5.7.2
_NEEDS_CONFIRM_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(发送|发消息|发邮件|发送文件)"),
    re.compile(r"(修改|编辑|更改|覆盖)"),
]


@dataclass
class PendingRule:
    """Rule temporarily stored in pending_rules:{rule_id} (Redis or local)."""
    rule: Rule
    trace_id: str
    created_at: float = field(default_factory=time.time)
    ttl_seconds: int = _PENDING_TTL_SECONDS

    @property
    def expired(self) -> bool:
        return (time.time() - self.created_at) > self.ttl_seconds


class TeachingModule:
    """Pure rule creation/confirmation engine. v5.x

    Coordinates between the RuleLearner, Redis pending storage, and the
    Orchestrator (which handles LLM-driven intent detection and dialog).

    Responsibilities:
      - Safety classification of proposed rules
      - Immediate learning for SAFE rules
      - Pending storage for NEEDS_CONFIRM rules
      - Confirmation: promote pending rules to OBSERVATION
      - Auto-rejection of DANGEROUS_AUTO_BLOCK rules
    """

    def __init__(
        self,
        rule_learner: RuleLearner,
        redis_client: Any = None,
    ) -> None:
        """
        Args:
            rule_learner: RuleLearner instance for rule storage and matching.
            redis_client: Redis client for pending_rules storage.
                          If None, uses in-memory dict (for testing).
        """
        self._learner: RuleLearner = rule_learner
        self._redis: Any = redis_client
        self._pending_local: dict[str, PendingRule] = {}
        self._lock: asyncio.Lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Rule creation — v5.x (LLM-driven)
    # ------------------------------------------------------------------

    async def teach(
        self,
        rule: dict[str, Any],
        trace_id: str = "",
    ) -> dict[str, Any]:
        """Process a user-taught rule through safety classification.

        v5.x: The Orchestrator's LLM produces a `rule` dict. This method
        validates safety and either learns, stores for confirmation, or blocks.

        Args:
            rule: dict with keys:
                condition_pattern (str): trigger pattern for the rule
                action_type (str): e.g. "voice_response", "mouse_click"
                safety_level (str, optional): pre-classified safety level;
                    if empty, _classify_safety() is called.
                trigger_phrase (str): original user phrase that triggered teaching
            trace_id: correlation id for logging

        Returns:
            dict with:
                action: "learned" | "needs_confirmation" | "blocked"
                message: human-readable status
                rule: the created Rule (if learned/pending)
                pending_id: rule_id for confirmation (if needs_confirmation)
                trace_id: the trace_id
        """
        tid: str = trace_id if trace_id else uuid.uuid4().hex

        condition_pattern: str = rule.get("condition_pattern", "")
        action_type: str = rule.get("action_type", "voice_response")
        safety_level: str = rule.get("safety_level", "")
        trigger_phrase: str = rule.get("trigger_phrase", "")

        # If safety level not provided by LLM, classify from available text.
        if not safety_level:
            classification_input: str = (
                f"{trigger_phrase} {condition_pattern} {action_type}"
            )
            safety_level = self._classify_safety(classification_input)

        # DANGEROUS_AUTO_BLOCK: reject immediately — v4.5.0 §5.7.2
        if safety_level == SafetyLevel.DANGEROUS_AUTO_BLOCK.value:
            logger.warning(
                "TeachingModule: blocked dangerous teaching attempt "
                "(trace_id=%s, trigger=%r)",
                tid, trigger_phrase,
            )
            return {
                "action": "blocked",
                "message": "这个操作可能涉及安全风险，我不能学习它。我们要一起保护你的安全～",
                "rule": None,
                "pending_id": None,
                "trace_id": tid,
            }

        # SAFE: check adjacent risk, then learn immediately.
        if safety_level == SafetyLevel.SAFE.value:
            trigger: str = condition_pattern or trigger_phrase or "unknown"
            if self._learner._detect_adjacent_risk(trigger, "voice_command"):
                logger.info(
                    "TeachingModule: SAFE rule upgraded to NEEDS_CONFIRM "
                    "due to adjacent risk (trace_id=%s, trigger=%r)",
                    tid, trigger,
                )
                safety_level = SafetyLevel.NEEDS_CONFIRM.value
                # Fall through to NEEDS_CONFIRM handling below.
            else:
                # v4.5.0 §5.7.3: SAFE → learn immediately.
                rule_obj: Rule = await self._learner.learn_from_interaction(
                    trigger=trigger,
                    action_type=action_type,
                    action_params={},
                    safety_level=safety_level,
                    source=RuleSource.USER_TEACHING.value,
                )
                return {
                    "action": "learned",
                    "message": f"记住了，以后说\"{condition_pattern}\"我就{action_type}",
                    "rule": rule_obj,
                    "pending_id": None,
                    "trace_id": tid,
                }

        # NEEDS_CONFIRM (including upgraded from SAFE by adjacent risk):
        # store in Redis/local pending, return pending_id for confirmation.
        iso_now: str = time.strftime(
            "%Y-%m-%dT%H:%M:%S.000+00:00", time.gmtime()
        )
        rule_obj = Rule(
            rule_id=uuid.uuid4().hex,
            name=f"pending_{condition_pattern[:30]}",
            priority=RulePriority.USER_TAUGHT.name,
            status=RuleStatus.OBSERVATION.value,
            condition=RuleCondition(
                trigger_type="voice_command",
                pattern=condition_pattern or trigger_phrase,
            ),
            action=RuleAction(
                type=action_type,
                params={},
                safety_level=safety_level,
            ),
            metadata=RuleMetadata(
                confidence=0.5,
                source=RuleSource.USER_TEACHING.value,
                created_at=iso_now,
            ),
        )

        pending = PendingRule(rule=rule_obj, trace_id=tid)
        await self._store_pending(pending)

        logger.info(
            "TeachingModule: rule pending confirmation (trace_id=%s, "
            "rule_id=%s, safety=%s, ttl=%ds)",
            tid, rule_obj.rule_id, safety_level, _PENDING_TTL_SECONDS,
        )
        return {
            "action": "needs_confirmation",
            "message": "你确定要让我学会这个操作吗？请回答'确定'或'取消'。",
            "rule": rule_obj,
            "pending_id": rule_obj.rule_id,
            "trace_id": tid,
        }

    # ------------------------------------------------------------------
    # Confirmation — v5.x
    # ------------------------------------------------------------------

    async def confirm_rule(self, rule_id: str) -> bool:
        """Promote a pending rule to OBSERVATION via learner.

        v5.x: Called by Orchestrator after LLM confirms user's affirmative
        response. Fetches the pending rule, adds to learner, removes from
        pending storage.

        Args:
            rule_id: ID of the pending rule to confirm.

        Returns:
            True if the rule was successfully confirmed and promoted.
            False if no such pending rule exists or it has expired.
        """
        async with self._lock:
            pending: Optional[PendingRule] = await self._fetch_pending(rule_id)

            if pending is None:
                logger.warning(
                    "TeachingModule.confirm_rule: no pending rule found "
                    "(rule_id=%s)",
                    rule_id,
                )
                return False

            if pending.expired:
                logger.info(
                    "TeachingModule.confirm_rule: pending rule expired "
                    "(rule_id=%s, trace_id=%s)",
                    rule_id, pending.trace_id,
                )
                await self._remove_pending(rule_id)
                return False

            # v4.5.0 §5.7.4: try/except — learner failure must not silently
            # drop the rule; keep it pending for retry.
            try:
                await self._learner.add_rule(pending.rule)
            except Exception:
                logger.warning(
                    "TeachingModule.confirm_rule: failed to add rule to "
                    "learner (rule_id=%s, trace_id=%s)",
                    rule_id, pending.trace_id,
                    exc_info=True,
                )
                return False

            await self._remove_pending(rule_id)
            logger.info(
                "TeachingModule: rule confirmed (rule_id=%s, trace_id=%s)",
                rule_id, pending.trace_id,
            )
            return True

    # ------------------------------------------------------------------
    # Safety classification — v4.5.0 §5.7.2
    # ------------------------------------------------------------------

    def _classify_safety(self, action_description: str) -> str:
        """Classify safety level of an action description. v4.5.0 §5.7.2.

        Checks action description against dangerous and needs-confirm
        keyword patterns (DELETE/SEND/PAYMENT/system ops).
        """
        for pattern in _DANGEROUS_PATTERNS:
            if pattern.search(action_description):
                return SafetyLevel.DANGEROUS_AUTO_BLOCK.value
        for pattern in _NEEDS_CONFIRM_PATTERNS:
            if pattern.search(action_description):
                return SafetyLevel.NEEDS_CONFIRM.value
        return SafetyLevel.SAFE.value

    # ------------------------------------------------------------------
    # Pending rule storage — Redis or local (for testing)
    # ------------------------------------------------------------------

    async def _store_pending(self, pending: PendingRule) -> None:
        """Store a pending rule keyed by rule_id.

        Uses Redis if available, else in-memory dict.
        """
        rule_id: str = pending.rule.rule_id
        if self._redis is not None:
            try:
                # v4.5.0 §5.7.4: Redis SET with TTL — expired keys
                # automatically cleaned up by Redis.
                import json
                key: str = f"pending_rules:{rule_id}"
                await asyncio.to_thread(
                    self._redis.set,
                    key,
                    json.dumps(pending.rule.to_dict(), ensure_ascii=False),
                    ex=_PENDING_TTL_SECONDS,
                )
            except Exception:
                # v4.5.0 §5.7.4: Redis failure → fall back to local storage.
                # Log at WARNING with trace_id.
                logger.warning(
                    "TeachingModule: Redis unavailable for pending rule "
                    "(rule_id=%s, trace_id=%s). Using local storage.",
                    rule_id, pending.trace_id,
                    exc_info=True,
                )
                self._pending_local[rule_id] = pending
        else:
            self._pending_local[rule_id] = pending

    async def _fetch_pending(self, rule_id: str) -> Optional[PendingRule]:
        """Fetch a pending rule by rule_id. Tries Redis first, then local.

        v4.5.0 §5.7.4: try/except — Redis errors fall through to local
        storage without crashing.
        """
        if self._redis is not None:
            try:
                import json
                key: str = f"pending_rules:{rule_id}"
                data: Optional[bytes] = await asyncio.to_thread(
                    self._redis.get, key,
                )
                if data is not None:
                    rule_dict: dict[str, Any] = json.loads(data)
                    rule: Rule = Rule.from_dict(rule_dict)
                    # trace_id is not serialized to Redis; use rule_id
                    # as fallback for logging purposes.
                    return PendingRule(
                        rule=rule,
                        trace_id=rule_id,
                    )
            except Exception:
                logger.warning(
                    "TeachingModule: failed to fetch pending rule from Redis "
                    "(rule_id=%s).",
                    rule_id,
                    exc_info=True,
                )
        return self._pending_local.get(rule_id)

    async def _remove_pending(self, rule_id: str) -> None:
        """Remove a pending rule from storage by rule_id.

        v4.5.0 §5.7.4: try/except — Redis failure is logged, local
        cleanup still proceeds.
        """
        if self._redis is not None:
            try:
                key: str = f"pending_rules:{rule_id}"
                await asyncio.to_thread(self._redis.delete, key)
            except Exception:
                logger.warning(
                    "TeachingModule: failed to remove pending rule from Redis "
                    "(rule_id=%s).",
                    rule_id,
                    exc_info=True,
                )
        self._pending_local.pop(rule_id, None)
