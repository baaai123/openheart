"""
Safety Classifier — v4.5.0 §5.7.2

Classifies decision commands into three safety levels:
  · SAFE                — no risk, execute directly.
  · NEEDS_CONFIRM       — moderate risk, require verbal user confirmation.
  · DANGEROUS_AUTO_BLOCK — high risk, auto-block and warn.

Classification is based on action types and keyword matching in the
voice_response text.  All decisions are logged with trace_id at WARNING
level when the level is not SAFE.

项目宪法 §1.3: 所有错误必须通过 WARNING 级别日志输出，并包含 trace_id。
"""
from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Safety level constants — 项目宪法 §2.1
# ---------------------------------------------------------------------------

SAFE: str = "SAFE"
NEEDS_CONFIRM: str = "NEEDS_CONFIRM"
DANGEROUS_AUTO_BLOCK: str = "DANGEROUS_AUTO_BLOCK"

VALID_SAFETY_LEVELS: frozenset[str] = frozenset({SAFE, NEEDS_CONFIRM, DANGEROUS_AUTO_BLOCK})

# ---------------------------------------------------------------------------
# Keyword tables — v4.5.0 §5.7.2
# ---------------------------------------------------------------------------

# DANGEROUS: payment, deleting data, system settings
_DANGEROUS_KEYWORDS: list[str] = [
    "支付", "付款", "转账", "购买", "下单", "扣款",
    "删除", "清空", "格式化", "销毁", "卸载",
    "系统设置", "注册表", "防火墙", "权限", "管理员",
    "root", "sudo", "rm -rf", "format", "delete all",
    "pay", "payment", "purchase", "transfer", "withdraw",
]

# NEEDS_CONFIRM: sending messages, modifying files
_NEEDS_CONFIRM_KEYWORDS: list[str] = [
    "发送", "转发", "上传", "发布", "提交",
    "修改", "编辑", "保存", "覆盖", "写入",
    "send", "post", "upload", "submit", "modify",
    "edit", "save", "overwrite", "write to",
]

# Compiled regexes for efficiency
_DANGEROUS_RE = re.compile(
    "|".join(re.escape(k) for k in _DANGEROUS_KEYWORDS),
    flags=re.IGNORECASE,
)
_NEEDS_CONFIRM_RE = re.compile(
    "|".join(re.escape(k) for k in _NEEDS_CONFIRM_KEYWORDS),
    flags=re.IGNORECASE,
)

# Action-type safety defaults (v4.5.0 §5.3.1 action.type)
# Any action not listed defaults to SAFE.
_ACTION_TYPE_DEFAULTS: dict[str, str] = {
    "mouse_click": SAFE,
    "mouse_move": SAFE,
    "keyboard_input": SAFE,
    "voice_response": SAFE,
    "animation_trigger": SAFE,
    "composite": SAFE,
}

# ---------------------------------------------------------------------------
# SafetyClassifier
# ---------------------------------------------------------------------------


class SafetyClassifier:
    """Classifies decision commands by safety level.

    Usage::

        classifier = SafetyClassifier()
        level = classifier.classify(decision_command, trace_id="...")
    """

    def __init__(self) -> None:
        """Initialise the classifier with compiled keyword tables."""
        # Keyword tables are module-level constants; no per-instance state needed.
        pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(
        self,
        decision_command: dict[str, Any],
        trace_id: str = "",
    ) -> str:
        """Return the safety level for *decision_command*.

        The classifier examines (in order of precedence):
          1. Any explicit ``safety_level`` field already present.
          2. Action types in ``command.actions``.
          3. Keywords in ``command.voice_response``.

        DANGEROUS_AUTO_BLOCK > NEEDS_CONFIRM > SAFE.

        Args:
            decision_command: Dict matching the decision command schema
                (see test_decision_contract.VALID_DECISION_COMMAND).
            trace_id: Correlation ID for logging.

        Returns:
            One of ``SAFE``, ``NEEDS_CONFIRM``, ``DANGEROUS_AUTO_BLOCK``.
        """
        # 1. Respect an explicit safety_level if already set and valid.
        explicit = decision_command.get("safety_level")
        if explicit in VALID_SAFETY_LEVELS:
            self._log_decision(explicit, trace_id, reason="explicit")
            return explicit

        command = decision_command.get("command", {})
        actions = command.get("actions", [])
        voice_response = command.get("voice_response", "")

        # 2. Check action types.
        level_from_actions = self._classify_actions(actions)

        # 3. Check voice_response text.
        level_from_text = self._classify_text(voice_response)

        # Precedence: DANGEROUS > NEEDS_CONFIRM > SAFE
        final_level = self._merge_levels(level_from_actions, level_from_text)

        self._log_decision(final_level, trace_id, reason="auto_classified")
        return final_level

    def get_confirmation_prompt(self, decision_command: dict[str, Any]) -> str:
        """Return the TTS prompt to ask the user for confirmation.

        v4.5.0 §5.7.2: 向用户口头确认后进入观察期.
        """
        action_desc = self._describe_actions(decision_command)
        return (
            f"这个操作{action_desc}可能需要确认一下哦。"
            f"你确定要执行吗？请回答'确定'或'取消'。"
        )

    def get_block_warning(self, decision_command: dict[str, Any]) -> str:
        """Return the TTS warning when a dangerous action is auto-blocked.

        v4.5.0 §5.7.2: 自动阻止，提示用户该操作不安全.
        """
        action_desc = self._describe_actions(decision_command)
        return (
            f"唔……这个操作{action_desc}看起来有点危险呢，"
            f"为了保护你的安全，我不能执行哦～"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_actions(actions: list[dict[str, Any]]) -> str:
        """Classify based on action types and params.

        Returns the most restrictive level found across all actions.
        """
        max_level = SAFE
        for action in actions:
            if isinstance(action, str):
                action_type = action
            else:
                action_type = action.get("type", "")
            params = {} if isinstance(action, str) else action.get("params", {})

            # Default level for this action type.
            level = _ACTION_TYPE_DEFAULTS.get(action_type, SAFE)

            # Inspect params for dangerous keywords.
            param_text = " ".join(str(v) for v in params.values() if isinstance(v, str))
            if _DANGEROUS_RE.search(param_text):
                level = DANGEROUS_AUTO_BLOCK
            elif _NEEDS_CONFIRM_RE.search(param_text) and level != DANGEROUS_AUTO_BLOCK:
                level = NEEDS_CONFIRM

            max_level = SafetyClassifier._merge_levels(max_level, level)
            if max_level == DANGEROUS_AUTO_BLOCK:
                break  # Can't get more restrictive.
        return max_level

    @staticmethod
    def _classify_text(text: str) -> str:
        """Classify based on keywords in *text*."""
        if _DANGEROUS_RE.search(text):
            return DANGEROUS_AUTO_BLOCK
        if _NEEDS_CONFIRM_RE.search(text):
            return NEEDS_CONFIRM
        return SAFE

    @staticmethod
    def _merge_levels(a: str, b: str) -> str:
        """Return the more restrictive of two safety levels."""
        if DANGEROUS_AUTO_BLOCK in (a, b):
            return DANGEROUS_AUTO_BLOCK
        if NEEDS_CONFIRM in (a, b):
            return NEEDS_CONFIRM
        return SAFE

    @staticmethod
    def _describe_actions(decision_command: dict[str, Any]) -> str:
        """Produce a short human-readable description of the actions."""
        actions = decision_command.get("command", {}).get("actions", [])
        if not actions:
            return ""
        types = [a.get("type", "unknown") for a in actions]
        return "（" + ", ".join(types) + "）"

    @staticmethod
    def _log_decision(level: str, trace_id: str, reason: str) -> None:
        """Log safety decisions at WARNING when not SAFE.

        项目宪法 §1.3: 所有错误必须通过 WARNING 级别日志输出，并包含 trace_id。
        Here we log all non-SAFE decisions (including auto-blocks) at WARNING.
        """
        if level != SAFE:
            logger.warning(
                "Safety decision: level=%s reason=%s trace_id=%s",
                level, reason, trace_id or "unknown",
            )
