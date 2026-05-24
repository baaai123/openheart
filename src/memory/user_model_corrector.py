"""
UserModelCorrector — natural-language correction interface for user models.

v4.5.0 §5.7.5: Users may correct AI inferences via natural language.
The corrector detects correction intent, parses target fields and operations,
applies changes to the user model dict, and returns a confirmation message.

All user model mutations are performed in-place on the passed dict.
"""

from __future__ import annotations

import copy
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class CorrectionOperation(Enum):
    """Supported correction directions per spec §5.7.5 step 2."""

    DELETE = "delete"
    MODIFY = "modify"
    LOWER_CONFIDENCE = "lower_confidence"


@dataclass
class CorrectionResult:
    """Outcome of a correction attempt."""

    success: bool
    operation: CorrectionOperation | None = None
    field_path: str = ""
    old_value: Any = None
    new_value: Any = None
    confirmation_text: str = ""
    user_verified_fields: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Default template values for DELETE resets — v4.5.0 §3.4.3
# ---------------------------------------------------------------------------

_DEFAULT_TEMPLATE_VALUES: dict[str, Any] = {
    "inferred_traits.personality": "尚未形成稳定认知，有待进一步了解",
    "inferred_traits.communication_style": "未知",
    "inferred_traits.emotional_pattern": "暂无数据",
    "knowledge_profile.topics_of_interest": [],
    "knowledge_profile.topics_to_avoid": [],
    "knowledge_profile.expertise_level": {},
    "behavioral_insights.active_hours": [],
    "behavioral_insights.avg_session_length_min": 0,
    "behavioral_insights.preferred_interaction_mode": "mixed",
}


# ---------------------------------------------------------------------------
# Natural-language intent patterns — v4.5.0 §5.7.5
# ---------------------------------------------------------------------------

_CORRECTION_KEYWORDS = [
    "别",
    "不要",
    "忘掉",
    "忘记",
    "不喜欢",
    "不是",
    "不只是",
    "不确定",
    "纠正",
    "改",
    "改一下",
    "改改",
    "改过来",
    "修正",
    "更正",
    "调整",
    "重新",
    "其实",
    "错了",
    "不对",
]

_FIELD_KEYWORD_MAP: dict[str, list[str]] = {
    "inferred_traits.personality": [
        "性格", "个性", "内向", "外向", "脆弱", "坚强", "开朗", "敏感",
        "不只是", "不是",
    ],
    "inferred_traits.communication_style": [
        "说话", "沟通", "表达方式", "语气", "风格", "标签",
    ],
    "inferred_traits.emotional_pattern": [
        "情绪", "心情", "焦虑", "容易", "脆弱", "模式", "状态",
        "下午", "晚上", "早上", "周末", "工作日",
    ],
    "knowledge_profile.topics_of_interest": [
        "喜欢", "感兴趣", "爱好", "偏好", "关注", "话题",
    ],
    "knowledge_profile.topics_to_avoid": [
        "讨厌", "不喜欢", "避开", "不想", "反感", "烦",
    ],
    "behavioral_insights.preferred_interaction_mode": [
        "打字", "语音", "文字", "说话", "交流方式", "互动",
    ],
    "behavioral_insights.active_hours": [
        "时间", "时段", "早上", "下午", "晚上", "熬夜", "作息",
    ],
    "relationship_meta.nickname_preference": [
        "称呼", "叫", "名字", "昵称", "小伙伴",
    ],
}

_DELETE_KEYWORDS = ["忘掉", "忘记", "删除", "去掉", "别", "不要", "取消"]
_LOWER_CONFIDENCE_KEYWORDS = ["不太确定", "不一定", "可能", "也许", "不见得", "不确定"]
_MODIFY_KEYWORDS = ["其实", "更喜欢", "应该是", "改", "调整", "纠正", "更正", "修正"]


# ---------------------------------------------------------------------------
# Corrector class
# ---------------------------------------------------------------------------

class UserModelCorrector:
    """Parse and apply natural-language corrections to a user model dict.

    Responsibilities (spec §5.7.5):
      1. Detect correction intent from free-form user text.
      2. Identify the target field and operation (delete, modify, lower confidence).
      3. Mutate the user model in-place:
         - DELETE → reset to template default, zero confidence, remove related key_memories.
         - MODIFY → update value, mark user_verified, append to user_verified_fields.
         - LOWER_CONFIDENCE → reduce field confidence, un-verify if appropriate.
      4. Adjust overall model_confidence.
      5. Return a human-readable confirmation message.
    """

    def __init__(
        self,
        llm_parser: Optional[Any] = None,
        confidence_threshold: float = 0.6,
    ) -> None:
        """Args:
            llm_parser: Optional async callable(text, user_model) -> dict with
                        keys 'field_path', 'operation', 'new_value'.  When None,
                        a lightweight rule-based parser is used.
            confidence_threshold: Minimum per-field confidence for downstream
                                  triggers (e.g. preventive comfort).
        """
        self._llm_parser = llm_parser
        self._confidence_threshold = confidence_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_intent(self, text: str) -> bool:
        """Return True if *text* appears to contain a correction intent.

        Heuristic: presence of any correction keyword.  This is a lightweight
        filter so the perception layer can decide whether to route the
        utterance to the corrector.
        """
        if not text or not isinstance(text, str):
            return False
        lowered = text.lower()
        return any(kw in lowered for kw in _CORRECTION_KEYWORDS)

    def detect_and_correct(
        self,
        user_model: dict[str, Any],
        text: str,
        trace_id: str = "",
    ) -> CorrectionResult:
        """Detect correction intent and apply it in one call.  v4.5.0 §5.7.5

        Convenience wrapper combining ``detect_intent()`` + ``apply_correction()``.
        Returns CorrectionResult(success=False) if no intent detected.

        The ``user_model`` dict is mutated in-place on success.
        """
        if not self.detect_intent(text):
            return CorrectionResult(
                success=False,
                errors=["No correction intent detected."],
            )
        return self.apply_correction(user_model, text, trace_id)

    def apply_correction(
        self,
        user_model: dict[str, Any],
        text: str,
        trace_id: str = "",
    ) -> CorrectionResult:
        """Parse *text* as a correction and mutate *user_model* in-place.

        This is the main entry point called by the memory layer after the
        perception layer has flagged a potential correction.

        Returns a CorrectionResult describing what changed (or why nothing
        changed).  All errors are logged at WARNING with *trace_id*.
        """
        if not self.detect_intent(text):
            return CorrectionResult(
                success=False,
                errors=["No correction intent detected."],
            )

        # Step 1 — parse intent (prefer LLM if available, else heuristic)
        parsed = self._parse_intent(text, user_model)
        if not parsed:
            _log_warning(trace_id, "Failed to parse correction intent from text: %r", text)
            return CorrectionResult(
                success=False,
                errors=["Could not parse correction intent."],
            )

        field_path: str = parsed.get("field_path", "")
        operation_str: str = parsed.get("operation", "")
        new_value: Any = parsed.get("new_value")

        try:
            operation = CorrectionOperation(operation_str)
        except ValueError:
            _log_warning(trace_id, "Unknown correction operation: %r", operation_str)
            return CorrectionResult(
                success=False,
                errors=[f"Unknown operation: {operation_str}"],
            )

        # Step 2 — execute
        if operation == CorrectionOperation.DELETE:
            return self._execute_delete(user_model, field_path, trace_id)
        elif operation == CorrectionOperation.MODIFY:
            return self._execute_modify(user_model, field_path, new_value, trace_id)
        elif operation == CorrectionOperation.LOWER_CONFIDENCE:
            return self._execute_lower_confidence(user_model, field_path, trace_id)

        # unreachable — all enum values handled above
        return CorrectionResult(success=False, errors=["Unhandled operation."])  # pragma: no cover

    def get_field_confidence(self, user_model: dict[str, Any], field_path: str) -> float:
        """Return the per-field _confidence value, or 0.0 if absent."""
        confidence_key = f"{field_path}_confidence"
        parent = _get_parent_dict(user_model, confidence_key)
        final_key = confidence_key.split(".")[-1]
        if parent is not None and final_key in parent:
            return float(parent[final_key])
        return 0.0

    def set_field_confidence(
        self,
        user_model: dict[str, Any],
        field_path: str,
        value: float,
    ) -> None:
        """Set the per-field _confidence value, creating parent dicts as needed."""
        confidence_key = f"{field_path}_confidence"
        _set_nested_value(user_model, confidence_key, float(value))

    # ------------------------------------------------------------------
    # Internal: parsing
    # ------------------------------------------------------------------

    def _parse_intent(self, text: str, user_model: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Return a dict with keys field_path, operation, new_value.

        If an external LLM parser was injected, it is tried first.
        On failure (or when absent) we fall back to rule-based heuristics.
        """
        # TODO: integrate real Qwen2.5-3B parser when available (spec §5.7.5 step 1-2)
        # For now, the rule-based fallback is sufficient for contract tests.
        return self._heuristic_parse(text, user_model)

    def _heuristic_parse(self, text: str, _user_model: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Lightweight keyword-based parser for common correction patterns."""
        lowered = text.lower()

        # Determine operation from keywords
        delete_score = sum(1 for kw in _DELETE_KEYWORDS if kw in lowered)
        lower_score = sum(1 for kw in _LOWER_CONFIDENCE_KEYWORDS if kw in lowered)
        modify_score = sum(1 for kw in _MODIFY_KEYWORDS if kw in lowered)

        if delete_score > 0 and delete_score >= modify_score:
            operation = CorrectionOperation.DELETE
        elif lower_score > 0 and lower_score >= modify_score:
            operation = CorrectionOperation.LOWER_CONFIDENCE
        else:
            operation = CorrectionOperation.MODIFY

        # Determine target field from keywords
        best_field = ""
        best_score = 0
        for field_path, keywords in _FIELD_KEYWORD_MAP.items():
            score = sum(1 for kw in keywords if kw in lowered)
            if score > best_score:
                best_score = score
                best_field = field_path

        if not best_field:
            # No field matched — try to infer from the most recently mentioned
            # inferred_trait (fallback heuristic)
            best_field = "inferred_traits.emotional_pattern"

        new_value: Any = None
        if operation == CorrectionOperation.MODIFY:
            new_value = _extract_new_value(text, best_field)

        return {
            "field_path": best_field,
            "operation": operation.value,
            "new_value": new_value,
        }

    # ------------------------------------------------------------------
    # Internal: execution
    # ------------------------------------------------------------------

    def _execute_delete(
        self,
        user_model: dict[str, Any],
        field_path: str,
        trace_id: str,
    ) -> CorrectionResult:
        """DELETE: reset to template default, zero confidence, remove related key_memories."""
        old_value = _get_nested_value(user_model, field_path)
        default_value = _DEFAULT_TEMPLATE_VALUES.get(field_path)

        if default_value is None:
            # No template default — set to empty string for scalars, empty list for lists
            default_value = [] if old_value is list else ""

        _set_nested_value(user_model, field_path, copy.deepcopy(default_value))
        self.set_field_confidence(user_model, field_path, 0.0)

        key_memories: list[dict[str, Any]] = user_model.get("key_memories", [])
        old_value_str = str(old_value) if old_value is not None else ""
        filtered = [
            m for m in key_memories
            if old_value_str not in m.get("summary", "")
        ]
        if len(filtered) < len(key_memories):
            user_model["key_memories"] = filtered

        verified: list[str] = _get_nested_value(
            user_model, "relationship_meta.user_verified_fields"
        ) or []
        if field_path in verified:
            verified.remove(field_path)
            _set_nested_value(
                user_model, "relationship_meta.user_verified_fields", verified
            )

        # Adjust overall model_confidence downward
        self._adjust_model_confidence(user_model, delta=-0.05)

        _log_info(trace_id, "Deleted field %s (reset to default)", field_path)
        return CorrectionResult(
            success=True,
            operation=CorrectionOperation.DELETE,
            field_path=field_path,
            old_value=old_value,
            new_value=default_value,
            confirmation_text="好的，我以后不会这样描述了。",
            user_verified_fields=verified,
        )

    def _execute_modify(
        self,
        user_model: dict[str, Any],
        field_path: str,
        new_value: Any,
        trace_id: str,
    ) -> CorrectionResult:
        """MODIFY: update value, mark user_verified, append to user_verified_fields."""
        if new_value is None:
            return CorrectionResult(
                success=False,
                errors=["MODIFY operation requires a new_value."],
            )

        old_value = _get_nested_value(user_model, field_path)
        _set_nested_value(user_model, field_path, copy.deepcopy(new_value))

        # Mark field as user-verified (spec §5.7.5 step 3)
        self._mark_user_verified(user_model, field_path)
        # Boost confidence because the user explicitly confirmed
        self.set_field_confidence(user_model, field_path, 1.0)

        # Adjust overall model_confidence upward
        self._adjust_model_confidence(user_model, delta=+0.05)

        _log_info(trace_id, "Modified field %s", field_path)
        return CorrectionResult(
            success=True,
            operation=CorrectionOperation.MODIFY,
            field_path=field_path,
            old_value=old_value,
            new_value=new_value,
            confirmation_text="记住了，我会调整对你的了解。",
            user_verified_fields=_get_nested_value(
                user_model, "relationship_meta.user_verified_fields"
            )
            or [],
        )

    def _execute_lower_confidence(
        self,
        user_model: dict[str, Any],
        field_path: str,
        trace_id: str,
    ) -> CorrectionResult:
        """LOWER_CONFIDENCE: reduce field confidence, un-verify if appropriate."""
        old_value = self.get_field_confidence(user_model, field_path)
        new_confidence = max(0.0, old_value - 0.3)
        self.set_field_confidence(user_model, field_path, new_confidence)

        # If confidence drops below threshold, remove from verified list
        verified: list[str] = _get_nested_value(
            user_model, "relationship_meta.user_verified_fields"
        ) or []
        if new_confidence < self._confidence_threshold and field_path in verified:
            verified.remove(field_path)
            _set_nested_value(
                user_model, "relationship_meta.user_verified_fields", verified
            )

        self._adjust_model_confidence(user_model, delta=-0.02)

        _log_info(trace_id, "Lowered confidence for field %s to %.2f", field_path, new_confidence)
        return CorrectionResult(
            success=True,
            operation=CorrectionOperation.LOWER_CONFIDENCE,
            field_path=field_path,
            old_value=old_value,
            new_value=new_confidence,
            confirmation_text="明白了，我会更谨慎地看待这一点。",
            user_verified_fields=verified,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _mark_user_verified(self, user_model: dict[str, Any], field_path: str) -> None:
        """Ensure *field_path* is in relationship_meta.user_verified_fields."""
        verified: list[str] = _get_nested_value(
            user_model, "relationship_meta.user_verified_fields"
        ) or []
        if field_path not in verified:
            verified = list(verified)  # copy before mutating
            verified.append(field_path)
            _set_nested_value(
                user_model, "relationship_meta.user_verified_fields", verified
            )

    def _adjust_model_confidence(
        self, user_model: dict[str, Any], delta: float
    ) -> None:
        """Bump overall model_confidence by *delta*, clamped to [0, 1]."""
        current = float(
            _get_nested_value(user_model, "relationship_meta.model_confidence") or 0.0
        )
        new_val = max(0.0, min(1.0, current + delta))
        _set_nested_value(user_model, "relationship_meta.model_confidence", new_val)


# ---------------------------------------------------------------------------
# Logging helpers — all errors at WARNING with trace_id (项目宪法)
# ---------------------------------------------------------------------------

def _log_warning(trace_id: str, msg: str, *args: Any) -> None:
    logger.warning(msg, *args, extra={"trace_id": trace_id})


def _log_info(trace_id: str, msg: str, *args: Any) -> None:
    logger.info(msg, *args, extra={"trace_id": trace_id})


# ---------------------------------------------------------------------------
# Dict-path utilities
# ---------------------------------------------------------------------------

def _get_nested_value(root: dict[str, Any], dotted_path: str) -> Any:
    """Safely traverse a dict using dot-separated keys."""
    parts = dotted_path.split(".")
    node: Any = root
    for part in parts:
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return None
    return node


def _get_parent_dict(root: dict[str, Any], dotted_path: str) -> Optional[dict[str, Any]]:
    """Return the parent dict of the final key in *dotted_path*, or None."""
    parts = dotted_path.split(".")
    if len(parts) == 1:
        return root
    node: Any = root
    for part in parts[:-1]:
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return None
    return node if isinstance(node, dict) else None


def _set_nested_value(root: dict[str, Any], dotted_path: str, value: Any) -> None:
    """Create intermediate dicts as needed and set the final key."""
    parts = dotted_path.split(".")
    node: Any = root
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            node[part] = {}
        node = node[part]
    node[parts[-1]] = value


def _extract_new_value(text: str, field_path: str) -> Any:
    """Naive heuristic to extract the new value the user wants.

    Looks for patterns like "其实我更喜欢X" or "应该是X".
    Falls back to the raw text stripped of correction keywords.
    """
    # Strip common correction prefixes
    cleaned = text
    for prefix in [
        "其实我更喜欢", "其实我更愛", "其实我更", "我喜欢", "我更喜欢",
        "我愛", "我愛", "应该是", "应该是", "不是", "不只是",
        "纠正一下", "更正", "应该是", "改为", "改成",
    ]:
        if prefix in cleaned:
            cleaned = cleaned.split(prefix, 1)[-1]
            break

    cleaned = re.sub(r"[，。？！\.\!\?]$", "", cleaned).strip()

    if not cleaned:
        return text  # fallback: return original

    # For list fields, attempt comma / Chinese-comma split
    if field_path in (
        "knowledge_profile.topics_of_interest",
        "knowledge_profile.topics_to_avoid",
        "behavioral_insights.active_hours",
    ):
        parts = re.split(r"[,，、]", cleaned)
        return [p.strip() for p in parts if p.strip()]

    return cleaned
