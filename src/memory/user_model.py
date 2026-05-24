"""
User Model — v4.5.0 §3.4

The UserModel is the system's holistic understanding of the user, stored
in cold memory as a dedicated node with versioning.  Generated/updated by
Qwen2.5-3B and consumed by Decision, Personality, Memory, and Prediction layers.

Key features:
  - Monotonic version management (int64, increment on update)
  - relationship_meta.model_confidence and user_verified_fields (§3.4.1)
  - Per-field _confidence companions (e.g., personality_confidence: 0.3)
  - New-user fallback template when cold_memory:initialized is false
  - Local-only storage — no cloud upload (§3.4 design note)
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# New-user fallback template — v4.5.0 §3.4.3
# ---------------------------------------------------------------------------

NEW_USER_FALLBACK_TEMPLATE: dict[str, Any] = {
    "inferred_traits": {
        "personality": "尚未形成稳定认知，有待进一步了解",
        "communication_style": "未知",
        "emotional_pattern": "暂无数据",
    },
    "knowledge_profile": {
        "topics_of_interest": [],
        "topics_to_avoid": [],
        "expertise_level": {},
    },
    "behavioral_insights": {
        "active_hours": [],
        "avg_session_length_min": 0,
        "preferred_interaction_mode": "mixed",
    },
    "key_memories": [],
    "relationship_meta": {
        "first_interaction_date": "",  # filled at creation time
        "total_interaction_hours": 0.0,
        "relationship_stage": "new",
        "nickname_preference": "",
        "model_confidence": 0.0,        # §3.4.1: overall confidence [0,1]
        "user_verified_fields": [],     # §3.4.1: fields confirmed by user
    },
}

# new_user_fallback System Prompt — v4.5.0 §5.4
NEW_USER_FALLBACK = (
    "你是第一次和我聊天的朋友，我还不太了解你，但我会用心倾听。"
)

# Injected values when UserModel version < 2 — v4.5.0 §5.4
NEW_USER_SYSTEM_PROMPT_VALUES: dict[str, str] = {
    "personality": "我正在慢慢了解你",
    "topics_of_interest": "目前还在发现中",
    "topics_to_avoid": "",
    "relationship_stage": "初次相识",
    "nickname": "",
}

# ---------------------------------------------------------------------------
# UserModel dataclass — v4.5.0 §3.4.1
# ---------------------------------------------------------------------------


@dataclass
class UserModel:
    """
    The system's comprehensive cognitive model of the user.

    Attributes:
        user_model_id: Unique UUID for this model instance.
        version: Monotonic int64 — incremented on each model update.
        generated_at: ISO8601 timestamp of last generation.
        inferred_traits: {personality, communication_style, emotional_pattern}.
        knowledge_profile: {topics_of_interest, topics_to_avoid, expertise_level}.
        behavioral_insights: {active_hours, avg_session_length_min,
            preferred_interaction_mode}.
        key_memories: List of significant memory references.
        relationship_meta: Contains first_interaction_date,
            total_interaction_hours, relationship_stage, nickname_preference,
            model_confidence, user_verified_fields (§3.4.1).
        _confidence_store: Per-field confidence values [0,1] — flattened
            to top-level keys (e.g., personality_confidence) in to_dict().
    """

    user_model_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    version: int = 1
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    inferred_traits: dict[str, str] = field(default_factory=dict)
    knowledge_profile: dict[str, Any] = field(default_factory=dict)
    behavioral_insights: dict[str, Any] = field(default_factory=dict)
    key_memories: list[dict[str, Any]] = field(default_factory=list)
    relationship_meta: dict[str, Any] = field(default_factory=dict)

    # Internal: field_path -> confidence [0,1] — §3.4.1
    _confidence_store: dict[str, float] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Factory — v4.5.0 §3.4.3  new user initialization
    # ------------------------------------------------------------------

    @classmethod
    def new_user(
        cls, first_interaction_date: Optional[str] = None
    ) -> "UserModel":
        """
        Create a fresh UserModel from the fallback template.

        Used when cold_memory:initialized is false OR when no cold-memory
        scenes satisfy importance>0.2 + within 30 days.

        v4.5.0 §3.4.3: version=1, template values pre-set.
        """
        from copy import deepcopy

        now = datetime.now(timezone.utc).isoformat()
        initial_date = first_interaction_date or now

        return cls(
            user_model_id=str(uuid.uuid4()),
            version=1,
            generated_at=now,
            inferred_traits=deepcopy(
                NEW_USER_FALLBACK_TEMPLATE["inferred_traits"]
            ),
            knowledge_profile=deepcopy(
                NEW_USER_FALLBACK_TEMPLATE["knowledge_profile"]
            ),
            behavioral_insights=deepcopy(
                NEW_USER_FALLBACK_TEMPLATE["behavioral_insights"]
            ),
            key_memories=[],
            relationship_meta={
                **deepcopy(
                    NEW_USER_FALLBACK_TEMPLATE["relationship_meta"]
                ),
                "first_interaction_date": initial_date,
            },
            _confidence_store={},
        )

    # ------------------------------------------------------------------
    # Version management — v4.5.0 §3.4.1 / §3.4.3
    # ------------------------------------------------------------------

    @property
    def is_new_user(self) -> bool:
        """
        True when version < 2 (only pre-set template data).

        v4.5.0 §5.4: System Prompt uses new_user_fallback for version < 2.
        As the model version rises, fields auto-switch to model outputs.
        """
        return self.version < 2

    def bump_version(self) -> None:
        """
        Increment version by 1 and refresh generated_at.

        v4.5.0 §3.4.3: version is monotonically increasing int64.
        Called after each Qwen2.5-3B model update.
        """
        self.version += 1
        self.generated_at = datetime.now(timezone.utc).isoformat()

    # ------------------------------------------------------------------
    # Confidence & verification — v4.5.0 §3.4.1
    # ------------------------------------------------------------------

    def set_confidence(self, field_path: str, confidence: float) -> None:
        """
        Set per-field model confidence.

        Args:
            field_path: Dot-separated path e.g. 'inferred_traits.personality'.
                Stored flattened as 'personality_confidence' in to_dict().
            confidence: [0.0, 1.0]; clamped internally.

        v4.5.0 §3.4.1: optional sibling _confidence for each inferable field.
        """
        _clamped = max(0.0, min(1.0, float(confidence)))
        self._confidence_store[field_path] = _clamped

    def get_confidence(self, field_path: str) -> Optional[float]:
        """Return per-field confidence, or None if not set."""
        return self._confidence_store.get(field_path)

    def is_field_verified(self, field_path: str) -> bool:
        """
        Check whether the user has explicitly confirmed/corrected a field.

        v4.5.0 §3.4.1: user_verified_fields tracks user-confirmed fields.
        """
        verified: list[str] = self.relationship_meta.get(
            "user_verified_fields", []
        )
        return field_path in verified

    def mark_field_verified(self, field_path: str) -> None:
        """
        Mark a field as verified by the user.

        Adds field_path to relationship_meta.user_verified_fields if not
        already present.
        """
        verified: list[str] = self.relationship_meta.setdefault(
            "user_verified_fields", []
        )
        if field_path not in verified:
            verified.append(field_path)

    @property
    def model_confidence(self) -> float:
        """
        Overall model confidence [0, 1] — v4.5.0 §3.4.1.

        Stored in relationship_meta.model_confidence. Defaults to 0.0.
        """
        return float(self.relationship_meta.get("model_confidence", 0.0))

    @model_confidence.setter
    def model_confidence(self, value: float) -> None:
        clamped = max(0.0, min(1.0, float(value)))
        self.relationship_meta["model_confidence"] = clamped

    # ------------------------------------------------------------------
    # System Prompt extraction — v4.5.0 §5.4
    # ------------------------------------------------------------------

    def get_system_prompt_values(self) -> dict[str, str]:
        """
        Return values for System Prompt user-model injection.

        If version < 2 (new user), returns fallback values per §5.4.
        Otherwise returns actual model field values.
        """
        if self.is_new_user:
            return dict(NEW_USER_SYSTEM_PROMPT_VALUES)

        traits = self.inferred_traits
        knowledge = self.knowledge_profile
        meta = self.relationship_meta

        return {
            "personality": traits.get("personality", ""),
            "topics_of_interest": ", ".join(
                knowledge.get("topics_of_interest", [])
            ),
            "topics_to_avoid": ", ".join(
                knowledge.get("topics_to_avoid", [])
            ),
            "relationship_stage": meta.get("relationship_stage", ""),
            "nickname": meta.get("nickname_preference", ""),
        }

    def get_summary_for_injection(self) -> str:
        """
        Generate a concise human-readable summary for System Prompt.

        v4.5.0 §5.4: injected as "你了解这位用户：…"
        """
        vals = self.get_system_prompt_values()
        topics_interest = vals["topics_of_interest"] or "正在了解中"
        topics_avoid = vals["topics_to_avoid"] or "暂无"
        nickname = vals["nickname"] or "（暂不称呼昵称）"

        return (
            "你了解这位用户：\n"
            f"  - 性格：{vals['personality']}\n"
            f"  - 近期关注：{topics_interest}\n"
            f"  - 避免话题：{topics_avoid}\n"
            f"  - 你们的关系阶段：{vals['relationship_stage']}\n"
            f"  - 你可以称呼他：{nickname}"
        )

    # ------------------------------------------------------------------
    # Serialization — v4.5.0 §3.4.1
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UserModel":
        """
        Deserialize from a dict (e.g., cold memory or mock).

        Recognises top-level _confidence_* keys and reconstructs
        _confidence_store from them.
        """
        valid_fields = {
            "user_model_id",
            "version",
            "generated_at",
            "inferred_traits",
            "knowledge_profile",
            "behavioral_insights",
            "key_memories",
            "relationship_meta",
        }
        core: dict[str, Any] = {}
        confidence_store: dict[str, float] = {}

        for k, v in data.items():
            if k == "_confidence_store" and isinstance(v, dict):
                # Re-hydrate if serialised as-is
                confidence_store.update(
                    {field: float(c) for field, c in v.items()}
                )
            elif k.endswith("_confidence") and k != "model_confidence":
                # Top-level confidence key → store field path
                field_name = k[: -len("_confidence")]
                confidence_store[field_name] = float(v)
            elif k in valid_fields:
                core[k] = v

        # Ensure relationship_meta has the v4.5.0 §3.4.1 additions
        rm = core.get("relationship_meta", {})
        if isinstance(rm, dict):
            rm.setdefault("model_confidence", 0.0)
            rm.setdefault("user_verified_fields", [])
        else:
            rm = {"model_confidence": 0.0, "user_verified_fields": []}
        core["relationship_meta"] = rm

        instance = cls(**core)
        instance._confidence_store = confidence_store
        return instance

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize to a flat dict for storage or consumption.

        Confidence values from _confidence_store are flattened to top-level
        keys matching the convention: field_path → field_name_confidence.

        Example:
            _confidence_store = {"personality": 0.3}
            → to_dict() includes "personality_confidence": 0.3
        """
        result: dict[str, Any] = asdict(self)
        # Flatten confidence store to top-level keys
        if self._confidence_store:
            for field_path, conf_value in self._confidence_store.items():
                # Use the last segment of the path as the key name
                key_name = field_path.rsplit(".", 1)[-1]
                result[f"{key_name}_confidence"] = conf_value
        # Remove internal store from output
        result.pop("_confidence_store", None)
        return result

    # ------------------------------------------------------------------
    # Dict-like access for consumers that expect dict[str, Any]
    # ------------------------------------------------------------------

    def __getitem__(self, key: str) -> Any:
        """Allow dict-style read access for compatibility."""
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        """Allow .get() access for dict-compatible consumers."""
        return getattr(self, key, default)


# ---------------------------------------------------------------------------
# cold_memory:initialized sentinel check — v4.5.0 §3.2.4 step 6, §3.4.3
# ---------------------------------------------------------------------------


def check_cold_memory_initialized(redis_client: Any) -> bool:
    """
    Check whether the cold_memory:initialized sentinel is set in Redis.

    Returns True if the sentinel value is truthy ("true" or "1").
    Returns False if the key is missing, the client is None, or any error
    occurs — defaulting to "new user" mode for safety.

    v4.5.0 §3.4.3: this check gates new-user template vs. model generation.
    §3.2.4 step 6: the key is set with no TTL on first successful sync.
    """
    if redis_client is None:
        logger.warning(
            "check_cold_memory_initialized: no Redis client available. "
            "Assuming cold memory is NOT initialized (new user mode)."
        )
        return False

    try:
        value = redis_client.get("cold_memory:initialized")
        if value is None:
            return False
        # Redis returns bytes; normalise
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        return str(value).lower() in ("true", "1")
    except Exception as e:
        # Non-fatal: if we can't check, assume not initialised for safety
        logger.warning(
            "Failed to read cold_memory:initialized sentinel: %s. "
            "Falling back to new user mode.",
            e,
        )
        return False
