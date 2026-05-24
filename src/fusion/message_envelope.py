"""
Unified message envelope — v4.5.0 §0.3

All layer-to-layer communication MUST use this format. Every message carries a
monotonic version counter scoped to its trace_id, enabling downstream modules
to detect gaps and replay.

Usage:
    from fusion.message_envelope import create_message, Message, Layer

    msg: MessageEnvelope = create_envelope(
        source_layer=Layer.PERCEPTION,
        source_component="visual_processor",
        payload_type=PayloadType.PERCEPTION_EVENT,
        payload={"text": "hello"},
    )
    d: dict = msg.to_dict()          # → JSON-serialisable dict
    msg2: Message = MessageEnvelope.from_dict(d)  # → round-trip
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums — v4.5.0 §0.3
# ---------------------------------------------------------------------------

class Layer(str, Enum):
    """Valid source_layer values — v4.5.0 §0.3."""
    PERCEPTION = "perception"
    FUSION = "fusion"
    MEMORY_HOT = "memory_hot"
    MEMORY_COLD = "memory_cold"
    PERSONALITY = "personality"
    DECISION = "decision"
    PREDICTION = "prediction"
    EXECUTION = "execution"


class PayloadType(str, Enum):
    """Valid payload_type values — v4.5.0 §0.3."""
    PERCEPTION_EVENT = "perception_event"
    SCENE = "scene"
    MEMORY_QUERY = "memory_query"
    PERSONALITY_CONFIG = "personality_config"
    DECISION_COMMAND = "decision_command"
    PREDICTION_TASK = "prediction_task"
    ACTION_SEQUENCE = "action_sequence"
    AFFECTIVE_EVENT = "affective_event"
    USER_MODEL_UPDATE = "user_model_update"


class EmotionCategory(str, Enum):
    """
    Emotion categories — v4.5.0 §0.3 constraint:
    Only joy, sadness, neutral are reliably produced by the current text
    sentiment analyser.  anger and surprise are placeholder enums —
    downstream modules MUST NOT branch on them unless config/sentiment.yaml
    has provider: "structbert".
    """
    JOY = "joy"
    SADNESS = "sadness"
    NEUTRAL = "neutral"
    ANGER = "anger"         # placeholder
    SURPRISE = "surprise"   # placeholder


# ---------------------------------------------------------------------------
# Exception — v4.5.0 §0.3 enforcement
# ---------------------------------------------------------------------------

class MessageValidationError(ValueError):
    """Raised when a message envelope fails structural validation."""


# ---------------------------------------------------------------------------
# Nestable dataclasses — metadata subtree of the envelope
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Emotion:
    """
    User emotion estimate from the perception layer.

    v4.5.0 §0.3: Emotion is the ONLY user-emotion source in the pipeline.
    category: Only joy/sadness/neutral are reliable (see EmotionCategory).
    intensity: Float [0, 1].
    source: Fixed to "text_sentiment" (voice feature extraction not yet enabled).
    confidence: Float [0, 1] — confidence of the emotion classification.
    """

    category: str
    intensity: float
    source: str = "text_sentiment"
    confidence: float = 0.0

    def __post_init__(self) -> None:
        if self.source != "text_sentiment":
            logger.warning(
                "emotion.source=%r — currently only text_sentiment is active "
                "(v4.5.0 §0.3). Voice feature extraction is reserved.",
                self.source,
            )


@dataclass(frozen=True)
class SceneContext:
    """
    Scene classification from the perception layer — v4.5.0 §0.3.

    primary_type: e.g. "code_editor", "browser", "desktop", "terminal"
    confidence: Float [0, 1].
    """

    primary_type: str
    confidence: float


@dataclass
class Metadata:
    """
    Message metadata — v4.5.0 §0.3.

    confidence: Float [0, 1] — overall confidence of the message content.
    latency_ms: Processing latency in milliseconds.
    degraded: True when this message was produced by a degradation path.
    fast_path: True when this message took the fast (shadow-skipped) path.
    emotion: User emotion estimate (Emotion dataclass).
    affective_flag: True when this message carries an emotion highlight.
    scene_context: Scene classification (SceneContext dataclass).
    user_model_version: Integer — version of the user model in use.
    """

    confidence: float
    latency_ms: float
    degraded: bool
    fast_path: bool
    emotion: Emotion
    affective_flag: bool
    scene_context: SceneContext
    user_model_version: int


# ---------------------------------------------------------------------------
# Core Message dataclass — v4.5.0 §0.3
# ---------------------------------------------------------------------------

@dataclass
class MessageEnvelope:
    """
    Unified message envelope for ALL layer-to-layer communication.

    v4.5.0 §0.3 — all fields required, no defaults for core identity fields.

    trace_id: UUID v4 — persists across the full interaction chain.
    source_layer: One of Layer enum values.
    source_component: Name of the producing component.
    timestamp: ISO 8601 with millisecond precision.
    version: Monotonic counter within the same trace_id.
    payload_type: One of PayloadType enum values.
    payload: Free-form dict carrying layer-specific data.
    metadata: Metadata dataclass (see above).
    """

    trace_id: str
    source_layer: str
    source_component: str
    timestamp: str
    version: int
    payload_type: str
    payload: dict[str, Any]
    metadata: Metadata

    @staticmethod
    def _iso_now() -> str:
        """Return current UTC time in ISO 8601 with millisecond precision."""
        return datetime.now(timezone.utc).isoformat(timespec="milliseconds")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the message envelope to a JSON-compatible dict.

        Returns a dict that can be passed to json.dumps() or sent over a
        WebSocket.  Metadata is recursively serialised via dataclasses.asdict.
        """
        return {
            "trace_id": self.trace_id,
            "source_layer": self.source_layer,
            "source_component": self.source_component,
            "timestamp": self.timestamp,
            "version": self.version,
            "payload_type": self.payload_type,
            "payload": self.payload,
            "metadata": asdict(self.metadata),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MessageEnvelope:
        """Deserialise a dict into a Message dataclass.

        Applies structural validation before construction.  Raises
        MessageValidationError on fatal structural issues; logs WARNING
        for non-fatal field anomalies (degraded messages, out-of-range
        confidence, etc.).

        Args:
            d: A dict matching the envelope format defined in v4.5.0 §0.3.

        Returns:
            A fully-constructed Message instance.

        Raises:
            MessageValidationError: when required fields are missing or
                types are wrong.
        """
        # v4.5.0 §0.3 — structural guard: every envelope MUST have these keys.
        required_top = (
            "trace_id", "source_layer", "source_component", "timestamp",
            "version", "payload_type", "payload", "metadata",
        )
        for key in required_top:
            if key not in d:
                raise MessageValidationError(
                    f"Missing required top-level field: {key}"
                )

        trace_id = d["trace_id"]
        # Validate trace_id is a well-formed UUID v4 string (log, not fail,
        # for backward compatibility).
        try:
            uid = uuid.UUID(str(d["trace_id"]))
            if uid.version != 4:
                logger.warning(
                    "Message envelope trace_id %r is UUID v%d, expected v4 "
                    "(v4.5.0 §0.3). Proceeding anyway.",
                    d["trace_id"], uid.version,
                )
        except (ValueError, AttributeError):
            # trace_id is not a valid UUID — log warning, do not reject
            # (degraded producers may emit non-UUID trace_ids).
            logger.warning(
                "Message envelope trace_id %r is not a valid UUID "
                "(v4.5.0 §0.3). Proceeding anyway — trace_id: %s",
                d["trace_id"], trace_id,
            )

        # Validate source_layer against the Layer enum.
        try:
            Layer(str(d["source_layer"]))
        except ValueError:
            logger.warning(
                "Unknown source_layer %r in message from %s (trace_id: %s). "
                "Valid values: %s",
                d["source_layer"], d.get("source_component", "?"),
                trace_id, [e.value for e in Layer],
            )

        # Validate payload_type against the PayloadType enum.
        try:
            PayloadType(str(d["payload_type"]))
        except ValueError:
            logger.warning(
                "Unknown payload_type %r in message from %s (trace_id: %s). "
                "Valid values: %s",
                d["payload_type"], d.get("source_component", "?"),
                trace_id, [e.value for e in PayloadType],
            )

        # Validate version is int.
        version = d["version"]
        if not isinstance(version, int):
            # v4.5.0 §0.3: version is int64 monotonic counter.
            raise MessageValidationError(
                f"version must be int, got {type(version).__name__} "
                f"(trace_id: {trace_id})"
            )

        # Build metadata subtree with structural validation.
        meta = d["metadata"]
        meta_required = (
            "confidence", "latency_ms", "degraded", "fast_path",
            "emotion", "affective_flag", "scene_context", "user_model_version",
        )
        for key in meta_required:
            if key not in meta:
                raise MessageValidationError(
                    f"Missing required metadata field: {key} (trace_id: {trace_id})"
                )

        # Validate degraded is boolean — v4.5.0 §0.3
        degraded = meta["degraded"]
        if not isinstance(degraded, bool):
            raise MessageValidationError(
                f"metadata.degraded must be bool, got {type(degraded).__name__} "
                f"(trace_id: {trace_id})"
            )
        if degraded:
            # Log every degraded message at WARNING per 项目宪法 §4.3.
            logger.warning(
                "Message with degraded=true from %s / %s (trace_id: %s)",
                d["source_layer"], d.get("source_component", "?"), trace_id,
            )

        # Validate fast_path is boolean.
        fast_path = meta["fast_path"]
        if not isinstance(fast_path, bool):
            raise MessageValidationError(
                f"metadata.fast_path must be bool, got {type(fast_path).__name__} "
                f"(trace_id: {trace_id})"
            )

        # Validate affective_flag is boolean.
        affective_flag = meta["affective_flag"]
        if not isinstance(affective_flag, bool):
            raise MessageValidationError(
                f"metadata.affective_flag must be bool, got "
                f"{type(affective_flag).__name__} (trace_id: {trace_id})"
            )

        # Validate confidence [0, 1].
        confidence = meta["confidence"]
        if not isinstance(confidence, (int, float)):
            raise MessageValidationError(
                f"metadata.confidence must be float, got "
                f"{type(confidence).__name__} (trace_id: {trace_id})"
            )
        if not (0.0 <= confidence <= 1.0):
            logger.warning(
                "metadata.confidence=%.3f out of [0,1] from %s / %s "
                "(trace_id: %s). Clamping.",
                confidence, d["source_layer"],
                d.get("source_component", "?"), trace_id,
            )
            # We do NOT clamp — preserve original for downstream inspection.

        # Build Emotion sub-dataclass — v4.5.0 §0.3 forbids emotion.type
        emo = meta["emotion"]
        if "type" in emo:
            logger.warning(
                "emotion.type detected in message envelope from %s — "
                "this field is FORBIDDEN per v4.5.0 §0.3 and 项目宪法 §2.2. "
                "Use emotion.category instead. (trace_id: %s)",
                d.get("source_component", "?"), trace_id,
            )
            # Remove the forbidden key so it does not propagate.
            emo = {k: v for k, v in emo.items() if k != "type"}

        if "category" not in emo:
            raise MessageValidationError(
                "metadata.emotion.category is required (v4.5.0 §0.3) "
                f"(trace_id: {trace_id})"
            )

        # Validate emotion.category is a known value.
        emotion_cat = str(emo["category"])
        try:
            EmotionCategory(emotion_cat)
        except ValueError:
            logger.warning(
                "Unknown emotion.category %r from %s (trace_id: %s). "
                "Valid values: %s",
                emotion_cat, d.get("source_component", "?"),
                trace_id, [e.value for e in EmotionCategory],
            )

        # v4.5.0 §0.3: anger and surprise are placeholder — log at WARNING
        # if downstream code is branching on them (we can't enforce that here,
        # but we can at least note their presence).
        if emotion_cat in (EmotionCategory.ANGER.value, EmotionCategory.SURPRISE.value):
            logger.warning(
                "emotion.category=%r is a placeholder enum (v4.5.0 §0.3). "
                "Downstream modules MUST NOT branch on it unless "
                "config/sentiment.yaml has provider: 'structbert'. "
                "(trace_id: %s)",
                emotion_cat, trace_id,
            )

        emotion = Emotion(
            category=emotion_cat,
            intensity=float(emo.get("intensity", 0.0)),
            source=str(emo.get("source", "text_sentiment")),
            confidence=float(emo.get("confidence", 0.0)),
        )

        # Build SceneContext.
        sc = meta["scene_context"]
        if not isinstance(sc, dict):
            raise MessageValidationError(
                f"metadata.scene_context must be dict, got "
                f"{type(sc).__name__} (trace_id: {trace_id})"
            )
        scene_context = SceneContext(
            primary_type=str(sc.get("primary_type", "unknown")),
            confidence=float(sc.get("confidence", 0.0)),
        )

        # Build Metadata.
        metadata = Metadata(
            confidence=float(confidence),
            latency_ms=float(meta["latency_ms"]),
            degraded=bool(degraded),
            fast_path=bool(fast_path),
            emotion=emotion,
            affective_flag=bool(affective_flag),
            scene_context=scene_context,
            user_model_version=int(meta["user_model_version"]),
        )

        return cls(
            trace_id=str(trace_id),
            source_layer=str(d["source_layer"]),
            source_component=str(d["source_component"]),
            timestamp=str(d["timestamp"]),
            version=int(version),
            payload_type=str(d["payload_type"]),
            payload=dict(d["payload"]),
            metadata=metadata,
        )


# ---------------------------------------------------------------------------
# Version counter — scoped to trace_id, monotonic
# ---------------------------------------------------------------------------

# Internal registry mapping trace_id → next version number.
# v4.5.0 §0.3: version monotonically increments within the same trace_id.
_version_counters: dict[str, int] = {}


def _next_version(trace_id: str) -> int:
    """Return and increment the version counter for the given trace_id.

    v4.5.0 §0.3: version is a monotonic counter within the same trace_id.
    The first call for a given trace_id returns 1.
    """
    global _version_counters
    current: int = _version_counters.get(trace_id, 0)
    current += 1
    _version_counters[trace_id] = current
    return current


# ---------------------------------------------------------------------------
# Factory function — the primary way to create envelopes
# ---------------------------------------------------------------------------


def create_envelope(
    source_layer: str | Layer,
    source_component: str,
    payload_type: str | PayloadType,
    payload: dict[str, Any],
    *,
    trace_id: str | None = None,
    confidence: float = 1.0,
    latency_ms: float = 0.0,
    degraded: bool = False,
    fast_path: bool = False,
    emotion_category: EmotionCategory = EmotionCategory.NEUTRAL,
    emotion_intensity: float = 0.0,
    emotion_confidence: float = 0.0,
    affective_flag: bool = False,
    scene_primary_type: str = "unknown",
    scene_confidence: float = 0.0,
    user_model_version: int = 0,
) -> MessageEnvelope:
    """Create a message envelope with auto-generated trace_id and version.

    v4.5.0 §0.3 Factory: trace_id is auto-generated (UUID v4), timestamp is
    current UTC ISO 8601 with milliseconds, version is auto-incremented per
    trace_id.

    Args:
        source_layer: Source layer (Layer enum or str).
        source_component: Name of the producing component.
        payload_type: Type of payload (PayloadType enum or str).
        payload: Free-form payload dict.
        trace_id: Optional explicit trace_id. If None, generates a new UUID v4.
        confidence: Overall confidence [0, 1]. Defaults to 1.0.
        latency_ms: Processing latency in ms. Defaults to 0.0.
        degraded: True if this message is from a degradation path.
        fast_path: True if taken the fast (shadow-skipped) path.
        emotion_category: User emotion category. Defaults to NEUTRAL.
        emotion_intensity: Emotion intensity [0, 1]. Defaults to 0.0.
        emotion_confidence: Emotion classifier confidence [0, 1]. Defaults to 0.0.
        affective_flag: True if this message carries an emotion highlight.
        scene_primary_type: Primary scene type (e.g. "code_editor").
        scene_confidence: Scene classification confidence [0, 1].
        user_model_version: Version of the user model in use.

    Returns:
        A fully-constructed Message instance.

    Example:
        >>> msg = create_envelope(
        ...     source_layer=Layer.PERCEPTION,
        ...     source_component="vision_fusion",
        ...     payload_type=PayloadType.PERCEPTION_EVENT,
        ...     payload={"text": "hello"},
        ... )
        >>> msg.version
        1
    """
    # Normalise enum args to their string values
    layer_str: str = source_layer.value if isinstance(source_layer, Layer) else source_layer
    pt_str: str = payload_type.value if isinstance(payload_type, PayloadType) else payload_type

    tid: str = trace_id if trace_id is not None else str(uuid.uuid4())
    version: int = _next_version(tid)

    # Log at WARNING if degraded (项目宪法 §4.3: all errors/degradations
    # must be logged at WARNING with trace_id).
    if degraded:
        logger.warning(
            "Creating degraded message from %s / %s (trace_id: %s, version: %d)",
            layer_str, source_component, tid, version,
        )

    # v4.5.0 §0.3: anger and surprise are placeholder — log at WARNING.
    if emotion_category in (EmotionCategory.ANGER, EmotionCategory.SURPRISE):
        logger.warning(
            "Creating message with placeholder emotion %s from %s / %s "
            "(trace_id: %s). Downstream MUST NOT branch on this.",
            emotion_category.value, layer_str, source_component, tid,
        )

    emotion: Emotion = Emotion(
        category=emotion_category.value,
        intensity=emotion_intensity,
        source="text_sentiment",
        confidence=emotion_confidence,
    )

    scene_context: SceneContext = SceneContext(
        primary_type=scene_primary_type,
        confidence=scene_confidence,
    )

    metadata: Metadata = Metadata(
        confidence=confidence,
        latency_ms=latency_ms,
        degraded=degraded,
        fast_path=fast_path,
        emotion=emotion,
        affective_flag=affective_flag,
        scene_context=scene_context,
        user_model_version=user_model_version,
    )

    return MessageEnvelope(
        trace_id=tid,
        source_layer=layer_str,
        source_component=source_component,
        timestamp=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        version=version,
        payload_type=pt_str,
        payload=payload,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Validation helper for downstream consumers
# ---------------------------------------------------------------------------


def validate_envelope(data: dict[str, Any]) -> MessageEnvelope:
    """Validate and construct a MessageEnvelope from a raw dict.

    Used by downstream layers when receiving messages from the bus.

    Args:
        data: Raw dict matching the envelope format (v4.5.0 §0.3).

    Returns:
        A validated MessageEnvelope instance.

    Raises:
        MessageValidationError: if any required field is missing or invalid.
    """
    return MessageEnvelope.from_dict(data)


# ---------------------------------------------------------------------------
# __main__ quick self-test — 实施方案 step 2.1 requirement
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Quick smoke test: create two messages, verify version monotonicity.
    msg1: MessageEnvelope = create_envelope(
        source_layer=Layer.PERCEPTION,
        source_component="smoke_test",
        payload_type=PayloadType.PERCEPTION_EVENT,
        payload={"test": "msg1"},
    )
    assert msg1.version == 1, f"Expected version 1, got {msg1.version}"

    msg2: MessageEnvelope = create_envelope(
        source_layer=Layer.PERCEPTION,
        source_component="smoke_test",
        payload_type=PayloadType.PERCEPTION_EVENT,
        payload={"test": "msg2"},
        trace_id=msg1.trace_id,  # same trace → version increments
    )
    assert msg2.version == 2, f"Expected version 2, got {msg2.version}"

    # Round-trip: dict → MessageEnvelope → dict.
    d: dict[str, Any] = msg1.to_dict()
    msg1_rt: MessageEnvelope = MessageEnvelope.from_dict(d)
    assert msg1_rt.trace_id == msg1.trace_id
    assert msg1_rt.version == msg1.version
    assert msg1_rt.metadata.emotion.category == msg1.metadata.emotion.category

    # Separate trace_id → version starts at 1 again.
    msg3: MessageEnvelope = create_envelope(
        source_layer=Layer.FUSION,
        source_component="smoke_test",
        payload_type=PayloadType.SCENE,
        payload={},
    )
    assert msg3.version == 1, f"Expected version 1 for new trace, got {msg3.version}"

    # Structural validation: from_dict rejects bad input.
    try:
        MessageEnvelope.from_dict({"bad": "envelope"})
        assert False, "Should have raised MessageValidationError"
    except MessageValidationError:
        pass  # expected

    # emotion.type is stripped.
    bad_emotion = {
        **msg1.to_dict(),
        "metadata": {
            **(msg1.to_dict()["metadata"]),
            "emotion": {"type": "bad", "category": "joy"},
        },
    }
    recovered: MessageEnvelope = MessageEnvelope.from_dict(bad_emotion)
    assert recovered.metadata.emotion.category == "joy"
    assert "type" not in recovered.to_dict()["metadata"]["emotion"], (
        "emotion.type should have been stripped"
    )

    print("message_envelope.py: all self-tests passed.")
