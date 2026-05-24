"""
Fusion layer — time_window, event_classifier, entity_fusion, scene_synthesis,
and shared message envelope.

v4.5.0 §2: Combines multi-modal perception events into structured Scene outputs.
"""

from .time_window import TimeSyncWindow, WindowedEvents
from .event_classifier import EventClassifier, ClassifiedEvents
from .entity_fusion import (
    EntityFusionEngine,
    EntityFusionResult,
    EntityObject,
    AlignedPair,
)
from .scene_synthesis import (
    SceneSynthesizer,
    ScenePayload,
    SceneMetadata,
    EntityRelation,
    EmotionSnapshot,
    SceneClass,
)

from .message_envelope import (
    MessageEnvelope,
    create_envelope,
    validate_envelope,
    Layer,
    PayloadType,
    EmotionCategory,
    Emotion,
    SceneContext,
    Metadata,
    MessageValidationError,
)

__all__ = [
    # Time window
    "TimeSyncWindow",
    "WindowedEvents",
    # Event classifier
    "EventClassifier",
    "ClassifiedEvents",
    # Entity fusion
    "EntityFusionEngine",
    "EntityFusionResult",
    "EntityObject",
    "AlignedPair",
    # Scene synthesis
    "SceneSynthesizer",
    "ScenePayload",
    "SceneMetadata",
    "EntityRelation",
    "EmotionSnapshot",
    "SceneClass",
    # Message envelope (shared)
    "MessageEnvelope",
    "create_envelope",
    "validate_envelope",
    "Layer",
    "PayloadType",
    "EmotionCategory",
    "Emotion",
    "SceneContext",
    "Metadata",
    "MessageValidationError",
]
