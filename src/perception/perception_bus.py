"""
PerceptionBus — unified perception-layer output bus.  v4.5.0 §1.5, §13

The PerceptionBus is the top-level entry point for the perception layer.
It:
  - Receives structured perception events from visual, audio, and sync modules.
  - Wraps every outgoing event in the unified message envelope (v4.5.0 §0.3).
  - Assigns trace_ids and maintains version monotonicity.
  - Passes RuntimeConfig (DI pattern) — never reads os.environ directly.

All downstream modules consume perception messages exclusively through this bus.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Awaitable

from src.config.runtime import RuntimeConfig                       # v4.5.0 §0.5
from src.fusion.message_envelope import (                          # v4.5.0 §0.3
    MessageEnvelope,
    create_envelope,
    Layer,
    PayloadType,
    EmotionCategory,
)

logger = logging.getLogger(__name__)

# Type alias for message subscribers.
Subscriber = Callable[[MessageEnvelope], Awaitable[None]]


class PerceptionBus:
    """Publish perception events as unified message envelopes.

    Design (v4.5.0 §1.5, §13):
      - Creates a fresh trace_id for each interaction session on start().
      - Wraps perception data (text, visual snapshot, audio segment, emotion,
        scene classification) into a MessageEnvelope.
      - Publishes envelopes to registered async subscribers.
      - Tracks version via the message_envelope module's per-trace_id counter.

    All methods are async-safe and non-blocking.
    """

    def __init__(self, runtime_config: RuntimeConfig) -> None:
        """
        Args:
            runtime_config: Immutable runtime config (DI pattern — never
                reads os.environ directly per 项目宪法 §3.3).
        """
        self._config: RuntimeConfig = runtime_config
        self._trace_id: str | None = None
        self._started: bool = False
        self._subscribers: list[Subscriber] = []
        self._lock: asyncio.Lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def trace_id(self) -> str | None:
        """Return the current session trace_id, or None if not started."""
        return self._trace_id

    @property
    def started(self) -> bool:
        return self._started

    async def start(self, trace_id: str | None = None) -> str:
        """Begin a new perception session with a fresh or supplied trace_id.

        v4.5.0 §0.3: trace_id persists across the full interaction chain.
        The perception_bus is the top-level entry point and generates the
        first trace_id (§0.3, §0.10).

        Args:
            trace_id: Optional explicit trace_id. If None, generates a new
                UUID v4.

        Returns:
            The assigned trace_id.
        """
        async with self._lock:
            import uuid
            self._trace_id = trace_id if trace_id is not None else str(uuid.uuid4())
            self._started = True
            logger.info(
                "PerceptionBus started with trace_id=%s (vram_tier=%s)",
                self._trace_id, self._config.vram_tier.value,
            )
            return self._trace_id

    async def stop(self) -> None:
        """End the current perception session."""
        async with self._lock:
            tid: str | None = self._trace_id
            self._trace_id = None
            self._started = False
            if tid:
                logger.info("PerceptionBus stopped (trace_id=%s)", tid)

    # ------------------------------------------------------------------
    # Subscriber management
    # ------------------------------------------------------------------

    async def subscribe(self, callback: Subscriber) -> None:
        """Register an async callback that receives every published MessageEnvelope."""
        async with self._lock:
            if callback not in self._subscribers:
                self._subscribers.append(callback)
                logger.debug("PerceptionBus subscriber added (%d total)",
                             len(self._subscribers))

    async def unsubscribe(self, callback: Subscriber) -> None:
        """Remove a previously registered subscriber."""
        async with self._lock:
            try:
                self._subscribers.remove(callback)
            except ValueError:
                # Subscriber not found — no-op, safe to ignore.
                pass

    # ------------------------------------------------------------------
    # Publishing methods
    # ------------------------------------------------------------------

    async def _publish(self, message: MessageEnvelope) -> None:
        """Deliver a message to all registered subscribers concurrently.

        Subscriber exceptions are caught and logged — one misbehaving
        subscriber must not block delivery to others (v4.5.0 §0.4).
        """
        if not self._subscribers:
            logger.debug(
                "PerceptionBus: no subscribers for message %s v%d (trace_id=%s)",
                message.payload_type, message.version, message.trace_id,
            )
            return

        async def _deliver(sub: Subscriber) -> None:
            try:
                await sub(message)
            except Exception:
                # Catch-all: a downstream subscriber must not take down
                # the perception pipeline.
                logger.warning(
                    "PerceptionBus subscriber raised exception for message "
                    "%s v%d (trace_id=%s)",
                    message.payload_type, message.version, message.trace_id,
                    exc_info=True,
                )

        # Fire all subscribers concurrently.
        tasks: list[asyncio.Task[None]] = [
            asyncio.create_task(_deliver(sub)) for sub in self._subscribers
        ]
        # Await all — exceptions are handled inside _deliver.
        for task in tasks:
            try:
                await task
            except Exception:
                # Should not reach here (caught inside _deliver), but
                # guard against Task-level failures.
                logger.warning(
                    "PerceptionBus delivery task failed unexpectedly",
                    exc_info=True,
                )

    def _ensure_started(self) -> str:
        """Return current trace_id, raising RuntimeError if not started."""
        if not self._started or self._trace_id is None:
            raise RuntimeError(
                "PerceptionBus is not started. Call await bus.start() first."
            )
        return self._trace_id

    async def publish_perception_event(
        self,
        component: str,
        payload: dict[str, Any],
        *,
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
    ) -> MessageEnvelope:
        """Publish a perception_event with all standard metadata.

        This is the primary publishing method for perception data.

        Args:
            component: Name of the producing perception component
                (e.g. "vision_fusion", "asr_stream").
            payload: Free-form perception data.
            confidence: Overall confidence [0, 1].
            latency_ms: Processing latency in ms.
            degraded: True if from a degradation path.
            fast_path: True if took the fast (shadow-skipped) path.
            emotion_category: User emotion estimate.
            emotion_intensity: Emotion intensity [0, 1].
            emotion_confidence: Emotion classifier confidence [0, 1].
            affective_flag: True if this message carries emotion highlight.
            scene_primary_type: Primary scene classification.
            scene_confidence: Scene classification confidence [0, 1].

        Returns:
            The published MessageEnvelope.
        """
        trace_id: str = self._ensure_started()

        message: MessageEnvelope = create_envelope(
            source_layer=Layer.PERCEPTION,
            source_component=component,
            payload_type=PayloadType.PERCEPTION_EVENT,
            payload=payload,
            trace_id=trace_id,
            confidence=confidence,
            latency_ms=latency_ms,
            degraded=degraded,
            fast_path=fast_path,
            emotion_category=emotion_category,
            emotion_intensity=emotion_intensity,
            emotion_confidence=emotion_confidence,
            affective_flag=affective_flag,
            scene_primary_type=scene_primary_type,
            scene_confidence=scene_confidence,
            user_model_version=0,  # Perception layer doesn't track user model version
        )

        await self._publish(message)
        return message

    async def publish_scene(
        self,
        component: str,
        scene_primary_type: str,
        scene_confidence: float,
        *,
        degraded: bool = False,
        fast_path: bool = False,
    ) -> MessageEnvelope:
        """Publish a scene classification event.

        Args:
            component: Name of the producing component.
            scene_primary_type: Scene type (e.g. "code_editor", "desktop").
            scene_confidence: Scene classifier confidence [0, 1].
            degraded: True if from a degradation path.
            fast_path: True if took the fast path.

        Returns:
            The published MessageEnvelope.
        """
        trace_id: str = self._ensure_started()

        message: MessageEnvelope = create_envelope(
            source_layer=Layer.PERCEPTION,
            source_component=component,
            payload_type=PayloadType.SCENE,
            payload={"scene": {"primary_type": scene_primary_type,
                               "confidence": scene_confidence}},
            trace_id=trace_id,
            degraded=degraded,
            fast_path=fast_path,
            scene_primary_type=scene_primary_type,
            scene_confidence=scene_confidence,
        )

        await self._publish(message)
        return message

    async def publish_affective_event(
        self,
        component: str,
        payload: dict[str, Any],
        emotion_category: EmotionCategory,
        emotion_intensity: float,
        *,
        emotion_confidence: float = 0.0,
        degraded: bool = False,
        fast_path: bool = False,
    ) -> MessageEnvelope:
        """Publish an affective (emotion-highlight) event.

        v4.5.0 §0.3: affective_flag is set to True automatically.

        Args:
            component: Name of the producing component.
            payload: Free-form payload.
            emotion_category: User emotion estimate.
            emotion_intensity: Emotion intensity [0, 1].
            emotion_confidence: Emotion classifier confidence [0, 1].
            degraded: True if from a degradation path.
            fast_path: True if took the fast path.

        Returns:
            The published MessageEnvelope.
        """
        trace_id: str = self._ensure_started()

        message: MessageEnvelope = create_envelope(
            source_layer=Layer.PERCEPTION,
            source_component=component,
            payload_type=PayloadType.AFFECTIVE_EVENT,
            payload=payload,
            trace_id=trace_id,
            emotion_category=emotion_category,
            emotion_intensity=emotion_intensity,
            emotion_confidence=emotion_confidence,
            affective_flag=True,
            degraded=degraded,
            fast_path=fast_path,
        )

        await self._publish(message)
        return message
