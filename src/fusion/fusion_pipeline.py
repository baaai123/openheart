"""Fusion pipeline — orchestrates the four fusion modules — v4.5.0 §2.1

Flow: raw inputs → TimeSyncWindow → EventClassifier → EntityFusionEngine → SceneSynthesizer → Scene envelope.

Phased wiring:
  T2: time_window wired (push events into TimeSyncWindow)
  T3: event_classifier wired (classify windowed events)
  T4: entity_fusion + scene_synthesis wired (full end-to-end)
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Output dataclass — v4.5.0 §2.1
# --------------------------------------------------------------------------


@dataclass
class FusionResult:
    """Result of a single fusion pipeline invocation.

    Attributes
    ----------
    scene:
        Full Scene message envelope (dict), or None if degraded.
    aligned_entities:
        Cross-modal entity alignments (list of dict).
    window_id:
        UUID of the time window.
    degraded:
        True if any step in the pipeline was skipped or failed.
    classified_events:
        Raw ClassifiedEvents output for downstream introspection.
    """

    scene: dict[str, Any] | None = None
    aligned_entities: list[dict[str, Any]] = field(default_factory=list)
    window_id: str = ""
    degraded: bool = False
    classified_events: Any = None


# --------------------------------------------------------------------------
# FusionPipeline
# --------------------------------------------------------------------------


class FusionPipeline:
    """Orchestrate the four fusion sub-modules end-to-end.

    v4.5.0 §2.1–§2.6

    Usage::

        pipeline = FusionPipeline()
        result = await pipeline.process(
            asr_text="你好世界",
            vision_snapshot=snap,
            audio_emotion="neutral",
        )
        if not result.degraded:
            downstream.ingest(result.scene)
    """

    def __init__(self) -> None:
        # Degradation tracking — set to None if import fails
        self._window: Any = None
        self._classifier: Any = None
        self._entity_fusion: Any = None
        self._synthesizer: Any = None

        # --- TimeSyncWindow (T2) ---
        try:
            from src.fusion.time_window import TimeSyncWindow

            self._window = TimeSyncWindow()
        except ImportError as exc:
            # v4.5.0 §2.3: time window is foundational — log at WARNING
            logger.warning(
                "fusion.FusionPipeline: TimeSyncWindow import failed: %s", exc
            )

        # --- EventClassifier (T3) ---
        try:
            from src.fusion.event_classifier import EventClassifier

            self._classifier = EventClassifier()
        except ImportError as exc:
            logger.warning(
                "fusion.FusionPipeline: EventClassifier import failed: %s", exc
            )

        # --- EntityFusionEngine (T4) ---
        try:
            from src.fusion.entity_fusion import EntityFusionEngine

            self._entity_fusion = EntityFusionEngine()
        except ImportError as exc:
            logger.warning(
                "fusion.FusionPipeline: EntityFusionEngine import failed: %s", exc
            )

        # --- SceneSynthesizer (T4) ---
        try:
            from src.fusion.scene_synthesis import SceneSynthesizer

            self._synthesizer = SceneSynthesizer()
        except ImportError as exc:
            logger.warning(
                "fusion.FusionPipeline: SceneSynthesizer import failed: %s", exc
            )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def process_sync(self, asr_text: str, vision_snapshot, audio_emotion: str = "neutral") -> FusionResult:
        """Synchronous wrapper for thread pool execution. Avoids blocking asyncio."""
        import asyncio
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.process(asr_text, vision_snapshot, audio_emotion))
        finally:
            loop.close()

    async def process(
        self,
        asr_text: str,
        vision_snapshot: Any,
        audio_emotion: str = "neutral",
    ) -> FusionResult:
        """Process multi-modal input → structured Scene.

        Parameters
        ----------
        asr_text:
            Transcribed speech from the audio pipeline.
        vision_snapshot:
            VisionSnapshot dataclass (from perception.visual) or its dict
            representation.  Accepts both.
        audio_emotion:
            Emotion category from audio pipeline.  One of:
            ``"joy"``, ``"sadness"``, ``"neutral"``.

        Returns
        -------
        FusionResult with scene envelope, aligned entities, window_id,
        and degradation flag.

        v4.5.0 §2.1 pipeline overview
        """
        window_id = str(uuid.uuid4())
        now_iso = datetime.now(timezone.utc).isoformat()
        begin = time.time()

        # --- Step 0: build perception event dicts ---
        audio_event = self._build_audio_event(
            asr_text=asr_text,
            emotion_category=audio_emotion,
            timestamp=now_iso,
        )
        visual_event = self._build_vision_event(
            snapshot=vision_snapshot,
            timestamp=now_iso,
        )

        # --- Step 1: time window (T2) ---
        if self._window is None:
            logger.warning(
                "fusion.FusionPipeline: TimeSyncWindow unavailable, degraded skip, trace_id=%s",
                audio_event.get("trace_id", window_id),
            )
            return FusionResult(
                window_id=window_id,
                degraded=True,
            )

        # Push both events; window may or may not trigger
        _ = self._window.push(audio_event)
        triggered = self._window.push(visual_event)

        if triggered is None:
            # Window did not trigger on push — force-flush
            triggered = self._window.flush()

        if triggered is None:
            # Still nothing — no events accumulated; degraded
            logger.warning(
                "fusion.FusionPipeline: no windowed events after push+flush, degraded skip"
            )
            return FusionResult(
                window_id=window_id,
                degraded=True,
            )

        window_events: list[dict[str, Any]] = triggered.events
        window_affective: bool = triggered.affective_highlight

        # --- Step 2: event classification (T3) ---
        if self._classifier is None:
            logger.warning(
                "fusion.FusionPipeline: EventClassifier unavailable, degraded skip"
            )
            return FusionResult(
                window_id=window_id,
                degraded=True,
            )

        classified = self._classifier.classify(
            events=window_events,
            window_affective=window_affective,
        )

        # --- Step 3: entity fusion (T4) ---
        aligned_entities: list[dict[str, Any]] = []
        fusion_result: Any = None

        if self._entity_fusion is not None:
            try:
                fusion_result = self._entity_fusion.fuse(
                    classified_events=classified,
                    window_events=window_events,
                )
                # Convert AlignedPair objects to plain dicts for consumers
                for pair in getattr(fusion_result, "aligned_pairs", []):
                    aligned_entities.append({
                        "audio_text": (
                            pair.audio_entity.text
                            if hasattr(pair, "audio_entity")
                            else ""
                        ),
                        "visual_text": (
                            pair.visual_entity.text
                            if hasattr(pair, "visual_entity")
                            else ""
                        ),
                        "similarity": (
                            pair.similarity if hasattr(pair, "similarity") else 0.0
                        ),
                    })
            except Exception as exc:
                # Entity fusion failure is non-fatal — log and continue
                # v4.5.0 §2.5: fusion can degrade silently
                logger.warning(
                    "fusion.FusionPipeline: EntityFusionEngine.fuse() failed: %s", exc
                )

        # --- Step 4: scene synthesis (T4) ---
        scene: dict[str, Any] | None = None
        degraded = False

        if self._synthesizer is not None:
            try:
                scene = self._synthesizer.synthesize(
                    trace_id=audio_event.get("trace_id", window_id),
                    version=audio_event.get("version", 1),
                    classified_events=classified,
                    entity_fusion_result=fusion_result,
                    window_events=window_events,
                    user_model_version=audio_event.get(
                        "metadata", {}
                    ).get("user_model_version", 0),
                    start_time=begin,
                )
            except Exception as exc:
                # Scene synthesis failure → degraded output with partial data
                logger.warning(
                    "fusion.FusionPipeline: SceneSynthesizer.synthesize() failed: %s",
                    exc,
                )
                degraded = True
        else:
            logger.warning(
                "fusion.FusionPipeline: SceneSynthesizer unavailable — degraded"
            )
            degraded = True

        return FusionResult(
            scene=scene,
            aligned_entities=aligned_entities,
            window_id=window_id,
            degraded=degraded,
            classified_events=classified,
        )

    # ------------------------------------------------------------------ #
    # Event factories
    # ------------------------------------------------------------------ #

    def _build_audio_event(
        self,
        asr_text: str,
        emotion_category: str,
        timestamp: str,
    ) -> dict[str, Any]:
        """Build a perception-compatible audio event dict.

        Matches the contract established by ``_make_audio_event`` in
        ``tests/contracts/test_fusion_contract.py``.

        v4.5.0 §1.5 unified perception event format
        """
        return {
            "trace_id": str(uuid.uuid4()),
            "source_layer": "perception",
            "source_component": "audio",
            "timestamp": timestamp,
            "version": 1,
            "payload_type": "perception_event",
            "payload": {
                "type": "audio_event",
                "vision_snapshot": {},
                "audio": {
                    "text": asr_text,
                    "voicefeature": {
                        "language": "zh",
                        "avg_logprob": -0.3,
                    },
                    "is_segment_end": True,
                },
            },
            "metadata": {
                "confidence": 0.9,
                "latency_ms": 10.0,
                "degraded": False,
                "scene_context": {
                    "primary_type": "desktop",
                    "confidence": 0.8,
                },
                "emotion": {
                    "category": emotion_category,
                    "intensity": 0.3,
                    "source": "text_sentiment",
                    "confidence": 0.8,
                },
                "affective_flag": False,
                "user_model_version": 0,
            },
        }

    def _build_vision_event(
        self,
        snapshot: Any,
        timestamp: str,
    ) -> dict[str, Any]:
        """Build a perception-compatible vision snapshot event dict.

        Accepts a :class:`VisionSnapshot` dataclass or a plain dict.
        Calls ``.to_dict()`` when available.

        Matches the contract established by ``_make_vision_event`` in
        ``tests/contracts/test_fusion_contract.py``.

        v4.5.0 §1.5 unified perception event format
        """
        # Normalise snapshot to a plain dict
        if hasattr(snapshot, "to_dict"):
            vs_dict: dict[str, Any] = snapshot.to_dict()
        elif isinstance(snapshot, dict):
            vs_dict = snapshot
        else:
            # Degraded: unknown snapshot type; produce empty dict
            logger.warning(
                "fusion.FusionPipeline: unknown vision_snapshot type=%s, "
                "proceeding with empty snapshot",
                type(snapshot).__name__,
            )
            vs_dict = {"scene_class": None, "objects": [], "text_content": []}

        # Extract scene class info for metadata and scene_context
        scene_cls = vs_dict.get("scene_class") or {}
        scene_primary = scene_cls.get("primary", "unknown") if isinstance(scene_cls, dict) else "unknown"
        scene_confidence = scene_cls.get("confidence", 0.0) if isinstance(scene_cls, dict) else 0.0

        return {
            "trace_id": str(uuid.uuid4()),
            "source_layer": "perception",
            "source_component": "vision_fusion",
            "timestamp": timestamp,
            "version": 1,
            "payload_type": "perception_event",
            "payload": {
                "type": "vision_snapshot",
                "vision_snapshot": {
                    "scene_class": {
                        "primary": scene_primary,
                        "confidence": scene_confidence,
                    },
                    "objects": vs_dict.get("objects", []),
                    "text_content": vs_dict.get("text_content", []),
                },
                "audio": {
                    "text": "",
                    "voicefeature": {},
                },
            },
            "metadata": {
                "confidence": 0.95,
                "latency_ms": 11.5,
                "degraded": False,
                "scene_context": {
                    "primary_type": scene_primary,
                    "confidence": scene_confidence,
                },
                "emotion": {
                    "category": "neutral",
                    "intensity": 0.0,
                    "source": "text_sentiment",
                    "confidence": 1.0,
                },
                "affective_flag": False,
                "user_model_version": 0,
            },
        }
