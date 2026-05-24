"""
Memory decay engine with emotional protection — v4.5.0 §3.3.2

Ebbinghaus-based forgetting curve with different decay rates per memory type:
  - EMOTION:  α=0.1 (very slow, emotional protection)
  - FACT:     α=0.5 (moderate)
  - ACTION:   α=0.8 (fast, temporary tasks)

Emotional protection: memories with emotion_intensity > 0.7 and positive
valence have α permanently fixed at 0.1 — they are almost never forgotten.

Formula (§3.3.2):
  importance(t) = initial_score * exp(-α * hours_since_last_access)
                * log(1 + access_count)
                * (1 + affective_weight)
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)

# v4.5.0 §3.3.2: decay check interval — 1 hour
DEFAULT_DECAY_CHECK_INTERVAL_SECONDS: int = 3600

# v4.5.0 §3.3.2: emotional protection threshold
EMOTIONAL_PROTECTION_INTENSITY_THRESHOLD: float = 0.7


class MemoryType(str, Enum):
    """Memory categories for decay rate selection — v4.5.0 §3.3.2."""
    EMOTION = "emotion"
    FACT = "fact"
    ACTION = "action"


# v4.5.0 §3.3.2: decay coefficients per memory type
DECAY_ALPHA: dict[MemoryType, float] = {
    MemoryType.EMOTION: 0.1,
    MemoryType.FACT: 0.5,
    MemoryType.ACTION: 0.8,
}

# Alpha applied when emotional protection is active.
EMOTIONAL_PROTECTION_ALPHA: float = 0.1


@dataclass
class DecayConfig:
    """Configuration for the memory decay engine — v4.5.0 §3.3.2."""
    decay_check_interval_seconds: int = DEFAULT_DECAY_CHECK_INTERVAL_SECONDS
    alpha: dict[MemoryType, float] = field(
        default_factory=lambda: dict(DECAY_ALPHA),
    )
    emotional_protection_threshold: float = EMOTIONAL_PROTECTION_INTENSITY_THRESHOLD
    emotional_protection_alpha: float = EMOTIONAL_PROTECTION_ALPHA
    enabled: bool = True


@dataclass
class DecayResult:
    """Outcome of a single memory entry after decay evaluation."""
    scene_id: str
    memory_type: MemoryType
    initial_score: float
    decayed_score: float
    hours_since_last_access: float
    access_count: int
    affective_weight: float
    emotionally_protected: bool
    alpha_used: float
    should_retain: bool  # True if score stays above retention threshold


def classify_memory_type(scene: dict[str, Any]) -> MemoryType:
    """Classify a scene's memory type from its affective_flag and content.

    v4.5.0 §3.3.2:
      - EMOTION: scenes with affective_flag=True or high emotion_intensity.
      - ACTION: scenes describing user operations, mouse/keyboard events.
      - FACT: everything else (factual observations, general knowledge scenes).
    """
    if scene.get("affective_flag"):
        return MemoryType.EMOTION

    # Check emotion in the payload or metadata.
    payload: dict[str, Any] = scene.get("payload", {}) or {}
    emotion: Optional[dict[str, Any]] = payload.get("emotion")
    if emotion and float(emotion.get("intensity", 0.0)) > 0.5:
        return MemoryType.EMOTION

    # Check metadata emotion.
    metadata: dict[str, Any] = scene.get("metadata", {}) or {}
    meta_emotion: Optional[dict[str, Any]] = metadata.get("emotion")
    if meta_emotion and float(meta_emotion.get("intensity", 0.0)) > 0.5:
        return MemoryType.EMOTION

    # Check if the scene is action-oriented.
    payload_type: str = str(scene.get("payload_type", "") or "")
    if payload_type in ("action_sequence", "decision_command"):
        return MemoryType.ACTION

    # Check events for mouse/keyboard actions.
    for event in scene.get("events", []):
        if not isinstance(event, dict):
            continue
        event_type: str = str(event.get("type", "") or "")
        if event_type in ("mouse_event", "keyboard_event", "click", "scroll"):
            return MemoryType.ACTION

    return MemoryType.FACT


def is_emotionally_protected(
    scene: dict[str, Any],
    threshold: float = EMOTIONAL_PROTECTION_INTENSITY_THRESHOLD,
) -> bool:
    """Check if a scene qualifies for emotional protection — v4.5.0 §3.3.2.

    A scene is emotionally protected when it has emotion_intensity > 0.7
    AND positive valence (category = joy).  Protected memories have alpha
    permanently fixed at 0.1.

    Returns True if emotional protection applies.
    """
    # Check emotion across multiple possible locations.
    metadata: dict[str, Any] = scene.get("metadata", {}) or {}
    emotion: Optional[dict[str, Any]] = metadata.get("emotion")
    if not emotion:
        payload: dict[str, Any] = scene.get("payload", {}) or {}
        emotion = payload.get("emotion")
    if not emotion:
        return False

    intensity: float = float(emotion.get("intensity", 0.0))
    category: str = str(emotion.get("category", "") or "")

    return intensity > threshold and category == "joy"


def compute_decayed_importance(
    scene: dict[str, Any],
    *,
    now: Optional[float] = None,
    alpha_override: Optional[float] = None,
) -> DecayResult:
    """Compute the decayed importance of a single memory scene.

    v4.5.0 §3.3.2 formula:
      importance(t) = initial_score
                    * exp(-α * hours_since_last_access)
                    * log(1 + access_count)
                    * (1 + affective_weight)

    Args:
        scene: Scene dict with importance_score, importance_components, and timestamp.
        now: Current time in epoch seconds. Defaults to time.time().
        alpha_override: If provided, overrides the memory-type default alpha.

    Returns:
        DecayResult with full decay analysis.
    """
    scene_id: str = str(scene.get("scene_id", "unknown"))
    memory_type: MemoryType = classify_memory_type(scene)
    initial_score: float = float(scene.get("importance_score", 0.5))

    # Get components.
    comps: dict[str, Any] = scene.get("importance_components", {}) or {}
    access_count: int = max(1, int(comps.get("access_count", 1)))
    affective_weight: float = float(comps.get("affective_bonus", 0.0))

    # Compute hours since last access.
    timestamp: Optional[str] = scene.get("timestamp")
    last_access: float = time.time()
    if timestamp:
        try:
            import datetime
            dt = datetime.datetime.fromisoformat(
                str(timestamp).replace("Z", "+00:00")
            )
            last_access = dt.timestamp()
        except (ValueError, TypeError):
            # Malformed timestamp — use current time so decay calc is safe.
            logger.debug(
                "Malformed timestamp for scene %s, using now for decay calc.",
                scene_id,
            )
            last_access = time.time()

    current_time: float = now if now is not None else time.time()
    hours_since_last_access: float = max(
        0.0, (current_time - last_access) / 3600.0
    )

    # Determine decay alpha with emotional protection.
    emotionally_protected: bool = is_emotionally_protected(scene)
    if alpha_override is not None:
        alpha: float = alpha_override
    elif emotionally_protected:
        alpha = EMOTIONAL_PROTECTION_ALPHA
    else:
        alpha = DECAY_ALPHA.get(memory_type, 0.5)

    # v4.5.0 §3.3.2 formula.
    time_factor: float = math.exp(-alpha * hours_since_last_access)
    access_factor: float = math.log(1.0 + float(access_count))
    affective_factor: float = 1.0 + affective_weight

    decayed_score: float = initial_score * time_factor * access_factor * affective_factor

    # Clamp to [0, 1].
    decayed_score = max(0.0, min(1.0, decayed_score))

    should_retain: bool = decayed_score >= 0.1

    return DecayResult(
        scene_id=scene_id,
        memory_type=memory_type,
        initial_score=initial_score,
        decayed_score=decayed_score,
        hours_since_last_access=hours_since_last_access,
        access_count=access_count,
        affective_weight=affective_weight,
        emotionally_protected=emotionally_protected,
        alpha_used=alpha,
        should_retain=should_retain,
    )


class MemoryDecayEngine:
    """Memory decay engine with emotional protection — v4.5.0 §3.3.2.

    Evaluates cold memory entries using the Ebbinghaus-based forgetting
    curve and marks entries for removal when their importance drops below
    the retention threshold.

    Emotional protection ensures positive high-intensity emotional memories
    are preserved almost permanently.
    """

    def __init__(
        self,
        cold_client: Any,
        config: Optional[DecayConfig] = None,
    ) -> None:
        """
        Args:
            cold_client: LanceDB client for cold memory access.
            config: Decay configuration. Uses defaults if None.
        """
        self._cold = cold_client
        self._config: DecayConfig = config or DecayConfig()
        self._last_check: float = time.time()

    @property
    def config(self) -> DecayConfig:
        return self._config

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    async def decay_cycle(self) -> list[DecayResult]:
        """Run one decay evaluation cycle on cold memory entries.

        v4.5.0 §3.3.2: executes every decay_check_interval (default 1 hour).
        Only processes entries that haven't been checked since last cycle.

        Returns a list of DecayResult for all evaluated entries.
        """
        if not self._config.enabled:
            logger.debug("MemoryDecayEngine: decay is disabled in config.")
            return []

        now: float = time.time()
        elapsed: float = now - self._last_check
        if elapsed < self._config.decay_check_interval_seconds:
            logger.debug(
                "MemoryDecayEngine: skipping decay cycle (elapsed=%.0fs < interval=%ds).",
                elapsed,
                self._config.decay_check_interval_seconds,
            )
            return []

        self._last_check = now

        try:
            entries: list[dict[str, Any]] = await self._fetch_cold_entries()
        except Exception:
            # LanceDB connectivity failure — log and skip this cycle.
            logger.warning(
                "MemoryDecayEngine: failed to fetch cold memory entries. "
                "Skipping decay cycle.",
                exc_info=True,
            )
            return []

        results: list[DecayResult] = []
        for entry in entries:
            try:
                result: DecayResult = compute_decayed_importance(entry, now=now)
                results.append(result)

                if not result.should_retain:
                    logger.info(
                        "MemoryDecayEngine: marking scene %s for removal "
                        "(type=%s, decayed=%.4f, alpha=%.2f, "
                        "protected=%s).",
                        result.scene_id,
                        result.memory_type.value,
                        result.decayed_score,
                        result.alpha_used,
                        result.emotionally_protected,
                    )
                    try:
                        await self._mark_for_removal(result.scene_id)
                    except Exception:
                        logger.warning(
                            "MemoryDecayEngine: failed to mark scene %s "
                            "for removal.",
                            result.scene_id,
                            exc_info=True,
                        )
                else:
                    # Update importance score in cold store.
                    try:
                        await self._update_score(entry, result.decayed_score)
                    except Exception:
                        logger.warning(
                            "MemoryDecayEngine: failed to update score for "
                            "scene %s.",
                            result.scene_id,
                            exc_info=True,
                        )

            except Exception:
                # Per-entry failure must not block the whole cycle.
                logger.warning(
                    "MemoryDecayEngine: failed to process entry during decay.",
                    exc_info=True,
                )

        logger.info(
            "MemoryDecayEngine cycle complete: evaluated=%d, "
            "mark_for_removal=%d, protected=%d.",
            len(results),
            sum(1 for r in results if not r.should_retain),
            sum(1 for r in results if r.emotionally_protected),
        )
        return results

    # ------------------------------------------------------------------
    # Abstracted I/O — inject real or mock clients
    # ------------------------------------------------------------------

    async def _fetch_cold_entries(self) -> list[dict[str, Any]]:
        """Fetch all cold memory entries for decay evaluation."""
        return []

    async def _mark_for_removal(self, scene_id: str) -> None:
        """Mark a cold memory entry for removal."""

    async def _update_score(self, entry: dict[str, Any], new_score: float) -> None:
        """Update the importance score of a cold memory entry."""
