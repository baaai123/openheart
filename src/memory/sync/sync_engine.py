"""
Hot-to-Cold memory sync engine — v4.5.0 §3.2.4

Incrementally synchronises scenes from Redis hot memory (hot:sync_queue Stream)
to LanceDB cold memory with:
  - Sensitive information filtering (phone, ID card, password)
  - Importance-based sorting (affective_bonus for emotional scenes)
  - cold_memory:initialized sentinel after first successful sync
  - Configurable sync interval
  - Crash recovery: resumes from last synced position on restart
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

# v4.5.0 §3.2.4: default sync interval 300s
DEFAULT_SYNC_INTERVAL_SECONDS: int = 300

# v4.5.0 §0.4 / 项目宪法 §5.1: sensitive patterns from thresholds.yaml
_SENSITIVE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"1[3-9]\d{9}"),
    re.compile(r"\d{17}[\dXx]"),
    re.compile(r"(?:password|passwd|pwd)\s*[:=]\s*\S+", re.IGNORECASE),
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SyncConfig:
    """Configuration for the Hot→Cold sync engine — v4.5.0 §3.2.4."""
    sync_interval_seconds: int = DEFAULT_SYNC_INTERVAL_SECONDS
    sensitive_patterns: list[re.Pattern[str]] = field(
        default_factory=lambda: list(_SENSITIVE_PATTERNS),
    )


@dataclass
class SyncResult:
    """Outcome of a single sync cycle."""
    scenes_synced: int = 0
    scenes_skipped_sensitive: int = 0
    scenes_skipped_error: int = 0
    cold_initialized_this_cycle: bool = False


# ---------------------------------------------------------------------------
# Sensitive information filtering
# ---------------------------------------------------------------------------

def _contains_sensitive(text: str, patterns: list[re.Pattern[str]]) -> bool:
    """Check if text matches any sensitive information pattern.

    v4.5.0 §3.2.4 step 2: 敏感信息过滤 — 电话、身份证、密码
    Returns True if sensitive content is detected.
    """
    if not text:
        return False
    for pattern in patterns:
        if pattern.search(text):
            logger.debug(
                "Sensitive pattern matched in sync candidate: %s",
                pattern.pattern,
            )
            return True
    return False


def filter_scene_sensitive(
    scene: dict[str, Any],
    patterns: Optional[list[re.Pattern[str]]] = None,
) -> bool:
    """Check a scene dict for sensitive content across summary and events.

    v4.5.0 §3.2.4 step 2: checks summary + text content of events.
    Returns True if the scene should be SKIPPED (contains sensitive data).
    """
    if patterns is None:
        patterns = _SENSITIVE_PATTERNS

    # Check summary field.
    summary: str = str(scene.get("summary", "") or "")
    if _contains_sensitive(summary, patterns):
        logger.warning(
            "Scene %s blocked from sync: sensitive content in summary",
            scene.get("scene_id", "?"),
        )
        return True

    # Check text content in events.
    for event in scene.get("events", []):
        if not isinstance(event, dict):
            continue
        for field_name in ("text_content", "content", "text", "audio_text"):
            text_val: str = str(event.get(field_name, "") or "")
            if _contains_sensitive(text_val, patterns):
                logger.warning(
                    "Scene %s blocked from sync: sensitive content in event %s",
                    scene.get("scene_id", "?"),
                    field_name,
                )
                return True

    return False


# ---------------------------------------------------------------------------
# Importance scoring
# ---------------------------------------------------------------------------

def compute_importance_score(
    scene: dict[str, Any],
) -> tuple[float, dict[str, float]]:
    """Compute importance score per v4.5.0 §3.2.4 step 3.

    Formula:
        score = access_count * recency_weight * relation_count_weight + affective_bonus

    affective_bonus = 0.5 when affective_flag is True.

    Returns (score, components_dict).
    """
    comps: dict[str, Any] = scene.get("importance_components", {})
    comps_default = dict(comps)

    access_count: int = int(comps.get("access_count", comps_default.get("access_count", 1)))
    recency_weight: float = float(
        comps.get("recency_weight", comps_default.get("recency_weight", 1.0))
    )
    relation_count_weight: float = float(
        comps.get("relation_count_weight", comps_default.get("relation_count_weight", 0.0))
    )
    affective_flag: bool = bool(scene.get("affective_flag", False))
    affective_bonus: float = 0.5 if affective_flag else 0.0

    score: float = (
        access_count * recency_weight * relation_count_weight + affective_bonus
    )

    components: dict[str, float] = {
        "access_count": float(access_count),
        "recency_weight": recency_weight,
        "relation_count_weight": relation_count_weight,
        "affective_bonus": affective_bonus,
    }
    return score, components


# ---------------------------------------------------------------------------
# Sync engine
# ---------------------------------------------------------------------------


class MemorySyncEngine:
    """Hot→Cold incremental sync engine — v4.5.0 §3.2.4.

    Coordinates:
      - Reading from Redis hot:sync_queue Stream
      - Sensitive information filtering
      - Importance-based sorting
      - Writing to LanceDB cold memory
      - cold_memory:initialized sentinel management
      - Crash recovery: resumes from last synced position

    All I/O operations are abstracted through injected clients so the engine
    can be tested without real Redis/LanceDB.
    """

    def __init__(
        self,
        hot_client: Any,
        cold_client: Any,
        config: Optional[SyncConfig] = None,
    ) -> None:
        """
        Args:
            hot_client: Redis client with Stream support.
            cold_client: LanceDB client for cold storage.
            config: Sync configuration. Uses defaults if None.
        """
        self._hot = hot_client
        self._cold = cold_client
        self._config: SyncConfig = config or SyncConfig()
        self._last_sync_id: str = "0"
        self._total_synced: int = 0

    @property
    def config(self) -> SyncConfig:
        return self._config

    @property
    def last_sync_id(self) -> str:
        """Last synced Stream entry ID for incremental reads."""
        return self._last_sync_id

    async def sync(self) -> SyncResult:
        """Execute one incremental sync cycle — v4.5.0 §3.2.4.

        Returns a SyncResult with counts.
        """
        result: SyncResult = SyncResult()

        # Step 1: Read new entries from hot:sync_queue Stream since last sync.
        try:
            stream_entries: list[tuple[str, dict[str, Any]]] = await self._read_pending()
        except Exception:
            # Redis/LanceDB connectivity failure — log and return empty.
            logger.warning(
                "MemorySyncEngine: failed to read from hot:sync_queue. "
                "Skipping this sync cycle.",
                exc_info=True,
            )
            return result

        if not stream_entries:
            logger.debug("MemorySyncEngine: no pending scenes in sync queue.")
            return result

        # Step 2: Read full Scene data for each entry and apply sensitive filter.
        scenes: list[dict[str, Any]] = []
        for entry_id, entry_data in stream_entries:
            scene_id: str = entry_data.get("scene_id", "")
            if not scene_id:
                self._last_sync_id = entry_id
                continue

            try:
                scene: Optional[dict[str, Any]] = await self._read_scene(scene_id)
            except Exception:
                # Individual scene read failure — skip, don't block pipeline.
                logger.warning(
                    "MemorySyncEngine: failed to read scene %s from hot memory.",
                    scene_id,
                    exc_info=True,
                )
                result.scenes_skipped_error += 1
                self._last_sync_id = entry_id
                continue

            if scene is None:
                self._last_sync_id = entry_id
                continue

            # v4.5.0 §3.2.4 step 2: sensitive info filtering.
            if filter_scene_sensitive(scene, self._config.sensitive_patterns):
                result.scenes_skipped_sensitive += 1
                self._last_sync_id = entry_id
                continue

            scenes.append(scene)
            self._last_sync_id = entry_id

        if not scenes:
            logger.debug(
                "MemorySyncEngine: all sync candidates filtered out "
                "(skipped_sensitive=%d, skipped_error=%d).",
                result.scenes_skipped_sensitive,
                result.scenes_skipped_error,
            )
            return result

        # Step 3: Sort by importance score (descending).
        scored: list[tuple[dict[str, Any], float, dict[str, float]]] = []
        for scene in scenes:
            score, components = compute_importance_score(scene)
            scene["importance_score"] = score
            scene.setdefault("importance_components", {}).update(components)
            scored.append((scene, score, components))

        scored.sort(key=lambda item: item[1], reverse=True)

        # Step 4: Write to LanceDB cold memory.
        synced_count: int = 0
        for scene, score, components in scored:
            try:
                await self._write_cold(scene)
                synced_count += 1
            except Exception:
                # Write failure for one scene — log, skip, continue with others.
                logger.warning(
                    "MemorySyncEngine: failed to write scene %s to cold memory.",
                    scene.get("scene_id", "?"),
                    exc_info=True,
                )
                result.scenes_skipped_error += 1

        result.scenes_synced = synced_count
        self._total_synced += synced_count

        # Step 5: Mark synced entries as consumed from the Stream.
        try:
            await self._acknowledge_synced()
        except Exception:
            logger.warning(
                "MemorySyncEngine: failed to acknowledge synced entries in "
                "hot:sync_queue. Will retry on next cycle.",
                exc_info=True,
            )

        # Step 6: cold_memory:initialized sentinel — v4.5.0 §3.2.4 step 6.
        if synced_count > 0:
            try:
                already_initialized: bool = await self._check_initialized()
            except Exception:
                already_initialized = True  # Assume already set on error.
            if not already_initialized:
                try:
                    await self._set_initialized()
                    result.cold_initialized_this_cycle = True
                    logger.info(
                        "MemorySyncEngine: cold_memory:initialized sentinel set "
                        "(first successful sync)."
                    )
                except Exception:
                    logger.warning(
                        "MemorySyncEngine: failed to set cold_memory:initialized.",
                        exc_info=True,
                    )

        logger.info(
            "MemorySyncEngine sync cycle complete: synced=%d, "
            "skipped_sensitive=%d, skipped_error=%d, total_synced=%d",
            result.scenes_synced,
            result.scenes_skipped_sensitive,
            result.scenes_skipped_error,
            self._total_synced,
        )
        return result

    # ------------------------------------------------------------------
    # Abstracted I/O — override in subclass or inject mock for testing
    # ------------------------------------------------------------------

    async def _read_pending(self) -> list[tuple[str, dict[str, Any]]]:
        """Read pending entries from Redis hot:sync_queue Stream."""
        return []

    async def _read_scene(self, scene_id: str) -> Optional[dict[str, Any]]:
        """Read a full Scene from Redis hot:scene:{scene_id}."""
        return None

    async def _write_cold(self, scene: dict[str, Any]) -> None:
        """Write a Scene to LanceDB cold memory."""

    async def _acknowledge_synced(self) -> None:
        """Remove synced entries from hot:sync_queue."""

    async def _check_initialized(self) -> bool:
        """Check if cold_memory:initialized sentinel exists."""
        return False

    async def _set_initialized(self) -> None:
        """Set cold_memory:initialized = true (no TTL)."""
