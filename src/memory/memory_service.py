"""MemoryService — unified hot/cold/sync/decay interface.

v4.5.0 §3: orchestrates Redis hot memory, LanceDB cold memory,
incremental sync (§3.2.4), and emotional decay (§3.3.2).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from src.memory.decay.decay_engine import (
    DecayConfig,
    MemoryDecayEngine,
)

logger = logging.getLogger(__name__)


class MemoryService:
    """Unified memory layer service — v4.5.0 §3.

    Holds references to the hot client (Redis-backed HotMemoryStore),
    cold client (LanceDB ColdMemoryStore), sync service, and decay engine.
    All I/O is delegated to injected clients so the service can be tested
    with mocks.

    Provides thin delegation wrappers so callers (e.g. DecisionBridge)
    only hold one ``_memory`` reference instead of separate ``_store`` +
    ``_cold_store``.
    """

    def __init__(
        self,
        hot_client: Optional[Any] = None,
        cold_client: Optional[Any] = None,
        sync_config: Optional[SyncConfig] = None,
        decay_config: Optional[DecayConfig] = None,
    ) -> None:
        self._hot = hot_client  # v4.5.0 §3.2: Redis-backed HotMemoryStore; None if unavailable
        self._cold = cold_client
        self._sync = None  # v5.x: no sync
        self._decay = None  # v5.x: no decay


    # -- direct property access for backward compatibility ------------

    @property
    def hot(self) -> Any:
        """Expose the hot-store client for direct access when needed."""
        return self._hot

    @property
    def cold(self) -> Any:
        """Expose the cold-store client for direct access when needed."""
        return self._cold

    # -- public API --------------------------------------------------

    async def sync_cycle(self) -> Any:
        """Run one incremental hot→cold sync cycle."""
        return None  # v5.x: no sync

    async def decay_cycle(self) -> list[Any]:
        """Run one memory decay evaluation cycle."""
        return []  # v5.x: no decay

    # -- Hot store delegation wrappers (v4.5.0 §3.2) ------------------

    def store_scene(self, scene: dict[str, Any]) -> bool:  # noqa: D401
        """Store a scene in hot memory. Delegates to HotMemoryStore.store_scene().

        v4.5.0 §3.2.1
        """
        if self._hot is None:
            return False
        return self._hot.store_scene(scene)

    def get_recent_context(self) -> list[str]:
        """Return recent context scene IDs from hot memory.

        Delegates to HotMemoryStore.get_context().
        v4.5.0 §3.2.4 / §5.1
        """
        if self._hot is None:
            return []
        return self._hot.get_context()

    def push_to_sync_queue(
        self,
        scene_id: str,
        summary: Optional[str] = None,
    ) -> bool:
        """Push a scene onto the hot→cold sync queue.

        Delegates to HotMemoryStore.push_sync_queue().
        v4.5.0 §3.2.4
        """
        if self._hot is None:
            return False
        metadata: dict[str, Any] = {}
        if summary:
            metadata["summary"] = summary
        return self._hot.push_sync_queue(scene_id, metadata)

    # -- Cold store delegation wrappers (v4.5.0 §3.3) -----------------

    async def get_memory_drawer(self, topic: str) -> str:
        """Retrieve past memories by topic — v4.5.0 §3.5.

        Supports queries like '还记得我们聊过的电影吗' by doing a
        semantic search over cold memory for the given topic and
        returning a condensed, privacy-filtered textual summary.
        """
        if self._cold is None:
            return ""

        try:
            fragments = await self._cold.semantic_search(topic, top_k=5, min_importance=0.1)
        except Exception:
            logger.warning(
                "MemoryService: get_memory_drawer semantic_search failed for topic %r",
                topic,
                exc_info=True,
            )
            return ""

        if not fragments:
            return ""

        # v4.5.0 §3.5: apply privacy filter to each fragment
        # Lazy import — only needed when cold memory is queried
        from src.memory.privacy_filter import filter_sensitive  # noqa: E402

        summaries: list[str] = []
        for frag in fragments:
            safe_summary = filter_sensitive(frag.scene_summary)
            if safe_summary:
                summaries.append(f"- {safe_summary}")

        if not summaries:
            return ""

        return "[相关记忆]\n" + "\n".join(summaries)

    async def get_recent_scenes(self, limit: int = 50) -> list[dict[str, Any]]:
        """Retrieve recent scenes from cold memory for user model generation.

        Delegates to ColdMemoryStore.get_recent_scenes().
        v4.5.0 §3.4.3
        """
        if self._cold is None:
            return []
        return await self._cold.get_recent_scenes(limit=limit)

    async def save_user_model(self, user_model: dict[str, Any]) -> bool:
        """Persist the user model to cold memory.

        Delegates to ColdMemoryStore.save_user_model().
        v4.5.0 §3.4.1
        """
        if self._cold is None:
            return False
        return await self._cold.save_user_model(user_model)
