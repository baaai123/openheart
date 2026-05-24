"""
SharedContext — thread-safe singleton shared memory layer for OpenHeart v5.x

Design:
- Singleton pattern: all layers (perception, decision, personality, execution)
  access the same state without passing references.
- Thread safety: ``threading.Lock()`` protects all read/write operations.
- Namespaces: 4 predefined namespaces matching OpenHeart layer architecture.
- ContextSnapshot: immutable freeze-dried copy for LLM context injection.

v4.5.0 §5: Cross-layer shared state management.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, ClassVar

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Namespace constants — one per OpenHeart layer
# ---------------------------------------------------------------------------
NS_PERCEPTION: str = "perception"
NS_DECISION: str = "decision"
NS_PERSONALITY: str = "personality"
NS_EXECUTION: str = "execution"

_NAMESPACES: tuple[str, ...] = (
    NS_PERCEPTION,
    NS_DECISION,
    NS_PERSONALITY,
    NS_EXECUTION,
)

# ---------------------------------------------------------------------------
# Module-level singleton instance (double-checked locking pattern)
# ---------------------------------------------------------------------------
_instance: SharedContext | None = None


class SharedContext:
    """Thread-safe singleton shared memory for cross-layer state exchange.

    Each namespace is a flat ``dict[str, Any]``. All public methods acquire
    ``self._lock`` to guarantee atomic read/write.
    """

    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self) -> None:
        """Initialize empty namespaces. No I/O in ``__init__``."""
        self._data: dict[str, dict[str, Any]] = {
            ns: {} for ns in _NAMESPACES
        }
        self._persist_path = None
        self._persist_task = None

    # ------------------------------------------------------------------
    # Singleton access
    # ------------------------------------------------------------------

    @classmethod
    def get_instance(cls) -> SharedContext:
        """Return singleton instance (double-checked locking).

        Creates the instance on first call; returns cached instance on
        subsequent calls. Safe for concurrent access from multiple threads.
        """
        # Reason: standard double-checked locking avoids lock contention
        # on the fast path after singleton initialisation.
        global _instance
        if _instance is None:
            with cls._lock:
                if _instance is None:
                    _instance = cls()
        return _instance  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Public API — thread-safe
    # ------------------------------------------------------------------

    def set(self, namespace: str, key: str, value: Any) -> None:
        """Thread-safe write of *key* → *value* under *namespace*.

        Args:
            namespace: One of the ``NS_*`` constants.
            key: String key for later retrieval.
            value: Any pickle-able Python object.
        """
        with self._lock:
            # setdefault handles unknown namespaces gracefully
            self._data.setdefault(namespace, {})[key] = value

    def get(self, namespace: str, key: str, default: Any = None) -> Any:
        """Thread-safe read of *key* under *namespace*.

        Args:
            namespace: One of the ``NS_*`` constants.
            key: String key to look up.
            default: Value returned when the key does not exist.

        Returns:
            The stored value, or *default* if missing.
        """
        with self._lock:
            return self._data.get(namespace, {}).get(key, default)

    def get_namespace(self, namespace: str) -> dict[str, Any]:
        """Return a **copy** of the entire namespace dict.

        Mutating the returned dict does **not** affect internal state.
        """
        with self._lock:
            return dict(self._data.get(namespace, {}))

    def delete(self, namespace: str, key: str) -> None:
        """Remove *key* from *namespace*. No-op when missing."""
        with self._lock:
            self._data.get(namespace, {}).pop(key, None)

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def snapshot(self) -> ContextSnapshot:
        """Return an immutable freeze-dried copy of all namespaces.

        Returns:
            A ``ContextSnapshot`` dataclass with a UTC ISO-8601 timestamp
            and deep copies of all namespace data.
        """
        with self._lock:
            return ContextSnapshot(
                timestamp=datetime.now(timezone.utc).isoformat(),
                snapshot_data={
                    ns: dict(data)
                    for ns, data in self._data.items()
                },
            )

    # ------------------------------------------------------------------
    # LanceDB persistence — v4.5.0 §5 (Task 7)
    # ------------------------------------------------------------------

    def enable_persistence(self, lance_db_path: str = "data/cold_memory") -> None:
        """Enable periodic LanceDB snapshot persistence.

        Spawns an async background task that snapshots the SharedContext
        every 60 seconds and writes all key-value pairs to a LanceDB
        ``shared_context_snapshots`` table.

        If lancedb is not installed, the background task logs a WARNING
        once and exits gracefully — no crash, no retry.

        Args:
            lance_db_path: Path to the LanceDB database directory.
        """
        self._persist_path = lance_db_path
        self._persist_task = asyncio.create_task(
            self._persist_loop()
        )

    async def _persist_loop(self) -> None:
        """Background task: snapshot → LanceDB every 60 s.

        Lazy-imports ``lancedb`` so the module stays importable even
        when LanceDB is not installed.  If the import fails the task
        exits immediately after logging one WARNING.
        """
        # v4.5.0 §5 — lazy import so import errors are handled once
        # and the loop never retries.
        try:
            import lancedb  # noqa: F401 — lazy import for persistence
        except ImportError:
            logger.warning(
                "lancedb not installed — SharedContext persistence disabled"
            )
            return

        TABLE_NAME: str = "shared_context_snapshots"
        assert self._persist_path is not None, (
            "_persist_path not set — enable_persistence() must be called "
            "before _persist_loop()"
        )
        persist_path: str = self._persist_path

        while True:
            await asyncio.sleep(60)
            snap: ContextSnapshot = self.snapshot()
            try:
                db = await lancedb.connect_async(persist_path)

                # Create table on first write, open on subsequent writes
                # — same pattern as ColdMemoryStore (memory_store.py L227-250).
                table_names = await db.table_names()
                if TABLE_NAME in table_names:
                    table = await db.open_table(TABLE_NAME)
                else:
                    table = await db.create_table(
                        TABLE_NAME,
                        data=[{
                            "timestamp": "",
                            "namespace": "",
                            "key": "",
                            "value_json": "",
                        }],
                    )

                for ns, data in snap.snapshot_data.items():
                    if not data:
                        continue
                    for key, val in data.items():
                        await table.add([{
                            "timestamp": snap.timestamp,
                            "namespace": ns,
                            "key": key,
                            "value_json": json.dumps(val, ensure_ascii=False),
                        }])
            except Exception:
                # Exception caught: any I/O or schema error during persist.
                # Safe to log and continue — next loop iteration will retry
                # with a fresh connection.
                logger.exception(
                    "SharedContext persist failed [namespace_count=%d]",
                    len(snap.snapshot_data),
                )

    @classmethod
    async def restore_from_lance(cls, lance_db_path: str) -> SharedContext:
        """Load the latest snapshot from LanceDB and populate the singleton.

        Connects to ``lance_db_path``, opens the ``shared_context_snapshots``
        table, groups records by timestamp, and applies all key-value pairs
        from the most recent snapshot.

        If the table does not exist or is empty the singleton is returned
        unmodified (fresh state).

        Returns:
            The singleton ``SharedContext`` instance (populated or fresh).
        """
        ctx = cls.get_instance()

        # v4.5.0 §5 — lazy import; if lancedb is unavailable return fresh ctx
        try:
            import lancedb  # noqa: F401
        except ImportError:
            logger.warning(
                "lancedb not installed — cannot restore SharedContext"
            )
            return ctx

        db = await lancedb.connect_async(lance_db_path)

        try:
            table = await db.open_table("shared_context_snapshots")
        except Exception:
            # open_table raises ValueError/Exception when table doesn't exist.
            # Safe: this just means no snapshots have been persisted yet.
            logger.info(
                "No shared_context_snapshots table in %s — starting fresh",
                lance_db_path,
            )
            return ctx

        # Fetch all records (small table — snapshots are compact)
        arrow_table = await table.to_arrow()
        if arrow_table.num_rows == 0:
            return ctx

        records: list[dict[str, Any]] = arrow_table.to_pylist()

        # Find the latest timestamp
        latest_ts: str = ""
        for r in records:
            ts = r.get("timestamp", "")
            if ts and ts > latest_ts:
                latest_ts = ts

        if not latest_ts:
            return ctx

        # Apply latest snapshot to ctx
        applied: int = 0
        for r in records:
            if r["timestamp"] != latest_ts:
                continue
            ns: str = r["namespace"]
            key: str = r["key"]
            raw: str = r["value_json"]
            try:
                val = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                # Raw string that isn't JSON — store as-is
                val = raw
            ctx.set(ns, key, val)
            applied += 1

        logger.info(
            "SharedContext restored from LanceDB (%d keys, ts=%s)",
            applied,
            latest_ts,
        )
        return ctx


@dataclass
class ContextSnapshot:
    """Immutable snapshot of shared context at a point in time.

    Attributes:
        timestamp: ISO-8601 UTC timestamp of when the snapshot was taken.
        snapshot_data: Frozen copy of all namespaces (``{ns: {key: value}}``).
    """

    timestamp: str
    snapshot_data: dict[str, dict[str, Any]]

    def to_llm_context(self) -> str:
        """Format snapshot as a compact LLM context block (≤500 chars).

        Example output::

            Current Context:
            [perception]
              active_window: vim
              cursor_pos: (42, 13)
            [decision]
              last_action: greet_user
            [personality]
              mood: curious
            [execution]
              tts_speaking: False
        """
        lines: list[str] = ["Current Context:"]

        for ns in _NAMESPACES:
            data = self.snapshot_data.get(ns, {})
            if not data:
                continue
            lines.append(f"[{ns}]")
            for key, value in data.items():
                val_str = str(value)
                # Truncate long values to avoid blowing context budget
                if len(val_str) > 80:
                    val_str = val_str[:77] + "..."
                lines.append(f"  {key}: {val_str}")

        result = "\n".join(lines)

        # Hard cap at 500 characters
        if len(result) > 500:
            result = result[:497] + "..."

        return result
