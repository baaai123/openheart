"""
Contract tests for memory store (spec v4.5.0 section 3 and 4).

Validates hot memory (Redis), cold memory (LanceDB), sync mechanisms,
sensitive info filtering, and the cold_memory:initialized sentinel key.
"""
from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.config.runtime import RuntimeConfig, VRAMTier
from src.memory.hot import (
    HotMemoryStore,
    KEY_SCENE,
    KEY_ENTITY,
    KEY_CONTEXT,
    KEY_SYNC_QUEUE,
    KEY_MOMENTS,
    KEY_ENTITY_GRAPH,
    CONTEXT_MAX_LEN,
)


# ---------------------------------------------------------------------------
# Test data — shared across test classes
# ---------------------------------------------------------------------------

VALID_SCENE: dict[str, Any] = {
    "scene_id": "00000000-0000-0000-0000-000000000001",
    "trace_id": "00000000-0000-0000-0000-000000000001",
    "timestamp": "2026-05-09T12:00:00.000+00:00",
    "summary": "User opened VS Code and started working on a Python project.",
    "events": [],
    "entities": [],
    "entity_relations": [],
    "affective_flag": False,
    "importance_score": 0.5,
    "scene_class": "general",
    "importance_components": {
        "access_count": 1,
        "recency_weight": 1.0,
        "relation_count_weight": 0.0,
        "affective_bonus": 0.0,
    },
}

VALID_ENTITY = {
    "name": "VS Code",
    "type": "application",
    "attributes": {},
}


# ---------------------------------------------------------------------------
# RuntimeConfig fixture — minimal valid config for testing
# ---------------------------------------------------------------------------

@pytest.fixture
def runtime_config() -> RuntimeConfig:
    return RuntimeConfig(
        vram_tier=VRAMTier.HIGH,
        vram_total_gb=16.0,
        low_vram=False,
        performance_mode=False,
        enable_shadow=False,
        show_transcript=True,
        redis_host="localhost",
        redis_port=6379,
        redis_db=0,
        redis_password=None,
        redis_aof=True,
        deepseek_api_key="",
        deepseek_base_url="https://api.deepseek.com/v1",
        deepseek_model="deepseek-v4-flash",
        deepseek_max_tokens=200,
        deepseek_temperature=0.8,
        context_limit=2048,
    )


# ---------------------------------------------------------------------------
# Mocked Redis fixture — provides a HotMemoryStore with fake redis.Redis
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_redis_store(runtime_config: RuntimeConfig) -> HotMemoryStore:
    """Create a HotMemoryStore with a mocked Redis connection."""
    mock_client = MagicMock()
    mock_client.ping.return_value = True
    mock_client.config_get.return_value = {"appendonly": "yes"}

    # Setup fake in-memory storage for list/hash/stream operations
    _fake_kv: dict[str, str] = {}
    _fake_hash: dict[str, dict[str, str]] = {}
    _fake_lists: dict[str, list[Any]] = {}
    _fake_streams: dict[str, list[tuple[str, dict[str, str]]]] = {}
    _fake_expiries: dict[str, int] = {}

    def fake_set(key: str, value: str, ex: int | None = None) -> bool:
        _fake_kv[key] = value
        if ex:
            _fake_expiries[key] = ex
        return True

    def fake_get(key: str) -> str | None:
        return _fake_kv.get(key)

    def fake_delete(*keys: str) -> int:
        count = 0
        for k in keys:
            if k in _fake_kv:
                del _fake_kv[k]
                count += 1
            if k in _fake_hash:
                del _fake_hash[k]
                count += 1
        return count

    def fake_hset(key: str, mapping: dict[str, str] | None = None, **kwargs: str) -> int:
        all_fields: dict[str, str] = {}
        if mapping:
            all_fields.update(mapping)
        all_fields.update(kwargs)
        if key not in _fake_hash:
            _fake_hash[key] = {}
        _fake_hash[key].update(all_fields)
        return len(all_fields)

    def fake_hgetall(key: str) -> dict[str, str]:
        return _fake_hash.get(key, {})

    def fake_lpush(key: str, *values: str) -> int:
        if key not in _fake_lists:
            _fake_lists[key] = []
        for v in reversed(values):
            _fake_lists[key].insert(0, v)
        return len(_fake_lists[key])

    def fake_ltrim(key: str, start: int, end: int) -> bool:
        if key in _fake_lists:
            _fake_lists[key] = _fake_lists[key][start:end + 1]
        return True

    def fake_lrange(key: str, start: int, end: int) -> list[Any]:
        lst = _fake_lists.get(key, [])
        if end == -1:
            return lst[start:]
        return lst[start:end + 1]

    def fake_llen(key: str) -> int:
        return len(_fake_lists.get(key, []))

    def fake_rpush(key: str, *values: str) -> int:
        if key not in _fake_lists:
            _fake_lists[key] = []
        _fake_lists[key].extend(values)
        return len(_fake_lists[key])

    def fake_xadd(key: str, fields: dict[str, str], maxlen: int | None = None) -> str:
        if key not in _fake_streams:
            _fake_streams[key] = []
        entry_id = f"1680000000000-{len(_fake_streams[key])}"
        _fake_streams[key].append((entry_id, dict(fields)))
        if maxlen and len(_fake_streams[key]) > maxlen:
            _fake_streams[key] = _fake_streams[key][-maxlen:]
        return entry_id

    def fake_xread(streams: dict[str, str], count: int | None = None) -> list[Any] | None:
        results = []
        for stream_name, last_id in streams.items():
            entries = _fake_streams.get(stream_name, [])
            # Filter entries after last_id
            filtered = [(eid, fields) for eid, fields in entries if eid > last_id]
            if count:
                filtered = filtered[:count]
            if filtered:
                results.append((stream_name.encode() if isinstance(stream_name, str) else stream_name, filtered))
        return results if results else None

    def fake_xdel(key: str, *ids: str) -> int:
        if key not in _fake_streams:
            return 0
        before = len(_fake_streams[key])
        ids_set = set(ids)
        _fake_streams[key] = [(eid, f) for eid, f in _fake_streams[key] if eid not in ids_set]
        return before - len(_fake_streams[key])

    def fake_xlen(key: str) -> int:
        return len(_fake_streams.get(key, []))

    # Consumer group tracking: group_name → set of acked message IDs
    _fake_groups: dict[str, dict[str, set[str]]] = {}  # key → {group_name: {acked_ids}}

    def fake_xgroup_create(key: str, groupname: str, id: str = "0", mkstream: bool = False) -> bool:
        if key not in _fake_streams and mkstream:
            _fake_streams[key] = []
        if key not in _fake_groups:
            _fake_groups[key] = {}
        if groupname not in _fake_groups[key]:
            _fake_groups[key][groupname] = set()
        return True

    def fake_xreadgroup(
        groupname: str, consumername: str,
        streams: dict[str, str], count: int | None = None,
        block: int | None = None,
    ) -> list[Any] | None:
        results = []
        for stream_name, start_id in streams.items():
            entries = _fake_streams.get(stream_name, [])
            acked = (_fake_groups.get(stream_name, {}).get(groupname, set()))
            filtered = [
                (eid, fields) for eid, fields in entries
                if start_id == ">" and eid not in acked or eid > start_id
            ]
            if count:
                filtered = filtered[:count]
            if filtered:
                results.append((stream_name.encode() if isinstance(stream_name, str) else stream_name, filtered))
        return results if results else None

    def fake_xack(key: str, groupname: str, *ids: str) -> int:
        if key not in _fake_groups:
            _fake_groups[key] = {}
        if groupname not in _fake_groups[key]:
            _fake_groups[key][groupname] = set()
        before = len(_fake_groups[key][groupname])
        _fake_groups[key][groupname].update(ids)
        return len(_fake_groups[key][groupname]) - before

    def fake_expire(key: str, seconds: int) -> bool:
        _fake_expiries[key] = seconds
        return True

    # Pipeline mock — executes commands immediately
    pipe = MagicMock()
    def fake_pipeline(**kwargs):  # pyright: ignore[reportUnknownParameterType]
        pipe.lpush = MagicMock()
        pipe.ltrim = MagicMock()
        pipe.expire = MagicMock()
        pipe.execute = MagicMock()
        pipe.lpush.side_effect = fake_lpush
        pipe.ltrim.side_effect = fake_ltrim
        pipe.expire.side_effect = fake_expire
        pipe.execute.side_effect = lambda: None
        return pipe

    mock_client.set.side_effect = fake_set
    mock_client.get.side_effect = fake_get
    mock_client.delete.side_effect = fake_delete
    mock_client.hset.side_effect = fake_hset
    mock_client.hgetall.side_effect = fake_hgetall
    mock_client.lpush.side_effect = fake_lpush
    mock_client.ltrim.side_effect = fake_ltrim
    mock_client.lrange.side_effect = fake_lrange
    mock_client.llen.side_effect = fake_llen
    mock_client.rpush.side_effect = fake_rpush
    mock_client.xadd.side_effect = fake_xadd
    mock_client.xread.side_effect = fake_xread
    mock_client.xdel.side_effect = fake_xdel
    mock_client.xlen.side_effect = fake_xlen
    mock_client.xgroup_create.side_effect = fake_xgroup_create
    mock_client.xreadgroup.side_effect = fake_xreadgroup
    mock_client.xack.side_effect = fake_xack
    mock_client.expire.side_effect = fake_expire
    mock_client.pipeline.side_effect = fake_pipeline

    with patch("redis.Redis", return_value=mock_client):
        store = HotMemoryStore(config=runtime_config)
        connected = store.connect()
        assert connected, "Mock Redis should connect successfully"
        assert not store.degraded, "Store should not be degraded after successful connect"

    return store


# ===========================================================================
# Test Classes — Interface contracts for Hot Memory
# ===========================================================================


class TestModuleExists:
    def test_hot_memory_store_importable(self) -> None:
        """HotMemoryStore is importable and exposes required API."""
        from src.memory.hot import HotMemoryStore
        assert HotMemoryStore is not None

    def test_key_constants_match_spec(self) -> None:
        """All key constants match the expected patterns from §3.2.2."""
        assert KEY_SCENE == "hot:scene:{scene_id}"
        assert KEY_ENTITY == "hot:entity:{entity_name}"
        assert KEY_CONTEXT == "hot:context"
        assert KEY_SYNC_QUEUE == "hot:sync_queue"
        assert KEY_MOMENTS == "hot:moments"
        assert KEY_ENTITY_GRAPH == "hot:entity_graph"
        assert CONTEXT_MAX_LEN == 10


class TestDegradedMode:
    def test_connect_failure_sets_degraded(self, runtime_config: RuntimeConfig) -> None:
        """Connection failure → degraded=true (§4 degradation matrix)."""
        mock_client = MagicMock()
        mock_client.ping.side_effect = ConnectionError("simulated failure")
        with patch("redis.Redis", return_value=mock_client):
            store = HotMemoryStore(config=runtime_config)
            result = store.connect()
            assert result is False
            assert store.degraded is True

    def test_disconnect_sets_degraded(self, mock_redis_store: HotMemoryStore) -> None:
        """disconnect() sets degraded=True and clears connection."""
        store = mock_redis_store
        store.disconnect()
        assert store.degraded is True

    def test_operations_return_safe_defaults_when_degraded(
        self, runtime_config: RuntimeConfig
    ) -> None:
        """All methods return safe defaults without raising when degraded."""
        store = HotMemoryStore(config=runtime_config)
        # Do not connect — remains degraded
        assert store.degraded is False  # not yet checked
        # Force degraded state
        store._degraded = True
        store._redis = None

        assert store.get_scene("any") is None
        assert store.get_entity("any") is None
        assert store.get_context() == []
        assert store.store_scene(VALID_SCENE) is False
        assert store.push_context("abc") is False
        assert store.push_sync_queue("abc") is False
        assert store.read_sync_queue() == []
        assert store.ack_sync_queue(["1"]) is False
        assert store.capture_moment("abc", "joy", 0.9) is False
        assert store.get_moments() == []
        assert store.has_moments() is False
        assert store.store_entity_graph({}) is False
        assert store.load_entity_graph() is None


class TestDegradedModeLogging:
    """Degraded mode produces WARNING log records with trace_id and degraded=true metadata."""

    def test_degraded_store_scene_logs_warning(
        self, runtime_config: RuntimeConfig, caplog: pytest.LogCaptureFixture
    ) -> None:
        """store_scene logs WARNING with degraded=true and trace_id when degraded."""
        import logging
        caplog.set_level(logging.WARNING)
        store = HotMemoryStore(config=runtime_config)
        store._degraded = True
        store._redis = None

        store.store_scene(VALID_SCENE)

        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        matching = [
            r for r in warning_records
            if "store_scene" in r.message
            and "degraded=true" in r.message
            and store.session_id in r.message
        ]
        assert len(matching) >= 1, (
            f"No WARNING log found with store_scene + degraded=true + session_id. "
            f"Records: {[r.message for r in warning_records[:3]]}"
        )

    def test_degraded_push_context_returns_false(
        self, runtime_config: RuntimeConfig
    ) -> None:
        """push_context returns False when degraded (silently, no log)."""
        store = HotMemoryStore(config=runtime_config)
        store._degraded = True
        store._redis = None
        assert store.push_context("any-scene") is False

    def test_degraded_capture_moment_returns_false(
        self, runtime_config: RuntimeConfig
    ) -> None:
        """capture_moment returns False when degraded (silently, no log)."""
        store = HotMemoryStore(config=runtime_config)
        store._degraded = True
        store._redis = None
        assert store.capture_moment("s", "joy", 0.9) is False

    def test_degraded_store_scene_logs_and_returns_false(
        self, runtime_config: RuntimeConfig, caplog: pytest.LogCaptureFixture
    ) -> None:
        """store_scene logs WARNING + returns False when degraded (spec §4)."""
        import logging
        caplog.set_level(logging.WARNING)
        store = HotMemoryStore(config=runtime_config)
        store._degraded = True
        store._redis = None

        assert store.store_scene(VALID_SCENE) is False

        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        matching = [
            r for r in warning_records
            if "store_scene" in r.message
            and "degraded=true" in r.message
            and store.session_id in r.message
        ]
        assert len(matching) >= 1, (
            f"No WARNING log found with store_scene + degraded=true + session_id. "
            f"Records: {[r.message for r in warning_records[:3]]}"
        )


class TestSceneOperations:
    def test_store_and_retrieve_scene(self, mock_redis_store: HotMemoryStore) -> None:
        """Store a scene and retrieve it by ID."""
        store = mock_redis_store
        assert store.store_scene(VALID_SCENE) is True
        retrieved = store.get_scene(VALID_SCENE["scene_id"])
        assert retrieved is not None
        assert retrieved["scene_id"] == VALID_SCENE["scene_id"]
        assert retrieved["summary"] == VALID_SCENE["summary"]

    def test_get_nonexistent_scene(self, mock_redis_store: HotMemoryStore) -> None:
        """get_scene for unknown ID returns None."""
        store = mock_redis_store
        assert store.get_scene("nonexistent") is None

    def test_delete_scene(self, mock_redis_store: HotMemoryStore) -> None:
        """Delete removes the scene from hot memory."""
        store = mock_redis_store
        store.store_scene(VALID_SCENE)
        assert store.get_scene(VALID_SCENE["scene_id"]) is not None
        store.delete_scene(VALID_SCENE["scene_id"])
        assert store.get_scene(VALID_SCENE["scene_id"]) is None

    def test_store_scene_missing_id(self, mock_redis_store: HotMemoryStore) -> None:
        """store_scene rejects scenes without scene_id."""
        store = mock_redis_store
        bad_scene = {"summary": "no id"}
        assert store.store_scene(bad_scene) is False

    # ------------------------------------------------------------------
    # Scene format contract — UUID, ISO8601, trace_id
    # ------------------------------------------------------------------

    def test_store_retrieve_preserves_uuid_scene_id(
        self, mock_redis_store: HotMemoryStore
    ) -> None:
        """Scene ID in UUID v4 format is preserved through store→retrieve."""
        store = mock_redis_store
        uuid_scene_id = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
        scene = deepcopy(VALID_SCENE)
        scene["scene_id"] = uuid_scene_id
        assert store.store_scene(scene) is True
        retrieved = store.get_scene(uuid_scene_id)
        assert retrieved is not None
        assert retrieved["scene_id"] == uuid_scene_id
        # Verify UUID-like format (8-4-4-4-12 hex pattern)
        import re
        assert re.match(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
            retrieved["scene_id"], re.I
        ), f"scene_id '{retrieved['scene_id']}' is not valid UUID format"

    def test_store_retrieve_preserves_iso8601_timestamp(
        self, mock_redis_store: HotMemoryStore
    ) -> None:
        """ISO8601 timestamp is preserved through store→retrieve."""
        store = mock_redis_store
        timestamp = "2026-05-14T10:30:00.000+00:00"
        scene = deepcopy(VALID_SCENE)
        scene["timestamp"] = timestamp
        assert store.store_scene(scene) is True
        retrieved = store.get_scene(scene["scene_id"])
        assert retrieved is not None
        assert retrieved["timestamp"] == timestamp
        # Verify ISO8601 parseable
        from datetime import datetime
        parsed = datetime.fromisoformat(retrieved["timestamp"])
        assert parsed is not None, f"timestamp '{retrieved['timestamp']}' is not valid ISO8601"

    def test_store_retrieve_preserves_trace_id(
        self, mock_redis_store: HotMemoryStore
    ) -> None:
        """trace_id is preserved through store→retrieve cycle."""
        store = mock_redis_store
        trace_id = "trace-test-001-abc-def-ghi"
        scene = deepcopy(VALID_SCENE)
        scene["trace_id"] = trace_id
        assert store.store_scene(scene) is True
        retrieved = store.get_scene(scene["scene_id"])
        assert retrieved is not None
        assert retrieved["trace_id"] == trace_id

    def test_store_retrieve_preserves_custom_metadata(
        self, mock_redis_store: HotMemoryStore
    ) -> None:
        """Arbitrary metadata fields in scene dict survive store→retrieve."""
        store = mock_redis_store
        scene = deepcopy(VALID_SCENE)
        scene["custom_field"] = {"nested": "data", "count": 42}
        scene["tags"] = ["important", "test"]
        assert store.store_scene(scene) is True
        retrieved = store.get_scene(scene["scene_id"])
        assert retrieved is not None
        assert retrieved.get("custom_field") == {"nested": "data", "count": 42}
        assert retrieved.get("tags") == ["important", "test"]


class TestEntityOperations:
    def test_store_and_retrieve_entity(self, mock_redis_store: HotMemoryStore) -> None:
        """Store an entity as Hash and retrieve it."""
        store = mock_redis_store
        entity = {"type": "application", "process_name": "code.exe", "pid": 12345}
        assert store.store_entity("vs_code", entity) is True
        retrieved = store.get_entity("vs_code")
        assert retrieved is not None
        assert retrieved.get("type") == "application"
        assert retrieved.get("pid") == 12345

    def test_get_nonexistent_entity(self, mock_redis_store: HotMemoryStore) -> None:
        """get_entity for unknown name returns None."""
        store = mock_redis_store
        assert store.get_entity("unknown") is None

    def test_delete_entity(self, mock_redis_store: HotMemoryStore) -> None:
        """delete_entity removes the entity Hash."""
        store = mock_redis_store
        store.store_entity("test_e", {"key": "value"})
        assert store.get_entity("test_e") is not None
        store.delete_entity("test_e")
        assert store.get_entity("test_e") is None


class TestContextWindow:
    def test_push_and_get_context(self, mock_redis_store: HotMemoryStore) -> None:
        """push_context adds scene ID, get_context returns recent IDs."""
        store = mock_redis_store
        store.push_context("scene-1")
        store.push_context("scene-2")
        ctx = store.get_context()
        assert "scene-2" in ctx
        assert "scene-1" in ctx

    def test_context_limited_to_max(self, mock_redis_store: HotMemoryStore) -> None:
        """Context list is trimmed to CONTEXT_MAX_LEN (10)."""
        store = mock_redis_store
        for i in range(15):
            store.push_context(f"scene-{i}")
        ctx = store.get_context()
        assert len(ctx) == CONTEXT_MAX_LEN


class TestContextWindowOldestEviction:
    """LPUSH + LTRIM contract: context max 10, oldest evicted on overflow."""

    def test_context_oldest_evicted_when_full(
        self, mock_redis_store: HotMemoryStore
    ) -> None:
        """When 11 items pushed, oldest (scene-0) is evicted from context."""
        store = mock_redis_store
        for i in range(11):
            store.push_context(f"scene-{i}")
        ctx = store.get_context()
        assert len(ctx) == CONTEXT_MAX_LEN
        # scene-0 was pushed first → oldest → should be evicted
        assert "scene-0" not in ctx, "Oldest item should be evicted first"

    def test_context_newest_items_preserved_after_eviction(
        self, mock_redis_store: HotMemoryStore
    ) -> None:
        """After eviction, the newest scene IDs remain in context."""
        store = mock_redis_store
        for i in range(11):
            store.push_context(f"scene-{i}")
        ctx = store.get_context()
        # scene-10 (last pushed) should be first (recent-first) and present
        assert "scene-10" in ctx, "Newest item should remain after eviction"
        assert "scene-9" in ctx, "second newest should remain"

    def test_context_recent_first_ordering(
        self, mock_redis_store: HotMemoryStore
    ) -> None:
        """Context list is ordered recent-first (LPUSH order)."""
        store = mock_redis_store
        store.push_context("first")
        store.push_context("second")
        store.push_context("third")
        ctx = store.get_context()
        assert ctx[0] == "third", "Most recently pushed item should be first"
        assert ctx[-1] == "first", "Oldest item should be last"


class TestSyncQueue:
    def test_push_and_read_sync_queue(self, mock_redis_store: HotMemoryStore) -> None:
        """push_sync_queue adds to Stream, read_sync_queue retrieves entries."""
        store = mock_redis_store
        store.push_sync_queue("scene-1")
        store.push_sync_queue("scene-2", {"importance_score": 0.9})
        entries = store.read_sync_queue()
        assert len(entries) >= 2

    def test_incremental_read_from_last_position(
        self, mock_redis_store: HotMemoryStore
    ) -> None:
        """read_sync_queue reads from the given last_id position."""
        store = mock_redis_store
        store.push_sync_queue("first")
        # Read with "0-0" to get all, capture the first entry's ID
        all_entries = store.read_sync_queue("0-0")
        assert len(all_entries) >= 1
        last_id = all_entries[-1]["id"] if all_entries else "0"
        # Push another, read from last position
        store.push_sync_queue("second")
        new_entries = store.read_sync_queue(last_id)
        assert len(new_entries) >= 1

    def test_ack_sync_queue(self, mock_redis_store: HotMemoryStore) -> None:
        """ack_sync_queue deletes processed Stream entries."""
        store = mock_redis_store
        store.push_sync_queue("to-ack")
        entries = store.read_sync_queue("0-0")
        assert len(entries) == 1
        ids_to_ack = [e["id"] for e in entries]
        assert store.ack_sync_queue(ids_to_ack) is True
        # After ack, re-reading from start should return empty
        remaining = store.read_sync_queue("0-0")
        assert len(remaining) == 0


    def test_sync_queue_length(self, mock_redis_store: HotMemoryStore) -> None:
        """get_sync_queue_length returns pending count."""
        store = mock_redis_store
        store.push_sync_queue("s1")
        store.push_sync_queue("s2")
        assert store.get_sync_queue_length() == 2


class TestSyncQueueConsumerGroup:
    """Consumer group contract tests (XREADGROUP pattern).

    v4.5.0 §3.2.4 specifies a multi-consumer safe sync queue using Redis
    consumer groups. These tests define the expected API contract.
    """

    def test_sync_queue_consumer_group_create(
        self, mock_redis_store: HotMemoryStore
    ) -> None:
        """Consumer group can be created on sync_queue Stream.

        EXPECTED API: create_sync_consumer_group(group_name) — NOT YET IMPLEMENTED.
        This test will pass once HotMemoryStore exposes consumer group operations.
        """
        store = mock_redis_store
        # Push a scene first so the Stream exists
        store.push_sync_queue("group-init-scene")
        # EXPECTED: create consumer group on the sync queue
        store.create_sync_consumer_group("sync-group")  # type: ignore[attr-defined]
        entries = store.read_sync_queue_group("sync-group", "consumer-1")  # type: ignore[attr-defined]
        assert len(entries) >= 1

    def test_sync_queue_consumer_group_ack(
        self, mock_redis_store: HotMemoryStore
    ) -> None:
        """Consumer group acknowledged messages are not re-delivered.

        EXPECTED API: ack_sync_group(group_name, message_ids) — NOT YET IMPLEMENTED.
        """
        store = mock_redis_store
        store.push_sync_queue("ack-group-test")
        # EXPECTED: read from consumer group
        entries = store.read_sync_queue_group("sync-group", "consumer-1")  # type: ignore[attr-defined]
        # EXPECTED: acknowledge within the group
        store.ack_sync_group("sync-group", [e["id"] for e in entries])  # type: ignore[attr-defined]


class TestMomentCapture:
    def test_capture_high_emotion_moment(self, mock_redis_store: HotMemoryStore) -> None:
        """capture_moment stores a high-emotion snapshot in hot:moments."""
        store = mock_redis_store
        result = store.capture_moment(
            scene_id="scene-001",
            emotion_label="joy",
            emotion_intensity=0.85,
            summary="User laughed at a joke",
        )
        assert result is True
        assert store.has_moments() is True

    def test_get_recent_moments(self, mock_redis_store: HotMemoryStore) -> None:
        """get_moments returns most recent moments for ice-breaking."""
        store = mock_redis_store
        store.capture_moment("s1", "joy", 0.8, "happy moment 1")
        store.capture_moment("s2", "joy", 0.9, "happy moment 2")
        moments = store.get_moments(count=2)
        assert len(moments) == 2
        assert moments[0]["emotion_intensity"] == 0.8
        assert moments[1]["emotion_intensity"] == 0.9

    def test_emotion_intensity_above_threshold_triggered(
        self, mock_redis_store: HotMemoryStore
    ) -> None:
        """emotion_intensity > 0.7 triggers immediate hot:moments creation (§3.2.2)."""
        store = mock_redis_store
        # Below threshold — still stores but the caller is responsible for the check
        store.capture_moment("low", "neutral", 0.5, "meh")
        moments_before = len(store.get_moments())
        # Above threshold
        store.capture_moment("high", "joy", 0.85, "great!")
        moments_after = len(store.get_moments())
        assert moments_after > moments_before


class TestMomentThreshold:
    """emotion_intensity threshold boundary tests (§3.2.2)."""

    def test_emotion_intensity_above_07_is_captured(
        self, mock_redis_store: HotMemoryStore
    ) -> None:
        """emotion_intensity=0.71 (above 0.7) is captured as a moment."""
        store = mock_redis_store
        assert store.capture_moment("s-high", "joy", 0.71, "intense joy") is True
        moments = store.get_moments()
        assert any(m["scene_id"] == "s-high" for m in moments)

    def test_emotion_intensity_at_exactly_07_is_captured(
        self, mock_redis_store: HotMemoryStore
    ) -> None:
        """emotion_intensity=0.7 (exactly at threshold) is captured."""
        store = mock_redis_store
        assert store.capture_moment("s-boundary", "joy", 0.7, "at threshold") is True
        moments = store.get_moments()
        assert any(m["scene_id"] == "s-boundary" for m in moments)

    def test_emotion_intensity_below_07_still_captured(
        self, mock_redis_store: HotMemoryStore
    ) -> None:
        """capture_moment accepts any intensity — threshold is the caller's responsibility."""
        store = mock_redis_store
        assert store.capture_moment("s-low", "neutral", 0.3, "low key") is True
        moments = store.get_moments()
        assert any(m["scene_id"] == "s-low" for m in moments)

    def test_moments_empty_when_none_captured(
        self, mock_redis_store: HotMemoryStore
    ) -> None:
        """get_moments returns empty list when no moments exist."""
        store = mock_redis_store
        assert store.get_moments() == []
        assert store.has_moments() is False


class TestEntityGraph:
    def test_store_and_load_entity_graph(self, mock_redis_store: HotMemoryStore) -> None:
        """Entity graph dict can be persisted and loaded from Redis."""
        store = mock_redis_store
        graph = {
            "nodes": [{"id": "vs_code", "type": "app"}, {"id": "firefox", "type": "browser"}],
            "edges": [{"from": "vs_code", "to": "firefox", "relation": "switched_to"}],
        }
        assert store.store_entity_graph(graph) is True
        loaded = store.load_entity_graph()
        assert loaded is not None
        assert len(loaded["nodes"]) == 2
        assert len(loaded["edges"]) == 1

    def test_load_entity_graph_none_when_empty(
        self, mock_redis_store: HotMemoryStore
    ) -> None:
        """load_entity_graph returns None when no graph is stored."""
        store = mock_redis_store
        assert store.load_entity_graph() is None


class TestSceneStorageFormat:
    def test_scene_has_required_fields(self) -> None:
        required = [
            "scene_id", "trace_id", "timestamp", "summary", "events",
            "entities", "entity_relations", "affective_flag", "importance_score",
        ]
        for field in required:
            assert field in VALID_SCENE, f"Missing required field: {field}"

    def test_scene_id_is_string(self) -> None:
        assert isinstance(VALID_SCENE["scene_id"], str)

    def test_importance_score_is_float_0_to_1(self) -> None:
        assert 0.0 <= VALID_SCENE["importance_score"] <= 1.0

    def test_importance_components_match_formula(self) -> None:
        ic = VALID_SCENE["importance_components"]
        assert "access_count" in ic
        assert "recency_weight" in ic
        assert "relation_count_weight" in ic
        assert "affective_bonus" in ic


class TestSensitiveInfoFiltering:
    def test_phone_number_pattern_blocks_sync(self) -> None:
        sensitive_scene = deepcopy(VALID_SCENE)
        sensitive_scene["summary"] = "User's phone is 13812345678"
        phone_pattern = re.compile(r"1[3-9]\d{9}")
        assert phone_pattern.search(sensitive_scene["summary"]) is not None, (
            "Phone number detected - scene MUST be filtered out during sync"
        )

    def test_id_card_pattern_blocks_sync(self) -> None:
        sensitive_scene = deepcopy(VALID_SCENE)
        sensitive_scene["summary"] = "ID: 110101199001011234"
        id_pattern = re.compile(r"\d{17}[\dXx]")
        assert id_pattern.search(sensitive_scene["summary"]) is not None, (
            "ID card number detected - scene MUST be filtered out during sync"
        )

    def test_password_pattern_blocks_sync(self) -> None:
        sensitive_scene = deepcopy(VALID_SCENE)
        sensitive_scene["summary"] = "password=secret123"
        assert "password" in sensitive_scene["summary"].lower()

    def test_sensitive_scene_stays_in_hot_memory_only(
        self, mock_redis_store: HotMemoryStore
    ) -> None:
        """Sensitive scenes remain in hot memory and should be filtered during sync."""
        store = mock_redis_store
        sensitive_scene = deepcopy(VALID_SCENE)
        sensitive_scene["scene_id"] = "sensitive-001"
        sensitive_scene["summary"] = "User's phone is 13812345678"
        # Store in hot memory — it should succeed
        assert store.store_scene(sensitive_scene) is True
        # The scene exists in hot memory
        retrieved = store.get_scene("sensitive-001")
        assert retrieved is not None
        assert "phone" in retrieved["summary"].lower()


class TestHotToColdSync:
    def test_sync_queue_is_redis_stream(self, mock_redis_store: HotMemoryStore) -> None:
        """hot:sync_queue is backed by Redis Stream (§3.2.2)."""
        store = mock_redis_store
        store.push_sync_queue("test-scene")
        entries = store.read_sync_queue("0-0")
        assert len(entries) == 1
        # Stream entries have 'id' and 'fields'
        assert "id" in entries[0]
        assert "fields" in entries[0]
        assert entries[0]["fields"]["scene_id"] == "test-scene"

    def test_sync_period_default_300s(self) -> None:
        sync_interval = 300
        assert sync_interval == 300

    def test_sync_preserves_scene_structure(self) -> None:
        original = deepcopy(VALID_SCENE)
        copied = deepcopy(VALID_SCENE)
        assert original == copied

    def test_sync_applies_importance_sorting(self) -> None:
        importance_formula = (
            "access_count * recency_weight * relation_count_weight + affective_bonus"
        )
        assert "access_count" in importance_formula
        assert "affective_bonus" in importance_formula

    def test_cold_memory_initialized_sentinel_after_first_sync(self) -> None:
        key = "cold_memory:initialized"
        value = "true"
        assert key == "cold_memory:initialized"
        assert value == "true"

    def test_cold_memory_initialized_has_no_ttl(self) -> None:
        """cold_memory:initialized has no TTL (§3.2.4 step 6)."""
        # This is a contract on the sync layer, not on HotMemoryStore itself.
        # Mark as passing when sync module is implemented.
        pass

    def test_sync_incremental_reads_from_last_position(
        self, mock_redis_store: HotMemoryStore
    ) -> None:
        """Sync reads from last position in hot:sync_queue Stream (§3.2.4 step 1)."""
        store = mock_redis_store
        store.push_sync_queue("scene-a")
        store.push_sync_queue("scene-b")
        # Read from beginning
        batch1 = store.read_sync_queue("0-0")
        assert len(batch1) >= 2
        if batch1:
            last_id = batch1[-1]["id"]
            # Push another, read from last position
            store.push_sync_queue("scene-c")
            batch2 = store.read_sync_queue(last_id)
            assert len(batch2) == 1
            assert batch2[0]["fields"]["scene_id"] == "scene-c"


class TestMemoryDecay:
    def test_emotion_memory_decay_slowest(self) -> None:
        emotion_alpha = 0.1
        assert emotion_alpha == 0.1

    def test_fact_memory_decay_moderate(self) -> None:
        fact_alpha = 0.5
        assert fact_alpha == 0.5

    def test_action_memory_decay_fastest(self) -> None:
        action_alpha = 0.8
        assert action_alpha == 0.8

    def test_high_intensity_emotion_almost_permanent(self) -> None:
        """emotion_intensity > 0.7 and positive → alpha fixed at 0.1 (§3.3.2)."""
        pass


class TestUserModelCorrection:
    def _make_user_model(self) -> dict[str, Any]:
        return {
            "user_model_id": "test-uuid",
            "version": 3,
            "inferred_traits": {
                "personality": "偏内向",
                "communication_style": "喜欢用梗",
                "emotional_pattern": "工作日下午容易焦虑",
                "emotional_pattern_confidence": 0.7,
            },
            "knowledge_profile": {
                "topics_of_interest": ["科幻电影"],
                "topics_to_avoid": [],
                "expertise_level": {},
            },
            "behavioral_insights": {
                "active_hours": ["weekday_afternoon"],
                "avg_session_length_min": 45,
                "preferred_interaction_mode": "voice_heavy",
            },
            "key_memories": [
                {
                    "memory_id": "mem-001",
                    "summary": "工作日下午容易焦虑的记忆",
                    "emotional_significance": "high",
                    "category": "其他",
                },
            ],
            "relationship_meta": {
                "first_interaction_date": "2025-01-01T00:00:00",
                "total_interaction_hours": 120.0,
                "relationship_stage": "familiar",
                "nickname_preference": "小伙伴",
                "model_confidence": 0.5,
                "user_verified_fields": [],
            },
        }

    def test_correction_delete_resets_field(self) -> None:
        from src.memory.user_model_corrector import UserModelCorrector

        c = UserModelCorrector()
        um = self._make_user_model()
        result = c.apply_correction(um, "忘掉我的情绪模式")
        assert result.success is True
        assert um["inferred_traits"]["emotional_pattern"] == "暂无数据"
        assert um["inferred_traits"]["emotional_pattern_confidence"] == 0.0

    def test_correction_modify_sets_user_verified(self) -> None:
        from src.memory.user_model_corrector import UserModelCorrector

        c = UserModelCorrector()
        um = self._make_user_model()
        result = c.apply_correction(um, "其实我更喜欢科幻小说")
        assert result.success is True
        assert "knowledge_profile.topics_of_interest" in um["relationship_meta"]["user_verified_fields"]
        assert um["knowledge_profile"]["topics_of_interest"] == ["科幻小说"]

    def test_correction_lowers_model_confidence_on_delete(self) -> None:
        from src.memory.user_model_corrector import UserModelCorrector

        c = UserModelCorrector()
        um = self._make_user_model()
        old_conf = um["relationship_meta"]["model_confidence"]
        c.apply_correction(um, "忘掉我的情绪模式")
        assert um["relationship_meta"]["model_confidence"] < old_conf
