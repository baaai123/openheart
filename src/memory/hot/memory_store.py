"""
HotMemoryStore — Redis-backed session memory (v4.5.0 §3.2).

Key patterns (v4.5.0 §3.2.2, 项目宪法 §2.3):
  hot:scene:{scene_id}   → Scene JSON (String, TTL: session)
  hot:entity:{name}      → Entity attributes (Hash, TTL: session)
  hot:context            → Recent 10 Scene IDs (List)
  hot:sync_queue         → Pending sync Scene IDs (Stream)
  hot:moments            → High-emotion Moment snapshots (List)
  hot:entity_graph       → Serialised networkx graph dict (String)

Redis AOF persistence with appendfsync everysec — max 30s data loss (§3.2.2).
Connection failures → degraded=true, all operations logged at WARNING with trace_id.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from src.config.runtime import RuntimeConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Redis key constants — v4.5.0 §3.2.2
# ---------------------------------------------------------------------------

KEY_SCENE = "hot:scene:{scene_id}"
KEY_ENTITY = "hot:entity:{entity_name}"
KEY_CONTEXT = "hot:context"
KEY_SYNC_QUEUE = "hot:sync_queue"
KEY_MOMENTS = "hot:moments"
KEY_ENTITY_GRAPH = "hot:entity_graph"

CONTEXT_MAX_LEN = 10
DEFAULT_SESSION_TTL = 86400  # 24 hours per spec — session lifetime


class HotMemoryStore:
    """Redis-backed hot (working) memory for a single session.

    All methods that require a live Redis connection check self._redis first.
    When the connection is unavailable, they return safe defaults and log
    at WARNING level with degraded=true metadata.

    v4.5.0 §3.2.1: Scenes flow into Redis → hot memory available.
    v4.5.0 §3.2.2: AOF persistence, Stream for sync queue.
    """

    def __init__(self, config: RuntimeConfig) -> None:
        self._config: RuntimeConfig = config
        self._redis: Any = None  # pyright: ignore[reportExplicitAny] — redis is lazily imported
        self._session_id: str = uuid.uuid4().hex
        self._degraded: bool = False
        self._load_memory_preferences()  # v4.5.0 §5.7 — load user memory preferences for importance weighting

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """Establish the Redis connection. Returns True on success.

        On failure sets degraded=True and logs at WARNING with trace_id.
        """
        # Lazy import — redis is only needed when hot memory is used
        # v4.5.0 §12: libs loaded on demand
        try:
            import redis  # type: ignore[import-untyped]  # pyright: ignore[reportMissingImports]
        except ImportError:  # redis-py not installed
            self._degraded = True
            logger.warning(
                "Redis client library not available — hot memory degraded. trace_id=%s degraded=true",
                self._session_id,
            )
            return False

        try:
            self._redis = redis.Redis(
                host=self._config.redis_host,
                port=self._config.redis_port,
                db=self._config.redis_db,
                password=self._config.redis_password,
                socket_connect_timeout=3,
                socket_keepalive=True,
                decode_responses=True,
            )
            # Smoke test — ping the server
            self._redis.ping()
            self._degraded = False
            logger.info(
                "HotMemoryStore connected to Redis %s:%d db=%d. session_id=%s",
                self._config.redis_host,
                self._config.redis_port,
                self._config.redis_db,
                self._session_id,
            )

            # Ensure AOF is enabled server-side if config expects it
            # v4.5.0 §3.2.2: AOF appendfsync everysec
            if self._config.redis_aof:
                self._verify_aof()

            return True

        except Exception as exc:
            # ConnectionError, TimeoutError, AuthenticationError, etc.
            # v4.5.0 §4 degradation: connection failure → degraded=true
            self._degraded = True
            self._redis = None
            logger.warning(
                "Redis connection failed — hot memory degraded. host=%s:%d error=%s trace_id=%s degraded=true",
                self._config.redis_host,
                self._config.redis_port,
                exc,
                self._session_id,
            )
            return False

    def disconnect(self) -> None:
        """Gracefully close the Redis connection."""
        if self._redis is not None:
            try:
                self._redis.close()  # type: ignore[union-attr]
            except Exception as exc:
                # Connection already broken — safe to ignore
                logger.debug(
                    "Error closing Redis connection (already closed?): %s",
                    exc,
                )
        self._redis = None
        self._degraded = True

    # ------------------------------------------------------------------
    # Scene operations — v4.5.0 §3.2.2
    # ------------------------------------------------------------------

    def store_scene(self, scene: dict[str, Any]) -> bool:
        """Store a Scene as JSON at hot:scene:{scene_id}.

        The scene dict must include at least 'scene_id' (str) and 'timestamp'.
        TTL is set to the session lifetime (24h default).

        v4.5.0 §5.7: applies memory preference bias to importance_score before store.

        Returns True if stored successfully, False on degraded or error.
        """
        if self._redis is None:
            logger.warning(
                "store_scene skipped — hot memory degraded. trace_id=%s degraded=true",
                self._session_id,
            )
            return False

        scene_id = scene.get("scene_id")
        if not scene_id:
            logger.warning(
                "store_scene rejected — missing scene_id. trace_id=%s degraded=true",
                self._session_id,
            )
            return False

        # v4.5.0 §5.7: compute and apply memory preference bias
        pref_bias = self._compute_preference_bias(scene.get("summary", ""))
        base_importance = scene.get("importance_score", 0.5)
        affective_weight = scene.get("affective_weight", 0.0)
        importance = base_importance * (1.0 + affective_weight + pref_bias)
        scene["importance_score"] = importance

        key = KEY_SCENE.format(scene_id=scene_id)
        try:
            payload = json.dumps(scene, ensure_ascii=False, default=str)
            self._redis.set(key, payload, ex=DEFAULT_SESSION_TTL)  # type: ignore[union-attr]
            logger.debug("store_scene: %s ttl=%ds", key, DEFAULT_SESSION_TTL)
            return True
        except Exception as exc:
            # redis.exceptions.ConnectionError, TypeError on json, etc.
            logger.warning(
                "store_scene failed key=%s error=%s trace_id=%s degraded=true",
                key, exc, self._session_id,
            )
            return False

    def get_scene(self, scene_id: str) -> dict[str, Any] | None:
        """Retrieve a Scene by ID from hot:scene:{scene_id}.

        Returns the deserialised scene dict or None if not found / degraded.
        """
        if self._redis is None:
            return None

        key = KEY_SCENE.format(scene_id=scene_id)
        try:
            raw = self._redis.get(key)  # type: ignore[union-attr]
            if raw is None:
                return None
            return json.loads(raw)  # pyright: ignore[reportUnknownVariableType]
        except Exception as exc:
            # ConnectionError, JSONDecodeError, etc.
            logger.warning(
                "get_scene failed key=%s error=%s trace_id=%s degraded=true",
                key, exc, self._session_id,
            )
            return None

    def delete_scene(self, scene_id: str) -> bool:
        """Remove a Scene from hot memory."""
        if self._redis is None:
            return False

        key = KEY_SCENE.format(scene_id=scene_id)
        try:
            self._redis.delete(key)  # type: ignore[union-attr]
            return True
        except Exception as exc:
            logger.warning(
                "delete_scene failed key=%s error=%s trace_id=%s degraded=true",
                key, exc, self._session_id,
            )
            return False

    # ------------------------------------------------------------------
    # Entity operations — v4.5.0 §3.2.2
    # ------------------------------------------------------------------

    def store_entity(self, name: str, entity: dict[str, Any]) -> bool:
        """Store entity attributes as a Hash at hot:entity:{name}.

        The entity dict is flattened into Redis Hash fields. Complex nested
        values are JSON-serialised within their fields. TTL = session lifetime.
        """
        if self._redis is None:
            return False

        key = KEY_ENTITY.format(entity_name=name)
        try:
            mapping: dict[str, str] = {}
            for k, v in entity.items():
                mapping[str(k)] = json.dumps(v, ensure_ascii=False, default=str) if not isinstance(v, str) else v
            self._redis.hset(key, mapping=mapping)  # type: ignore[union-attr]
            self._redis.expire(key, DEFAULT_SESSION_TTL)  # type: ignore[union-attr]
            logger.debug("store_entity: %s fields=%d", key, len(mapping))
            return True
        except Exception as exc:
            logger.warning(
                "store_entity failed key=%s error=%s trace_id=%s degraded=true",
                key, exc, self._session_id,
            )
            return False

    def get_entity(self, name: str) -> dict[str, Any] | None:
        """Retrieve entity attributes from hot:entity:{name}.

        Returns the deserialised dict or None if not found / degraded.
        """
        if self._redis is None:
            return None

        key = KEY_ENTITY.format(entity_name=name)
        try:
            raw = self._redis.hgetall(key)  # type: ignore[union-attr]
            if not raw:
                return None
            result: dict[str, Any] = {}
            for k, v in raw.items():
                try:
                    result[k] = json.loads(v)  # pyright: ignore[reportUnknownVariableType]
                except (json.JSONDecodeError, TypeError):
                    result[k] = v
            return result
        except Exception as exc:
            logger.warning(
                "get_entity failed key=%s error=%s trace_id=%s degraded=true",
                key, exc, self._session_id,
            )
            return None

    def delete_entity(self, name: str) -> bool:
        """Remove an entity from hot memory."""
        if self._redis is None:
            return False

        key = KEY_ENTITY.format(entity_name=name)
        try:
            self._redis.delete(key)  # type: ignore[union-attr]
            return True
        except Exception as exc:
            logger.warning(
                "delete_entity failed key=%s error=%s trace_id=%s degraded=true",
                key, exc, self._session_id,
            )
            return False

    # ------------------------------------------------------------------
    # Context window — v4.5.0 §3.2.2
    # ------------------------------------------------------------------

    def push_context(self, scene_id: str) -> bool:
        """Push a scene_id onto the hot:context list (LPUSH).

        The list is trimmed to CONTEXT_MAX_LEN (10) entries.
        """
        if self._redis is None:
            return False

        try:
            pipe = self._redis.pipeline()  # type: ignore[union-attr]
            pipe.lpush(KEY_CONTEXT, scene_id)
            pipe.ltrim(KEY_CONTEXT, 0, CONTEXT_MAX_LEN - 1)
            pipe.expire(KEY_CONTEXT, DEFAULT_SESSION_TTL)
            pipe.execute()  # pyright: ignore[reportUnknownMemberType]
            logger.debug("push_context: %s (max=%d)", scene_id, CONTEXT_MAX_LEN)
            return True
        except Exception as exc:
            logger.warning(
                "push_context failed scene_id=%s error=%s trace_id=%s degraded=true",
                scene_id, exc, self._session_id,
            )
            return False

    def get_context(self) -> list[str]:
        """Return the recent context scene ID list (most recent first)."""
        if self._redis is None:
            return []

        try:
            ids = self._redis.lrange(KEY_CONTEXT, 0, -1)  # type: ignore[union-attr]
            return list(ids) if ids else []
        except Exception as exc:
            logger.warning(
                "get_context failed error=%s trace_id=%s degraded=true",
                exc, self._session_id,
            )
            return []

    # ------------------------------------------------------------------
    # Sync queue (Redis Stream) — v4.5.0 §3.2.2, §3.2.4
    # ------------------------------------------------------------------

    def push_sync_queue(self, scene_id: str, metadata: dict[str, Any] | None = None) -> bool:
        """Push a scene_id onto hot:sync_queue Redis Stream.

        The scene_id is stored as a Stream entry field. Additional metadata
        (e.g. importance_score) may be attached.

        v4.5.0 §3.2.4: Scene IDs flow through this Stream for incremental
        hot→cold sync. Consumer reads from last position.
        """
        if self._redis is None:
            return False

        entry: dict[str, str] = {"scene_id": scene_id}
        if metadata:
            for k, v in metadata.items():
                entry[str(k)] = json.dumps(v, ensure_ascii=False, default=str) if not isinstance(v, str) else str(v)

        try:
            self._redis.xadd(KEY_SYNC_QUEUE, entry, maxlen=10000)  # type: ignore[union-attr]
            logger.debug("push_sync_queue: %s", scene_id)
            return True
        except Exception as exc:
            logger.warning(
                "push_sync_queue failed scene_id=%s error=%s trace_id=%s degraded=true",
                scene_id, exc, self._session_id,
            )
            return False

    def read_sync_queue(self, last_id: str = "0", count: int = 100) -> list[dict[str, Any]]:
        """Read entries from hot:sync_queue Stream since last_id.

        Returns list of {id: message_id, fields: {...}} dicts.
        v4.5.0 §3.2.4 step 1: incremental read from last position.
        """
        if self._redis is None:
            return []

        try:
            raw = self._redis.xread({KEY_SYNC_QUEUE: last_id}, count=count)  # type: ignore[union-attr]
            results: list[dict[str, Any]] = []
            if raw:
                for _stream_name, messages in raw:
                    for msg_id, fields in messages:
                        # Deserialise JSON-encoded values back
                        deserialised: dict[str, Any] = {}
                        for k, v in fields.items():
                            try:
                                deserialised[k] = json.loads(v)  # pyright: ignore[reportUnknownVariableType]
                            except (json.JSONDecodeError, TypeError):
                                deserialised[k] = v
                        results.append({"id": msg_id, "fields": deserialised})
            return results
        except Exception as exc:
            logger.warning(
                "read_sync_queue failed last_id=%s error=%s trace_id=%s degraded=true",
                last_id, exc, self._session_id,
            )
            return []

    def ack_sync_queue(self, message_ids: list[str]) -> bool:
        """Acknowledge (delete) processed messages from the sync queue Stream.

        v4.5.0 §3.2.4 step 5: after successful sync, delete entries.
        """
        if self._redis is None or not message_ids:
            return False

        try:
            self._redis.xdel(KEY_SYNC_QUEUE, *message_ids)  # type: ignore[union-attr]
            logger.debug("ack_sync_queue: deleted %d entries", len(message_ids))
            return True
        except Exception as exc:
            logger.warning(
                "ack_sync_queue failed count=%d error=%s trace_id=%s degraded=true",
                len(message_ids), exc, self._session_id,
            )
            return False

    def get_sync_queue_length(self) -> int:
        """Return the number of pending entries in the sync queue Stream."""
        if self._redis is None:
            return 0

        try:
            return self._redis.xlen(KEY_SYNC_QUEUE)  # type: ignore[union-attr]
        except Exception as exc:
            logger.warning(
                "get_sync_queue_length failed error=%s trace_id=%s degraded=true",
                exc, self._session_id,
            )
            return 0

    # ------------------------------------------------------------------
    # Sync queue consumer groups — v4.5.0 §3.2.4
    # ------------------------------------------------------------------

    def create_sync_consumer_group(self, group_name: str) -> bool:
        """Create a Redis Stream consumer group on hot:sync_queue.

        Uses MKSTREAM to auto-create the Stream if it doesn't exist.
        Idempotent: ignores BUSYGROUP error if group already exists.

        v4.5.0 §3.2.4: Consumer groups enable multi-consumer sync workers
        that read from the sync queue without losing positions.
        """
        if self._redis is None:
            return False

        try:
            # XGROUP CREATE with MKSTREAM — auto-creates Stream if absent
            # BUSYGROUP (group already exists) is safe to ignore
            self._redis.xgroup_create(  # type: ignore[union-attr]
                KEY_SYNC_QUEUE, group_name, id="0", mkstream=True
            )
            logger.info(
                "Consumer group '%s' created on %s", group_name, KEY_SYNC_QUEUE
            )
            return True
        except Exception as exc:
            # redis.exceptions.ResponseError: BUSYGROUP — group exists
            if "BUSYGROUP" in str(exc):
                logger.debug(
                    "Consumer group '%s' already exists on %s",
                    group_name, KEY_SYNC_QUEUE,
                )
                return True
            logger.warning(
                "create_sync_consumer_group failed group=%s error=%s trace_id=%s degraded=true",
                group_name, exc, self._session_id,
            )
            return False

    def read_sync_queue_group(
        self, group_name: str, consumer_id: str, count: int = 100
    ) -> list[dict[str, Any]]:
        """Read pending entries from hot:sync_queue via consumer group.

        Uses XREADGROUP with '>' to read only new (undelivered) messages.
        Returns a list of {id: message_id, fields: {...}} dicts.

        v4.5.0 §3.2.4 step 2: consumer group read for incremental sync.
        """
        if self._redis is None:
            return []

        try:
            raw = self._redis.xreadgroup(  # type: ignore[union-attr]
                groupname=group_name,
                consumername=consumer_id,
                streams={KEY_SYNC_QUEUE: ">"},
                count=count,
                block=1000,  # 1s block — returns empty if no data
            )
            results: list[dict[str, Any]] = []
            if raw:
                for _stream_name, messages in raw:
                    for msg_id, fields in messages:
                        deserialised: dict[str, Any] = {}
                        for k, v in fields.items():
                            try:
                                deserialised[k] = json.loads(v)
                            except (json.JSONDecodeError, TypeError):
                                deserialised[k] = v
                        results.append({"id": msg_id, "fields": deserialised})
            return results
        except Exception as exc:
            logger.warning(
                "read_sync_queue_group failed group=%s consumer=%s error=%s trace_id=%s degraded=true",
                group_name, consumer_id, exc, self._session_id,
            )
            return []

    def ack_sync_group(self, group_name: str, message_ids: list[str]) -> bool:
        """Acknowledge (XACK) consumed messages in a consumer group.

        After XACK, messages are removed from the group's PEL
        (pending entries list) and won't be re-delivered.

        v4.5.0 §3.2.4 step 5: acknowledge after successful sync.
        """
        if self._redis is None or not message_ids:
            return False

        try:
            acked = self._redis.xack(  # type: ignore[union-attr]
                KEY_SYNC_QUEUE, group_name, *message_ids
            )
            logger.debug(
                "ack_sync_group: group=%s acked=%d/%d",
                group_name, acked, len(message_ids),
            )
            return True
        except Exception as exc:
            logger.warning(
                "ack_sync_group failed group=%s count=%d error=%s trace_id=%s degraded=true",
                group_name, len(message_ids), exc, self._session_id,
            )
            return False

    # ------------------------------------------------------------------
    # Moments — v4.5.0 §3.2.2
    # ------------------------------------------------------------------

    def capture_moment(self, scene_id: str, emotion_label: str,
                       emotion_intensity: float, summary: str = "") -> bool:
        """Capture a high-emotion moment as a JSON snapshot in hot:moments.

        v4.5.0 §3.2.2: emotion_intensity > 0.7 triggers immediate capture.
        The moment is RPUSHed onto the hot:moments list for ice-breaking use.
        """
        if self._redis is None:
            return False

        moment: dict[str, Any] = {
            "scene_id": scene_id,
            "emotion": emotion_label,
            "emotion_intensity": emotion_intensity,
            "summary": summary,
            "captured_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            payload = json.dumps(moment, ensure_ascii=False, default=str)
            self._redis.rpush(KEY_MOMENTS, payload)  # type: ignore[union-attr]
            self._redis.expire(KEY_MOMENTS, DEFAULT_SESSION_TTL)  # type: ignore[union-attr]
            logger.info(
                "capture_moment: scene_id=%s emotion=%s intensity=%.2f trace_id=%s",
                scene_id, emotion_label, emotion_intensity, self._session_id,
            )
            return True
        except Exception as exc:
            logger.warning(
                "capture_moment failed scene_id=%s error=%s trace_id=%s degraded=true",
                scene_id, exc, self._session_id,
            )
            return False

    def get_moments(self, count: int = 5) -> list[dict[str, Any]]:
        """Retrieve the most recent moments from hot:moments (LRANGE from right).

        Used for active ice-breaking: the companion references warm moments.

        v4.5.0 §5.7: re-ranks results by preference_bias computed from scene_summary.
        """
        if self._redis is None:
            return []

        try:
            end = -1
            start = max(-count, -(self._redis.llen(KEY_MOMENTS) or 0))  # type: ignore[union-attr]
            raw = self._redis.lrange(KEY_MOMENTS, start, end)  # type: ignore[union-attr]
            moments: list[dict[str, Any]] = []
            for item in (raw or []):
                try:
                    moments.append(json.loads(item))  # pyright: ignore[reportUnknownVariableType]
                except json.JSONDecodeError:
                    continue

            # v4.5.0 §5.7: re-rank moments by preference bias
            if self._memory_prefs:
                scored = [
                    (
                        m,
                        self._compute_preference_bias(m.get("scene_summary", "")) * 0.3,
                    )
                    for m in moments
                ]
                scored.sort(key=lambda x: x[1], reverse=True)
                return [item for item, _ in scored]

            return moments
        except Exception as exc:
            logger.warning(
                "get_moments failed error=%s trace_id=%s degraded=true",
                exc, self._session_id,
            )
            return []

    def has_moments(self) -> bool:
        """Check if any moments exist in hot:moments."""
        if self._redis is None:
            return False

        try:
            length = self._redis.llen(KEY_MOMENTS)  # type: ignore[union-attr]
            return length > 0 if length else False
        except Exception as exc:
            logger.warning(
                "has_moments failed error=%s trace_id=%s degraded=true",
                exc, self._session_id,
            )
            return False

    # ------------------------------------------------------------------
    # Entity graph persistence — v4.5.0 §3.2.3
    # ------------------------------------------------------------------

    def store_entity_graph(self, graph_data: dict[str, Any]) -> bool:
        """Persist a serialised networkx DiGraph dict at hot:entity_graph.

        v4.5.0 §3.2.3: the graph can be stored as a Python dict in Redis
        and rebuilt on demand to avoid maintaining it in-memory permanently.
        """
        if self._redis is None:
            return False

        try:
            payload = json.dumps(graph_data, ensure_ascii=False, default=str)
            self._redis.set(KEY_ENTITY_GRAPH, payload, ex=DEFAULT_SESSION_TTL)  # type: ignore[union-attr]
            logger.debug("store_entity_graph: %d keys in dict", len(graph_data))
            return True
        except Exception as exc:
            logger.warning(
                "store_entity_graph failed error=%s trace_id=%s degraded=true",
                exc, self._session_id,
            )
            return False

    def load_entity_graph(self) -> dict[str, Any] | None:
        """Load the serialised entity graph dict from Redis.

        Returns None if no graph is stored or connection is degraded.
        """
        if self._redis is None:
            return None

        try:
            raw = self._redis.get(KEY_ENTITY_GRAPH)  # type: ignore[union-attr]
            if raw is None:
                return None
            return json.loads(raw)  # pyright: ignore[reportUnknownVariableType]
        except Exception as exc:
            logger.warning(
                "load_entity_graph failed error=%s trace_id=%s degraded=true",
                exc, self._session_id,
            )
            return None

    # ------------------------------------------------------------------
    # Health / metadata
    # ------------------------------------------------------------------

    @property
    def degraded(self) -> bool:
        """True when the Redis connection is unavailable (degraded mode)."""
        return self._degraded

    @property
    def connected(self) -> bool:
        """True when Redis is connected and ready."""
        return self._redis is not None and not self._degraded

    @property
    def session_id(self) -> str:
        """Unique session identifier for this hot memory instance."""
        return self._session_id

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _verify_aof(self) -> None:
        """Verify that the Redis server has AOF persistence enabled.

        Logs a warning if AOF is off but the config expects it.
        v4.5.0 §3.2.2: AOF with appendfsync everysec.
        """
        if self._redis is None:
            return

        try:
            persistence = self._redis.config_get("appendonly")  # type: ignore[union-attr]
            # config_get returns dict like {"appendonly": "yes" or "no"}
            aof_value = persistence.get("appendonly", "no") if isinstance(persistence, dict) else "no"
            if aof_value.lower() != "yes":
                logger.warning(
                    "Redis AOF is not enabled on server. Data loss >30s is possible. "
                    "Set 'appendonly yes' in redis.conf. trace_id=%s",
                    self._session_id,
                )
            else:
                logger.info("Redis AOF persistence confirmed: appendonly=%s", aof_value)
        except Exception as exc:
            # config_get may fail if Redis version < 2.0 or permissions missing
            logger.warning(
                "Unable to verify Redis AOF status: %s. trace_id=%s",
                exc, self._session_id,
            )

    # ------------------------------------------------------------------
    # Memory preference weighting — v4.5.0 §5.7
    # ------------------------------------------------------------------

    def _load_memory_preferences(self) -> None:
        """Load memory_preferences from config/baseline.json.

        On any failure (missing file, bad JSON, missing key), sets
        self._memory_prefs = None for graceful degradation — all
        preference_bias computations return 0.0.
        """
        try:
            with open("config/baseline.json") as f:
                baseline = json.load(f)
            self._memory_prefs = baseline.get("memory_preferences", None)
            if self._memory_prefs:
                logger.debug(
                    "Loaded memory_preferences: %d positive, %d negative groups",
                    len(self._memory_prefs.get("positive", [])),
                    len(self._memory_prefs.get("negative", [])),
                )
            else:
                logger.debug("No memory_preferences found in baseline.json — using neutral weighting")
        except Exception as exc:
            # FileNotFoundError, JSONDecodeError, KeyError — all safe to degrade
            self._memory_prefs = None
            logger.warning(
                "Failed to load memory_preferences from baseline.json: %s. "
                "Preference bias disabled. trace_id=%s degraded=true",
                exc, self._session_id,
            )

    def _compute_preference_bias(self, text: str) -> float:
        """Compute preference_bias = clamp(Σ(keyword_match × weight), -0.3, 0.3).

        Iterates positive and negative keyword groups from memory_preferences.
        Each keyword match adds its weight to the bias.
        Final result is clamped to [-0.3, 0.3].

        Returns 0.0 if memory_preferences was not loaded or text is empty.
        """
        if not self._memory_prefs or not text:
            return 0.0

        bias = 0.0
        for entry in self._memory_prefs.get("positive", []):
            for kw in entry.get("keywords", []):
                if kw in text:
                    bias += entry.get("weight", 0.0)

        for entry in self._memory_prefs.get("negative", []):
            for kw in entry.get("keywords", []):
                if kw in text:
                    bias += entry.get("weight", 0.0)

        return max(-0.3, min(0.3, bias))
