"""
StateBus — Redis Stream-based state reporting bus for execution channels.
v4.5.0 §7.6

Each execution channel reports its status to dedicated Redis Streams. All
channels also report to a shared state:global stream for cross-channel
coordination. Consumer groups enable multi-consumer subscriptions.

Stream layout (项目宪法 §2.3):
  state:avatar   — Live2D rendering state (fps, expression, animation)
  state:keyboard — mouse / keyboard state
  state:voice    — TTS playback state
  state:global   — global broadcast from all channels

MAXLEN ~10000 per stream (v4.5.0 §7.6). Consumer group mode with ACK for
critical messages.
"""
from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.config.runtime import RuntimeConfig  # v4.5.0 §0.5

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stream name constants — 项目宪法 §2.3
# ---------------------------------------------------------------------------

STREAM_AVATAR = "state:avatar"
STREAM_KEYBOARD = "state:keyboard"
STREAM_VOICE = "state:voice"
STREAM_GLOBAL = "state:global"

ALL_STREAMS: tuple[str, ...] = (STREAM_AVATAR, STREAM_KEYBOARD, STREAM_VOICE, STREAM_GLOBAL)

# Default MAXLEN — v4.5.0 §7.6
DEFAULT_MAXLEN = 10000

# Default consumer group name
DEFAULT_CONSUMER_GROUP = "execution"

# ---------------------------------------------------------------------------
# Status message dataclass — v4.5.0 §7.6
# ---------------------------------------------------------------------------


@dataclass
class StateMessage:
    """A status report from an execution channel.

    Fields follow the pattern established in HotMemoryStore status reporting.
    """

    channel: str
    """Reporting channel: avatar, mouse, voice."""

    stream: str
    """Target stream: state:avatar, state:keyboard, state:voice, state:global."""

    status: str
    """Operational status: running, idle, error, completed, interrupted."""

    data: dict[str, Any] = field(default_factory=dict)  # pyright: ignore[reportExplicitAny]
    """Channel-specific payload (fps, expression, position, playback_state, etc.)."""

    message_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    """Unique message identifier for ACK tracking."""

    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    critical: bool = False
    """True for messages that require ACK (v4.5.0 §7.6)."""

    def to_dict(self) -> dict[str, Any]:  # pyright: ignore[reportExplicitAny]
        """Serialize to JSON-compatible dict."""
        return {
            "channel": self.channel,
            "stream": self.stream,
            "status": self.status,
            "data": self.data,
            "message_id": self.message_id,
            "timestamp": self.timestamp,
            "critical": self.critical,
        }

    def to_json(self) -> str:
        """Serialize to JSON string for Redis Stream entry."""
        return json.dumps(self.to_dict(), ensure_ascii=False)


# ---------------------------------------------------------------------------
# StateBus — v4.5.0 §7.6
# ---------------------------------------------------------------------------


class StateBus:
    """Redis Stream-based state reporting bus.

    v4.5.0 §7.6: All execution channels report status via Redis Streams.
    The bus manages stream creation (with MAXLEN), publishing, subscription
    via consumer groups, and ACK tracking for critical messages.

    Degradation: if Redis is unavailable, all operations log at WARNING
    level with degraded=true and return safe defaults.
    """

    def __init__(self, config: RuntimeConfig) -> None:
        self._config: RuntimeConfig = config
        self._redis: Any = None  # pyright: ignore[reportExplicitAny] — lazy import
        self._degraded: bool = False
        self._consumer_groups: set[str] = set()

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """Establish Redis connection for the state bus.

        Lazy-imports redis-py. On failure, sets degraded=True and logs
        at WARNING with trace_id.

        Returns:
            True if connected successfully.
        """
        try:
            import redis  # type: ignore[import-untyped]  # pyright: ignore[reportMissingImports]
        except ImportError:
            self._degraded = True
            logger.warning(
                "redis-py not installed — StateBus degraded. trace_id=TBD degraded=true"
            )
            return False

        try:
            self._redis = redis.Redis(
                host=self._config.redis_host,
                port=self._config.redis_port,
                db=self._config.redis_db,
                password=self._config.redis_password,
                decode_responses=True,
            )
            self._redis.ping()  # pyright: ignore[reportUnknownMemberType]
            logger.info("StateBus connected to Redis %s:%d", self._config.redis_host, self._config.redis_port)  # pyright: ignore[reportUnknownMemberType]
            return True
        except Exception:
            # redis.Redis can raise ConnectionError, TimeoutError, or ResponseError
            self._degraded = True
            logger.warning(
                "StateBus failed to connect to Redis — degraded. trace_id=TBD degraded=true",
                exc_info=True,
            )
            self._redis = None
            return False

    # ------------------------------------------------------------------
    # Stream publishing — v4.5.0 §7.6
    # ------------------------------------------------------------------

    def publish(self, message: StateMessage) -> str | None:
        """Publish a status message to a Redis Stream.

        v4.5.0 §7.6: Messages are added to the stream with MAXLEN ~10000,
        trimming older entries automatically.

        Args:
            message: StateMessage to publish.

        Returns:
            The Redis message ID if published successfully, None if degraded.
        """
        if self._degraded or self._redis is None:
            logger.warning(
                "StateBus degraded — cannot publish to %s. trace_id=TBD degraded=true",
                message.stream,
            )
            return None

        try:
            entry = {message.message_id: message.to_json()}
            msg_id: str = self._redis.xadd(  # pyright: ignore[reportUnknownMemberType]
                message.stream,
                entry,
                maxlen=DEFAULT_MAXLEN,
                approximate=True,
            )
            if message.critical:
                logger.debug(
                    "Critical message %s published to %s (requires ACK)",
                    message.message_id,
                    message.stream,
                )
            return msg_id
        except Exception:
            # redis-py may raise ResponseError on stream issues
            logger.warning(
                "Failed to publish to %s — trace_id=TBD degraded=true",
                message.stream,
                exc_info=True,
            )
            return None

    def publish_raw(
        self,
        stream: str,
        channel: str,
        status: str,
        data: dict[str, Any] | None = None,  # pyright: ignore[reportExplicitAny]
        critical: bool = False,
    ) -> str | None:
        """Convenience method to publish without constructing a StateMessage.

        Args:
            stream: Target stream name.
            channel: Reporting channel.
            status: Operational status.
            data: Channel-specific payload.
            critical: Whether ACK is required.

        Returns:
            Redis message ID or None if degraded.
        """
        msg = StateMessage(
            channel=channel,
            stream=stream,
            status=status,
            data=data or {},
            critical=critical,
        )
        return self.publish(msg)

    # ------------------------------------------------------------------
    # Multi-stream broadcast
    # ------------------------------------------------------------------

    def broadcast_to_global(self, channel: str, status: str, data: dict[str, Any] | None = None) -> None:  # pyright: ignore[reportExplicitAny]
        """Publish to state:global AND the channel-specific stream.

        All channels must report to state:global (v4.5.0 §7.2.3).

        Args:
            channel: Reporting channel.
            status: Operational status.
            data: Channel-specific payload.
        """
        payload = data or {}

        # Channel-specific stream
        channel_stream = self._stream_for_channel(channel)
        if channel_stream:
            _ = self.publish_raw(channel_stream, channel, status, payload)

        # Always broadcast to global
        _ = self.publish_raw(STREAM_GLOBAL, channel, status, payload)

    @staticmethod
    def _stream_for_channel(channel: str) -> str | None:
        """Map a channel name to its dedicated stream."""
        mapping: dict[str, str] = {
            "avatar": STREAM_AVATAR,
            "mouse": STREAM_KEYBOARD,
            "voice": STREAM_VOICE,
        }
        return mapping.get(channel)

    # ------------------------------------------------------------------
    # Subscription / consumer group — v4.5.0 §7.6
    # ------------------------------------------------------------------

    def create_consumer_group(
        self, stream: str, group: str = DEFAULT_CONSUMER_GROUP, start_id: str = "0"
    ) -> bool:
        """Create a consumer group on a stream if it doesn't exist.

        v4.5.0 §7.6: Consumer group mode for multi-consumer subscriptions.

        Args:
            stream: Stream name.
            group: Consumer group name.
            start_id: ID to start consuming from (default: beginning).

        Returns:
            True if group was created or already exists.
        """
        if self._degraded or self._redis is None:
            return False

        try:
            self._redis.xgroup_create(  # pyright: ignore[reportUnknownMemberType]
                stream, group, id=start_id, mkstream=True
            )
            self._consumer_groups.add(f"{stream}:{group}")
            return True
        except Exception:
            # redis.exceptions.ResponseError if group already exists — that's OK
            # Other exceptions (connection issues) are logged
            if "BUSYGROUP" in str(Exception):
                self._consumer_groups.add(f"{stream}:{group}")
                return True
            logger.warning(
                "Failed to create consumer group %s on %s — trace_id=TBD",
                group,
                stream,
                exc_info=True,
            )
            return False

    def read_pending(
        self,
        stream: str,
        group: str = DEFAULT_CONSUMER_GROUP,
        consumer: str = "default",
        count: int = 10,
    ) -> list[dict[str, Any]]:  # pyright: ignore[reportExplicitAny]
        """Read pending (unacknowledged) messages from a consumer group.

        Args:
            stream: Stream name.
            group: Consumer group name.
            consumer: Consumer name within group.
            count: Max messages to read.

        Returns:
            List of pending message dicts.
        """
        if self._degraded or self._redis is None:
            return []

        try:
            pending: list[Any] = self._redis.xpending_range(  # pyright: ignore[reportUnknownMemberType]
                stream, group, min="-", max="+", count=count, consumername=consumer
            )
            return list(pending) if pending else []
        except Exception:
            logger.warning(
                "Failed to read pending messages from %s/%s — trace_id=TBD",
                stream,
                group,
                exc_info=True,
            )
            return []

    def acknowledge(
        self, stream: str, group: str, message_id: str
    ) -> bool:
        """Acknowledge a message in a consumer group.

        v4.5.0 §7.6: Critical messages require ACK.

        Args:
            stream: Stream name.
            group: Consumer group name.
            message_id: ID of message to acknowledge.

        Returns:
            True if acknowledged successfully.
        """
        if self._degraded or self._redis is None:
            return False

        try:
            count: int = self._redis.xack(  # pyright: ignore[reportUnknownMemberType]
                stream, group, message_id
            )
            return count > 0
        except Exception:
            logger.warning(
                "Failed to ACK message %s on %s/%s — trace_id=TBD",
                message_id,
                stream,
                group,
                exc_info=True,
            )
            return False

    def read_stream(
        self,
        stream: str,
        group: str = DEFAULT_CONSUMER_GROUP,
        consumer: str = "default",
        count: int = 10,
        block_ms: int | None = None,
    ) -> list[dict[str, Any]]:  # pyright: ignore[reportExplicitAny]
        """Read new messages from a stream via consumer group.

        Args:
            stream: Stream name.
            group: Consumer group name.
            consumer: Consumer name within group.
            count: Max messages to read.
            block_ms: Block waiting for messages (None = non-blocking).

        Returns:
            List of message dicts with stream, id, and data fields.
        """
        if self._degraded or self._redis is None:
            return []

        try:
            results: list[Any] = self._redis.xreadgroup(  # pyright: ignore[reportUnknownMemberType]
                group, consumer, {stream: ">"}, count=count, block=block_ms
            )
            if not results:
                return []

            messages: list[dict[str, Any]] = []  # pyright: ignore[reportExplicitAny]
            for stream_name, entries in results:  # type: ignore[union-attr]  # pyright: ignore[reportUnknownVariableType]
                for msg_id, fields in entries:  # type: ignore[union-attr]  # pyright: ignore[reportUnknownVariableType]
                    try:
                        data = json.loads(
                            list(fields.values())[0]  # type: ignore[union-attr]  # pyright: ignore[reportUnknownArgumentType,reportUnknownMemberType]
                        )
                    except (json.JSONDecodeError, IndexError):
                        data = {}
                    messages.append({"stream": stream_name, "id": msg_id, "data": data})
            return messages
        except Exception:
            logger.warning(
                "Failed to read from stream %s — trace_id=TBD",
                stream,
                exc_info=True,
            )
            return []

    # ------------------------------------------------------------------
    # Degradation status
    # ------------------------------------------------------------------

    @property
    def is_degraded(self) -> bool:
        """Whether the state bus is in degraded mode (Redis unavailable)."""
        return self._degraded

    @property
    def is_connected(self) -> bool:
        """Whether Redis is connected and operational."""
        return not self._degraded and self._redis is not None
