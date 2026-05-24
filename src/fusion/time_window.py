"""Adaptive time window for fusion layer — v4.5.0 §2.3

Groups incoming perception events into WindowedEvents based on trigger
conditions: voice segment end, visual change, max timeout, minimum
protection, and affective highlighting.
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Constants — v4.5.0 §2.3
# --------------------------------------------------------------------------

DEFAULT_MAX_WINDOW_MS: float = 800.0
DEFAULT_MIN_WINDOW_MS: float = 150.0
AFFECTIVE_MAX_WINDOW_MS: float = 600.0
VISUAL_CHANGE_THRESHOLD: int = 3

# --------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------


@dataclass
class WindowedEvents:
    """Output of the time sync window — v4.5.0 §2.3"""

    window_id: str
    window_start: str  # ISO 8601
    window_end: str  # ISO 8601
    events: list[dict[str, Any]]
    trigger_reason: str  # voice_segment_end | visual_change | max_timeout
    affective_highlight: bool


def _iso_now() -> str:
    """Return current UTC time as ISO 8601 string."""
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _extract_timestamp_float(ts: str | None) -> float:
    """Safely parse an ISO 8601 string into a Unix timestamp (float seconds).

    Falls back to time.time() on failure so the windowing loop does not
    dead-lock on a single malformed timestamp.
    """
    if ts is None:
        return time.time()
    try:
        import datetime
        dt = datetime.datetime.fromisoformat(ts)
        return dt.timestamp()
    except (ValueError, TypeError):
        # Malformed timestamp — use current time to keep pipeline moving.
        logger.warning(
            "fusion.time_window: malformed timestamp %r, falling back to now", ts
        )
        return time.time()


def _count_event_types(
    events: list[dict[str, Any]],
) -> tuple[set[str], set[str]]:
    """Extract distinct object types and UI categories from vision events.

    Returns (object_types, ui_types).
    """
    object_types: set[str] = set()
    ui_types: set[str] = set()

    for evt in events:
        payload = evt.get("payload", {})
        if payload.get("type") != "vision_snapshot":
            continue
        vs = payload.get("vision_snapshot", {})
        for obj in vs.get("objects", []):
            label = obj.get("label") or obj.get("class_name")
            if label:
                object_types.add(str(label))
        for txt in vs.get("text_content", []):
            ui_label = txt.get("type") or txt.get("label")
            if ui_label:
                ui_types.add(str(ui_label))

    return object_types, ui_types


# --------------------------------------------------------------------------
# TimeSyncWindow
# --------------------------------------------------------------------------


class TimeSyncWindow:
    """Adaptive time window that groups perception events for fusion.

    Parameters
    ----------
    max_window_ms: float
        Maximum window duration in milliseconds (default 800).
    min_window_ms: float
        Minimum window duration in milliseconds (default 150).
    visual_change_threshold: int
        Number of new object/UI type changes to trigger on (default 3).

    v4.5.0 §2.3
    """

    def __init__(
        self,
        max_window_ms: float = DEFAULT_MAX_WINDOW_MS,
        min_window_ms: float = DEFAULT_MIN_WINDOW_MS,
        visual_change_threshold: int = VISUAL_CHANGE_THRESHOLD,
    ) -> None:
        self.max_window_ms = max_window_ms
        self.min_window_ms = min_window_ms
        self.visual_change_threshold = visual_change_threshold

        # Internal state
        self._buffer: list[dict[str, Any]] = []
        self._last_output_time: float = 0.0
        self._has_output: bool = False  # True after first flush
        self._last_object_types: set[str] = set()
        self._last_ui_types: set[str] = set()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def push(self, event: dict[str, Any]) -> WindowedEvents | None:
        """Feed one perception event into the window.

        Returns a WindowedEvents batch if the window triggers, else None.
        The caller should drain the window immediately after a trigger.

        v4.5.0 §2.3: trigger conditions
        """
        # Check if event has a metadata timestamp
        ts_str = event.get("timestamp")
        event_time = _extract_timestamp_float(ts_str)

        # If buffer is empty, set start time
        if not self._buffer:
            self._last_output_time = event_time
            self._last_object_types, self._last_ui_types = _count_event_types(
                [event]
            )

        self._buffer.append(event)

        # Only evaluate triggers after minimum window protection
        # (skip protection on first event — no prior output to protect)
        if self._has_output:
            elapsed_since_output = (event_time - self._last_output_time) * 1000.0
            if elapsed_since_output < self.min_window_ms:
                return None

        # Determine effective max window (affective highlight shortens it)
        has_affective = any(
            e.get("metadata", {}).get("affective_flag", False) is True
            for e in self._buffer
        )
        effective_max = (
            AFFECTIVE_MAX_WINDOW_MS if has_affective else self.max_window_ms
        )

        # Trigger condition 1: voice segment end
        trigger_reason = self._check_voice_segment_end()
        if trigger_reason:
            return self._flush(trigger_reason)

        # Trigger condition 2: significant visual change
        trigger_reason = self._check_visual_change()
        if trigger_reason:
            return self._flush(trigger_reason)

        # Trigger condition 3: max window timeout
        window_start_time = (
            _extract_timestamp_float(self._buffer[0].get("timestamp"))
            if self._buffer
            else event_time
        )
        elapsed_window = (event_time - window_start_time) * 1000.0
        if elapsed_window >= effective_max:
            return self._flush("max_timeout")

        return None

    def flush(self) -> WindowedEvents | None:
        """Force-flush the current buffer regardless of trigger conditions.

        Returns None if the buffer is empty.
        """
        if not self._buffer:
            return None
        return self._flush("forced_flush")

    # ------------------------------------------------------------------ #
    # Internal trigger checks
    # ------------------------------------------------------------------ #

    def _check_voice_segment_end(self) -> str | None:
        """Check if the latest audio event marks a voice segment end.

        A voice segment end is indicated by:
        - An audio event whose payload.audio.text is non-empty and the
          perception layer signals `voice_segment_end` through an
          `is_segment_end` flag in the audio payload, OR
        - Heuristic: the last audio event has text but no newer audio
          event follows (implied segment boundary).

        Returns "voice_segment_end" or None.
        """
        audio_events = [
            e
            for e in self._buffer
            if e.get("payload", {}).get("type") == "audio_event"
        ]
        if not audio_events:
            return None

        last_audio = audio_events[-1]
        audio_payload = last_audio.get("payload", {}).get("audio", {})

        # Explicit segment end flag (perception layer may set this)
        explicit = audio_payload.get("is_segment_end")
        if explicit is True:
            return "voice_segment_end"
        if explicit is False:
            # Explicitly marked as not-segment-end — do not auto-trigger
            return None

        # Heuristic (only when flag is absent): if the last audio has text
        # content, treat as potential segment boundary.
        if audio_payload.get("text") and len(audio_payload["text"].strip()) > 0:
            return "voice_segment_end"

        return None

    def _check_visual_change(self) -> str | None:
        """Check for significant visual category change (≥ threshold new types).

        Returns "visual_change" or None.
        """
        object_types, ui_types = _count_event_types(self._buffer)

        new_objects = object_types - self._last_object_types
        new_ui = ui_types - self._last_ui_types
        total_change = len(new_objects) + len(new_ui)

        if total_change >= self.visual_change_threshold:
            return "visual_change"

        return None

    # ------------------------------------------------------------------ #
    # Buffer management
    # ------------------------------------------------------------------ #

    def _flush(self, trigger_reason: str) -> WindowedEvents:
        """Drain the buffer and produce a WindowedEvents batch."""
        events = list(self._buffer)
        self._buffer.clear()

        # Update visual state trackers for next window
        self._last_object_types, self._last_ui_types = _count_event_types(
            events
        )

        # Extract time range
        times = []
        for e in events:
            ts = e.get("timestamp")
            if ts:
                t = _extract_timestamp_float(ts)
                times.append(t)

        if times:
            window_start_ts = min(times)
            window_end_ts = max(times)
        else:
            now = time.time()
            window_start_ts = now
            window_end_ts = now

        import datetime

        window_start = datetime.datetime.fromtimestamp(
            window_start_ts, tz=datetime.timezone.utc
        ).isoformat()
        window_end = datetime.datetime.fromtimestamp(
            window_end_ts, tz=datetime.timezone.utc
        ).isoformat()

        has_affective = any(
            e.get("metadata", {}).get("affective_flag", False) is True
            for e in events
        )

        result = WindowedEvents(
            window_id=str(uuid.uuid4()),
            window_start=window_start,
            window_end=window_end,
            events=events,
            trigger_reason=trigger_reason,
            affective_highlight=has_affective,
        )

        self._last_output_time = window_end_ts
        self._has_output = True

        logger.debug(
            "fusion.time_window: flushed window %s with %d events, "
            "reason=%s, affective=%s",
            result.window_id,
            len(events),
            trigger_reason,
            has_affective,
        )

        return result

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        """Reset internal state completely (e.g. on session restart)."""
        self._buffer.clear()
        self._last_output_time = 0.0
        self._has_output = False
        self._last_object_types.clear()
        self._last_ui_types.clear()
