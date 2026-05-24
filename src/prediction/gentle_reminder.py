"""Gentle Reminder — proactive comfort and well-being checks.

v4.5.0 §6: Monitors user emotional patterns and interaction cadence,
triggering lightweight template-based reminders when confidence thresholds
are met.  No LLM calls — all output is template + dynamic parameter fill.

Reminder types (§6.3):
  * time_greeting   — first daily interaction, time-of-day greeting
  * health_reminder — continuous interaction > 6 hours (once per day)
  * memory_warm     — random high-emotion Moment from cold memory
  * silent_company  — user silent > 10 minutes
  * preventive_comfort — emotional_pattern confidence ≥ 0.6 or user_verified

Key constraint (§5.7.5 / §6.3):
  preventive_comfort requires emotional_pattern._confidence ≥ 0.6
  OR the "emotional_pattern" field present in user_verified_fields.
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reminder type enum — spec §6.3
# ---------------------------------------------------------------------------

class ReminderType(str, Enum):
    """v4.5.0 §6.3 — reminder categories."""
    time_greeting = "time_greeting"
    health_reminder = "health_reminder"
    memory_warm = "memory_warm"
    silent_company = "silent_company"
    preventive_comfort = "preventive_comfort"


# ---------------------------------------------------------------------------
# Reminder output dataclass
# ---------------------------------------------------------------------------

@dataclass
class Reminder:
    """A single reminder recommendation with metadata.

    Attributes
    ----------
    reminder_type : ReminderType
        Which reminder category this belongs to.
    text : str
        The rendered reminder text (template filled with dynamic params).
    emotion : str
        Suggested emotion for TTS/delivery ("joy", "sadness", "neutral").
    priority : int
        Higher = more important. User actions override reminders.
    trace_id : str
        UUID for logging and trace correlation across layers.
    skip_decision : bool
        Always True — reminders bypass the decision layer (§6.4).
    """
    reminder_type: ReminderType
    text: str
    emotion: str = "neutral"
    priority: int = 0
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    skip_decision: bool = True


# ---------------------------------------------------------------------------
# Reminder template store — spec §6.4: "预置模板 + 动态参数填充生成，不调用大模型"
# ---------------------------------------------------------------------------

_TIME_GREETINGS: dict[str, str] = {
    "morning": "早安，今天也要元气满满哦～",
    "afternoon": "午安，下午也要加油哦～",
    "evening": "傍晚了，今天辛苦啦～",
    "night": "夜深了，记得早点休息哦～",
}

_HEALTH_REMINDER_TEXT = "该起来走动一下啦，我帮你盯着屏幕～"
_SILENT_COMPANY_TEXT = "（静静地陪在你身边）"
_PREVENTIVE_COMFORT_TEMPLATE = "说起来，{pattern_hint}。要不要歇一下？"


# ---------------------------------------------------------------------------
# Default thresholds
# ---------------------------------------------------------------------------

DEFAULT_IDLE_SECONDS = 10.0          # §6.2: max idle_threshold
DEFAULT_HEALTH_HOURS = 6.0           # §6.3: continuous interaction threshold
DEFAULT_SILENT_MINUTES = 10.0        # §6.3: silent companion trigger
DEFAULT_CONFIDENCE_THRESHOLD = 0.6   # §5.7.5: emotional_pattern confidence floor
DEFAULT_WEATHER_TIMEOUT_MS = 500     # §6.3: wttr.in timeout


# ---------------------------------------------------------------------------
# GentleReminder
# ---------------------------------------------------------------------------

class GentleReminder:
    """Proactive reminder engine — template-based, no LLM.

    Parameters
    ----------
    idle_threshold: float
        Seconds of idle before reminders are considered (§6.2, default 10.0).
    health_hours: float
        Continuous interaction hours before health reminder (§6.3, default 6.0).
    silent_minutes: float
        Minutes of user silence before silent companion (§6.3, default 10.0).
    confidence_threshold: float
        Minimum emotional_pattern._confidence for preventive comfort (§5.7.5).
    weather_timeout_ms: int
        wttr.in API timeout in milliseconds (§6.3, default 500).
    """

    # ------------------------------------------------------------------ #
    idle_threshold: float
    health_hours: float
    silent_minutes: float
    confidence_threshold: float
    weather_timeout_ms: int

    _last_health_reminder_date: Optional[str]
    _last_greeting_date: Optional[str]
    _session_start: float

    def __init__(
        self,
        idle_threshold: float = DEFAULT_IDLE_SECONDS,
        health_hours: float = DEFAULT_HEALTH_HOURS,
        silent_minutes: float = DEFAULT_SILENT_MINUTES,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        weather_timeout_ms: int = DEFAULT_WEATHER_TIMEOUT_MS,
    ) -> None:
        self.idle_threshold = idle_threshold
        self.health_hours = health_hours
        self.silent_minutes = silent_minutes
        self.confidence_threshold = confidence_threshold
        self.weather_timeout_ms = weather_timeout_ms

        self._last_health_reminder_date = None
        self._last_greeting_date = None
        self._session_start = time.time()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def evaluate(
        self,
        user_model: Optional[dict[str, Any]] = None,
        idle_seconds: float = 0.0,
        session_duration_hours: float = 0.0,
        silent_minutes: float = 0.0,
        cold_moment: Optional[dict[str, Any]] = None,
    ) -> list[Reminder]:
        """Evaluate all reminder conditions and return triggered reminders.

        Parameters
        ----------
        user_model:
            The current UserModel dict (§3.4.1).  Must contain
            ``inferred_traits``, ``relationship_meta``, and optionally
            ``_confidence`` suffixed sibling fields.
        idle_seconds:
            Seconds since the decision layer last issued a command (§6.2).
        session_duration_hours:
            Total hours of continuous interaction this session.
        silent_minutes:
            Minutes since last user interaction (voice / text input).
        cold_moment:
            A high-emotion Moment from cold memory for memory_warm reminders.

        Returns
        -------
        list[Reminder]
            The triggered reminders, sorted by priority (highest first).
            Empty list if no conditions are met.
        """
        reminders: list[Reminder] = []

        if idle_seconds < self.idle_threshold:
            return reminders  # too soon — skip all checks

        # ---- Time greeting (once per day) ----
        today = datetime.now().strftime("%Y-%m-%d")
        if self._last_greeting_date != today:
            reminder = self._check_time_greeting()
            if reminder is not None:
                reminders.append(reminder)
                self._last_greeting_date = today

        # ---- Health reminder (once per day) ----
        if session_duration_hours >= self.health_hours:
            if self._last_health_reminder_date != today:
                reminders.append(self._build_health_reminder())
                self._last_health_reminder_date = today

        # ---- Silent companion ----
        if silent_minutes >= self.silent_minutes:
            reminders.append(self._build_silent_reminder())

        # ---- Memory warm ----
        if cold_moment is not None:
            reminders.append(self._build_memory_reminder(cold_moment))

        # ---- Preventive comfort ----
        comfort = self._check_preventive_comfort(user_model)
        if comfort is not None:
            reminders.append(comfort)

        reminders.sort(key=lambda r: r.priority, reverse=True)
        return reminders

    def _check_preventive_comfort(
        self, user_model: Optional[dict[str, Any]]
    ) -> Optional[Reminder]:
        """Evaluate preventive comfort conditions (§6.3, §5.7.5).

        Requires emotional_pattern._confidence ≥ confidence_threshold
        OR "emotional_pattern" in user_verified_fields.
        """
        if user_model is None:
            return None

        inferred = user_model.get("inferred_traits", {})
        if not isinstance(inferred, dict):
            return None  # v4.5.0 §2.0: malformed model → skip

        pattern = inferred.get("emotional_pattern", "")
        if not pattern or pattern == "暂无数据":
            return None  # v4.5.0 §3.4.1: new-user template → skip

        # Confidence check: §5.7.5
        confidence = user_model.get("emotional_pattern_confidence")
        if confidence is None:
            # Check the inferred_traits._confidence sibling
            confidence = inferred.get("emotional_pattern_confidence")

        relationship_meta = user_model.get("relationship_meta", {})
        if not isinstance(relationship_meta, dict):
            relationship_meta = {}
        user_verified_fields: list[str] = relationship_meta.get(
            "user_verified_fields", []
        )
        if not isinstance(user_verified_fields, list):
            user_verified_fields = []

        confidence_ok = (
            isinstance(confidence, (int, float)) and confidence >= self.confidence_threshold
        )
        verified_ok = "emotional_pattern" in user_verified_fields

        if not (confidence_ok or verified_ok):
            logger.debug(
                "preventive_comfort skipped: confidence=%.2f < %.2f, verified=%s",
                float(confidence) if isinstance(confidence, (int, float)) else 0.0,
                self.confidence_threshold,
                verified_ok,
            )
            return None

        # Check time-window match — simplified: use the pattern string
        # as a hint; real implementation would parse active_hours and
        # match against current time (§6.3).
        hint = self._extract_pattern_hint(pattern)

        trace_id = str(uuid.uuid4())
        logger.info(
            "preventive_comfort triggered trace_id=%s confidence=%.2f verified=%s",
            trace_id,
            float(confidence) if isinstance(confidence, (int, float)) else 0.0,
            verified_ok,
        )

        return Reminder(
            reminder_type=ReminderType.preventive_comfort,
            text=_PREVENTIVE_COMFORT_TEMPLATE.format(pattern_hint=hint),
            emotion="neutral",
            priority=10,
            trace_id=trace_id,
        )

    # ------------------------------------------------------------------ #
    # Internal builders
    # ------------------------------------------------------------------ #

    def _check_time_greeting(self) -> Optional[Reminder]:
        hour = datetime.now().hour
        if 6 <= hour < 12:
            greeting = _TIME_GREETINGS["morning"]
        elif 12 <= hour < 14:
            greeting = _TIME_GREETINGS["afternoon"]
        elif 14 <= hour < 19:
            greeting = _TIME_GREETINGS["afternoon"]
        elif 19 <= hour < 23:
            greeting = _TIME_GREETINGS["evening"]
        else:
            greeting = _TIME_GREETINGS["night"]

        trace_id = str(uuid.uuid4())
        logger.info("time_greeting trace_id=%s hour=%d", trace_id, hour)
        return Reminder(
            reminder_type=ReminderType.time_greeting,
            text=greeting,
            emotion="joy",
            priority=5,
            trace_id=trace_id,
        )

    def _build_health_reminder(self) -> Reminder:
        trace_id = str(uuid.uuid4())
        logger.info("health_reminder trace_id=%s", trace_id)
        return Reminder(
            reminder_type=ReminderType.health_reminder,
            text=_HEALTH_REMINDER_TEXT,
            emotion="neutral",
            priority=15,
            trace_id=trace_id,
        )

    def _build_silent_reminder(self) -> Reminder:
        trace_id = str(uuid.uuid4())
        logger.info("silent_company trace_id=%s", trace_id)
        return Reminder(
            reminder_type=ReminderType.silent_company,
            text=_SILENT_COMPANY_TEXT,
            emotion="neutral",
            priority=1,
            trace_id=trace_id,
        )

    def _build_memory_reminder(self, moment: dict[str, Any]) -> Reminder:
        trace_id = str(uuid.uuid4())
        logger.info("memory_warm trace_id=%s", trace_id)
        summary = moment.get("summary", "那个温暖的瞬间")
        text = f"说起来，上次你{summary}时笑得好开心，那感觉真不错～"
        return Reminder(
            reminder_type=ReminderType.memory_warm,
            text=text,
            emotion="joy",
            priority=8,
            trace_id=trace_id,
        )

    @staticmethod
    def _extract_pattern_hint(pattern: str) -> str:
        """Extract a short hint from the emotional_pattern string.

        If the pattern is very long, truncate to ~20 characters
        for a natural-sounding insertion.
        """
        if len(pattern) <= 20:
            return pattern
        return pattern[:18] + "…"

    # ------------------------------------------------------------------ #
    # Direct generate — for DecisionBridge comfort injection (§5.7.5 / §6.3)
    # ------------------------------------------------------------------ #

    def generate(self, emotional_pattern: dict[str, Any]) -> str:
        """Generate preventive comfort text from emotional pattern data.

        v4.5.0 §6.3: Template-based, no LLM. Returns a natural-sounding
        comfort prompt for injection into the LLM system prompt via
        DecisionBridge.

        Args:
            emotional_pattern: Dict with keys like ``trend``, ``trend_description``,
                ``raw``.  The ``trend_description`` is preferred as the pattern hint;
                falls back to ``raw`` or a generic default.

        Returns:
            Rendered comfort text (e.g. "说起来，最近有点低落。要不要歇一下？").
        """
        pattern_text = emotional_pattern.get("trend_description", "")
        if not pattern_text:
            pattern_text = str(emotional_pattern.get("raw", "最近是不是有点累了"))
        hint = self._extract_pattern_hint(pattern_text)
        return _PREVENTIVE_COMFORT_TEMPLATE.format(pattern_hint=hint)

    async def fetch_weather(self) -> Optional[str]:
        """Attempt to fetch weather via wttr.in (optional, §6.3).

        Returns a short weather summary string, or None on failure/timeout.
        """
        try:
            import aiohttp
        except ImportError:
            # Catches ImportError: aiohttp not installed.
            # Safe: weather is optional — reminder works without it.
            logger.debug("aiohttp not available — skipping weather fetch")
            return None

        try:
            # wttr.in '?format=3' gives "City: weather, temp" one-liner
            timeout_sec = self.weather_timeout_ms / 1000.0
            timeout = aiohttp.ClientTimeout(total=timeout_sec)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(
                    "https://wttr.in/?format=3", headers={"User-Agent": "curl/7"}
                ) as resp:
                    if resp.status == 200:
                        text = await resp.text()
                        return text.strip()
                    return None
        except Exception:
            # Catches aiohttp.ClientError, TimeoutError, or any network issue.
            # Safe: weather is optional — reminder works without it.
            logger.debug("weather fetch failed (optional) — skipping")
            return None

    def reset_daily_counters(self) -> None:
        """Reset daily state (health_reminder counter, greeting flag).

        Called on date change to allow reminders to fire again.
        """
        today = datetime.now().strftime("%Y-%m-%d")
        if self._last_health_reminder_date != today:
            self._last_health_reminder_date = None
        if self._last_greeting_date != today:
            self._last_greeting_date = None
