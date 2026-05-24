"""Contract tests for GentleReminder (spec v4.5.0 §6)."""
from __future__ import annotations

import logging
import pytest

from tests.contracts.conftest import require_module

require_module("src.prediction.gentle_reminder", "GentleReminder")

from src.prediction.gentle_reminder import GentleReminder, Reminder, ReminderType  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_user_model(
    *,
    emotional_pattern: str = "工作日下午容易焦虑",
    confidence: float = 0.7,
    user_verified_fields: list[str] | None = None,
    model_confidence: float = 0.5,
) -> dict:
    """Build a minimal user model dict matching spec §3.4.1 schema."""
    return {
        "user_model_id": "test-uuid",
        "version": 3,
        "inferred_traits": {
            "personality": "偏内向",
            "communication_style": "喜欢用梗",
            "emotional_pattern": emotional_pattern,
        },
        "emotional_pattern_confidence": confidence,
        "knowledge_profile": {
            "topics_of_interest": [],
            "topics_to_avoid": [],
            "expertise_level": {},
        },
        "behavioral_insights": {
            "active_hours": ["weekday_afternoon"],
            "avg_session_length_min": 45,
            "preferred_interaction_mode": "voice_heavy",
        },
        "relationship_meta": {
            "first_interaction_date": "2025-01-01T00:00:00",
            "total_interaction_hours": 120.0,
            "relationship_stage": "familiar",
            "nickname_preference": "小伙伴",
            "model_confidence": model_confidence,
            "user_verified_fields": (
                user_verified_fields if user_verified_fields is not None else []
            ),
        },
    }


# ---------------------------------------------------------------------------
# Module existence
# ---------------------------------------------------------------------------

class TestGentleReminderExists:
    def test_class_is_importable(self):
        assert GentleReminder is not None

    def test_reminder_type_enum_has_all_categories(self):
        """v4.5.0 §6.3: five reminder types."""
        expected = {"time_greeting", "health_reminder", "memory_warm",
                     "silent_company", "preventive_comfort"}
        assert set(ReminderType.__members__.keys()) == expected

    def test_reminder_dataclass_fields(self):
        r = Reminder(reminder_type=ReminderType.time_greeting, text="hello")
        assert r.reminder_type == ReminderType.time_greeting
        assert r.text == "hello"
        assert r.emotion == "neutral"
        assert r.skip_decision is True
        assert isinstance(r.trace_id, str) and len(r.trace_id) > 0


# ---------------------------------------------------------------------------
# evaluate() — idle guard
# ---------------------------------------------------------------------------

class TestEvaluateIdleGuard:
    def test_below_idle_threshold_returns_empty(self):
        """§6.2: no reminders fire when idle < threshold."""
        gr = GentleReminder(idle_threshold=5.0)
        result = gr.evaluate(idle_seconds=2.0)
        assert result == []

    def test_at_idle_threshold_allows_checks(self):
        gr = GentleReminder(idle_threshold=5.0)
        result = gr.evaluate(idle_seconds=5.0)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Preventive comfort — confidence checks (§5.7.5, §6.3)
# ---------------------------------------------------------------------------

class TestPreventiveComfortConfidence:
    def test_high_confidence_triggers(self):
        """§5.7.5: emotional_pattern_confidence >= 0.6 triggers."""
        gr = GentleReminder(idle_threshold=1.0, confidence_threshold=0.6)
        um = _make_user_model(confidence=0.7)
        result = gr.evaluate(user_model=um, idle_seconds=10.0)
        types = [r.reminder_type for r in result]
        assert ReminderType.preventive_comfort in types

    def test_low_confidence_skips(self):
        """§5.7.5: confidence < 0.6 does NOT trigger unless verified."""
        gr = GentleReminder(idle_threshold=1.0, confidence_threshold=0.6)
        um = _make_user_model(confidence=0.3)
        result = gr.evaluate(user_model=um, idle_seconds=10.0)
        types = [r.reminder_type for r in result]
        assert ReminderType.preventive_comfort not in types

    def test_user_verified_bypasses_confidence(self):
        """§5.7.5: user_verified_fields containing 'emotional_pattern'
        allows trigger even with low confidence."""
        gr = GentleReminder(idle_threshold=1.0, confidence_threshold=0.6)
        um = _make_user_model(
            confidence=0.3,
            user_verified_fields=["emotional_pattern"],
        )
        result = gr.evaluate(user_model=um, idle_seconds=10.0)
        types = [r.reminder_type for r in result]
        assert ReminderType.preventive_comfort in types

    def test_new_user_placeholder_skips(self):
        """§3.4.1: new user template '暂无数据' does NOT trigger."""
        gr = GentleReminder(idle_threshold=1.0)
        um = _make_user_model(
            emotional_pattern="暂无数据",
            confidence=0.8,
        )
        result = gr.evaluate(user_model=um, idle_seconds=10.0)
        types = [r.reminder_type for r in result]
        assert ReminderType.preventive_comfort not in types

    def test_missing_emotional_pattern_skips(self):
        """None / empty emotional_pattern skips."""
        gr = GentleReminder(idle_threshold=1.0)
        um = _make_user_model(emotional_pattern="", confidence=0.7)
        result = gr.evaluate(user_model=um, idle_seconds=10.0)
        types = [r.reminder_type for r in result]
        assert ReminderType.preventive_comfort not in types

    def test_none_user_model_skips(self):
        gr = GentleReminder(idle_threshold=1.0)
        result = gr.evaluate(user_model=None, idle_seconds=10.0)
        types = [r.reminder_type for r in result]
        assert ReminderType.preventive_comfort not in types


# ---------------------------------------------------------------------------
# Health reminder (§6.3) — once per day
# ---------------------------------------------------------------------------

class TestHealthReminder:
    def test_triggers_after_6_hours(self):
        gr = GentleReminder(idle_threshold=1.0, health_hours=6.0)
        result = gr.evaluate(idle_seconds=10.0, session_duration_hours=7.0)
        types = [r.reminder_type for r in result]
        assert ReminderType.health_reminder in types

    def test_does_not_trigger_below_threshold(self):
        gr = GentleReminder(idle_threshold=1.0, health_hours=6.0)
        result = gr.evaluate(idle_seconds=10.0, session_duration_hours=3.0)
        types = [r.reminder_type for r in result]
        assert ReminderType.health_reminder not in types

    def test_once_per_day(self):
        """§6.4: health reminder fires at most once per day."""
        gr = GentleReminder(idle_threshold=1.0, health_hours=1.0)
        # First call
        result1 = gr.evaluate(idle_seconds=10.0, session_duration_hours=2.0)
        health_count1 = sum(1 for r in result1 if r.reminder_type == ReminderType.health_reminder)
        assert health_count1 == 1
        # Second call — same day, should not trigger again
        result2 = gr.evaluate(idle_seconds=10.0, session_duration_hours=2.0)
        health_count2 = sum(1 for r in result2 if r.reminder_type == ReminderType.health_reminder)
        assert health_count2 == 0


# ---------------------------------------------------------------------------
# Silent companion (§6.3)
# ---------------------------------------------------------------------------

class TestSilentCompanion:
    def test_triggers_after_10_minutes(self):
        gr = GentleReminder(idle_threshold=1.0, silent_minutes=10.0)
        result = gr.evaluate(idle_seconds=10.0, silent_minutes=12.0)
        types = [r.reminder_type for r in result]
        assert ReminderType.silent_company in types

    def test_does_not_trigger_early(self):
        gr = GentleReminder(idle_threshold=1.0, silent_minutes=10.0)
        result = gr.evaluate(idle_seconds=10.0, silent_minutes=5.0)
        types = [r.reminder_type for r in result]
        assert ReminderType.silent_company not in types


# ---------------------------------------------------------------------------
# Time greeting (§6.3)
# ---------------------------------------------------------------------------

class TestTimeGreeting:
    def test_triggers_when_idle(self):
        gr = GentleReminder(idle_threshold=1.0)
        result = gr.evaluate(idle_seconds=5.0)
        types = [r.reminder_type for r in result]
        assert ReminderType.time_greeting in types

    def test_once_per_day(self):
        gr = GentleReminder(idle_threshold=1.0)
        result1 = gr.evaluate(idle_seconds=5.0)
        greeting_count1 = sum(1 for r in result1 if r.reminder_type == ReminderType.time_greeting)
        assert greeting_count1 == 1
        result2 = gr.evaluate(idle_seconds=5.0)
        greeting_count2 = sum(1 for r in result2 if r.reminder_type == ReminderType.time_greeting)
        assert greeting_count2 == 0


# ---------------------------------------------------------------------------
# Memory warm (§6.3)
# ---------------------------------------------------------------------------

class TestMemoryWarm:
    def test_triggers_with_cold_moment(self):
        gr = GentleReminder(idle_threshold=1.0)
        moment = {"summary": "聊《流浪地球》", "emotional_significance": "high"}
        result = gr.evaluate(idle_seconds=10.0, cold_moment=moment)
        types = [r.reminder_type for r in result]
        assert ReminderType.memory_warm in types


# ---------------------------------------------------------------------------
# Reminder attributes
# ---------------------------------------------------------------------------

class TestReminderAttributes:
    def test_all_reminders_have_trace_id(self):
        gr = GentleReminder(idle_threshold=1.0)
        um = _make_user_model(confidence=0.8)
        result = gr.evaluate(user_model=um, idle_seconds=10.0)
        assert len(result) > 0
        for r in result:
            assert isinstance(r.trace_id, str)
            assert len(r.trace_id) > 0

    def test_all_reminders_skip_decision(self):
        """§6.4: reminders bypass decision layer."""
        gr = GentleReminder(idle_threshold=1.0)
        um = _make_user_model(confidence=0.8)
        result = gr.evaluate(user_model=um, idle_seconds=10.0)
        for r in result:
            assert r.skip_decision is True

    def test_reminders_have_valid_emotion(self):
        allowed = {"joy", "sadness", "neutral"}
        gr = GentleReminder(idle_threshold=1.0)
        um = _make_user_model(confidence=0.8)
        result = gr.evaluate(user_model=um, idle_seconds=10.0, silent_minutes=15.0,
                            session_duration_hours=7.0)
        for r in result:
            assert r.emotion in allowed


# ---------------------------------------------------------------------------
# Logging with trace_id
# ---------------------------------------------------------------------------

class TestLogging:
    def test_preventive_comfort_logs_trace_id(self, caplog):
        gr = GentleReminder(idle_threshold=1.0, confidence_threshold=0.6)
        um = _make_user_model(confidence=0.75)
        with caplog.at_level(logging.INFO, logger="src.prediction.gentle_reminder"):
            gr.evaluate(user_model=um, idle_seconds=10.0)
        # Should have at least one log with trace_id
        trace_logs = [r for r in caplog.records if hasattr(r, "message")]
        assert any("trace_id" in r.message.lower() for r in trace_logs)


# ---------------------------------------------------------------------------
# Default parameters
# ---------------------------------------------------------------------------

class TestDefaults:
    def test_default_idle_threshold(self):
        gr = GentleReminder()
        assert gr.idle_threshold == 10.0  # §6.2 max

    def test_default_confidence_threshold(self):
        gr = GentleReminder()
        assert gr.confidence_threshold == 0.6  # §5.7.5

    def test_default_health_hours(self):
        gr = GentleReminder()
        assert gr.health_hours == 6.0  # §6.3
