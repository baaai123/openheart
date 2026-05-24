"""Contract tests for CompanionMetrics (spec v4.5.0 §8.1)."""
from __future__ import annotations

import pytest

from tests.contracts.conftest import require_module

require_module("src.prediction.companion_metrics", "CompanionMetrics")

from src.prediction.companion_metrics import CompanionMetrics, MetricsSnapshot  # noqa: E402


class TestCompanionMetricsExists:
    def test_class_is_importable(self):
        assert CompanionMetrics is not None

    def test_snapshot_class_is_importable(self):
        assert MetricsSnapshot is not None


class TestCompanionMetricsRecording:
    def test_record_interaction_increments_counter(self):
        cm = CompanionMetrics()
        cm.record_interaction()
        assert cm._interaction_count == 1

    def test_record_latency_appends_value(self):
        cm = CompanionMetrics()
        cm.record_latency(150.0)
        assert len(cm._latencies) == 1
        assert cm._latencies[0] == 150.0

    def test_record_latency_clamps_max(self):
        cm = CompanionMetrics(max_latency_ms=1000.0)
        cm.record_latency(2000.0)
        assert cm._latencies[0] == 1000.0

    def test_record_latency_rejects_negative(self):
        cm = CompanionMetrics()
        cm.record_latency(-50.0)
        assert cm._latencies[0] == 0.0

    def test_record_degradation_increments_counter(self):
        cm = CompanionMetrics()
        cm.record_degradation()
        assert cm._degradation_count == 1

    def test_record_emotion_counts_joy_and_sadness(self):
        cm = CompanionMetrics()
        cm.record_emotion("joy")
        cm.record_emotion("sadness")
        assert cm._emotion_hits == 2

    def test_record_emotion_ignores_neutral(self):
        cm = CompanionMetrics()
        cm.record_emotion("neutral")
        assert cm._emotion_hits == 0

    def test_record_emotion_ignores_placeholder_anger(self):
        cm = CompanionMetrics()
        cm.record_emotion("anger")
        assert cm._emotion_hits == 0

    def test_record_emotion_ignores_placeholder_surprise(self):
        cm = CompanionMetrics()
        cm.record_emotion("surprise")
        assert cm._emotion_hits == 0


class TestCompanionMetricsComputation:
    def test_response_latency_zero_when_empty(self):
        cm = CompanionMetrics()
        assert cm.response_latency_ms == 0.0

    def test_response_latency_average(self):
        cm = CompanionMetrics()
        cm.record_latency(100.0)
        cm.record_latency(200.0)
        assert cm.response_latency_ms == 150.0

    def test_emotion_match_rate_zero_when_empty(self):
        cm = CompanionMetrics()
        assert cm.emotion_match_rate == 0.0

    def test_emotion_match_rate_calculation(self):
        cm = CompanionMetrics()
        cm.record_interaction()
        cm.record_emotion("joy")
        cm.record_interaction()
        cm.record_emotion("neutral")
        assert cm.emotion_match_rate == 0.5

    def test_degradation_frequency_zero_when_empty(self):
        cm = CompanionMetrics()
        assert cm.degradation_frequency == 0.0

    def test_degradation_frequency_calculation(self):
        cm = CompanionMetrics()
        cm.record_interaction()
        cm.record_interaction()
        cm.record_degradation()
        assert cm.degradation_frequency == 0.5

    def test_user_engagement_score_zero_when_empty(self):
        cm = CompanionMetrics()
        assert cm.user_engagement_score == 0.0

    def test_user_engagement_score_increases_with_interactions(self):
        cm = CompanionMetrics()
        for _ in range(10):
            cm.record_interaction()
            cm.record_latency(100.0)
            cm.record_emotion("joy")
        assert cm.user_engagement_score > 0.0
        assert cm.user_engagement_score <= 1.0


class TestCompanionMetricsSnapshot:
    def test_snapshot_returns_current_values(self):
        cm = CompanionMetrics()
        cm.record_interaction()
        cm.record_latency(200.0)
        cm.record_emotion("joy")
        snap = cm.snapshot()
        assert snap.interaction_count == 1
        assert snap.response_latency_ms == 200.0
        assert snap.emotion_match_rate == 1.0
        assert snap.degradation_count == 0

    def test_snapshot_timestamp_is_iso8601(self):
        cm = CompanionMetrics()
        snap = cm.snapshot()
        assert "T" in snap.timestamp


class TestCompanionMetricsReport:
    def test_generate_report_has_required_keys(self):
        cm = CompanionMetrics()
        cm.record_interaction()
        report = cm.generate_report()
        required = {
            "date",
            "interaction_count",
            "response_latency_ms",
            "emotion_match_rate",
            "user_engagement_score",
            "degradation_frequency",
            "degradation_count",
        }
        assert required.issubset(report.keys())

    def test_generate_report_date_is_today(self):
        from datetime import datetime, timezone

        cm = CompanionMetrics()
        report = cm.generate_report()
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        assert report["date"] == today


class TestCompanionMetricsReset:
    def test_reset_clears_all_counters(self):
        cm = CompanionMetrics()
        cm.record_interaction()
        cm.record_latency(100.0)
        cm.record_emotion("joy")
        cm.record_degradation()
        cm.reset()
        assert cm._interaction_count == 0
        assert len(cm._latencies) == 0
        assert cm._emotion_hits == 0
        assert cm._degradation_count == 0
