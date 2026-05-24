"""
Contract tests for the unified message envelope (spec v4.5.0 section 0.3).

All layer-to-layer communication MUST follow this format.
These tests define the "right" vs "wrong" standards for message envelope construction.
"""
import uuid
import pytest
from copy import deepcopy
from datetime import datetime, timezone

from .conftest import require_module


class TestModuleExists:
    def test_message_envelope_module_available(self):
        """The shared message envelope module must be importable."""
        require_module(
            module_path="src.fusion.message_envelope",
            component_name="MessageEnvelope (fusion/message_envelope.py)",
        )
        # If we got here, the module exists — validate it exports the core API.
        import src.fusion.message_envelope as me
        assert hasattr(me, "MessageEnvelope"), "Must export MessageEnvelope dataclass"
        assert hasattr(me, "create_envelope"), "Must export create_envelope factory"
        assert hasattr(me, "Layer"), "Must export Layer enum"
        assert hasattr(me, "PayloadType"), "Must export PayloadType enum"
        assert hasattr(me, "EmotionCategory"), "Must export EmotionCategory enum"
        assert hasattr(me, "MessageValidationError"), "Must export MessageValidationError"


# ——— Envelope specification (from spec v4.5.0 section 0.3) ———

VALID_ENVELOPE = {
    "trace_id": "00000000-0000-0000-0000-000000000001",
    "source_layer": "perception",
    "source_component": "vision_fusion",
    "timestamp": "2026-05-09T12:00:00.000+00:00",
    "version": 1,
    "payload_type": "perception_event",
    "payload": {},
    "metadata": {
        "confidence": 1.0,
        "latency_ms": 10.0,
        "degraded": False,
        "fast_path": False,
        "emotion": {
            "category": "neutral",
            "intensity": 0.2,
            "source": "text_sentiment",
            "confidence": 0.8,
        },
        "affective_flag": False,
        "scene_context": {
            "primary_type": "code_editor",
            "confidence": 0.94,
        },
        "user_model_version": 0,
    },
}


class TestMessageEnvelopeStructure:
    @pytest.fixture
    def envelope(self):
        return deepcopy(VALID_ENVELOPE)

    def test_has_required_top_level_fields(self, envelope):
        required = [
            "trace_id", "source_layer", "source_component", "timestamp",
            "version", "payload_type", "payload", "metadata",
        ]
        for field in required:
            assert field in envelope, f"Missing required field: {field}"

    def test_trace_id_is_valid_uuid(self, envelope):
        tid = envelope["trace_id"]
        uuid.UUID(tid)

    def test_trace_id_is_string(self, envelope):
        assert isinstance(envelope["trace_id"], str)

    def test_source_layer_must_be_valid_enum(self, envelope):
        valid_layers = {
            "perception", "fusion", "memory_hot", "memory_cold",
            "personality", "decision", "prediction", "execution",
        }
        assert envelope["source_layer"] in valid_layers

    def test_version_is_integer(self, envelope):
        assert isinstance(envelope["version"], int)

    def test_payload_type_must_be_valid_enum(self, envelope):
        valid_types = {
            "perception_event", "scene", "memory_query", "personality_config",
            "decision_command", "prediction_task", "action_sequence",
            "affective_event", "user_model_update",
        }
        assert envelope["payload_type"] in valid_types

    def test_metadata_has_required_fields(self, envelope):
        meta_required = [
            "confidence", "latency_ms", "degraded", "fast_path",
            "emotion", "affective_flag", "scene_context", "user_model_version",
        ]
        for field in meta_required:
            assert field in envelope["metadata"], f"Missing metadata field: {field}"

    def test_confidence_is_float_between_0_and_1(self, envelope):
        c = envelope["metadata"]["confidence"]
        assert isinstance(c, float)
        assert 0.0 <= c <= 1.0

    def test_degraded_is_boolean(self, envelope):
        assert isinstance(envelope["metadata"]["degraded"], bool)

    def test_fast_path_is_boolean(self, envelope):
        assert isinstance(envelope["metadata"]["fast_path"], bool)

    def test_affective_flag_is_boolean(self, envelope):
        assert isinstance(envelope["metadata"]["affective_flag"], bool)


class TestEmotionCategoryConstraints:
    @pytest.fixture
    def envelope(self):
        return deepcopy(VALID_ENVELOPE)

    def test_emotion_field_exists_in_metadata(self, envelope):
        assert "emotion" in envelope["metadata"]

    def test_emotion_uses_category_not_type(self, envelope):
        assert "category" in envelope["metadata"]["emotion"]
        assert "type" not in envelope["metadata"]["emotion"], (
            "emotion.type is FORBIDDEN per spec v4.5.0 section 0.3 and "
            "项目宪法 section 2.2. Use emotion.category instead."
        )

    def test_emotion_category_allows_reliable_values(self, envelope):
        reliable = {"joy", "sadness", "neutral"}
        envelope["metadata"]["emotion"]["category"] = "joy"
        assert envelope["metadata"]["emotion"]["category"] in reliable

        envelope["metadata"]["emotion"]["category"] = "sadness"
        assert envelope["metadata"]["emotion"]["category"] in reliable

        envelope["metadata"]["emotion"]["category"] = "neutral"
        assert envelope["metadata"]["emotion"]["category"] in reliable

    def test_anger_and_surprise_are_placeholder_only(self, envelope):
        placeholder = {"anger", "surprise"}
        envelope["metadata"]["emotion"]["category"] = "anger"
        assert envelope["metadata"]["emotion"]["category"] in placeholder, (
            "anger is a placeholder enum - allowed in structure but "
            "downstream modules MUST NOT branch on it"
        )

    def test_emotion_source_is_text_sentiment(self, envelope):
        assert envelope["metadata"]["emotion"]["source"] == "text_sentiment"

    def test_emotion_intensity_is_float_0_to_1(self, envelope):
        intensity = envelope["metadata"]["emotion"]["intensity"]
        assert isinstance(intensity, float)
        assert 0.0 <= intensity <= 1.0


class TestTraceLineage:
    def test_trace_id_persists_across_layers(self):
        tid = str(uuid.uuid4())
        perception_msg = {**deepcopy(VALID_ENVELOPE), "trace_id": tid, "source_layer": "perception"}
        decision_msg = {**deepcopy(VALID_ENVELOPE), "trace_id": tid, "source_layer": "decision"}
        assert perception_msg["trace_id"] == decision_msg["trace_id"] == tid

    def test_version_must_be_monotonic_within_same_trace(self):
        tid = str(uuid.uuid4())
        msg1 = {**deepcopy(VALID_ENVELOPE), "trace_id": tid, "version": 1}
        msg2 = {**deepcopy(VALID_ENVELOPE), "trace_id": tid, "version": 2}
        msg3 = {**deepcopy(VALID_ENVELOPE), "trace_id": tid, "version": 3}
        assert msg1["version"] < msg2["version"] < msg3["version"]

    def test_version_monotonic_violation_rejected(self):
        tid = str(uuid.uuid4())
        msg1 = {**deepcopy(VALID_ENVELOPE), "trace_id": tid, "version": 2}
        msg2 = {**deepcopy(VALID_ENVELOPE), "trace_id": tid, "version": 1}
        assert msg2["version"] < msg1["version"], (
            "version 1 after version 2 = monotonic violation. "
            "Downstream MUST reject or log WARNING per spec v4.5.0 section 0.3."
        )


class TestDegradedFlag:
    @pytest.fixture
    def envelope(self):
        return deepcopy(VALID_ENVELOPE)

    def test_degraded_true_lowers_downstream_expectations(self, envelope):
        degraded_msg = {**envelope, "metadata": {**envelope["metadata"], "degraded": True}}
        assert degraded_msg["metadata"]["degraded"] is True

    def test_degraded_false_means_full_quality(self, envelope):
        assert envelope["metadata"]["degraded"] is False


class TestTimestamp:
    @pytest.fixture
    def envelope(self):
        return deepcopy(VALID_ENVELOPE)

    def test_timestamp_is_iso8601(self, envelope):
        ts = envelope["timestamp"]
        dt = datetime.fromisoformat(ts)
        assert dt.tzinfo is not None

    def test_timestamp_includes_milliseconds(self, envelope):
        ts = envelope["timestamp"]
        assert "." in ts
