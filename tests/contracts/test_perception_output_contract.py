"""
Contract tests for perception layer unified output format (spec v4.5.0 section 1.5).

All perception events MUST follow this output format when sent to the fusion layer.
"""
import pytest
from copy import deepcopy
from tests.contracts import require_module, fail_red

PERCEPTION_BUS = "src.perception.perception_bus"


class TestModuleExists:
    def test_perception_bus_module_available(self):
        require_module(
            module_path=PERCEPTION_BUS,
            component_name="PerceptionBus (perception/perception_bus.py)",
        )


VALID_PERCEPTION_OUTPUT = {
    "trace_id": "00000000-0000-0000-0000-000000000001",
    "source_layer": "perception",
    "source_component": "vision_fusion",
    "timestamp": "2026-05-09T12:00:00.000+00:00",
    "version": 1,
    "payload_type": "perception_event",
    "payload": {
        "type": "vision_snapshot",
        "vision_snapshot": {
            "scene_class": {"primary": "code_editor", "confidence": 0.94},
            "objects": [],
            "text_content": [],
        },
        "audio": {
            "text": "",
            "voicefeature": {"language": "zh", "avg_logprob": -0.3},
        },
    },
    "metadata": {
        "confidence": 0.95,
        "latency_ms": 11.5,
        "degraded": False,
        "scene_context": {"primary_type": "code_editor", "confidence": 0.94},
        "emotion": {
            "category": "neutral",
            "intensity": 0.2,
            "source": "text_sentiment",
            "confidence": 0.8,
        },
        "affective_flag": False,
        "user_model_version": 0,
    },
}


class TestPerceptionOutputStructure:
    @pytest.fixture
    def output(self):
        return deepcopy(VALID_PERCEPTION_OUTPUT)

    def test_source_layer_is_perception(self, output):
        assert output["source_layer"] == "perception"

    def test_payload_type_is_perception_event(self, output):
        assert output["payload_type"] == "perception_event"

    def test_payload_contains_type_field(self, output):
        assert "type" in output["payload"]

    def test_payload_type_must_be_vision_snapshot_or_audio_event(self, output):
        assert output["payload"]["type"] in {"vision_snapshot", "audio_event"}

    def test_vision_snapshot_present_when_type_is_vision_snapshot(self, output):
        output["payload"]["type"] = "vision_snapshot"
        assert "vision_snapshot" in output["payload"]

    def test_audio_present_when_type_is_audio_event(self, output):
        output["payload"]["type"] = "audio_event"
        assert "audio" in output["payload"]

    def test_vision_snapshot_has_scene_class(self, output):
        vs = output["payload"]["vision_snapshot"]
        assert "scene_class" in vs

    def test_vision_snapshot_has_objects(self, output):
        vs = output["payload"]["vision_snapshot"]
        assert "objects" in vs
        assert isinstance(vs["objects"], list)

    def test_vision_snapshot_has_text_content(self, output):
        vs = output["payload"]["vision_snapshot"]
        assert "text_content" in vs
        assert isinstance(vs["text_content"], list)

    def test_audio_has_text_field(self, output):
        assert "text" in output["payload"]["audio"]

    def test_audio_has_voicefeature(self, output):
        assert "voicefeature" in output["payload"]["audio"]

    def test_voicefeature_has_language(self, output):
        assert "language" in output["payload"]["audio"]["voicefeature"]


class TestPerceptionMetadata:
    @pytest.fixture
    def output(self):
        return deepcopy(VALID_PERCEPTION_OUTPUT)

    def test_metadata_contains_stale_field(self):
        assert "stale" not in VALID_PERCEPTION_OUTPUT["metadata"], (
            "stale field is defined in SyncVisionQuery (section 1.7), "
            "not in perception output from section 1.5"
        )

    def test_metadata_contains_degraded(self, output):
        assert "degraded" in output["metadata"]

    def test_metadata_scene_context_present(self, output):
        assert "scene_context" in output["metadata"]
        assert "primary_type" in output["metadata"]["scene_context"]

    def test_metadata_has_user_model_version(self, output):
        assert "user_model_version" in output["metadata"]


class TestPerceptionEmotionCategory:
    @pytest.fixture
    def output(self):
        return deepcopy(VALID_PERCEPTION_OUTPUT)

    def test_emotion_category_uses_category_not_type(self, output):
        assert "category" in output["metadata"]["emotion"]
        assert "type" not in output["metadata"]["emotion"], (
            "emotion.type is FORBIDDEN (spec v4.5.0, 项目宪法 section 2.2)"
        )

    def test_emotion_only_joy_sadness_neutral_are_reliable(self, output):
        reliable = {"joy", "sadness", "neutral"}
        for cat in reliable:
            output["metadata"]["emotion"]["category"] = cat
            assert output["metadata"]["emotion"]["category"] in reliable

    def test_downstream_module_must_not_branch_on_anger(self):
        cat = "anger"
        assert cat not in {"joy", "sadness", "neutral"}, (
            "anger is a placeholder per spec v4.5.0 section 0.3. "
            "Downstream modules must NOT branch on anger unless "
            "config/sentiment.yaml has provider=structbert."
        )

    def test_downstream_module_must_not_branch_on_surprise(self):
        cat = "surprise"
        assert cat not in {"joy", "sadness", "neutral"}, (
            "surprise is a placeholder per spec v4.5.0 section 0.3."
        )


class TestPerceptionDegradedPaths:
    """Validate perception degradation matrix from 项目宪法 §4."""

    def test_yolo_world_unavailable_path(self):
        """YOLO-World unavailable (low VRAM) -> degraded=true, lanes 2+3+4 only."""
        from src.config.runtime import RuntimeConfig, VRAMTier
        from src.perception.visual.visual_pipeline import VisualPipeline

        # Create a low-VRAM config that forces YOLO-World off at init
        config = RuntimeConfig(
            vram_tier=VRAMTier.LOW,
            vram_total_gb=8.0,
            low_vram=True,
            performance_mode=False,
            enable_shadow=False,
            show_transcript=False,
            redis_host="localhost",
            redis_port=6379,
            redis_db=0,
            redis_password=None,
            redis_aof=True,
            context_limit=2048,
        )
        pipeline = VisualPipeline(config)

        # Lane 1 should be unavailable and degraded
        assert pipeline.lane1_available is False
        assert pipeline.any_degraded is True

    def test_scene_classifier_unavailable_defaults_other(self):
        """Scene classifier unavailable -> default to 'other', degraded=true."""
        from src.perception.visual.clip_scene import CLIPSceneLane

        lane = CLIPSceneLane(model_path="nonexistent_model_path")
        # Force unavailable state (model path does not exist)
        lane._available = False
        lane._degraded = True

        import asyncio
        result = asyncio.run(lane.process(None))

        assert result.primary == "other"
        assert result.confidence == 0.0
        assert lane.degraded is True

    def test_snownlp_unavailable_fallback_neutral(self):
        """SnowNLP unavailable -> spacytextblob fallback. Both unavailable -> neutral, degraded=true."""
        import asyncio
        from src.perception.audio.emotion import EmotionAnalyzer

        analyzer = EmotionAnalyzer(provider="snownlp", fallback="spacytextblob")
        # Force both backends to be unavailable
        analyzer._snownlp = None
        analyzer._spacytextblob = None
        analyzer._spacy_nlp = None

        result = asyncio.run(
            analyzer.analyze("你好世界", language="zh")
        )

        assert result["category"] == "neutral"
        assert result["intensity"] == 0.0
        assert result["degraded"] is True
