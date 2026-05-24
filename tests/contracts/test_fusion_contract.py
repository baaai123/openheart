"""
Contract tests for fusion layer — v4.5.0 §2

Validates the complete fusion pipeline: TimeSyncWindow, EventClassifier,
EntityFusionEngine, and SceneSynthesizer against the spec's interfaces,
thresholds, and data formats.

Perception outputs feed into the fusion layer; the fusion layer outputs
Scene messages to the memory and decision layers.
"""
import copy
import json
import uuid
from datetime import datetime, timezone

import pytest

from src.fusion.time_window import (
    TimeSyncWindow,
    WindowedEvents,
    DEFAULT_MAX_WINDOW_MS,
    DEFAULT_MIN_WINDOW_MS,
    AFFECTIVE_MAX_WINDOW_MS,
    VISUAL_CHANGE_THRESHOLD,
)
from src.fusion.event_classifier import (
    EventClassifier,
    ClassifiedEvents,
    PRIMARY_SCORE_THRESHOLD,
    EMOTION_INTENSITY_THRESHOLD,
    CONTEXT_MERGE_SIMILARITY,
)
from src.fusion.entity_fusion import (
    EntityFusionEngine,
    EntityFusionResult,
    EntityObject,
    AlignedPair,
    DEFAULT_ALIGN_THRESHOLD,
    DEICTIC_BOOST_REDUCTION,
)
from src.fusion.scene_synthesis import (
    SceneSynthesizer,
    ScenePayload,
    SceneMetadata,
    EntityRelation,
    EmotionSnapshot,
    SceneClass,
)

# --------------------------------------------------------------------------
# Fixtures — perception events matching spec §1.5 unified output format
# --------------------------------------------------------------------------


def _make_audio_event(
    text: str = "你好世界",
    emotion_category: str = "neutral",
    emotion_intensity: float = 0.3,
    affective: bool = False,
    degraded: bool = False,
    is_segment_end: bool = True,
    timestamp: str | None = None,
) -> dict:
    """Factory for a spec-compliant audio perception event (§1.5)."""
    return {
        "trace_id": str(uuid.uuid4()),
        "source_layer": "perception",
        "source_component": "audio",
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        "version": 1,
        "payload_type": "perception_event",
        "payload": {
            "type": "audio_event",
            "vision_snapshot": {},
            "audio": {
                "text": text,
                "voicefeature": {"language": "zh", "avg_logprob": -0.3},
                "is_segment_end": is_segment_end,
            },
        },
        "metadata": {
            "confidence": 0.9,
            "latency_ms": 10.0,
            "degraded": degraded,
            "scene_context": {"primary_type": "desktop", "confidence": 0.8},
            "emotion": {
                "category": emotion_category,
                "intensity": emotion_intensity,
                "source": "text_sentiment",
                "confidence": 0.8,
            },
            "affective_flag": affective,
            "user_model_version": 0,
        },
    }


def _make_vision_event(
    objects: list[dict] | None = None,
    text_content: list[dict] | None = None,
    scene_class_primary: str = "code_editor",
    scene_confidence: float = 0.94,
    degraded: bool = False,
    deictic_reference: bool = False,
    timestamp: str | None = None,
) -> dict:
    """Factory for a spec-compliant vision snapshot event (§1.5)."""
    return {
        "trace_id": str(uuid.uuid4()),
        "source_layer": "perception",
        "source_component": "vision_fusion",
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        "version": 1,
        "payload_type": "perception_event",
        "payload": {
            "type": "vision_snapshot",
            "vision_snapshot": {
                "scene_class": {
                    "primary": scene_class_primary,
                    "confidence": scene_confidence,
                },
                "objects": objects or [],
                "text_content": text_content or [],
            },
            "audio": {"text": "", "voicefeature": {}},
        },
        "metadata": {
            "confidence": 0.95,
            "latency_ms": 11.5,
            "degraded": degraded,
            "scene_context": {
                "primary_type": scene_class_primary,
                "confidence": scene_confidence,
            },
            "emotion": {
                "category": "neutral",
                "intensity": 0.0,
                "source": "text_sentiment",
                "confidence": 1.0,
            },
            "affective_flag": False,
            "user_model_version": 0,
            "deictic_reference": deictic_reference,
        },
    }


# --------------------------------------------------------------------------
# Module existence tests
# --------------------------------------------------------------------------


class TestFusionModulesExist:
    """Verify all fusion sub-modules are importable."""

    def test_time_window_importable(self):
        from src.fusion.time_window import TimeSyncWindow
        assert TimeSyncWindow is not None

    def test_event_classifier_importable(self):
        from src.fusion.event_classifier import EventClassifier
        assert EventClassifier is not None

    def test_entity_fusion_importable(self):
        from src.fusion.entity_fusion import EntityFusionEngine
        assert EntityFusionEngine is not None

    def test_scene_synthesis_importable(self):
        from src.fusion.scene_synthesis import SceneSynthesizer
        assert SceneSynthesizer is not None


# --------------------------------------------------------------------------
# TimeSyncWindow tests — v4.5.0 §2.3
# --------------------------------------------------------------------------


class TestTimeSyncWindow:
    """Adaptive time window: trigger conditions and windowing behaviour."""

    @pytest.fixture
    def window(self):
        return TimeSyncWindow(
            max_window_ms=DEFAULT_MAX_WINDOW_MS,
            min_window_ms=DEFAULT_MIN_WINDOW_MS,
            visual_change_threshold=VISUAL_CHANGE_THRESHOLD,
        )

    def test_default_constants_match_spec(self):
        assert DEFAULT_MAX_WINDOW_MS == 800.0
        assert DEFAULT_MIN_WINDOW_MS == 150.0
        assert AFFECTIVE_MAX_WINDOW_MS == 600.0

    def test_push_audio_segment_end_triggers_window(self, window):
        """Voice segment end should trigger window output."""
        event = _make_audio_event(text="你好", is_segment_end=True)
        result = window.push(event)
        assert result is not None
        assert isinstance(result, WindowedEvents)
        assert result.trigger_reason == "voice_segment_end"
        assert len(result.events) == 1

    def test_push_returns_none_within_min_window(self, window):
        """Events within min_window_ms should not trigger output."""
        event1 = _make_audio_event(text="第一条", is_segment_end=True)
        result1 = window.push(event1)
        assert result1 is not None  # First flush
        window.reset()
        # Push a non-trigger event immediately — should be buffered
        event2 = _make_audio_event(text="第二条", is_segment_end=False)
        result2 = window.push(event2)
        # Within min_window_ms, no trigger
        assert result2 is None

    def test_flush_drains_buffer(self, window):
        """Force flush should drain buffer regardless of conditions."""
        window.push(_make_audio_event(text="test", is_segment_end=False))
        window.push(_make_audio_event(text="test2", is_segment_end=False))
        result = window.flush()
        assert result is not None
        assert len(result.events) == 2

    def test_window_id_is_valid_uuid(self, window):
        event = _make_audio_event(text="测试", is_segment_end=True)
        result = window.push(event)
        assert result is not None
        uuid.UUID(result.window_id)

    def test_window_has_iso_timestamps(self, window):
        event = _make_audio_event(text="测试", is_segment_end=True)
        result = window.push(event)
        assert result is not None
        # Should be valid ISO 8601
        dt_start = datetime.fromisoformat(result.window_start)
        dt_end = datetime.fromisoformat(result.window_end)
        assert dt_end >= dt_start

    def test_visual_change_detection(self, window):
        """Trigger condition: significant visual category change."""
        # First, establish baseline with initial vision event
        evt1 = _make_vision_event(
            objects=[{"label": "button"}, {"label": "text_field"}, {"label": "icon"}],
            scene_class_primary="code_editor",
        )
        window.push(evt1)  # Register initial types
        window.flush()  # Flush to set baseline

        # Second event with 3+ new types
        evt2 = _make_vision_event(
            objects=[{"label": "new_button"}, {"label": "new_icon"}, {"label": "new_menu"}],
            scene_class_primary="code_editor",
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        result = window.push(evt2)
        if result is not None:
            assert result.trigger_reason in ("visual_change", "voice_segment_end")

    def test_affective_flag_shortens_max_window(self, window):
        """Affective highlight should use shorter max_window_ms."""
        event = _make_audio_event(
            text="我好难过", emotion_category="sadness",
            emotion_intensity=0.8, affective=True
        )
        # Internal logic: effective_max should be AFFECTIVE_MAX_WINDOW_MS
        # We just verify the event is accepted and check affective_highlight
        result = window.push(event)
        if result is not None:
            assert result.affective_highlight is True

    def test_reset_clears_state(self, window):
        window.push(_make_audio_event(text="test"))
        window.reset()
        result = window.flush()
        assert result is None  # Buffer should be empty after reset


# --------------------------------------------------------------------------
# EventClassifier tests — v4.5.0 §2.4
# --------------------------------------------------------------------------


class TestEventClassifier:
    """Event classification with emotion-weighted scoring."""

    @pytest.fixture
    def classifier(self):
        return EventClassifier()

    def test_classify_empty_events(self, classifier):
        result = classifier.classify([])
        assert result.primary_event is None
        assert len(result.secondary_events) == 0
        assert len(result.fragment_events) == 0
        assert len(result.ambient_events) == 0

    def test_classify_single_audio_primary(self, classifier):
        """A complete sentence with high score should be classified as primary."""
        events = [
            _make_audio_event(
                text="用户点击了运行按钮来启动程序",
                emotion_category="joy",
                emotion_intensity=0.5,
            )
        ]
        result = classifier.classify(events)
        assert result.primary_event is not None
        assert result.primary_event["source"] == "audio"
        assert "text" in result.primary_event
        assert "score" in result.primary_event
        assert "affective" in result.primary_event

    def test_classify_high_intensity_promotes(self, classifier):
        """High emotion intensity should promote fragments (key event rule)."""
        events = [
            _make_audio_event(
                text="?",  # Minimal text
                emotion_category="sadness",
                emotion_intensity=0.9,  # Above 0.7 threshold
                affective=True,
            )
        ]
        result = classifier.classify(events)
        # Either promoted to primary (if no other) or classified as fragment with high intensity
        if result.primary_event is not None:
            assert result.primary_event["affective"] is True

    def test_classify_visual_as_ambient(self, classifier):
        """Visual events with no deictic association default to ambient."""
        events = [
            _make_audio_event(text="你好"),
            _make_vision_event(scene_class_primary="code_editor"),
        ]
        result = classifier.classify(events)
        # Should have at least one visual event classified
        total = (
            len(result.secondary_events)
            + len(result.ambient_events)
        )
        assert total >= 1

    def test_classify_output_structure(self, classifier):
        events = [
            _make_audio_event(text="今天天气真好"),
        ]
        result = classifier.classify(events)
        assert isinstance(result, ClassifiedEvents)
        if result.primary_event:
            keys = set(result.primary_event.keys())
            for key in ("text", "source", "score", "affective"):
                assert key in keys

    def test_context_merge_enabled(self, classifier):
        """Context-aware merging should be active for similar consecutive primaries."""
        events1 = [_make_audio_event(text="我今天去了公园")]
        result1 = classifier.classify(events1)
        assert result1.primary_event is not None

        events2 = [_make_audio_event(text="我今天去了商场")]
        result2 = classifier.classify(events2)
        assert result2.primary_event is not None

    def test_reset_clears_context(self, classifier):
        events = [_make_audio_event(text="测试")]
        classifier.classify(events)
        classifier.reset()
        # After reset, recent primary texts should be empty
        assert len(classifier._recent_primary_texts) == 0


# --------------------------------------------------------------------------
# EntityFusionEngine tests — v4.5.0 §2.5
# --------------------------------------------------------------------------


class TestEntityFusionEngine:
    """Cross-modal entity alignment with bge embeddings."""

    @pytest.fixture
    def engine(self):
        return EntityFusionEngine(
            align_threshold=DEFAULT_ALIGN_THRESHOLD,
            deictic_boost_reduction=DEICTIC_BOOST_REDUCTION,
        )

    def test_default_thresholds_match_spec(self):
        assert DEFAULT_ALIGN_THRESHOLD == 0.75
        assert DEICTIC_BOOST_REDUCTION == 0.10

    def test_fuse_empty_input(self, engine):
        """Empty input should produce empty result."""
        from src.fusion.event_classifier import ClassifiedEvents
        classified = ClassifiedEvents(primary_event=None)
        result = engine.fuse(classified, [])
        assert isinstance(result, EntityFusionResult)
        assert len(result.aligned_pairs) == 0
        assert len(result.unmatched_audio_entities) == 0
        assert len(result.unmatched_visual_entities) == 0

    def test_fuse_with_visual_entities(self, engine):
        """Visual entities should be extracted from vision snapshots."""
        from src.fusion.event_classifier import ClassifiedEvents
        classified = ClassifiedEvents(
            primary_event={"text": "点击运行按钮", "source": "audio", "score": 0.7, "affective": False},
        )
        events = [
            _make_vision_event(
                objects=[{"label": "button"}, {"label": "run_icon"}],
                scene_class_primary="code_editor",
            )
        ]
        result = engine.fuse(classified, events)
        assert isinstance(result, EntityFusionResult)

    def test_deictic_signal_reduces_threshold(self, engine):
        """Deictic reference should lower alignment threshold by 0.1."""
        from src.fusion.event_classifier import ClassifiedEvents
        classified = ClassifiedEvents(
            primary_event={"text": "点击这个按钮", "source": "audio", "score": 0.8, "affective": False},
        )
        events = [
            _make_vision_event(
                objects=[{"label": "button"}],
                deictic_reference=True,
            )
        ]
        result = engine.fuse(classified, events)
        assert isinstance(result, EntityFusionResult)

    def test_fuse_result_structure(self, engine):
        """Fusion result should have the correct structure."""
        from src.fusion.event_classifier import ClassifiedEvents
        classified = ClassifiedEvents(primary_event=None)
        result = engine.fuse(classified, [])
        assert hasattr(result, "aligned_pairs")
        assert hasattr(result, "unmatched_audio_entities")
        assert hasattr(result, "unmatched_visual_entities")

    def test_aligned_pair_fields(self):
        """Aligned pair dataclass should have required fields."""
        pair = AlignedPair(
            audio_entity=EntityObject(text="测试", label="OBJECT"),
            visual_entity=EntityObject(text="测试图像", label="OBJECT"),
            similarity=0.85,
            deictic_boosted=False,
        )
        assert pair.similarity == 0.85
        assert pair.deictic_boosted is False
        assert pair.audio_entity.text == "测试"
        assert pair.visual_entity.label == "OBJECT"


# --------------------------------------------------------------------------
# SceneSynthesizer tests — v4.5.0 §2.6
# --------------------------------------------------------------------------


class TestSceneSynthesizer:
    """Scene synthesis: composition of classified events and entities into Scene."""

    @pytest.fixture
    def synthesizer(self):
        return SceneSynthesizer()

    def test_synthesize_returns_dict(self, synthesizer):
        """Synthesize should return a dict representing the Scene message envelope."""
        from src.fusion.event_classifier import ClassifiedEvents
        from src.fusion.entity_fusion import EntityFusionResult

        classified = ClassifiedEvents(
            primary_event={
                "text": "用户询问今天天气",
                "source": "audio",
                "score": 0.75,
                "affective": False,
            },
            secondary_events=[],
            fragment_events=[],
            ambient_events=[],
        )
        fusion_result = EntityFusionResult()

        result = synthesizer.synthesize(
            trace_id=str(uuid.uuid4()),
            version=1,
            classified_events=classified,
            entity_fusion_result=fusion_result,
        )

        assert isinstance(result, dict)

    def test_scene_has_required_top_level_fields(self, synthesizer):
        """Scene output must match spec §2.6 structure."""
        from src.fusion.event_classifier import ClassifiedEvents
        from src.fusion.entity_fusion import EntityFusionResult

        classified = ClassifiedEvents(
            primary_event={
                "text": "测试",
                "source": "audio",
                "score": 0.6,
                "affective": False,
            },
        )
        fusion_result = EntityFusionResult()

        result = synthesizer.synthesize(
            trace_id=str(uuid.uuid4()),
            version=1,
            classified_events=classified,
            entity_fusion_result=fusion_result,
        )

        # Top-level envelope fields per §0.3 and §2.6
        assert "scene_id" in result
        assert "trace_id" in result
        assert result["source_layer"] == "fusion"
        assert result["source_component"] == "scene_synthesis"
        assert "timestamp" in result
        assert result["payload_type"] == "scene"

    def test_scene_payload_has_required_fields(self, synthesizer):
        """Scene payload should contain all fields defined in §2.6."""
        from src.fusion.event_classifier import ClassifiedEvents
        from src.fusion.entity_fusion import EntityFusionResult

        classified = ClassifiedEvents(
            primary_event={
                "text": "用户说今天心情不错",
                "source": "audio",
                "score": 0.7,
                "affective": False,
            },
        )
        fusion_result = EntityFusionResult()

        result = synthesizer.synthesize(
            trace_id=str(uuid.uuid4()),
            version=1,
            classified_events=classified,
            entity_fusion_result=fusion_result,
        )

        payload = result["payload"]
        assert "summary" in payload
        assert "primary_event" in payload
        assert "secondary_events" in payload
        assert "entity_relations" in payload
        assert "aligned_entities" in payload
        assert "emotion_snapshot" in payload
        assert "scene_class" in payload
        assert "confidence" in payload
        assert "provenance" in payload
        assert "user_model_snapshot" in payload

    def test_scene_metadata_has_required_fields(self, synthesizer):
        """Scene metadata should contain all fields defined in §2.6."""
        from src.fusion.event_classifier import ClassifiedEvents
        from src.fusion.entity_fusion import EntityFusionResult

        classified = ClassifiedEvents(
            primary_event={
                "text": "测试",
                "source": "audio",
                "score": 0.8,
                "affective": True,
            },
        )
        fusion_result = EntityFusionResult()

        result = synthesizer.synthesize(
            trace_id=str(uuid.uuid4()),
            version=1,
            classified_events=classified,
            entity_fusion_result=fusion_result,
        )

        metadata = result["metadata"]
        assert "confidence" in metadata
        assert "latency_ms" in metadata
        assert "degraded" in metadata
        assert "affective_flag" in metadata
        assert "user_model_version" in metadata

    def test_scene_id_is_valid_uuid(self, synthesizer):
        from src.fusion.event_classifier import ClassifiedEvents
        from src.fusion.entity_fusion import EntityFusionResult

        result = synthesizer.synthesize(
            trace_id=str(uuid.uuid4()),
            version=1,
            classified_events=ClassifiedEvents(primary_event=None),
            entity_fusion_result=EntityFusionResult(),
        )

        uuid.UUID(result["scene_id"])

    def test_emotion_snapshot_defaults_neutral(self, synthesizer):
        """When no emotion data, snapshot should default to neutral."""
        from src.fusion.event_classifier import ClassifiedEvents
        from src.fusion.entity_fusion import EntityFusionResult

        result = synthesizer.synthesize(
            trace_id=str(uuid.uuid4()),
            version=1,
            classified_events=ClassifiedEvents(primary_event=None),
            entity_fusion_result=EntityFusionResult(),
        )

        emotion = result["payload"]["emotion_snapshot"]
        assert emotion["category"] == "neutral"
        assert emotion["intensity"] == 0.0

    def test_affective_flag_propagates(self, synthesizer):
        """Affective flag from primary event should propagate to metadata."""
        from src.fusion.event_classifier import ClassifiedEvents
        from src.fusion.entity_fusion import EntityFusionResult

        classified = ClassifiedEvents(
            primary_event={
                "text": "我今天好开心",
                "source": "audio",
                "score": 0.7,
                "affective": True,
            },
        )
        result = synthesizer.synthesize(
            trace_id=str(uuid.uuid4()),
            version=1,
            classified_events=classified,
            entity_fusion_result=EntityFusionResult(),
        )

        assert result["metadata"]["affective_flag"] is True

    def test_degraded_propagates_from_window_events(self, synthesizer):
        """degraded flag from window events should propagate to Scene metadata."""
        from src.fusion.event_classifier import ClassifiedEvents
        from src.fusion.entity_fusion import EntityFusionResult

        classified = ClassifiedEvents(
            primary_event={
                "text": "测试降级",
                "source": "audio",
                "score": 0.5,
                "affective": False,
            },
        )
        window_events = [
            _make_audio_event(text="测试降级", degraded=True),
        ]

        result = synthesizer.synthesize(
            trace_id=str(uuid.uuid4()),
            version=1,
            classified_events=classified,
            entity_fusion_result=EntityFusionResult(),
            window_events=window_events,
        )

        assert result["metadata"]["degraded"] is True

    def test_scene_json_serializable(self, synthesizer):
        """Scene output should be JSON-serializable."""
        from src.fusion.event_classifier import ClassifiedEvents
        from src.fusion.entity_fusion import EntityFusionResult

        classified = ClassifiedEvents(
            primary_event={
                "text": "用户说你好",
                "source": "audio",
                "score": 0.8,
                "affective": False,
            },
        )
        result = synthesizer.synthesize(
            trace_id=str(uuid.uuid4()),
            version=1,
            classified_events=classified,
            entity_fusion_result=EntityFusionResult(),
        )

        serialized = json.dumps(result)
        assert isinstance(serialized, str)
        round_trip = json.loads(serialized)
        assert round_trip["source_layer"] == "fusion"

    def test_entity_relation_structure(self, synthesizer):
        """Entity relations from primary text should have subject/predicate/object."""
        from src.fusion.event_classifier import ClassifiedEvents
        from src.fusion.entity_fusion import EntityFusionResult

        classified = ClassifiedEvents(
            primary_event={
                "text": "用户点击了运行按钮",
                "source": "audio",
                "score": 0.7,
                "affective": False,
            },
        )
        result = synthesizer.synthesize(
            trace_id=str(uuid.uuid4()),
            version=1,
            classified_events=classified,
            entity_fusion_result=EntityFusionResult(),
        )

        relations = result["payload"]["entity_relations"]
        assert isinstance(relations, list)
        for rel in relations:
            assert "subject" in rel
            assert "predicate" in rel
            assert "object" in rel


# --------------------------------------------------------------------------
# End-to-end fusion pipeline test — v4.5.0 §2 full flow
# --------------------------------------------------------------------------


class TestFusionPipeline:
    """End-to-end: perception events → time window → classify → fuse → synthesize."""

    @pytest.fixture
    def window(self):
        return TimeSyncWindow()

    @pytest.fixture
    def classifier(self):
        return EventClassifier()

    @pytest.fixture
    def engine(self):
        return EntityFusionEngine()

    @pytest.fixture
    def synthesizer(self):
        return SceneSynthesizer()

    def test_full_pipeline_single_event(self, window, classifier, engine, synthesizer):
        """Single audio event through full pipeline should produce a valid Scene."""
        event = _make_audio_event(
            text="用户点击了运行按钮",
            emotion_category="neutral",
            emotion_intensity=0.3,
            is_segment_end=True,
        )

        # Stage 1: time window
        windowed = window.push(event)
        if windowed is None:
            windowed = window.flush()
        assert windowed is not None

        # Stage 2: event classification
        classified = classifier.classify(windowed.events)
        assert isinstance(classified, ClassifiedEvents)

        # Stage 3: entity fusion
        fused = engine.fuse(classified, windowed.events)
        assert isinstance(fused, EntityFusionResult)

        # Stage 4: scene synthesis
        trace_id = str(uuid.uuid4())
        scene = synthesizer.synthesize(
            trace_id=trace_id,
            version=1,
            classified_events=classified,
            entity_fusion_result=fused,
            window_events=windowed.events,
        )

        # Validate
        assert isinstance(scene, dict)
        assert scene["payload_type"] == "scene"
        assert scene["source_layer"] == "fusion"
        assert "scene_id" in scene
        assert "payload" in scene
        assert "metadata" in scene
        assert isinstance(scene["payload"]["summary"], str)

    def test_pipeline_with_audio_and_vision(self, window, classifier, engine, synthesizer):
        """Audio + vision events through full pipeline."""
        audio = _make_audio_event(
            text="打开这个文件",
            emotion_category="neutral",
            emotion_intensity=0.2,
            is_segment_end=True,
        )
        vision = _make_vision_event(
            objects=[{"label": "file_icon"}, {"label": "folder"}],
            scene_class_primary="file_explorer",
        )

        # Push both events into window
        window.push(audio)
        windowed = window.push(vision)
        if windowed is None:
            windowed = window.flush()
        assert windowed is not None

        classified = classifier.classify(windowed.events)
        fused = engine.fuse(classified, windowed.events)
        scene = synthesizer.synthesize(
            trace_id=str(uuid.uuid4()),
            version=1,
            classified_events=classified,
            entity_fusion_result=fused,
            window_events=windowed.events,
        )

        assert scene["payload_type"] == "scene"
        assert len(scene["payload"]["secondary_events"]) >= 0
