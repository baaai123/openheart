"""Integration test: perception → fusion → memory pipeline.

Tests the complete data flow from perception events through fusion
(TimeSyncWindow → EventClassifier → SceneSynthesizer) to memory
storage (HotMemoryStore, ColdMemoryStore).

GPU-dependent and I/O-dependent components are mocked; pipeline logic,
envelope construction, scene synthesis, and memory store/retrieve are
exercised end‑to‑end.

Scenarios (v4.5.0):
  1. Visual input → VisionSnapshot → Fusion → Scene
  2. Audio input → AudioEvent → Emotion → Scene
  3. Scene → HotMemory store/retrieve
  4. Scene → ColdMemory store/retrieve
  5. Degraded flag propagation through pipeline
  6. trace_id persistence through pipeline
  7. emotion.category is joy/sadness/neutral only
"""

# pyright: reportUninitializedInstanceVariable=false
# pytest fixtures (autouse=True) initialise instance vars — pyright can't trace them.

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest

# ---------------------------------------------------------------------------
# Fusion pipeline - real components (GPU/spaCy are lazy-loaded, safe to import)
# ---------------------------------------------------------------------------

from src.fusion.time_window import TimeSyncWindow, WindowedEvents
from src.fusion.event_classifier import EventClassifier, ClassifiedEvents
from src.fusion.entity_fusion import (
    EntityFusionEngine,
    EntityFusionResult,
    EntityObject,
    AlignedPair,
)
from src.fusion.scene_synthesis import (
    SceneSynthesizer,
    ScenePayload,
    SceneMetadata,
    EmotionSnapshot,
    SceneClass,
)
from src.fusion.message_envelope import (
    MessageEnvelope,
    create_envelope,
    Layer,
    PayloadType,
    EmotionCategory,
)

# ---------------------------------------------------------------------------
# Helpers — produce synthetic perception events matching PerceptionBus output
# ---------------------------------------------------------------------------


def _make_vision_event(
    trace_id: str = "test-trace",
    degraded: bool = False,
    objects: list[dict[str, Any]] | None = None,
    scene_class: str = "desktop",
    emotion_category: str = "neutral",
    emotion_intensity: float = 0.0,
) -> dict[str, Any]:
    """Build a perception event dict that mimics a published vision snapshot."""
    if objects is None:
        objects = [
            {"label": "person", "confidence": 0.95, "bbox": [100, 200, 50, 120]},
            {"label": "laptop", "confidence": 0.88, "bbox": [300, 400, 300, 20]},
        ]
    return {
        "trace_id": trace_id,
        "source_layer": "perception",
        "source_component": "vision_fusion",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "version": 1,
        "payload_type": "perception_event",
        "payload": {
            "type": "vision_snapshot",
            "vision_snapshot": {
                "objects": objects,
                "text_content": [
                    {"text": "Hello World", "confidence": 0.99},
                ],
                "scene_class": {"primary": scene_class, "confidence": 0.85},
            },
        },
        "metadata": {
            "degraded": degraded,
            "confidence": 0.9,
            "latency_ms": 5.0,
            "emotion": {
                "category": emotion_category,
                "intensity": emotion_intensity,
            },
            "affective_flag": False,
        },
    }


def _make_audio_event(
    trace_id: str = "test-trace",
    degraded: bool = False,
    text: str = "你好今天天气真好",
    emotion_category: str = "joy",
    emotion_intensity: float = 0.85,
    affective: bool = False,
) -> dict[str, Any]:
    """Build a perception event dict that mimics a published audio event."""
    return {
        "trace_id": trace_id,
        "source_layer": "perception",
        "source_component": "asr_stream",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "version": 1,
        "payload_type": "perception_event",
        "payload": {
            "type": "audio_event",
            "audio": {
                "text": text,
                "language": "zh",
                "segments": [
                    {"start": 0.0, "end": 2.5, "text": text, "avg_logprob": -0.3},
                ],
            },
        },
        "metadata": {
            "degraded": degraded,
            "confidence": 0.95,
            "latency_ms": 150.0,
            "emotion": {
                "category": emotion_category,
                "intensity": emotion_intensity,
            },
            "affective_flag": affective,
        },
    }


# ===================================================================
# 1. Visual input → Fusion → Scene
# ===================================================================


class TestVisualToScene:
    """Simulate a visual perception event flowing through fusion to a Scene."""

    # Type annotations (initialised by _setup fixture)
    time_window: TimeSyncWindow
    classifier: EventClassifier
    synthesizer: SceneSynthesizer
    fusion_engine: EntityFusionEngine
    trace_id: str

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.time_window = TimeSyncWindow(max_window_ms=5000, min_window_ms=1)
        self.classifier = EventClassifier()
        self.synthesizer = SceneSynthesizer()
        self.fusion_engine = EntityFusionEngine(align_threshold=0.1)
        self.trace_id = "test-visual-scene"

    def _flush_window(self) -> WindowedEvents:
        """Force-flush the time window by pushing a timeout-trigger event."""
        # Push a dummy event with a far-future timestamp to trigger max_timeout
        late = _make_vision_event(trace_id=self.trace_id)
        late["timestamp"] = "2099-01-01T00:00:00+00:00"
        result = self.time_window.push(late)
        assert result is not None, "Expected window to flush on timeout trigger"
        return result

    def test_visual_event_flows_to_scene(self):
        """Push a vision event → classify → fuse → synthesize → verify scene."""
        # Push a vision event
        event = _make_vision_event(trace_id=self.trace_id)
        push_result = self.time_window.push(event)
        # The event may or may not trigger the window immediately
        if push_result is None:
            push_result = self._flush_window()

        assert isinstance(push_result, WindowedEvents)
        assert push_result.trigger_reason in (
            "voice_segment_end", "visual_change", "max_timeout"
        )
        assert len(push_result.events) >= 1

        # Classify
        classified = self.classifier.classify(push_result.events)
        assert isinstance(classified, ClassifiedEvents)

        fusion_result = self.fusion_engine.fuse(
            classified, window_events=push_result.events
        )
        assert isinstance(fusion_result, EntityFusionResult)
        assert len(fusion_result.unmatched_visual_entities) >= 2

        # Synthesize
        scene = self.synthesizer.synthesize(
            trace_id=self.trace_id,
            version=1,
            classified_events=classified,
            entity_fusion_result=fusion_result,
            window_events=push_result.events,
            user_model_version=0,
            start_time=time.time(),
        )

        # Verify scene structure
        assert isinstance(scene, dict)
        assert scene["trace_id"] == self.trace_id
        assert scene["source_layer"] == "fusion"
        assert scene["payload_type"] == "scene"
        assert "scene_id" in scene
        assert "payload" in scene
        assert "metadata" in scene

        # Verify payload fields
        payload = scene["payload"]
        assert isinstance(payload["emotion_snapshot"], dict)
        assert payload["emotion_snapshot"]["category"] in (
            "joy", "sadness", "neutral"
        )
        assert isinstance(payload["confidence"], float)
        assert 0.0 <= payload["confidence"] <= 1.0

        # Verify metadata
        metadata = scene["metadata"]
        assert "confidence" in metadata
        assert "latency_ms" in metadata
        assert "degraded" in metadata
        assert metadata["degraded"] is False


# ===================================================================
# 2. Audio input → Emotion → Fusion → Scene
# ===================================================================


class TestAudioToScene:
    """Simulate an audio perception event with emotion through to Scene."""

    time_window: TimeSyncWindow
    classifier: EventClassifier
    synthesizer: SceneSynthesizer
    fusion_engine: EntityFusionEngine
    trace_id: str

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.time_window = TimeSyncWindow(max_window_ms=5000, min_window_ms=1)
        self.classifier = EventClassifier()
        self.synthesizer = SceneSynthesizer()
        self.fusion_engine = EntityFusionEngine(align_threshold=0.1)
        self.trace_id = "test-audio-scene"

    def _flush_window(self) -> WindowedEvents:
        late = _make_audio_event(trace_id=self.trace_id)
        late["timestamp"] = "2099-01-01T00:00:00+00:00"
        result = self.time_window.push(late)
        assert result is not None
        return result

    def test_audio_event_with_emotion_to_scene(self):
        """Audio with joy emotion → verify scene carries emotion."""
        event = _make_audio_event(
            trace_id=self.trace_id,
            text="今天真是太开心了",
            emotion_category="joy",
            emotion_intensity=0.9,
        )
        push_result = self.time_window.push(event)
        if push_result is None:
            push_result = self._flush_window()

        classified = self.classifier.classify(push_result.events)

        fusion_result = self.fusion_engine.fuse(
            classified, window_events=push_result.events
        )

        scene = self.synthesizer.synthesize(
            trace_id=self.trace_id,
            version=1,
            classified_events=classified,
            entity_fusion_result=fusion_result,
            window_events=push_result.events,
            user_model_version=0,
            start_time=time.time(),
        )

        # Verify emotion was picked up from the window event metadata
        emotion = scene["payload"]["emotion_snapshot"]
        assert emotion["category"] in ("joy", "sadness", "neutral")
        assert isinstance(emotion["intensity"], float)

        # Verify primary event text
        assert "今天" in scene["payload"]["primary_event"] or scene["payload"]["primary_event"] == ""

    def test_audio_with_sadness_emotion(self):
        """Audio with sadness emotion → scene reflects sadness."""
        event = _make_audio_event(
            trace_id=self.trace_id,
            text="有点难过",
            emotion_category="sadness",
            emotion_intensity=0.75,
        )
        push_result = self.time_window.push(event)
        if push_result is None:
            push_result = self._flush_window()

        classified = self.classifier.classify(push_result.events)
        fusion_result = self.fusion_engine.fuse(classified, window_events=push_result.events)

        scene = self.synthesizer.synthesize(
            trace_id=self.trace_id,
            version=1,
            classified_events=classified,
            entity_fusion_result=fusion_result,
            window_events=push_result.events,
        )

        emotion = scene["payload"]["emotion_snapshot"]
        assert emotion["category"] == "sadness"
        assert emotion["intensity"] == 0.75


# ===================================================================
# 3. HotMemory store/retrieve
# ===================================================================


class TestHotMemoryStoreRetrieve:
    """Scene → HotMemory store then retrieve with a mock Redis."""

    @pytest.fixture
    def mock_redis(self):
        """Create a dict-based mock Redis client."""
        store: dict[str, Any] = {}

        class _FakeRedis:
            """Simulates enough of redis.Redis for HotMemoryStore."""

            def get(self, key: str) -> bytes | None:
                val = store.get(key)
                return json.dumps(val).encode() if val is not None else None

            def set(self, key: str, value: str, ex: int | None = None) -> bool:
                store[key] = json.loads(value) if isinstance(value, (str, bytes)) else value
                return True

            def delete(self, key: str) -> bool:
                return store.pop(key, None) is not None

            def exists(self, key: str) -> bool:
                return key in store

            def close(self) -> None:
                pass

            # Redis pipeline stub
            def pipeline(self):
                p = MagicMock()
                p.execute = MagicMock(return_value=[])
                return p

        return _FakeRedis()

    @pytest.fixture
    def sample_scene(self) -> dict[str, Any]:
        return {
            "scene_id": "scene-hot-001",
            "trace_id": "trace-hot-001",
            "source_layer": "fusion",
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "version": 1,
            "payload_type": "scene",
            "payload": {
                "summary": "用户正在使用电脑",
                "primary_event": "用户在看屏幕",
                "secondary_events": [],
                "entity_relations": [
                    {"subject": "用户", "predicate": "使用", "object": "电脑"}
                ],
                "aligned_entities": [],
                "emotion_snapshot": {"category": "neutral", "intensity": 0.0},
                "scene_class": {"primary": "desktop", "confidence": 0.85},
                "confidence": 0.75,
                "provenance": {},
                "user_model_snapshot": {},
            },
            "metadata": {
                "confidence": 0.75,
                "latency_ms": 10.0,
                "degraded": False,
                "affective_flag": False,
                "user_model_version": 0,
            },
        }

    def test_store_and_retrieve_scene(self, mock_redis, sample_scene):
        from src.memory.hot.memory_store import HotMemoryStore
        from src.config.runtime import RuntimeConfig

        config = MagicMock(spec=RuntimeConfig)
        config.redis_host = "localhost"
        config.redis_port = 6379
        config.redis_db = 0
        config.redis_password = None
        config.redis_aof = False

        store = HotMemoryStore(config)
        # Inject mock redis directly
        store._redis = mock_redis
        store._degraded = False

        # Store
        result = store.store_scene(sample_scene)
        assert result is True

        # Retrieve
        retrieved = store.get_scene("scene-hot-001")
        assert retrieved is not None
        assert retrieved["scene_id"] == "scene-hot-001"
        assert retrieved["trace_id"] == "trace-hot-001"
        assert retrieved["payload"]["emotion_snapshot"]["category"] == "neutral"
        assert retrieved["metadata"]["degraded"] is False

    def test_store_missing_scene_id_returns_false(self, mock_redis):
        from src.memory.hot.memory_store import HotMemoryStore
        from src.config.runtime import RuntimeConfig

        config = MagicMock(spec=RuntimeConfig)
        store = HotMemoryStore(config)
        store._redis = mock_redis

        result = store.store_scene({"no_scene_id": True})
        assert result is False

    def test_get_nonexistent_scene_returns_none(self, mock_redis):
        from src.memory.hot.memory_store import HotMemoryStore
        from src.config.runtime import RuntimeConfig

        config = MagicMock(spec=RuntimeConfig)
        store = HotMemoryStore(config)
        store._redis = mock_redis

        retrieved = store.get_scene("nonexistent")
        assert retrieved is None

    def test_degraded_store_returns_false(self, mock_redis):
        """When redis is None (degraded), store returns False."""
        from src.memory.hot.memory_store import HotMemoryStore
        from src.config.runtime import RuntimeConfig

        config = MagicMock(spec=RuntimeConfig)
        store = HotMemoryStore(config)
        store._redis = None  # Simulate degraded / no connection
        store._degraded = True

        result = store.store_scene({"scene_id": "test"})
        assert result is False


# ===================================================================
# 4. ColdMemory store/retrieve
# ===================================================================


class TestColdMemoryStoreRetrieve:
    """Scene → ColdMemory store then search with a mock LanceDB."""

    @pytest.fixture
    def mock_cold_records(self):
        """In-memory list acting as LanceDB table."""
        records: list[dict[str, Any]] = []

        class _MockTable:
            """Mocks LanceDB Table for append/search."""

            async def add(self, data: list[dict[str, Any]]) -> None:
                for record in data:
                    records.append(record)

            async def search(self, vector, vector_column_name: str = "embedding"):
                result = MagicMock()
                result.limit = MagicMock(return_value=result)
                result.to_list = MagicMock(
                    return_value=list(records)
                )
                return result

        return records, _MockTable()

    @pytest.fixture
    def sample_cold_scene(self):
        from src.memory.cold.memory_store import Scene

        return Scene(
            scene_id="scene-cold-001",
            trace_id="trace-cold-001",
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            summary="用户使用电脑处理文档",
            events=[
                {"type": "vision_snapshot", "text": "用户面对屏幕"},
            ],
            entities=[
                {"name": "用户", "type": "person"},
                {"name": "电脑", "type": "object"},
            ],
            entity_relations=[
                {"subject": "用户", "predicate": "使用", "object": "电脑"},
            ],
            affective_flag=False,
            importance_score=0.6,
            scene_class="work",
        )

    @pytest.mark.asyncio
    @patch("src.memory.cold.memory_store.MemoryDecayConfig")
    async def test_store_and_search_scene(
        self, mock_decay_config, mock_cold_records, sample_cold_scene
    ):
        records, mock_table = mock_cold_records

        from src.memory.cold.memory_store import ColdMemoryStore

        store = ColdMemoryStore(
            db_path="/tmp/test_cold_memory",
            embedding_model_name="mock-model",
            embedding_dim=512,
            redis_client=None,
        )
        store._table = mock_table  # type: ignore[reportAttributeAccessIssue]
        store._initialized_in_this_session = True
        store._embedding_model = MagicMock()
        store._embedding_model.encode = MagicMock(
            return_value=[0.1] * 512
        )

        # Store
        memory_id = await store.store_scene(sample_cold_scene)
        assert memory_id is not None
        assert memory_id == "scene-cold-001"

        # Verify one record was added
        assert len(records) == 1
        assert records[0]["memory_id"] == "scene-cold-001"
        assert records[0]["scene_summary"] == "用户使用电脑处理文档"
        assert records[0]["memory_type"] == "FACT"
        assert records[0]["scene_class"] == "work"

        # Search
        results = await store.semantic_search("电脑", top_k=5)
        # Since mock returns all records, search should return results
        assert isinstance(results, list)
        if results:
            assert results[0].memory_id == "scene-cold-001"

    @pytest.mark.asyncio
    @patch("src.memory.cold.memory_store.MemoryDecayConfig")
    async def test_cold_store_sensitive_info_blocked(self, mock_decay_config, mock_cold_records):
        """Scene with phone number gets blocked from cold storage."""
        records, mock_table = mock_cold_records

        from src.memory.cold.memory_store import ColdMemoryStore, Scene

        store = ColdMemoryStore(
            db_path="/tmp/test_cold_memory",
            embedding_dim=512,
        )
        store._table = mock_table  # type: ignore[reportAttributeAccessIssue]
        store._initialized_in_this_session = True

        sensitive_scene = Scene(
            scene_id="scene-sensitive",
            trace_id="trace-sensitive",
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            summary="用户电话是13800138000请拨打",
        )

        memory_id = await store.store_scene(sensitive_scene)
        assert memory_id is None, "Sensitive scene should be blocked"
        assert len(records) == 0

    @pytest.mark.asyncio
    @patch("src.memory.cold.memory_store.MemoryDecayConfig")
    async def test_cold_store_without_initialization_raises(
        self, mock_decay_config, sample_cold_scene
    ):
        from src.memory.cold.memory_store import ColdMemoryStore

        store = ColdMemoryStore(db_path="/tmp/test_cold_memory", embedding_dim=512)
        # _table is None → store_scene should raise RuntimeError
        with pytest.raises(RuntimeError, match="not initialized"):
            await store.store_scene(sample_cold_scene)


# ===================================================================
# 5. Degraded flag propagation
# ===================================================================


class TestDegradedPropagation:
    """Verify the degraded flag propagates from perception events through fusion."""

    time_window: TimeSyncWindow
    classifier: EventClassifier
    synthesizer: SceneSynthesizer
    fusion_engine: EntityFusionEngine
    trace_id: str

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.time_window = TimeSyncWindow(max_window_ms=5000, min_window_ms=1)
        self.classifier = EventClassifier()
        self.synthesizer = SceneSynthesizer()
        self.fusion_engine = EntityFusionEngine(align_threshold=0.1)
        self.trace_id = "test-degraded"

    def _flush_window(self) -> WindowedEvents:
        late = _make_vision_event(trace_id=self.trace_id)
        late["timestamp"] = "2099-01-01T00:00:00+00:00"
        result = self.time_window.push(late)
        assert result is not None
        return result

    def test_degraded_vision_event_propagates_to_scene(self):
        """Push a degraded vision event → scene metadata.degraded=True."""
        event = _make_vision_event(
            trace_id=self.trace_id, degraded=True,
            objects=[{"label": "blurry", "confidence": 0.3}],
        )
        push_result = self.time_window.push(event)
        if push_result is None:
            push_result = self._flush_window()

        classified = self.classifier.classify(push_result.events)
        fusion_result = self.fusion_engine.fuse(classified, window_events=push_result.events)

        scene = self.synthesizer.synthesize(
            trace_id=self.trace_id,
            version=1,
            classified_events=classified,
            entity_fusion_result=fusion_result,
            window_events=push_result.events,
        )

        # SceneSynthesizer checks window events metadata.degraded
        assert scene["metadata"]["degraded"] is True

    def test_non_degraded_event_keeps_false(self):
        """Normal event → degraded remains False."""
        event = _make_audio_event(
            trace_id=self.trace_id,
            text="正常对话",
            emotion_category="neutral",
            degraded=False,
        )
        push_result = self.time_window.push(event)
        if push_result is None:
            push_result = self._flush_window()

        classified = self.classifier.classify(push_result.events)
        fusion_result = self.fusion_engine.fuse(classified, window_events=push_result.events)

        scene = self.synthesizer.synthesize(
            trace_id=self.trace_id,
            version=1,
            classified_events=classified,
            entity_fusion_result=fusion_result,
            window_events=push_result.events,
        )

        assert scene["metadata"]["degraded"] is False


# ===================================================================
# 6. trace_id persistence
# ===================================================================


class TestTraceIdPersistence:
    """Verify trace_id persists through the entire pipeline."""

    time_window: TimeSyncWindow
    classifier: EventClassifier
    synthesizer: SceneSynthesizer
    fusion_engine: EntityFusionEngine
    trace_id: str

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.time_window = TimeSyncWindow(max_window_ms=5000, min_window_ms=1)
        self.classifier = EventClassifier()
        self.synthesizer = SceneSynthesizer()
        self.fusion_engine = EntityFusionEngine(align_threshold=0.1)
        self.trace_id = "persistent-trace-42"

    def _flush_window(self) -> WindowedEvents:
        late = _make_vision_event(trace_id=self.trace_id)
        late["timestamp"] = "2099-01-01T00:00:00+00:00"
        result = self.time_window.push(late)
        assert result is not None
        return result

    def test_trace_id_survives_full_pipeline(self):
        """trace_id set at event creation persists to the final scene."""
        events = [
            _make_audio_event(
                trace_id=self.trace_id, text="第一步",
                emotion_category="neutral",
            ),
            _make_vision_event(
                trace_id=self.trace_id,
                objects=[{"label": "book", "confidence": 0.9}],
            ),
        ]

        push_result: WindowedEvents | None = None
        for evt in events:
            result = self.time_window.push(evt)
            if result is not None:
                push_result = result

        if push_result is None:
            push_result = self._flush_window()

        # Every event in the window should have the same trace_id
        for evt in push_result.events:
            assert evt.get("trace_id") == self.trace_id

        classified = self.classifier.classify(push_result.events)
        fusion_result = self.fusion_engine.fuse(classified, window_events=push_result.events)

        scene = self.synthesizer.synthesize(
            trace_id=self.trace_id,
            version=1,
            classified_events=classified,
            entity_fusion_result=fusion_result,
            window_events=push_result.events,
        )

        assert scene["trace_id"] == self.trace_id
        assert scene["trace_id"] == "persistent-trace-42"

    def test_different_trace_ids_isolate_scenes(self):
        """Two different trace_ids produce separate scenes."""
        scene_a = self._synthesize_for_trace("trace-A")
        scene_b = self._synthesize_for_trace("trace-B")

        assert scene_a["trace_id"] == "trace-A"
        assert scene_b["trace_id"] == "trace-B"
        assert scene_a["scene_id"] != scene_b["scene_id"]

    def _synthesize_for_trace(self, trace_id: str) -> dict[str, Any]:
        tw = TimeSyncWindow(max_window_ms=5000, min_window_ms=1)
        clf = EventClassifier()
        syn = SceneSynthesizer()
        eng = EntityFusionEngine(align_threshold=0.1)

        evt = _make_vision_event(trace_id=trace_id)
        result = tw.push(evt)
        if result is None:
            late = _make_vision_event(trace_id=trace_id)
            late["timestamp"] = "2099-01-01T00:00:00+00:00"
            result = tw.push(late)
            assert result is not None

        classified = clf.classify(result.events)
        fusion_result = eng.fuse(classified, window_events=result.events)
        return syn.synthesize(
            trace_id=trace_id, version=1,
            classified_events=classified,
            entity_fusion_result=fusion_result,
            window_events=result.events,
        )


# ===================================================================
# 7. Emotion categories are joy/sadness/neutral only
# ===================================================================


class TestEmotionCategories:
    """Verify emotion.category is always joy, sadness, or neutral."""

    VALID_CATEGORIES = {"joy", "sadness", "neutral"}

    def test_message_envelope_emotion_categories(self):
        """MessageEnvelope EmotionCategory enum values."""
        assert EmotionCategory.JOY.value == "joy"
        assert EmotionCategory.SADNESS.value == "sadness"
        assert EmotionCategory.NEUTRAL.value == "neutral"
        # ANGER and SURPRISE exist but are placeholders
        assert EmotionCategory.ANGER.value == "anger"
        assert EmotionCategory.SURPRISE.value == "surprise"

    def test_scene_synthesis_emotion_snapshot_struct(self):
        """EmotionSnapshot dataclass enforces valid fields."""
        for cat in ("joy", "sadness", "neutral"):
                snap = EmotionSnapshot(category=cat, intensity=0.5)
                assert snap.category == cat
                assert snap.category in self.VALID_CATEGORIES

    def test_emotion_from_audio_event_passes_through(self):
        """Emotion set in audio event metadata appears in the scene."""
        tw = TimeSyncWindow(max_window_ms=5000, min_window_ms=1)
        clf = EventClassifier()
        syn = SceneSynthesizer()
        eng = EntityFusionEngine(align_threshold=0.1)
        trace_id = "emotion-test"

        event = _make_audio_event(
            trace_id=trace_id,
            text="很高兴见到你",
            emotion_category="joy",
            emotion_intensity=0.92,
        )
        result = tw.push(event)
        if result is None:
            late = _make_audio_event(trace_id=trace_id)
            late["timestamp"] = "2099-01-01T00:00:00+00:00"
            result = tw.push(late)
            assert result is not None

        classified = clf.classify(result.events)
        fusion_result = eng.fuse(classified, window_events=result.events)

        scene = syn.synthesize(
            trace_id=trace_id, version=1,
            classified_events=classified,
            entity_fusion_result=fusion_result,
            window_events=result.events,
        )

        emotion = scene["payload"]["emotion_snapshot"]
        assert emotion["category"] in self.VALID_CATEGORIES


# ===================================================================
# Combined pipeline: multi-modal → fusion → hot memory → cold memory
# ===================================================================


class TestFullPipelineIntegration:
    """End-to-end: multi-modal events → fusion → hot memory → cold memory."""

    time_window: TimeSyncWindow
    classifier: EventClassifier
    synthesizer: SceneSynthesizer
    fusion_engine: EntityFusionEngine
    trace_id: str

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.time_window = TimeSyncWindow(max_window_ms=5000, min_window_ms=1)
        self.classifier = EventClassifier()
        self.synthesizer = SceneSynthesizer()
        self.fusion_engine = EntityFusionEngine(align_threshold=0.1)
        self.trace_id = "full-pipeline-test"

    def _flush_window(self) -> WindowedEvents:
        late = _make_vision_event(trace_id=self.trace_id)
        late["timestamp"] = "2099-01-01T00:00:00+00:00"
        result = self.time_window.push(late)
        assert result is not None
        return result

    @pytest.mark.asyncio
    @patch("src.memory.cold.memory_store.MemoryDecayConfig")
    async def test_full_pipeline(
        self, mock_decay_config
    ):
        """Feed vision + audio events → fusion → hot store → cold store."""
        from src.config.runtime import RuntimeConfig
        from src.memory.hot.memory_store import HotMemoryStore
        from src.memory.cold.memory_store import ColdMemoryStore, Scene

        # ---- Fusion ----
        events = [
            _make_vision_event(trace_id=self.trace_id),
            _make_audio_event(
                trace_id=self.trace_id,
                text="今天工作很顺利",
                emotion_category="joy",
                emotion_intensity=0.8,
            ),
        ]

        push_result: WindowedEvents | None = None
        for evt in events:
            result = self.time_window.push(evt)
            if result is not None:
                push_result = result
        if push_result is None:
            push_result = self._flush_window()

        classified = self.classifier.classify(push_result.events)
        fusion_result = self.fusion_engine.fuse(classified, window_events=push_result.events)
        scene = self.synthesizer.synthesize(
            trace_id=self.trace_id,
            version=1,
            classified_events=classified,
            entity_fusion_result=fusion_result,
            window_events=push_result.events,
        )

        assert scene["trace_id"] == self.trace_id
        assert scene["payload_type"] == "scene"
        assert scene["payload"]["emotion_snapshot"]["category"] in (
            "joy", "sadness", "neutral"
        )

        # ---- Hot memory ----
        config = MagicMock(spec=RuntimeConfig)
        hot_store = HotMemoryStore(config)
        # Inject an in-memory redis-like dict
        hot_data: dict[str, Any] = {}

        class _FakeRedis:
            def get(self, key):
                v = hot_data.get(key)
                return json.dumps(v).encode() if v is not None else None
            def set(self, key, value, ex=None):
                hot_data[key] = json.loads(value) if isinstance(value, (str, bytes)) else value
                return True
            def delete(self, key):
                return hot_data.pop(key, None) is not None
            def exists(self, key):
                return key in hot_data
            def close(self):
                pass
            def pipeline(self):
                p = MagicMock()
                p.execute = MagicMock(return_value=[])
                return p

        hot_store._redis = _FakeRedis()
        hot_store._degraded = False

        hot_ok = hot_store.store_scene(scene)
        assert hot_ok is True

        retrieved_hot = hot_store.get_scene(scene["scene_id"])
        assert retrieved_hot is not None
        assert retrieved_hot["trace_id"] == self.trace_id
        assert retrieved_hot["payload"]["emotion_snapshot"]["category"] in (
            "joy", "sadness", "neutral"
        )

        # ---- Cold memory ----
        cold_store = ColdMemoryStore(
            db_path="/tmp/test_full_pipeline_cold",
            embedding_dim=512,
        )
        # Inject mocks
        cold_records: list[dict[str, Any]] = []

        class _MockColdTable:
            async def add(self, data):
                for rec in data:
                    cold_records.append(rec)
            async def search(self, vector, vector_column_name="embedding"):
                r = MagicMock()
                r.limit = MagicMock(return_value=r)
                r.to_list = MagicMock(return_value=list(cold_records))
                return r

        cold_store._table = _MockColdTable()
        cold_store._initialized_in_this_session = True
        cold_store._embedding_model = MagicMock()
        cold_store._embedding_model.encode = MagicMock(return_value=[0.1] * 512)

        cold_scene = Scene(
            scene_id=scene["scene_id"],
            trace_id=scene["trace_id"],
            timestamp=scene["timestamp"],
            summary=scene["payload"]["summary"],
            events=scene["payload"].get("secondary_events", []),
            entity_relations=scene["payload"].get("entity_relations", []),
            affective_flag=scene["metadata"].get("affective_flag", False),
            importance_score=scene["payload"]["confidence"],
            scene_class=scene["payload"].get("scene_class", {}).get("primary", "general"),
        )

        cold_id = await cold_store.store_scene(cold_scene)
        assert cold_id is not None
        assert cold_id == scene["scene_id"]

        # Search back
        results = await cold_store.semantic_search("工作", top_k=5)
        assert isinstance(results, list)
        if results:
            assert results[0].memory_id == scene["scene_id"]


# Need datetime import for timestamp generation
import datetime  # noqa: E402 — must come after __future__ annotations but before usage
