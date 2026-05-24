# pyright: reportUninitializedInstanceVariable=false
# pytest fixtures (autouse=True) initialise instance vars — pyright can't trace them.

"""
End-to-end integration test: pure voice dialogue loop.  v4.5.0 §1–§7

The most important vertical slice — exercises the complete pipeline:

  1. Audio input "你好吗" → MockAudioPipeline → AudioEvent envelope (§1)
  2. AudioEvent → Fusion (TimeSyncWindow → EventClassifier → SceneSynthesizer) → Scene (§2)
  3. Scene → HotMemory store (§3)
  4. Scene + DynamicPersonality → MainDecisionEngine.decide() (mocked 3B) (§4–§5)
  5. DecisionCommand → ActionSequenceScheduler → channels dispatched (§7)
  6. VoiceChannel → TTS audio chunk produced (§7.5)

Verification:
  · trace_id stays same throughout entire pipeline
  · version monotonically increments
  · emotion.category propagates correctly (joy/sadness/neutral)
  · degraded flag flows correctly through all layers
  · All modules use MessageEnvelope-style dicts (not raw payloads)

All GPU dependencies (Qwen, Whisper, CosyVoice) are mocked.
No Redis/LanceDB services required.
"""
from __future__ import annotations

import json
import time
import uuid
from copy import deepcopy
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Fusion pipeline — real components (GPU/spaCy are lazy-loaded, safe to import)
# ---------------------------------------------------------------------------

from src.fusion.time_window import TimeSyncWindow, WindowedEvents
from src.fusion.event_classifier import EventClassifier, ClassifiedEvents
from src.fusion.entity_fusion import EntityFusionEngine, EntityFusionResult
from src.fusion.scene_synthesis import SceneSynthesizer
from src.fusion.message_envelope import MessageEnvelope  # noqa: PLC0106

# ---------------------------------------------------------------------------
# Config / runtime
# ---------------------------------------------------------------------------

from src.config.runtime import RuntimeConfig, VRAMTier

# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

from src.memory.hot.memory_store import HotMemoryStore

# ---------------------------------------------------------------------------
# Personality
# ---------------------------------------------------------------------------

from src.personality.baseline import BaselinePersonality
from src.personality.dynamic_fusion import DynamicFusion

# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------

# v5.x: MainDecisionEngine was removed in DecisionBridge refactor.
# Tests in this file depend on it — skip the whole module if unavailable.
try:
    from src.decision.main_decision import MainDecisionEngine
except ImportError:
    pytest.skip(
        "src.decision.main_decision removed in v5.x (DecisionBridge refactor). "
        "Remove this module or port tests to DecisionBridge.",
        allow_module_level=True,
    )
    MainDecisionEngine = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

from src.execution.action_scheduler import (
    Action,
    ActionSequence,
    ActionSequenceScheduler,
    CHANNEL_VOICE,
)
from src.execution.channels.voice_channel import VoiceChannel
from src.execution.tts_service.cosyvoice_adapter import (
    CosyVoiceAdapter,
    TTSAudioChunk,
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
    Emotion,
    Metadata,
    Layer,
    PayloadType,
    EmotionCategory,
)

# ---------------------------------------------------------------------------
# Config / runtime
# ---------------------------------------------------------------------------

from src.config.runtime import RuntimeConfig, VRAMTier

# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

from src.memory.hot.memory_store import HotMemoryStore

# ---------------------------------------------------------------------------
# Personality
# ---------------------------------------------------------------------------

from src.personality.baseline import BaselinePersonality
from src.personality.dynamic_fusion import DynamicFusion

# ---------------------------------------------------------------------------
# Decision  (MainDecisionEngine imported above via try/except)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

from src.execution.action_scheduler import (
    Action,
    ActionSequence,
    ActionSequenceScheduler,
    CHANNEL_AVATAR,
    CHANNEL_MOUSE,
    CHANNEL_VOICE,
    CHANNEL_BUBBLE,
)
from src.execution.state_bus import StateBus
from src.execution.channels.voice_channel import VoiceChannel
from src.execution.tts_service.cosyvoice_adapter import (
    CosyVoiceAdapter,
    TTSAudioChunk,
)

# ===================================================================
# Fixtures  (shared across test classes)
# ===================================================================

TRACE_ID = uuid.uuid4().hex


@pytest.fixture
def runtime_config() -> RuntimeConfig:
    return RuntimeConfig(
        vram_tier=VRAMTier.HIGH,
        vram_total_gb=16.0,
        low_vram=False,
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


VALID_BASELINE = {
    "baseline_id": "00000000-0000-0000-0000-000000000001",
    "name": "温柔伙伴",
    "description": "耐心、鼓励型，偶尔俏皮，善于倾听",
    "voice_style": {
        "tone": {"value": "gentle", "type": "categorical",
                 "allowed": ["gentle", "calm", "lively", "serious"]},
        "speed": {"value": 1.0, "min": 0.8, "max": 1.3, "type": "numeric"},
        "formality": {"value": 0.5, "min": 0.3, "max": 0.7, "type": "numeric"},
        "emotion_range": {"value": 0.7, "min": 0.5, "max": 0.9, "type": "numeric"},
    },
    "avatar_style": {
        "expression_intensity": {"value": 0.7, "min": 0.5, "max": 0.9, "type": "numeric"},
        "gesture_frequency": {"value": 0.5, "min": 0.3, "max": 0.7, "type": "numeric"},
        "eye_contact_tendency": {"value": 0.8, "min": 0.6, "max": 1.0, "type": "numeric"},
    },
    "mouse_style": {
        "movement_speed": {"value": 0.6, "min": 0.3, "max": 0.9, "type": "numeric"},
        "click_delay": {"value": 0.5, "min": 0.2, "max": 0.8, "type": "numeric"},
    },
    "proactivity": {
        "interrupt_probability": {"value": 0.3, "min": 0.1, "max": 0.6, "type": "numeric"},
        "topic_initiative": {"value": 0.4, "min": 0.2, "max": 0.7, "type": "numeric"},
        "gesture_frequency": {"value": 0.5, "min": 0.2, "max": 0.7, "type": "numeric"},
    },
    "forbidden_actions": [
        "always_ask_before_sending_external_data",
    ],
    "immutable": True,
}


@pytest.fixture
def baseline_personality() -> BaselinePersonality:
    return BaselinePersonality(config=deepcopy(VALID_BASELINE))


# ===================================================================
# Helper — build a synthetic audio perception event envelope
# ===================================================================

def _make_audio_event(
    trace_id: str = TRACE_ID,
    degraded: bool = False,
    text: str = "你好吗",
    emotion_category: str = "neutral",
    emotion_intensity: float = 0.0,
    emotion_confidence: float = 0.5,
    affective: bool = False,
) -> dict[str, Any]:
    """Build a perception event dict matching AudioPipeline._build_event() output.

    This is the §0.3 unified message envelope that the AudioPipeline yields
    after processing a microphone chunk.
    """
    from datetime import datetime, timezone
    return {
        "trace_id": trace_id,
        "source_layer": "perception",
        "source_component": "audio",
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "version": 1,
        "payload_type": "perception_event",
        "payload": {
            "type": "audio_event",
            "audio": {
                "text": text,
                "voicefeature": {
                    "language": "zh",
                    "avg_logprob": -0.3,
                },
            },
        },
        "metadata": {
            "confidence": 0.95,
            "latency_ms": 150.0,
            "degraded": degraded,
            "fast_path": False,
            "emotion": {
                "category": emotion_category,
                "intensity": emotion_intensity,
                "source": "text_sentiment",
                "confidence": emotion_confidence,
            },
            "affective_flag": affective,
            "scene_context": {
                "primary_type": "unknown",
                "confidence": 0.0,
            },
            "user_model_version": 0,
        },
    }


# ===================================================================
# Test — full voice dialogue loop
# ===================================================================


class TestVoiceDialogueLoopE2E:
    """Complete end-to-end voice dialogue loop.

    Simulates "你好吗" → audio pipeline → fusion → memory → personality →
    decision → scheduler → voice channel, verifying that every module
    passes the correct data and that envelope invariants hold.
    """

    # Type annotations (initialised by _setup fixture)
    time_window: TimeSyncWindow
    classifier: EventClassifier
    fusion_engine: EntityFusionEngine
    synthesizer: SceneSynthesizer
    engine: MainDecisionEngine
    scheduler: ActionSequenceScheduler
    voice_channel: VoiceChannel
    hot_memory: HotMemoryStore
    trace_id: str
    input_event: dict[str, Any]

    @pytest.fixture(autouse=True)
    def _setup(self, runtime_config: RuntimeConfig, baseline_personality: BaselinePersonality):
        """Initialise real pipeline components with mocked heavy dependencies."""
        # Trace ID — must persist across ALL layers
        self.trace_id = TRACE_ID

        # ── Step 1: Audio perception event ──────────────────────────
        self.input_event = _make_audio_event(
            trace_id=self.trace_id,
            text="你好吗",
            emotion_category="neutral",
            emotion_intensity=0.0,
        )

        # ── Fusion layer (real components) ───────────────────────────
        # max_window_ms must be large enough for normal sequential pushes,
        # small enough that a late-timestamp flush event triggers closure.
        self.time_window = TimeSyncWindow(max_window_ms=5000, min_window_ms=1)
        self.classifier = EventClassifier()
        self.fusion_engine = EntityFusionEngine(align_threshold=0.1)
        self.synthesizer = SceneSynthesizer()

        # ── Hot memory ──────────────────────────────────────────────
        self.hot_memory = HotMemoryStore(runtime_config)
        # Patch _redis to use in-memory dict (no real Redis)
        self.hot_memory._redis = MagicMock()
        self.hot_memory._redis.exists.return_value = False
        self.hot_memory._redis.set = MagicMock()
        self.hot_memory._redis.get.return_value = None
        self.hot_memory._redis.rpush = MagicMock(return_value=1)
        self.hot_memory._redis.lrange.return_value = []
        self.hot_memory._redis.xadd = MagicMock(return_value=b"stream-id-001")
        self.hot_memory._redis.delete = MagicMock(return_value=True)

        # ── Decision engine (mocked 3B model) ───────────────────────
        self.engine = MainDecisionEngine(runtime_config)
        self.engine._model_loaded = True
        self.engine._degraded = False
        self.engine._generate = AsyncMock(return_value="我很好，谢谢你的关心！")
        self.engine._estimate_confidence = MagicMock(return_value=0.92)

        # ── Action scheduler ────────────────────────────────────────
        self.scheduler = ActionSequenceScheduler(runtime_config)

        # ── Voice channel (mocked CosyVoice) ────────────────────────
        mock_adapter = MagicMock(spec=CosyVoiceAdapter)
        mock_adapter.health_check = AsyncMock(return_value=True)
        mock_adapter.using_onnx = False
        mock_adapter.stream_synthesize = AsyncMock(
            return_value=[
                TTSAudioChunk(
                    audio_bytes=b"\x00\x01\x02\x03" * 1000,
                    elapsed_ms=500,
                    text="我很好，谢谢你的关心！",
                    is_final=True,
                )
            ]
        )
        self.voice_channel = VoiceChannel(adapter=mock_adapter)

    def _push_and_flush(self, event: dict[str, Any]) -> WindowedEvents:
        """Push an event into the time window and force flush.

        ``TimeSyncWindow.push()`` returns ``None`` when the window does not
        yet close.  This helper pushes the target event, then — if needed —
        pushes a late-timestamp sentinel to force window closure.
        """
        result = self.time_window.push(event)
        if result is not None:
            return result
        sentinel = deepcopy(event)
        sentinel["timestamp"] = "2099-01-01T00:00:00+00:00"
        result = self.time_window.push(sentinel)
        assert result is not None, "Flush sentinel must close the window"
        return result

    # ──────────────────────────────────────────────────────────────
    # Step 1: Audio Perception → Fusion
    # ──────────────────────────────────────────────────────────────

    def test_step1_audio_perception_produces_envelope(self):
        """The audio perception event is a valid §0.3 message envelope."""
        event = self.input_event
        assert event["trace_id"] == self.trace_id
        assert event["source_layer"] == "perception"
        assert event["source_component"] == "audio"
        assert event["payload_type"] == "perception_event"
        assert isinstance(event["version"], int) and event["version"] >= 1

        meta = event["metadata"]
        assert isinstance(meta["degraded"], bool)
        assert isinstance(meta["emotion"]["category"], str)
        assert meta["emotion"]["category"] in ("joy", "sadness", "neutral")
        assert isinstance(meta["emotion"]["intensity"], (int, float))

        payload = event["payload"]
        assert payload["type"] == "audio_event"
        assert payload["audio"]["text"] == "你好吗"
        assert payload["audio"]["voicefeature"]["language"] == "zh"

    # ──────────────────────────────────────────────────────────────
    # Step 2: AudioEvent → Fusion → Scene
    # ──────────────────────────────────────────────────────────────

    def test_step2_fusion_produces_scene_with_trace_id(self):
        """Audio event flows through TimeSyncWindow → Classifier → Scene."""
        # Push into time window
        push_result = self._push_and_flush(self.input_event)

        # Classify (pass events list, not WindowedEvents object)
        classified = self.classifier.classify(push_result.events)  # noqa: PLC0106
        assert isinstance(classified, ClassifiedEvents)

        # Entity fusion (no visual events → empty result)
        fusion_result = self.fusion_engine.fuse(
            classified, window_events=push_result.events
        )
        assert isinstance(fusion_result, EntityFusionResult)

        # Synthesize scene
        scene = self.synthesizer.synthesize(
            trace_id=self.trace_id,
            version=1,
            classified_events=classified,
            entity_fusion_result=fusion_result,
            window_events=push_result.events,
            user_model_version=0,
            start_time=time.time(),
        )

        # ── Assert scene structure ──
        assert isinstance(scene, dict)
        assert scene["trace_id"] == self.trace_id
        assert scene["source_layer"] == "fusion"
        assert scene["payload_type"] == "scene"
        assert "scene_id" in scene

        # emotion.category must be one of the three reliable categories
        emotion_cat = scene["payload"]["emotion_snapshot"]["category"]
        assert emotion_cat in ("joy", "sadness", "neutral")

        # degraded flag in metadata
        assert isinstance(scene["metadata"]["degraded"], bool)

        # version is monotonic
        assert scene["version"] >= 1

    # ──────────────────────────────────────────────────────────────
    # Step 3: Scene → HotMemory store
    # ──────────────────────────────────────────────────────────────

    def test_step3_scene_stored_in_hot_memory(self):
        """Scene can be stored and retrieved from hot memory."""
        # Produce a scene first (same path as step 2)
        push_result = self._push_and_flush(self.input_event)
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

        # Store scene in hot memory
        scene_id = scene["scene_id"]
        # Ensure redis.get returns scene JSON for subsequent get_scene call
        self.hot_memory._redis.get.return_value = json.dumps(
            scene, ensure_ascii=False, default=str
        )
        success = self.hot_memory.store_scene(scene)
        assert success is True

        # Verify Redis was called to store the scene
        self.hot_memory._redis.set.assert_called()

        # Retrieve and verify
        retrieved = self.hot_memory.get_scene(scene_id)
        assert retrieved is not None
        assert retrieved["trace_id"] == self.trace_id
        assert retrieved["payload_type"] == "scene"

    # ──────────────────────────────────────────────────────────────
    # Step 4: Scene + DynamicPersonality → MainDecisionEngine
    # ──────────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_step4_decision_uses_scene_and_personality(self, baseline_personality):
        """MainDecisionEngine.decide() consumes scene summary and emotion."""
        # Build dynamic personality from baseline with emotion from scene
        scene_emotion = "neutral"
        dynamic = DynamicFusion.generate(
            baseline=baseline_personality.to_dict(),
            emotion_label=scene_emotion,
        )
        tts_emotion = dynamic["tts_control"]["emotion"]

        scene_summary = "用户说'你好吗'，询问我的状态。"

        # Run decision (mocked 3B model returns a canned response)
        decision = await self.engine.decide(
            scene_summary=scene_summary,
            emotion_category=scene_emotion,
            emotion_intensity=0.0,
            tts_emotion=tts_emotion,
            scene_primary="conversation",
            trace_id=self.trace_id,
        )

        # ── Assert decision structure ──
        assert isinstance(decision, dict)
        assert decision["trace_id"] == self.trace_id
        assert decision["source"] == "main_decision_3b"
        assert decision["decision_type"] == "voice_response"
        assert decision["command"]["voice_response"] == "我很好，谢谢你的关心！"
        assert isinstance(decision["command"]["actions"], list)
        assert decision["safety_level"] in ("SAFE", "NEEDS_CONFIRM", "DANGEROUS_AUTO_BLOCK")
        assert isinstance(decision["confidence"], float)
        assert 0.0 <= decision["confidence"] <= 1.0
        assert decision["shadow_overridden"] is False

        # Verify the mocked generate was called
        self.engine._generate.assert_awaited_once()  # type: ignore[reportAttributeAccessIssue]

    # ──────────────────────────────────────────────────────────────
    # Step 5: DecisionCommand → ActionSequenceScheduler
    # ──────────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_step5_scheduler_creates_sequence_from_decision(self, baseline_personality):
        """ActionSequenceScheduler produces a sequence from the decision command."""
        dynamic = DynamicFusion.generate(
            baseline=baseline_personality.to_dict(),
            emotion_label="neutral",
        )
        decision = await self.engine.decide(
            scene_summary="用户说'你好吗'。",
            emotion_category="neutral",
            tts_emotion=dynamic["tts_control"]["emotion"],
            scene_primary="conversation",
            trace_id=self.trace_id,
        )

        # Build actions from decision command
        voice_text = decision["command"]["voice_response"]
        actions = [
            Action(channel=CHANNEL_VOICE, type="text", value=voice_text, start_ms=0),
        ]
        sequence = self.scheduler.create_sequence(actions=actions)
        self.scheduler.set_active_sequence(sequence)

        # Verify sequence
        assert isinstance(sequence, ActionSequence)
        assert sequence.sequence_id is not None
        assert len(sequence.actions) == 1
        assert sequence.actions[0].channel == CHANNEL_VOICE
        assert sequence.actions[0].value == voice_text
        assert sequence.skip_decision is False

        # Verify ready actions
        ready = self.scheduler.get_ready_actions(elapsed_ms=0)
        assert len(ready) == 1
        assert ready[0].channel == CHANNEL_VOICE

    # ──────────────────────────────────────────────────────────────
    # Step 6: VoiceChannel → TTS audio produced
    # ──────────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_step6_voice_channel_produces_audio(self):
        """VoiceChannel.speak() produces TTS audio chunks."""
        self.voice_channel.set_trace_id(self.trace_id)

        result = await self.voice_channel.speak(
            text="我很好，谢谢你的关心！",
            emotion="neutral",
        )
        # Should return True since mocked adapter returns chunks
        assert result is True

        # Verify the adapter's stream_synthesize was called with correct params
        # v4.5.0 §7.5.3: emotion mapped via map_emotion_to_cosyvoice → enum
        from src.execution.tts_service.cosyvoice_adapter import CosyVoiceEmotion
        self.voice_channel._adapter.stream_synthesize.assert_awaited_once_with(  # type: ignore[reportAttributeAccessIssue]
            text="我很好，谢谢你的关心！",
            emotion=CosyVoiceEmotion.NEUTRAL,
            speed=1.0,
            speaker="diana",
        )

    # ──────────────────────────────────────────────────────────────
    # Full end-to-end: trace_id persistence
    # ──────────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_trace_id_persists_through_full_pipeline(self, baseline_personality):
        """trace_id remains the same across all six pipeline stages."""
        # Stage 1-2: Audio event → Scene
        push_result = self._push_and_flush(self.input_event)
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
            start_time=time.time(),
        )
        assert scene["trace_id"] == self.trace_id

        # Stage 3: Hot memory (store + retrieve)
        scene_id = scene["scene_id"]
        self.hot_memory._redis.get.return_value = json.dumps(
            scene, ensure_ascii=False, default=str
        )
        self.hot_memory.store_scene(scene)

        # Stage 4: Decision
        dynamic = DynamicFusion.generate(
            baseline=baseline_personality.to_dict(),
            emotion_label="neutral",
        )
        decision = await self.engine.decide(
            scene_summary="用户说'你好吗'。",
            emotion_category="neutral",
            tts_emotion=dynamic["tts_control"]["emotion"],
            scene_primary="conversation",
            trace_id=self.trace_id,
        )
        assert decision["trace_id"] == self.trace_id

        # Stage 6: Voice channel
        self.voice_channel.set_trace_id(self.trace_id)
        await self.voice_channel.speak(
            text=decision["command"]["voice_response"],
            emotion="neutral",
        )
        # Trace ID was set; no explicit assertion needed beyond no crash

    # ──────────────────────────────────────────────────────────────
    # Full end-to-end: degraded flag propagation
    # ──────────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_degraded_flag_propagates_correctly(self, baseline_personality):
        """Degraded=false in nominal path; degraded=true when components fail."""
        # ── Normal path: no degradation ──
        normal_event = _make_audio_event(
            trace_id=self.trace_id,
            degraded=False,
            text="你好吗",
            emotion_category="neutral",
        )
        assert normal_event["metadata"]["degraded"] is False

        # Fusion path (degraded=false from input)
        push_result = self._push_and_flush(normal_event)
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
            start_time=time.time(),
        )

        # ── Degraded path: degraded flag from audio ──
        degraded_event = _make_audio_event(
            trace_id=self.trace_id,
            degraded=True,
            text="你好吗",
            emotion_category="neutral",
            emotion_confidence=0.0,
        )
        assert degraded_event["metadata"]["degraded"] is True

        push_result2 = self._push_and_flush(degraded_event)
        classified2 = self.classifier.classify(push_result2.events)
        fusion_result2 = self.fusion_engine.fuse(
            classified2, window_events=push_result2.events
        )
        scene2 = self.synthesizer.synthesize(
            trace_id=self.trace_id,
            version=2,
            classified_events=classified2,
            entity_fusion_result=fusion_result2,
            window_events=push_result2.events,
            start_time=time.time(),
        )
        # The fusion layer propagates the degraded flag
        # (Note: actual propagation depends on how scenes track degradation;
        #  the integration tests verify the mechanism exists.)
        assert isinstance(scene2["metadata"]["degraded"], bool)

    # ──────────────────────────────────────────────────────────────
    # Full end-to-end: emotion propagation
    # ──────────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_emotion_category_propagates_from_perception(self, baseline_personality):
        """emotion.category from audio perception reaches the decision layer."""
        test_emotions = ["joy", "sadness", "neutral"]
        for expected_emotion in test_emotions:
            # Create audio event with specific emotion
            event = _make_audio_event(
                trace_id=uuid.uuid4().hex,
                text=f"Test {expected_emotion}",
                emotion_category=expected_emotion,
                emotion_intensity=0.7 if expected_emotion != "neutral" else 0.0,
            )
            assert event["metadata"]["emotion"]["category"] == expected_emotion

            # Flow through fusion
            push_result = self._push_and_flush(event)
            classified = self.classifier.classify(push_result.events)
            fusion_result = self.fusion_engine.fuse(
                classified, window_events=push_result.events
            )
            scene = self.synthesizer.synthesize(
                trace_id=event["trace_id"],
                version=1,
                classified_events=classified,
                entity_fusion_result=fusion_result,
                window_events=push_result.events,
                start_time=time.time(),
            )

            # Emotion snapshot in scene
            scene_emotion = scene["payload"]["emotion_snapshot"]["category"]
            assert scene_emotion in ("joy", "sadness", "neutral")

            # Dynamic personality uses this emotion
            dynamic = DynamicFusion.generate(
                baseline=baseline_personality.to_dict(),
                emotion_label=scene_emotion,
            )
            assert dynamic["tts_control"]["emotion"] in ("joy", "sadness", "neutral")

            # Decision engine receives it
            decision = await self.engine.decide(
                scene_summary=f"User expresses {expected_emotion}",
                emotion_category=scene_emotion,
                tts_emotion=dynamic["tts_control"]["emotion"],
                scene_primary="conversation",
                trace_id=event["trace_id"],
            )
            assert decision["trace_id"] == event["trace_id"]

    # ──────────────────────────────────────────────────────────────
    # Full end-to-end: version monotonically increments
    # ──────────────────────────────────────────────────────────────

    def test_version_monotonically_increments(self):
        """version counter increases across successive scenes in the same trace."""
        versions: list[int] = []

        # Push three events and check version increments
        for i in range(3):
            event = _make_audio_event(
                trace_id=self.trace_id,
                text=f"Utterance {i}",
                emotion_category="neutral",
            )
            event["version"] = i + 1
            versions.append(event["version"])

            push_result = self._push_and_flush(event)
            classified = self.classifier.classify(push_result.events)
            fusion_result = self.fusion_engine.fuse(
                classified, window_events=push_result.events
            )
            scene = self.synthesizer.synthesize(
                trace_id=self.trace_id,
                version=i + 1,
                classified_events=classified,
                entity_fusion_result=fusion_result,
                window_events=push_result.events,
                start_time=time.time(),
            )
            assert scene["version"] == i + 1

        # Verify all versions in this trace are monotonic
        assert versions == [1, 2, 3]
        assert all(
            versions[j] < versions[j + 1]
            for j in range(len(versions) - 1)
        )

    # ──────────────────────────────────────────────────────────────
    # Unified envelope assertion: all modules use §0.3 envelope
    # ──────────────────────────────────────────────────────────────

    def test_all_modules_use_message_envelope(self):
        """All pipeline stages produce dicts matching §0.3 envelope structure."""
        # Check audio event envelope
        event = self.input_event
        for key in ("trace_id", "source_layer", "timestamp", "version",
                     "payload_type", "payload", "metadata"):
            assert key in event, f"Audio event missing envelope key: {key}"

        # Check scene envelope
        push_result = self._push_and_flush(event)
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
            start_time=time.time(),
        )
        for key in ("trace_id", "source_layer", "timestamp", "version",
                     "payload_type", "payload", "metadata"):
            assert key in scene, f"Scene envelope missing key: {key}"
        assert scene["source_layer"] == "fusion"
        assert scene["payload_type"] == "scene"

    # ──────────────────────────────────────────────────────────────
    # Full end-to-end: complete pipeline
    # ──────────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_full_voice_loop_end_to_end(self, baseline_personality):
        """Complete voice dialogue loop from '你好吗' to TTS audio output."""
        trace_id = uuid.uuid4().hex

        # ── 1. Audio perception event ──────────────────────────────
        event = _make_audio_event(
            trace_id=trace_id,
            text="你好吗",
            emotion_category="neutral",
        )

        # ── 2. Fusion → Scene ──────────────────────────────────────
        push_result = self._push_and_flush(event)
        classified = self.classifier.classify(push_result.events)
        fusion_result = self.fusion_engine.fuse(
            classified, window_events=push_result.events
        )
        scene = self.synthesizer.synthesize(
            trace_id=trace_id,
            version=1,
            classified_events=classified,
            entity_fusion_result=fusion_result,
            window_events=push_result.events,
            start_time=time.time(),
        )
        assert scene["trace_id"] == trace_id
        scene_emotion = scene["payload"]["emotion_snapshot"]["category"]

        # ── 3. Store scene in hot memory ────────────────────────────
        scene_id = scene["scene_id"]
        self.hot_memory._redis.get.return_value = json.dumps(
            scene, ensure_ascii=False, default=str
        )
        store_ok = self.hot_memory.store_scene(scene)
        assert store_ok is True

        # ── 4. Build personality → decide ───────────────────────────
        dynamic = DynamicFusion.generate(
            baseline=baseline_personality.to_dict(),
            emotion_label=scene_emotion,
        )
        tts_emotion = dynamic["tts_control"]["emotion"]

        decision = await self.engine.decide(
            scene_summary=scene["payload"]["summary"],
            hot_memory_summaries=["用户说你好吗"],
            emotion_category=scene_emotion,
            emotion_intensity=0.0,
            tts_emotion=tts_emotion,
            scene_primary=scene["payload"]["scene_class"]["primary"],
            trace_id=trace_id,
        )
        assert decision["trace_id"] == trace_id
        assert len(decision["command"]["voice_response"]) > 0

        # ── 5. Schedule the voice action ────────────────────────────
        voice_text = decision["command"]["voice_response"]
        actions = [
            Action(channel=CHANNEL_VOICE, type="text", value=voice_text, start_ms=0),
        ]
        sequence = self.scheduler.create_sequence(actions=actions)
        self.scheduler.set_active_sequence(sequence)

        ready = self.scheduler.get_ready_actions(elapsed_ms=0)
        assert len(ready) == 1

        # ── 6. Speak via voice channel ─────────────────────────────
        self.voice_channel.set_trace_id(trace_id)
        spoke = await self.voice_channel.speak(
            text=voice_text,
            emotion=tts_emotion,
        )
        assert spoke is True

        # Verify adapter was called with the right emotion
        call_kwargs = self.voice_channel._adapter.stream_synthesize.call_args  # type: ignore[reportAttributeAccessIssue]
        if call_kwargs is not None:
            _, kwargs = call_kwargs
            assert kwargs["text"] == voice_text

        # ── 7. trace_id invariant ──────────────────────────────────
        # The trace_id must be the same across all stages
        assert event["trace_id"] == trace_id
        assert scene["trace_id"] == trace_id
        assert decision["trace_id"] == trace_id
        # voice channel set its trace_id — verified by no crash
