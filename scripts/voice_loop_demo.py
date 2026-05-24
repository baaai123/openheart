#!/usr/bin/env python3
"""
voice_loop_demo.py — one-shot voice dialogue loop CLI.  v4.5.0 §1–§7

Simulates the complete voice interaction pipeline end-to-end:

  Text → AudioEvent → Fusion → Scene → HotMemory → DynamicPersonality
  → MainDecision → ActionSequence → Voice TTS

Modes:
  --mode mock:  mocked Qwen 3B + CosyVoice (no GPU, no model files required)
  --mode real:  actual Qwen 3B inference + CosyVoice TTS (requires downloaded
                models + GPU)

Usage:
  python scripts/voice_loop_demo.py --text "你好" --mode mock
  python scripts/voice_loop_demo.py --text "你今天开心吗" --mode real
  python scripts/voice_loop_demo.py --text "Hello" --mode mock --vram-tier low
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

# ── Project imports ───────────────────────────────────────────────────────
# Fusion pipeline — real components (lazy GPU loading, safe to import)
from fusion.time_window import TimeSyncWindow
from fusion.event_classifier import EventClassifier
from fusion.entity_fusion import EntityFusionEngine
from fusion.scene_synthesis import SceneSynthesizer

# Config
from config.runtime import RuntimeConfig, VRAMTier

# Memory
from memory.hot.memory_store import HotMemoryStore

# Personality
from personality.baseline import BaselinePersonality
from personality.dynamic_fusion import DynamicFusion

# Decision
from decision.main_decision import MainDecisionEngine

# Execution
from execution.action_scheduler import (
    ActionSequenceScheduler,
)
from execution.channels.voice_channel import VoiceChannel
from execution.tts_service.cosyvoice_adapter import (
    CosyVoiceAdapter,
    TTSAudioChunk,
)

# ── Logging ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s  %(name)s  %(message)s",
)
logger = logging.getLogger("voice_loop_demo")
logger.setLevel(logging.INFO)


# ── Constants ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# v4.5.0 项目宪法 §2.2: only joy, sadness, neutral are reliable outputs
VALID_EMOTIONS = frozenset({"joy", "sadness", "neutral"})


# =====================================================================
#  Helpers
# =====================================================================

def _make_audio_event(
    trace_id: str,
    text: str,
    emotion_category: str = "neutral",
    emotion_intensity: float = 0.0,
    emotion_confidence: float = 0.5,
    degraded: bool = False,
) -> dict[str, Any]:
    """Build a perception-event message envelope from text input.

    Matches the shape produced by AudioPipeline._build_event() in the
    real perception layer (v4.5.0 §1.4, §0.3).
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
                    "language": "zh" if any("\u4e00" <= ch <= "\u9fff" for ch in text) else "en",
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
            "affective_flag": False,
            "scene_context": {
                "primary_type": "unknown",
                "confidence": 0.0,
            },
            "user_model_version": 0,
        },
    }


class EmotionResult:
    category: str
    intensity: float
    confidence: float

    def __init__(self, category: str, intensity: float, confidence: float) -> None:
        self.category = category
        self.intensity = intensity
        self.confidence = confidence


def _detect_emotion(text: str) -> EmotionResult:
    """Simple keyword-based emotion detection for demo purposes.

    In real operation this is replaced by EmotionAnalyzer (SnowNLP /
    spaCytextblob / StructBERT).  Only joy/sadness/neutral are reliable.
    """
    text_lower = text.lower()
    joy_kw = {"开心", "高兴", "哈哈", "好", "棒", "喜欢", "爱", "happy", "great", "love", "wonderful", "nice"}
    sad_kw = {"难过", "伤心", "悲伤", "不好", "累", "烦", "sad", "bad", "tired", "angry", "hate"}

    joy_score = sum(1 for kw in joy_kw if kw in text_lower)
    sad_score = sum(1 for kw in sad_kw if kw in text_lower)

    if joy_score > sad_score:
        return EmotionResult("joy", min(0.5 + joy_score * 0.15, 1.0), 0.7)
    elif sad_score > joy_score:
        return EmotionResult("sadness", min(0.5 + sad_score * 0.15, 1.0), 0.7)
    else:
        return EmotionResult("neutral", 0.0, 0.5)


def _make_scene_summary(text: str, emotion: str) -> str:
    """Build a human-readable scene summary for the decision engine."""
    emotion_zh = {"joy": "开心", "sadness": "难过", "neutral": "平静"}.get(emotion, "平静")
    return f"用户说'{text}'，情绪{emotion_zh}。场景类型：对话。"


def _load_baseline() -> dict[str, object]:
    """Load the default personality baseline from config/baseline.json."""
    cfg_path = PROJECT_ROOT / "config" / "baseline.json"
    with open(cfg_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# =====================================================================
#  Mock / Real pipeline factory
# =====================================================================

class VoiceLoopPipeline:
    """Holds all pipeline components for the voice dialogue loop."""

    config: RuntimeConfig
    mode: str
    baseline_data: dict[str, object]
    trace_id: str
    time_window: TimeSyncWindow
    classifier: EventClassifier
    fusion_engine: EntityFusionEngine
    synthesizer: SceneSynthesizer
    hot_memory: HotMemoryStore
    baseline: BaselinePersonality
    dynamic: dict[str, Any]
    engine: MainDecisionEngine
    scheduler: ActionSequenceScheduler
    voice_channel: VoiceChannel | None

    def __init__(
        self,
        runtime_config: RuntimeConfig,
        mode: str,
        baseline_data: dict[str, object],
    ) -> None:
        self.config = runtime_config
        self.mode = mode
        self.baseline_data = baseline_data
        self.trace_id = uuid.uuid4().hex

        # ── Fusion layer (always real — no GPU needed) ────────────────
        self.time_window = TimeSyncWindow(max_window_ms=5000, min_window_ms=1)
        self.classifier = EventClassifier()
        self.fusion_engine = EntityFusionEngine(align_threshold=0.1)
        self.synthesizer = SceneSynthesizer()

        # ── Hot memory (mock if no Redis) ────────────────────────────
        self.hot_memory = HotMemoryStore(runtime_config)
        self._setup_mock_redis()

        # ── Personality ───────────────────────────────────────────────
        self.baseline = BaselinePersonality(config=deepcopy(baseline_data))  # type: ignore[arg-type]
        self.dynamic = {}

        # ── Decision engine ───────────────────────────────────────────
        self.engine = MainDecisionEngine(runtime_config)

        # ── Execution ─────────────────────────────────────────────────
        self.scheduler = ActionSequenceScheduler(runtime_config)
        self.voice_channel = None

    def _setup_mock_redis(self) -> None:
        """Replace real Redis with an in-memory mock."""
        mock = MagicMock()
        mock.exists.return_value = False
        mock.set = MagicMock()
        mock.get.return_value = None
        mock.rpush = MagicMock(return_value=1)
        mock.lrange.return_value = []
        mock.xadd = MagicMock(return_value=b"stream-id-001")
        mock.delete = MagicMock(return_value=True)
        self.hot_memory._redis = mock

    async def setup_decision_and_voice(self) -> None:
        """Initialize decision engine and voice channel based on mode."""
        if self.mode == "mock":
            self._setup_mock_decision()
            self._setup_mock_voice()
        else:
            await self._setup_real_decision()
            await self._setup_real_voice()

    # ── Mock mode ───────────────────────────────────────────────────

    def _setup_mock_decision(self) -> None:
        """Use mocked Qwen 3B output."""
        self.engine._model_loaded = True
        self.engine._degraded = False
        self.engine._generate = AsyncMock(
            return_value="我很好，谢谢你的关心！今天有什么想聊的吗？"
        )
        self.engine._estimate_confidence = MagicMock(return_value=0.92)

    def _setup_mock_voice(self) -> None:
        """Use mocked CosyVoice."""
        mock_adapter = MagicMock(spec=CosyVoiceAdapter)
        mock_adapter.health_check = AsyncMock(return_value=True)
        mock_adapter.using_onnx = False
        mock_adapter.stream_synthesize = AsyncMock(
            return_value=[
                TTSAudioChunk(
                    audio_bytes=b"\x00\x01\x02\x03" * 1000,
                    elapsed_ms=500,
                    text="Mock TTS output — no actual audio generated.",
                    is_final=True,
                )
            ]
        )
        self.voice_channel = VoiceChannel(adapter=mock_adapter)

    # ── Real mode ───────────────────────────────────────────────────

    async def _setup_real_decision(self) -> None:
        """Load actual Qwen 2.5-3B model."""
        logger.info("Loading Qwen2.5-3B model (this may take a minute)...")
        ok = self.engine.load_model()
        if not ok:
            logger.warning("Qwen3B model unavailable — falling back to degraded mode.")
            # Degraded: engine returns canned responses
            self.engine._model_loaded = False
            self.engine._degraded = True

    async def _setup_real_voice(self) -> None:
        """Use real CosyVoice adapter (mock if unavailable)."""
        try:
            adapter = CosyVoiceAdapter(config=self.config)
            healthy = await adapter.health_check()
            if not healthy:
                logger.warning("CosyVoice health check failed — using mock TTS.")
                self._setup_mock_voice()
                return
            self.voice_channel = VoiceChannel(adapter=adapter)
            logger.info("CosyVoice adapter ready.")
        except Exception as exc:
            # Catches: ImportError (cosyvoice package missing),
            # OSError (model files not found),
            # RuntimeError (GPU or gRPC connection error).
            # Safe: degrades to mock TTS.
            logger.warning("CosyVoice unavailable (%s) — using mock TTS.", exc)
            self._setup_mock_voice()

    # ── Pipeline steps ──────────────────────────────────────────────

    def make_event(self, text: str, emotion: str, intensity: float, confidence: float) -> dict[str, Any]:
        """Step 1: Create the AudioEvent envelope from text input."""
        return _make_audio_event(
            trace_id=self.trace_id,
            text=text,
            emotion_category=emotion,
            emotion_intensity=intensity,
            emotion_confidence=confidence,
        )

    def fusion(self, event: dict[str, Any]) -> dict[str, Any]:
        """Step 2: Push event through TimeSyncWindow → Classifier → Fusion → Scene."""
        # Push into time window; force flush with sentinel if needed
        result = self.time_window.push(event)
        if result is None:
            sentinel = deepcopy(event)
            sentinel["timestamp"] = "2099-01-01T00:00:00+00:00"
            result = self.time_window.push(sentinel)
            assert result is not None, "Flush sentinel must close the window"

        classified = self.classifier.classify(result.events)
        fusion_result = self.fusion_engine.fuse(classified, window_events=result.events)

        scene = self.synthesizer.synthesize(
            trace_id=self.trace_id,
            version=1,
            classified_events=classified,
            entity_fusion_result=fusion_result,
            window_events=result.events,
            user_model_version=0,
            start_time=time.time(),
        )
        return scene

    def store_scene(self, scene: dict[str, Any]) -> bool:
        """Step 3: Store scene in hot memory."""
        # Make redis.get return this scene for subsequent retrieval
        self.hot_memory._redis.get.return_value = json.dumps(
            scene, ensure_ascii=False, default=str
        )
        return self.hot_memory.store_scene(scene)

    def update_personality(self, emotion_label: str) -> dict[str, Any]:
        """Step 4: Build dynamic personality from baseline + emotion."""
        self.dynamic = DynamicFusion.generate(
            baseline=self.baseline.to_dict(),
            emotion_label=emotion_label,
        )
        return self.dynamic

    async def decide(self, scene_summary: str, emotion: str) -> dict[str, Any]:
        """Step 5: Run the decision engine to get a response command."""
        tts_emotion = self.dynamic.get("tts_control", {}).get("emotion", emotion)
        decision = await self.engine.decide(
            scene_summary=scene_summary,
            emotion_category=emotion,
            emotion_intensity=0.0,
            tts_emotion=tts_emotion,
            scene_primary="conversation",
            trace_id=self.trace_id,
        )
        return decision

    async def speak(self, decision: dict[str, Any]) -> bool:
        """Step 6: Voice output via TTS."""
        if self.voice_channel is None:
            logger.error("Voice channel not initialized.")
            return False

        voice_text = decision.get("command", {}).get("voice_response", "")
        tts_emotion = self.dynamic.get("tts_control", {}).get("emotion", "neutral")
        self.voice_channel.set_trace_id(self.trace_id)

        result = await self.voice_channel.speak(
            text=voice_text,
            emotion=tts_emotion,
        )
        return result


# =====================================================================
#  Argument parsing
# =====================================================================

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OpenHeart — voice dialogue loop demo  v4.5.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/voice_loop_demo.py --text \"你好\" --mode mock\n"
            "  python scripts/voice_loop_demo.py --text \"你今天开心吗\" --mode real\n"
        ),
    )
    parser.add_argument(
        "--text",
        type=str,
        default="你好",
        help="Input text to simulate (default: 你好)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["mock", "real"],
        default="mock",
        help="Pipeline mode: mock (no GPU) or real (actual models)",
    )
    parser.add_argument(
        "--vram-tier",
        type=str,
        choices=["high", "low"],
        default="high",
        help="VRAM configuration tier (default: high)",
    )
    parser.add_argument(
        "--emotion",
        type=str,
        choices=list(VALID_EMOTIONS),
        default=None,
        help="Override emotion detection (default: auto-detect from text)",
    )
    return parser.parse_args(argv)


# =====================================================================
#  Main loop
# =====================================================================

async def main() -> None:
    args = parse_args()

    # ── Runtime config ────────────────────────────────────────────────
    config = RuntimeConfig(
        vram_tier=VRAMTier.HIGH if args.vram_tier == "high" else VRAMTier.LOW,
        vram_total_gb=16.0 if args.vram_tier == "high" else 7.5,
        low_vram=args.vram_tier == "low",
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

    # ── Load personality baseline ─────────────────────────────────────
    baseline_data = _load_baseline()

    # ── Build pipeline ────────────────────────────────────────────────
    pipeline = VoiceLoopPipeline(
        runtime_config=config,
        mode=args.mode,
        baseline_data=baseline_data,
    )

    # Initialize decision engine and voice channel
    await pipeline.setup_decision_and_voice()

    overall_start = time.time()

    # ── Step 1: Emotion detection ─────────────────────────────────────
    t0 = time.time()
    if args.emotion:
        emotion_info = EmotionResult(args.emotion, 0.5, 1.0)
    else:
        emotion_info = _detect_emotion(args.text)
    emotion_latency = (time.time() - t0) * 1000

    # ── Step 2: Build AudioEvent → Fusion → Scene ────────────────────
    t0 = time.time()
    event = pipeline.make_event(
        text=args.text,
        emotion=emotion_info.category,
        intensity=emotion_info.intensity,
        confidence=emotion_info.confidence,
    )
    scene = pipeline.fusion(event)
    fusion_latency = (time.time() - t0) * 1000

    # ── Step 3: Store scene in hot memory ─────────────────────────────
    t0 = time.time()
    pipeline.store_scene(scene)
    memory_latency = (time.time() - t0) * 1000

    # ── Step 4: Dynamic personality ───────────────────────────────────
    t0 = time.time()
    dynamic = pipeline.update_personality(emotion_label=emotion_info.category)
    personality_latency = (time.time() - t0) * 1000

    # ── Step 5: Decision ──────────────────────────────────────────────
    scene_summary = _make_scene_summary(args.text, emotion_info.category)
    t0 = time.time()
    decision = await pipeline.decide(
        scene_summary=scene_summary,
        emotion=emotion_info.category,
    )
    decision_latency = (time.time() - t0) * 1000
    response_text = decision.get("command", {}).get("voice_response", "")

    # ── Step 6: TTS voice output ──────────────────────────────────────
    t0 = time.time()
    spoke = await pipeline.speak(decision)
    tts_latency = (time.time() - t0) * 1000

    overall_latency = (time.time() - overall_start) * 1000

    # ── Report ────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  OpenHeart — Voice Dialogue Loop Report")
    print("=" * 60)
    print(f"  Mode:             {args.mode}")
    print(f"  VRAM Tier:        {args.vram_tier}")
    print(f"  Trace ID:         {pipeline.trace_id}")
    print(f"  Input Text:       {args.text}")
    print(f"  Detected Emotion: {emotion_info.category}  "
          f"(intensity={emotion_info.intensity:.2f}, "
          f"confidence={emotion_info.confidence:.2f})")
    print(f"  Response Text:    {response_text}")
    print(f"  Decision Safety:  {decision.get('safety_level', 'N/A')}")
    print(f"  Decision Conf:    {decision.get('confidence', 0.0):.2f}")
    print(f"  TTS Emitted:      {'yes' if spoke else 'no'}")
    print()
    print("  ── Latency Breakdown ──")
    print(f"  Emotion Detection:  {emotion_latency:8.1f} ms")
    print(f"  Fusion → Scene:     {fusion_latency:8.1f} ms")
    print(f"  HotMemory Store:    {memory_latency:8.1f} ms")
    print(f"  DynamicPersonality: {personality_latency:8.1f} ms")
    print(f"  Decision Engine:    {decision_latency:8.1f} ms")
    print(f"  TTS Synthesis:      {tts_latency:8.1f} ms")
    print(f"  ─────────────────────")
    print(f"  Total:              {overall_latency:8.1f} ms")
    print("=" * 60)
    print()


def entry_point() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    entry_point()
