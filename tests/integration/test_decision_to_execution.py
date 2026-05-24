"""
Integration tests: Personality → Decision → Execution pipeline.  v4.5.0 §4–§7

Validates the end-to-end flow:
  1. Scene + DynamicPersonality → DecisionCommand (mock 3B model)
  2. DecisionCommand → ActionSequenceScheduler creates sequence
  3. ActionSequence dispatched to avatar/mouse/voice channels
  4. skip_decision=True external actions have lower priority
  5. User interrupt stops all channels
  6. Safety level enforcement (SAFE / NEEDS_CONFIRM / DANGEROUS_AUTO_BLOCK)
  7. Emotion injection into decision sampling params

All GPU models are mocked — no GPU required.
"""
from __future__ import annotations

import uuid
from copy import deepcopy
from unittest.mock import MagicMock, PropertyMock, patch, AsyncMock

import pytest

from src.config.runtime import RuntimeConfig, VRAMTier
from src.personality.baseline import BaselinePersonality
from src.personality.dynamic_fusion import DynamicFusion
from src.personality.emotion_adj import SubjectiveEmotionClassifier
from src.personality.persona_auditor import PersonaAuditor
from src.personality.preference_shift import PreferenceShift
# v5.x: MainDecisionEngine was removed in DecisionBridge refactor.
# Tests in this file depend on it — skip the whole module if unavailable.
try:
    from src.decision.main_decision import (
        MainDecisionEngine,
        SafetyLevel,
    )
except ImportError:
    pytest.skip(
        "src.decision.main_decision removed in v5.x (DecisionBridge refactor). "
        "Remove this module or port tests to DecisionBridge.",
        allow_module_level=True,
    )
    MainDecisionEngine = None  # type: ignore[assignment]
    SafetyLevel = None  # type: ignore[assignment]
from src.decision.safety_classifier import (
    SafetyClassifier,
    SAFE,
    NEEDS_CONFIRM,
    DANGEROUS_AUTO_BLOCK,
)
from src.execution.action_scheduler import (
    Action,
    ActionSequence,
    ActionSequenceScheduler,
    CHANNEL_AVATAR,
    CHANNEL_MOUSE,
    CHANNEL_VOICE,
    CHANNEL_BUBBLE,
)
from src.execution.state_bus import StateBus, StateMessage, STREAM_GLOBAL




# ---------------------------------------------------------------------------
# Baseline fixture — matches config/baseline.json
# ---------------------------------------------------------------------------

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
        "movement_speed": {"value": 0.6, "min": 0.4, "max": 0.8, "type": "numeric"},
        "precision_mode": {"value": 0.3, "min": 0.1, "max": 0.5, "type": "numeric"},
        "hover_before_click": {"value": True, "type": "boolean"},
    },
    "signature_phrases": ["没事的～", "你做得很好呀", "需要我帮什么忙吗？"],
    "safety_constraints": [
        "never_use_profanity",
        "never_execute_destructive_action_without_confirmation",
        "always_ask_before_sending_external_data",
    ],
    "immutable": True,
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

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


@pytest.fixture
def baseline_personality() -> BaselinePersonality:
    return BaselinePersonality(config=deepcopy(VALID_BASELINE))


@pytest.fixture
def scheduler(runtime_config: RuntimeConfig) -> ActionSequenceScheduler:
    return ActionSequenceScheduler(runtime_config)


@pytest.fixture
def state_bus(runtime_config: RuntimeConfig) -> StateBus:
    sb = StateBus(runtime_config)
    sb._degraded = True  # Don't try to connect to Redis in tests
    return sb


# ---------------------------------------------------------------------------
# 1. Scene + DynamicPersonality → DecisionCommand (mock 3B model)
# ---------------------------------------------------------------------------

class TestPersonalityToDecision:
    """Validate that DynamicPersonality feeds into MainDecisionEngine.decide()."""

    def test_dynamic_personality_generation(self, baseline_personality: BaselinePersonality):
        """DynamicFusion.generate() produces a valid dynamic personality from baseline."""
        dynamic = DynamicFusion.generate(
            baseline=baseline_personality.to_dict(),
            emotion_label="joy",
        )
        assert "version" in dynamic
        assert "fused_at" in dynamic
        assert "tts_control" in dynamic
        assert dynamic["tts_control"]["emotion"] == "joy"
        for dim in ("voice_style", "avatar_style", "mouse_style"):
            assert dim in dynamic
            assert isinstance(dynamic[dim], dict)

    def test_dynamic_personality_emotion_defaults_to_neutral(
        self, baseline_personality: BaselinePersonality,
    ):
        """Unknown emotion labels fall back to neutral."""
        dynamic = DynamicFusion.generate(
            baseline=baseline_personality.to_dict(),
            emotion_label="anger",
        )
        assert dynamic["tts_control"]["emotion"] == "neutral"

    def test_main_decision_engine_constructs(self, runtime_config: RuntimeConfig):
        """MainDecisionEngine can be constructed from a RuntimeConfig."""
        engine = MainDecisionEngine(runtime_config)
        assert engine._config is runtime_config
        assert engine._model_loaded is False
        assert engine._degraded is False

    @pytest.mark.asyncio
    async def test_personality_to_decision_command(
        self, runtime_config: RuntimeConfig, baseline_personality: BaselinePersonality,
    ):
        """Scene summary + dynamic personality → structured DecisionCommand."""
        engine = MainDecisionEngine(runtime_config)

        dynamic = DynamicFusion.generate(
            baseline=baseline_personality.to_dict(),
            emotion_label="joy",
        )
        persona_summary = (
            f"情绪: {dynamic['tts_control']['emotion']}, "
            f"语速: {dynamic['voice_style']['speed']}, "
            f"表达强度: {dynamic['avatar_style']['expression_intensity']}"
        )

        engine._model_loaded = True
        engine._degraded = False
        engine._generate = AsyncMock(  # type: ignore[method-assign]
            return_value="好的，我看到你在编辑文档。需要我帮你调整格式吗？"
        )
        engine._estimate_confidence = MagicMock(  # type: ignore[method-assign]
            return_value=0.87
        )

        trace_id = uuid.uuid4().hex
        decision = await engine.decide(
            scene_summary="用户正在编辑一个Markdown文档，光标在标题处。",
            dynamic_persona_summary=persona_summary,
            signature_phrases="没事的～",
            emotion_category="joy",
            emotion_intensity=0.6,
            tts_emotion="joy",
            scene_primary="work",
            trace_id=trace_id,
        )

        assert isinstance(decision, dict)
        assert decision["decision_type"] == "voice_response"
        assert "voice_response" in decision["command"]
        assert isinstance(decision["confidence"], float)
        assert 0.0 <= decision["confidence"] <= 1.0
        assert decision["safety_level"] in (SAFE, NEEDS_CONFIRM, DANGEROUS_AUTO_BLOCK)
        assert decision["trace_id"] == trace_id
        assert decision["source"] == "main_decision_3b"

    @pytest.mark.asyncio
    async def test_decision_contain_actions_list(
        self, runtime_config: RuntimeConfig,
    ):
        """Decision command contains an actions list for the scheduler."""
        engine = MainDecisionEngine(runtime_config)
        engine._model_loaded = True
        engine._degraded = False
        engine._generate = AsyncMock(  # type: ignore[method-assign]
            return_value="我来帮你打开那个文件夹。"
        )
        engine._estimate_confidence = MagicMock(  # type: ignore[method-assign]
            return_value=0.92
        )
        decision = await engine.decide(
            scene_summary="用户指向了屏幕左侧的文件夹图标。",
            trace_id=uuid.uuid4().hex,
        )
        assert isinstance(decision["command"]["actions"], list)


# ---------------------------------------------------------------------------
# 2. DecisionCommand → ActionSequenceScheduler creates sequence
# ---------------------------------------------------------------------------

class TestDecisionToActionSequence:
    """Validate that decisions are translated into scheduled action sequences."""

    def test_create_sequence_from_actions(self, scheduler: ActionSequenceScheduler):
        """ActionSequenceScheduler.create_sequence() produces a valid ActionSequence."""
        actions = [
            Action(channel=CHANNEL_AVATAR, type="expression", value="smile", start_ms=0, duration_ms=3000),
            Action(channel=CHANNEL_VOICE, type="text", value="你好！", start_ms=0),
            Action(channel=CHANNEL_MOUSE, type="move_to", target={"x": 500, "y": 300}, start_ms=200, deadline_ms=1200),
        ]
        sequence = scheduler.create_sequence(actions=actions, text="你好！")
        assert isinstance(sequence, ActionSequence)
        assert sequence.sequence_id is not None
        assert len(sequence.actions) == 3
        assert sequence.skip_decision is False
        assert sequence.source == "decision"

    def test_create_sequence_from_dict(self, scheduler: ActionSequenceScheduler):
        """create_sequence_from_dict() parses dict data into Action objects."""
        actions_data = [
            {"channel": "avatar", "type": "expression", "value": "smile", "start_ms": 0, "duration_ms": 3000},
            {"channel": "mouse", "type": "move_to", "target": {"x": 100, "y": 200}, "start_ms": 500, "deadline_ms": 2000},
        ]
        sequence = scheduler.create_sequence_from_dict(
            actions_data=actions_data, text="测试序列"
        )
        assert len(sequence.actions) == 2
        assert sequence.actions[0].channel == CHANNEL_AVATAR
        assert sequence.actions[0].type == "expression"
        assert sequence.actions[0].value == "smile"
        assert sequence.actions[1].channel == CHANNEL_MOUSE
        assert sequence.actions[1].target == {"x": 100, "y": 200}

    def test_sequence_round_trip_to_dict(self, scheduler: ActionSequenceScheduler):
        """ActionSequence.to_dict() produces a serializable dict."""
        actions = [
            Action(channel=CHANNEL_AVATAR, type="expression", value="smile", start_ms=0, duration_ms=3000),
        ]
        sequence = scheduler.create_sequence(actions=actions)
        d = sequence.to_dict()
        assert "sequence_id" in d
        assert "actions" in d
        assert len(d["actions"]) == 1
        assert d["actions"][0]["channel"] == CHANNEL_AVATAR
        assert d["actions"][0]["type"] == "expression"


# ---------------------------------------------------------------------------
# 3. ActionSequence dispatched to avatar/mouse/voice channels
# ---------------------------------------------------------------------------

class TestDispatchToChannels:
    """Validate channel-specific dispatch from the scheduler."""

    def test_get_channel_actions_avatar(self, scheduler: ActionSequenceScheduler):
        """get_channel_actions() filters actions for avatar channel."""
        actions = [
            Action(channel=CHANNEL_AVATAR, type="expression", value="smile", start_ms=0, duration_ms=3000),
            Action(channel=CHANNEL_VOICE, type="text", value="你好", start_ms=0),
            Action(channel=CHANNEL_MOUSE, type="move_to", target={"x": 500, "y": 300}, start_ms=200),
        ]
        scheduler.set_active_sequence(scheduler.create_sequence(actions=actions))
        avatar_actions = scheduler.get_channel_actions(CHANNEL_AVATAR, elapsed_ms=0)
        assert len(avatar_actions) == 1
        assert avatar_actions[0].channel == CHANNEL_AVATAR
        assert avatar_actions[0].type == "expression"

    def test_get_channel_actions_multi_channel(self, scheduler: ActionSequenceScheduler):
        """get_ready_actions() returns all actions whose start_ms <= elapsed_ms."""
        actions = [
            Action(channel=CHANNEL_AVATAR, type="expression", value="nod", start_ms=0, duration_ms=2000),
            Action(channel=CHANNEL_VOICE, type="text", value="好的", start_ms=100),
            Action(channel=CHANNEL_MOUSE, type="click", target={"x": 400, "y": 250}, start_ms=300),
            Action(channel=CHANNEL_AVATAR, type="expression", value="smile", start_ms=500, duration_ms=3000),
        ]
        scheduler.set_active_sequence(scheduler.create_sequence(actions=actions))
        # At elapsed=200ms: first 2 actions ready
        ready = scheduler.get_ready_actions(elapsed_ms=200)
        assert len(ready) == 2
        assert ready[0].type == "expression"
        assert ready[1].type == "text"
        # At elapsed=600ms: all 4 actions ready
        ready = scheduler.get_ready_actions(elapsed_ms=600)
        assert len(ready) == 4

    def test_dispatch_to_all_channels(self, scheduler: ActionSequenceScheduler):
        """Actions target all valid channels."""
        actions = [
            Action(channel=CHANNEL_AVATAR, type="expression", value="smile", start_ms=0),
            Action(channel=CHANNEL_MOUSE, type="move_to", target={"x": 100, "y": 100}, start_ms=0),
            Action(channel=CHANNEL_VOICE, type="text", value="hello", start_ms=0),
            Action(channel=CHANNEL_BUBBLE, type="text", value="fallback", start_ms=0),
        ]
        scheduler.set_active_sequence(scheduler.create_sequence(actions=actions))
        for ch in (CHANNEL_AVATAR, CHANNEL_MOUSE, CHANNEL_VOICE, CHANNEL_BUBBLE):
            ch_actions = scheduler.get_channel_actions(ch, elapsed_ms=0)
            assert len(ch_actions) == 1
            assert ch_actions[0].channel == ch


# ---------------------------------------------------------------------------
# 4. skip_decision=True external actions have lower priority
# ---------------------------------------------------------------------------

class TestSkipDecisionPriority:
    """External sequences (skip_decision=True) yield to user-triggered sequences."""

    def test_external_sequence_submission(self, scheduler: ActionSequenceScheduler):
        """submit_external() creates a sequence with skip_decision=True."""
        actions = [
            Action(channel=CHANNEL_AVATAR, type="expression", value="idle", start_ms=0),
        ]
        ext_seq = scheduler.submit_external(actions, source="prediction")
        assert ext_seq.skip_decision is True
        assert ext_seq.source == "prediction"

    def test_user_sequence_has_priority_over_external(self, scheduler: ActionSequenceScheduler):
        """User-triggered (skip_decision=False) actions returned before external."""
        external_actions = [
            Action(channel=CHANNEL_AVATAR, type="expression", value="idle", start_ms=0),
        ]
        scheduler.submit_external(external_actions, source="prediction")

        user_actions = [
            Action(channel=CHANNEL_VOICE, type="text", value="用户触发回复", start_ms=0),
            Action(channel=CHANNEL_AVATAR, type="expression", value="smile", start_ms=100),
        ]
        user_seq = scheduler.create_sequence(actions=user_actions)
        scheduler.set_active_sequence(user_seq)

        # get_ready_actions should return user-triggered actions, not external
        ready = scheduler.get_ready_actions(elapsed_ms=0)
        assert len(ready) == 1
        assert ready[0].value == "用户触发回复"

        ready = scheduler.get_ready_actions(elapsed_ms=200)
        assert len(ready) == 2
        assert all(a.value != "idle" for a in ready)

    def test_external_served_when_no_active_sequence(self, scheduler: ActionSequenceScheduler):
        """External sequences are served when no user-triggered sequence is active."""
        external_actions = [
            Action(channel=CHANNEL_MOUSE, type="move_to", target={"x": 300, "y": 200}, start_ms=0),
        ]
        scheduler.submit_external(external_actions, source="prediction")
        ready = scheduler.get_ready_actions(elapsed_ms=0)
        assert len(ready) == 1
        assert ready[0].target == {"x": 300, "y": 200}


# ---------------------------------------------------------------------------
# 5. User interrupt stops all channels
# ---------------------------------------------------------------------------

class TestUserInterrupt:
    """User mid-speech interrupt stops dispatch to all channels."""

    def test_interrupt_clears_ready_actions(self, scheduler: ActionSequenceScheduler):
        """After interrupt, get_ready_actions returns empty list."""
        actions = [
            Action(channel=CHANNEL_VOICE, type="text", value="回复内容", start_ms=0),
        ]
        scheduler.set_active_sequence(scheduler.create_sequence(actions=actions))
        scheduler.interrupt()
        ready = scheduler.get_ready_actions(elapsed_ms=100)
        assert ready == []

    def test_interrupt_clears_external_sequences(self, scheduler: ActionSequenceScheduler):
        """Interrupt clears all external (skip_decision) sequences."""
        scheduler.submit_external(
            [Action(channel=CHANNEL_AVATAR, type="expression", value="wave", start_ms=0)],
            source="prediction",
        )
        scheduler.interrupt()
        # No active sequence and external sequences are cleared → empty
        ready = scheduler.get_ready_actions(elapsed_ms=0)
        assert ready == []

    def test_reset_interrupt_restores_dispatch(self, scheduler: ActionSequenceScheduler):
        """After reset_interrupt(), dispatch resumes normally."""
        actions = [
            Action(channel=CHANNEL_VOICE, type="text", value="再开", start_ms=0),
        ]
        scheduler.set_active_sequence(scheduler.create_sequence(actions=actions))
        scheduler.interrupt()
        assert scheduler.is_interrupted is True
        assert scheduler.get_ready_actions(elapsed_ms=0) == []

        scheduler.reset_interrupt()
        assert scheduler.is_interrupted is False
        ready = scheduler.get_ready_actions(elapsed_ms=0)
        assert len(ready) == 1
        assert ready[0].value == "再开"

    def test_channel_specific_stop(self, scheduler: ActionSequenceScheduler):
        """All channels stop on interrupt — no channel gets preferential treatment."""
        actions = [
            Action(channel=CHANNEL_AVATAR, type="expression", value="smile", start_ms=0),
            Action(channel=CHANNEL_MOUSE, type="click", target={"x": 100, "y": 100}, start_ms=50),
            Action(channel=CHANNEL_VOICE, type="text", value="讲话", start_ms=100),
        ]
        scheduler.set_active_sequence(scheduler.create_sequence(actions=actions))
        scheduler.interrupt()
        for ch in (CHANNEL_AVATAR, CHANNEL_MOUSE, CHANNEL_VOICE):
            assert scheduler.get_channel_actions(ch, elapsed_ms=200) == []


# ---------------------------------------------------------------------------
# 6. Safety level enforcement
# ---------------------------------------------------------------------------

class TestSafetyLevelEnforcement:
    """Safety levels SAFE / NEEDS_CONFIRM / DANGEROUS_AUTO_BLOCK are correctly enforced."""

    def test_classify_safe(self, runtime_config: RuntimeConfig):
        """A benign command is classified as SAFE."""
        engine = MainDecisionEngine(runtime_config)
        level = engine._classify_safety("今天天气真好，我们一起散步吧。")
        assert level == SAFE

    def test_classify_needs_confirm(self, runtime_config: RuntimeConfig):
        """Commands with modification keywords need confirmation."""
        engine = MainDecisionEngine(runtime_config)
        level = engine._classify_safety("我来帮你修改这个文件。")
        assert level == NEEDS_CONFIRM

    def test_classify_dangerous_auto_block(self, runtime_config: RuntimeConfig):
        """Commands with dangerous keywords are auto-blocked."""
        engine = MainDecisionEngine(runtime_config)
        level = engine._classify_safety("我来帮你格式化硬盘。")
        assert level == DANGEROUS_AUTO_BLOCK

    def test_safety_classifier_safe(self):
        """SafetyClassifier.classify() returns SAFE for benign commands."""
        classifier = SafetyClassifier()
        result = classifier.classify({"command": {"voice_response": "我们一起学习吧。"}})
        assert result == SAFE

    def test_safety_classifier_classify_no_explicit_level(self):
        """Without explicit safety_level, classifier checks voice_response keywords."""
        classifier = SafetyClassifier()
        result = classifier.classify({"command": {"voice_response": "我帮你保存这个文件。"}})
        assert result == NEEDS_CONFIRM

    def test_safety_classifier_classify_dangerous(self):
        """Classifier matches dangerous keywords in voice_response."""
        classifier = SafetyClassifier()
        result = classifier.classify({"command": {"voice_response": "删除所有数据。"}})
        assert result == DANGEROUS_AUTO_BLOCK

    def test_safety_classifier_classify_explicit_level(self):
        """An explicit safety_level field is respected."""
        classifier = SafetyClassifier()
        result = classifier.classify({"safety_level": "SAFE", "command": {"voice_response": "删除所有数据。"}})
        assert result == SAFE

    def test_safety_classifier_classify_actions_dangerous(self):
        """Action params containing dangerous keywords trigger DANGEROUS_AUTO_BLOCK."""
        classifier = SafetyClassifier()
        result = classifier.classify({"command": {
            "voice_response": "执行操作",
            "actions": [{"type": "mouse_click", "params": {"command": "rm -rf /"}}],
        }})
        assert result == DANGEROUS_AUTO_BLOCK

    def test_enforce_safety_safe_passthrough(self, runtime_config: RuntimeConfig):
        """SAFE level: decision command passes through unchanged."""
        engine = MainDecisionEngine(runtime_config)
        decision = {
            "decision_type": "voice_response",
            "command": {"voice_response": "好的，我来帮你。", "actions": [{"type": "voice_response"}]},
            "safety_level": SAFE,
        }
        result = engine._enforce_safety(decision, trace_id="test")
        assert result["command"]["actions"] != []  # unchanged
        assert "awaiting_confirmation" not in result.get("metadata", {})

    def test_enforce_safety_needs_confirm(self, runtime_config: RuntimeConfig):
        """NEEDS_CONFIRM level: actions are replaced with confirmation prompt."""
        engine = MainDecisionEngine(runtime_config)
        decision = {
            "decision_type": "voice_response",
            "command": {"voice_response": "我来修改文件。", "actions": [{"type": "mouse_click"}]},
            "safety_level": NEEDS_CONFIRM,
            "trace_id": "test-confirm",
        }
        result = engine._enforce_safety(decision, trace_id="test-confirm")
        assert result["command"]["actions"] == []  # cleared
        assert result["command"]["voice_response"] != ""  # confirmation text
        assert result.get("metadata", {}).get("awaiting_confirmation") is True

    def test_enforce_safety_dangerous_auto_block(self, runtime_config: RuntimeConfig):
        """DANGEROUS_AUTO_BLOCK: actions are cleared and warning is issued."""
        engine = MainDecisionEngine(runtime_config)
        decision = {
            "decision_type": "voice_response",
            "command": {"voice_response": "删除所有文件。", "actions": [{"type": "mouse_click"}]},
            "safety_level": DANGEROUS_AUTO_BLOCK,
        }
        result = engine._enforce_safety(decision, trace_id="test-block")
        assert result["command"]["actions"] == []
        assert "degraded" in result
        assert "危险" in result["command"]["voice_response"]

    def test_safety_level_three_values(self):
        """All three safety level constants are distinct strings."""
        assert SAFE == "SAFE"
        assert NEEDS_CONFIRM == "NEEDS_CONFIRM"
        assert DANGEROUS_AUTO_BLOCK == "DANGEROUS_AUTO_BLOCK"
        assert len({SAFE, NEEDS_CONFIRM, DANGEROUS_AUTO_BLOCK}) == 3


# ---------------------------------------------------------------------------
# 7. Emotion injection into decision sampling params
# ---------------------------------------------------------------------------

class TestEmotionInjection:
    """Emotion labels affect generation parameters and DynamicPersonality fusion."""

    def test_resolve_gen_params_neutral(self, runtime_config: RuntimeConfig):
        """Neutral emotion returns default sampling params."""
        engine = MainDecisionEngine(runtime_config)
        params = engine._resolve_generation_params("neutral")
        assert params["temperature"] == 0.8
        assert params["top_p"] == 0.9
        assert params["repetition_penalty"] == 1.0

    def test_resolve_gen_params_joy(self, runtime_config: RuntimeConfig):
        """Joy emotion returns higher temperature params."""
        engine = MainDecisionEngine(runtime_config)
        params = engine._resolve_generation_params("joy")
        assert params["temperature"] == 0.95
        assert params["top_p"] == 0.95

    def test_resolve_gen_params_sadness(self, runtime_config: RuntimeConfig):
        """Sadness returns lower temperature params."""
        engine = MainDecisionEngine(runtime_config)
        params = engine._resolve_generation_params("sadness")
        assert params["temperature"] == 0.65
        assert params["top_p"] == 0.85

    def test_unknown_emotion_falls_back_to_neutral(self, runtime_config: RuntimeConfig):
        """Unknown emotion labels fall back to neutral params."""
        engine = MainDecisionEngine(runtime_config)
        params = engine._resolve_generation_params("nonexistent_emotion")
        assert params["temperature"] == 0.8  # neutral default

    def test_emotion_affects_dynamic_personality(
        self, baseline_personality: BaselinePersonality,
    ):
        """Different emotions produce different dynamic personality values."""
        joy_dynamic = DynamicFusion.generate(
            baseline=baseline_personality.to_dict(),
            emotion_label="joy",
        )
        sad_dynamic = DynamicFusion.generate(
            baseline=baseline_personality.to_dict(),
            emotion_label="sadness",
        )
        neutral_dynamic = DynamicFusion.generate(
            baseline=baseline_personality.to_dict(),
            emotion_label="neutral",
        )
        # Joy has higher speed than sadness
        assert joy_dynamic["voice_style"]["speed"] > sad_dynamic["voice_style"]["speed"]
        # Joy has higher expression intensity than sadness
        assert (
            joy_dynamic["avatar_style"]["expression_intensity"]
            > sad_dynamic["avatar_style"]["expression_intensity"]
        )
        # Neutral is in between or at baseline (speed=1.0 with λ=0.2 → stays 1.0)
        assert neutral_dynamic["voice_style"]["speed"] == 1.0

    @pytest.mark.asyncio
    async def test_emotion_passes_through_decision(
        self, runtime_config: RuntimeConfig,
    ):
        """Emotion label flows through decide() and affects the output command structure."""
        engine = MainDecisionEngine(runtime_config)
        engine._model_loaded = True
        engine._degraded = False
        engine._generate = AsyncMock(  # type: ignore[method-assign]
            return_value="很开心能帮到你！(´▽｀)"
        )
        engine._estimate_confidence = MagicMock(  # type: ignore[method-assign]
            return_value=0.90
        )

        # decide() receives tts_emotion and passes to _resolve_generation_params internally
        decision = await engine.decide(
            scene_summary="用户说'谢谢你的帮助'",
            emotion_category="joy",
            tts_emotion="joy",
            trace_id=uuid.uuid4().hex,
        )
        assert decision["decision_type"] == "voice_response"
        assert isinstance(decision["confidence"], float)

    def test_emotion_params_file_loaded(self, runtime_config: RuntimeConfig):
        """Emotion params from config file are loaded at construction."""
        engine = MainDecisionEngine(runtime_config)
        assert "joy" in engine._emotion_params
        assert "sadness" in engine._emotion_params
        assert "neutral" in engine._emotion_params
        assert engine._emotion_params["neutral"]["temperature"] == 0.8
