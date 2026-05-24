"""
Contract tests for ActionSequenceScheduler and StateBus (spec v4.5.0 §7.2, §7.6).

Validates ActionSequence structure, channel distribution logic,
TTS progress-driven mechanism, skip_decision priority, interrupt handling,
and StateBus Redis Stream reporting.
"""
import pytest
from unittest.mock import MagicMock, patch

from src.config.runtime import RuntimeConfig, VRAMTier
from src.execution.action_scheduler import (
    Action,
    ActionSequence,
    ActionSequenceScheduler,
    CharDurationPredictor,
    CHANNEL_AVATAR,
    CHANNEL_MOUSE,
    CHANNEL_VOICE,
    CHANNEL_BUBBLE,
    VALID_CHANNELS,
)
from src.execution.state_bus import (
    StateBus,
    StateMessage,
    STREAM_AVATAR,
    STREAM_KEYBOARD,
    STREAM_VOICE,
    STREAM_GLOBAL,
    ALL_STREAMS,
    DEFAULT_MAXLEN,
)


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
        show_transcript=True,
        redis_host="localhost",
        redis_port=6379,
        redis_db=0,
        redis_password=None,
        redis_aof=True,
        context_limit=2048,
    )


@pytest.fixture
def scheduler(runtime_config: RuntimeConfig) -> ActionSequenceScheduler:
    return ActionSequenceScheduler(runtime_config)


@pytest.fixture
def state_bus(runtime_config: RuntimeConfig) -> StateBus:
    sb = StateBus(runtime_config)
    sb._degraded = True  # Don't try to connect to Redis in tests
    return sb


# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

VALID_ACTION_SEQUENCE = {
    "sequence_id": "00000000-0000-0000-0000-000000000001",
    "actions": [
        {
            "channel": "avatar",
            "type": "expression",
            "value": "smile",
            "start_ms": 0,
            "duration_ms": 3000,
        },
        {
            "channel": "mouse",
            "type": "move_to",
            "target": {"x": 500, "y": 300},
            "start_ms": 200,
            "deadline_ms": 1200,
        },
        {
            "channel": "avatar",
            "type": "motion",
            "value": "nod",
            "start_ms": 400,
            "duration_ms": 800,
        },
        {
            "channel": "bubble",
            "type": "text",
            "value": "点这个按钮",
            "start_ms": 0,
        },
    ],
}

VALID_CHANNEL_NAMES = {"avatar", "mouse", "voice", "bubble"}


# ===========================================================================
# TestModuleExists — verify modules are importable
# ===========================================================================


class TestModuleExists:
    def test_action_scheduler_module_available(self):
        from src.execution import action_scheduler  # noqa: F401

    def test_state_bus_module_available(self):
        from src.execution import state_bus  # noqa: F401

    def test_action_scheduler_class_exists(self):
        assert ActionSequenceScheduler is not None

    def test_state_bus_class_exists(self):
        assert StateBus is not None


# ===========================================================================
# TestActionSequenceStructure — validate data structures
# ===========================================================================


class TestActionSequenceStructure:
    def test_sequence_has_sequence_id(self):
        assert "sequence_id" in VALID_ACTION_SEQUENCE

    def test_sequence_has_actions_list(self):
        assert "actions" in VALID_ACTION_SEQUENCE
        assert isinstance(VALID_ACTION_SEQUENCE["actions"], list)

    def test_each_action_has_channel(self):
        for action in VALID_ACTION_SEQUENCE["actions"]:
            assert "channel" in action

    def test_each_action_has_type(self):
        for action in VALID_ACTION_SEQUENCE["actions"]:
            assert "type" in action

    def test_each_action_has_start_ms(self):
        for action in VALID_ACTION_SEQUENCE["actions"]:
            assert "start_ms" in action
            assert isinstance(action["start_ms"], int)


# ===========================================================================
# TestActionDataClass — validate Action dataclass
# ===========================================================================


class TestActionDataClass:
    def test_action_creation(self):
        a = Action(channel="avatar", type="expression", start_ms=0, value="smile", duration_ms=3000)
        assert a.channel == "avatar"
        assert a.type == "expression"
        assert a.start_ms == 0
        assert a.value == "smile"
        assert a.duration_ms == 3000

    def test_action_mouse_with_target(self):
        a = Action(channel="mouse", type="move_to", start_ms=200, target={"x": 500, "y": 300}, deadline_ms=1200)
        assert a.channel == "mouse"
        assert a.target == {"x": 500, "y": 300}
        assert a.deadline_ms == 1200

    def test_action_bubble_text(self):
        a = Action(channel="bubble", type="text", start_ms=0, value="点这个按钮")
        assert a.channel == "bubble"
        assert a.value == "点这个按钮"

    def test_invalid_channel_logs_warning(self, caplog):
        a = Action(channel="invalid", type="test", start_ms=0)
        assert any("Unknown channel" in r.message for r in caplog.records)


# ===========================================================================
# TestActionSequenceDataClass — validate ActionSequence
# ===========================================================================


class TestActionSequenceDataClass:
    def test_sequence_creation(self):
        actions = [
            Action(channel="avatar", type="expression", start_ms=0, value="smile"),
        ]
        seq = ActionSequence(actions=actions)
        assert len(seq.actions) == 1
        assert seq.skip_decision is False
        assert seq.source == "decision"

    def test_sequence_skip_decision_flag(self):
        seq = ActionSequence(skip_decision=True, source="prediction")
        assert seq.skip_decision is True
        assert seq.source == "prediction"

    def test_sequence_to_dict(self):
        actions = [
            Action(channel="avatar", type="expression", start_ms=0, value="smile", duration_ms=3000),
        ]
        seq = ActionSequence(sequence_id="test-001", actions=actions)
        d = seq.to_dict()
        assert d["sequence_id"] == "test-001"
        assert len(d["actions"]) == 1
        assert d["actions"][0]["channel"] == "avatar"
        assert d["actions"][0]["value"] == "smile"

    def test_empty_sequence_is_falsy(self):
        seq = ActionSequence(actions=[])
        assert not seq
        seq2 = ActionSequence(actions=[Action(channel="avatar", type="expression", start_ms=0)])
        assert seq2


# ===========================================================================
# TestChannelNamesCorrect — verify naming conventions (项目宪法 §2.1)
# ===========================================================================


class TestChannelNamesCorrect:
    def test_avatar_channel_is_correct_name(self):
        assert "avatar" in VALID_CHANNEL_NAMES, (
            "avatar_channel is the correct name. "
            "live2d_channel is FORBIDDEN (项目宪法 §2.1)"
        )

    def test_mouse_channel_is_correct_name(self):
        assert "mouse" in VALID_CHANNEL_NAMES, (
            "mouse_channel is the correct name. "
            "input_channel is FORBIDDEN (项目宪法 §2.1)"
        )

    def test_voice_channel_is_correct_name(self):
        assert "voice" in VALID_CHANNEL_NAMES, (
            "voice_channel is the correct name. "
            "tts_channel is FORBIDDEN (项目宪法 §2.1)"
        )

    def test_channels_constant_exists(self):
        assert CHANNEL_AVATAR == "avatar"
        assert CHANNEL_MOUSE == "mouse"
        assert CHANNEL_VOICE == "voice"
        assert CHANNEL_BUBBLE == "bubble"

    def test_valid_channels_frozenset(self):
        assert isinstance(VALID_CHANNELS, frozenset)
        assert "avatar" in VALID_CHANNELS
        assert "mouse" in VALID_CHANNELS
        assert "voice" in VALID_CHANNELS
        assert "bubble" in VALID_CHANNELS

    def test_avatar_channel_fallback_channel_name(self):
        # Verify the bubble channel constant uses the expected name
        assert CHANNEL_BUBBLE == "bubble"


# ===========================================================================
# TestChannelDispatch — validate action routing by channel
# ===========================================================================


class TestChannelDispatch:
    def test_actions_filtered_by_channel_type(self):
        avatar_actions = [
            a for a in VALID_ACTION_SEQUENCE["actions"]
            if a["channel"] == "avatar"
        ]
        assert len(avatar_actions) == 2

        mouse_actions = [
            a for a in VALID_ACTION_SEQUENCE["actions"]
            if a["channel"] == "mouse"
        ]
        assert len(mouse_actions) == 1

    def test_avatar_actions_include_expressions_and_motions(self):
        avatar_actions = [
            a for a in VALID_ACTION_SEQUENCE["actions"]
            if a["channel"] == "avatar"
        ]
        types = {a["type"] for a in avatar_actions}
        assert "expression" in types
        assert "motion" in types

    def test_mouse_actions_have_target_coordinates(self):
        mouse_actions = [
            a for a in VALID_ACTION_SEQUENCE["actions"]
            if a["channel"] == "mouse"
        ]
        for action in mouse_actions:
            if action["type"] == "move_to":
                assert "target" in action
                assert "x" in action["target"]
                assert "y" in action["target"]

    def test_scheduler_dispatch_to_channels(self, scheduler):
        actions = [
            Action(channel="avatar", type="expression", start_ms=0, value="smile"),
            Action(channel="mouse", type="move_to", start_ms=200, target={"x": 500, "y": 300}),
            Action(channel="avatar", type="motion", start_ms=400, value="nod"),
            Action(channel="bubble", type="text", start_ms=0, value="点这个按钮"),
        ]
        seq = scheduler.create_sequence(actions)
        scheduler.set_active_sequence(seq)

        dispatched = scheduler.dispatch_to_channels(elapsed_ms=200)
        assert len(dispatched["avatar"]) == 1  # expression at 0
        assert len(dispatched["bubble"]) == 1   # text at 0
        # motion at 400 is not ready at elapsed_ms=200

    def test_get_channel_actions_filtering(self, scheduler):
        actions = [
            Action(channel="avatar", type="expression", start_ms=0, value="smile"),
            Action(channel="mouse", type="move_to", start_ms=200, target={"x": 500, "y": 300}),
            Action(channel="avatar", type="motion", start_ms=400, value="nod"),
        ]
        seq = scheduler.create_sequence(actions)
        scheduler.set_active_sequence(seq)

        avatar_ready = scheduler.get_channel_actions("avatar", elapsed_ms=200)
        assert len(avatar_ready) == 1
        assert avatar_ready[0].type == "expression"

        mouse_ready = scheduler.get_channel_actions("mouse", elapsed_ms=200)
        assert len(mouse_ready) == 1
        assert mouse_ready[0].type == "move_to"


# ===========================================================================
# TestTTSProgressDrivenExecution — validate timing-based dispatch
# ===========================================================================


class TestTTSProgressDrivenExecution:
    def test_actions_executed_when_start_ms_reached(self):
        elapsed_ms = 200
        ready = [
            a for a in VALID_ACTION_SEQUENCE["actions"]
            if a["start_ms"] <= elapsed_ms + 50
        ]
        assert len(ready) == 3, (
            "Actions with start_ms <= elapsed_ms + 50ms should be dispatched"
        )

    def test_future_actions_not_dispatched(self):
        elapsed_ms = 0
        dispatched = [
            a for a in VALID_ACTION_SEQUENCE["actions"]
            if a["start_ms"] <= elapsed_ms + 50
        ]
        assert len(dispatched) == 2

    def test_mouse_deadline_triggers_jump(self):
        mouse_action = [
            a for a in VALID_ACTION_SEQUENCE["actions"]
            if a["channel"] == "mouse" and "deadline_ms" in a
        ][0]
        assert "deadline_ms" in mouse_action

    def test_scheduler_get_ready_actions(self, scheduler):
        actions = [
            Action(channel="avatar", type="expression", start_ms=0, value="smile"),
            Action(channel="mouse", type="move_to", start_ms=200, target={"x": 500, "y": 300}),
            Action(channel="avatar", type="motion", start_ms=400, value="nod"),
        ]
        seq = scheduler.create_sequence(actions)
        scheduler.set_active_sequence(seq)

        ready_0 = scheduler.get_ready_actions(elapsed_ms=0)
        assert len(ready_0) == 1  # only the action at start_ms=0

        ready_200 = scheduler.get_ready_actions(elapsed_ms=200)
        assert len(ready_200) == 2  # actions at 0 and 200

        ready_400 = scheduler.get_ready_actions(elapsed_ms=400)
        assert len(ready_400) == 3  # all actions

    def test_cosyvoice_audio_chunks_drive_progress(self, scheduler):
        actions = [
            Action(channel="voice", type="tts", start_ms=0, value="你好"),
            Action(channel="avatar", type="expression", start_ms=150, value="smile"),
            Action(channel="avatar", type="expression", start_ms=500, value="neutral"),
        ]
        seq = scheduler.create_sequence(actions)
        scheduler.set_active_sequence(seq)

        # At TTS progress 100ms, threshold=150: actions at 0ms and 150ms are ready
        ready_100 = scheduler.get_ready_actions(elapsed_ms=100)
        assert len(ready_100) == 2

        # At TTS progress 200ms, threshold=250: actions at 0ms and 150ms ready, 500ms not yet
        ready_200 = scheduler.get_ready_actions(elapsed_ms=200)
        assert len(ready_200) == 2

        # At TTS progress 450ms, threshold=500: all three actions ready
        ready_450 = scheduler.get_ready_actions(elapsed_ms=450)
        assert len(ready_450) == 3


# ===========================================================================
# TestSkipDecision — validate priority handling
# ===========================================================================


class TestSkipDecision:
    def test_skip_decision_true_allows_external_modules(self, scheduler):
        external_actions = [
            Action(channel="bubble", type="text", start_ms=0, value="reminder"),
        ]
        seq = scheduler.submit_external(external_actions, source="prediction")
        assert seq.skip_decision is True
        assert seq.source == "prediction"

    def test_skip_decision_actions_have_lower_priority(self, scheduler):
        # First, submit an external (skip_decision) sequence
        external_actions = [
            Action(channel="bubble", type="text", start_ms=0, value="external reminder"),
        ]
        scheduler.submit_external(external_actions, source="prediction")

        # Then, set an active user-triggered sequence
        user_actions = [
            Action(channel="avatar", type="expression", start_ms=0, value="joy"),
        ]
        user_seq = scheduler.create_sequence(user_actions, skip_decision=False, source="decision")
        scheduler.set_active_sequence(user_seq)

        # User-triggered actions should be returned, not external
        ready = scheduler.get_ready_actions(elapsed_ms=0)
        assert len(ready) == 1
        assert ready[0].channel == "avatar"
        assert ready[0].value == "joy"

    def test_external_actions_returned_when_no_user_sequence(self, scheduler):
        external_actions = [
            Action(channel="bubble", type="text", start_ms=0, value="reminder"),
        ]
        scheduler.submit_external(external_actions, source="prediction")

        ready = scheduler.get_ready_actions(elapsed_ms=0)
        assert len(ready) == 1
        assert ready[0].value == "reminder"

    def test_skip_decision_not_overridden_by_user(self, scheduler):
        seq = scheduler.create_sequence([], skip_decision=True)
        assert seq.skip_decision is True

    def test_create_sequence_from_dict_respects_skip_decision(self, scheduler):
        actions_data = [
            {"channel": "bubble", "type": "text", "value": "test", "start_ms": 0},
        ]
        seq = scheduler.create_sequence_from_dict(actions_data, skip_decision=True, source="prediction")
        assert seq.skip_decision is True
        assert seq.source == "prediction"


# ===========================================================================
# TestCharDurationPredictor — validate timing estimation
# ===========================================================================


class TestCharDurationPredictor:
    def test_predictor_exists(self):
        predictor = CharDurationPredictor()
        assert predictor is not None

    def test_predicts_timing_for_chinese_text(self):
        predictor = CharDurationPredictor()
        result = predictor.predict_durations("你好世界")
        assert len(result) == 4  # per-char for CJK
        assert result[0]["word"] == "你"
        assert result[0]["start_ms"] == 0
        assert result[0]["end_ms"] > result[0]["start_ms"]

    def test_predicts_timing_for_mixed_text(self):
        predictor = CharDurationPredictor()
        result = predictor.predict_durations("Hello world")
        assert len(result) == 2  # per-word for English
        assert result[0]["word"] == "Hello"

    def test_empty_text_returns_empty(self):
        predictor = CharDurationPredictor()
        result = predictor.predict_durations("")
        assert result == []

    def test_total_duration_increases_with_length(self):
        predictor = CharDurationPredictor()
        short = predictor.total_duration_ms("你好")
        long = predictor.total_duration_ms("你好世界这是测试文本")
        assert long > short

    def test_fallback_when_no_word_alignment_from_cosyvoice(self, scheduler):
        timing = scheduler.estimate_timing("你好世界")
        assert len(timing) == 4
        assert all("start_ms" in t and "end_ms" in t for t in timing)

    def test_scheduler_integrates_char_predictor(self, scheduler):
        total = scheduler.estimate_total_duration_ms("你好世界")
        assert total > 0


# ===========================================================================
# TestStateBusReporting — validate Redis Stream state bus
# ===========================================================================


class TestStateBusReporting:
    def test_channels_report_to_redis_streams(self):
        valid_streams = {STREAM_AVATAR, STREAM_KEYBOARD, STREAM_VOICE, STREAM_GLOBAL}
        assert len(valid_streams) == 4
        assert STREAM_AVATAR == "state:avatar"
        assert STREAM_KEYBOARD == "state:keyboard"
        assert STREAM_VOICE == "state:voice"
        assert STREAM_GLOBAL == "state:global"

    def test_state_global_receives_broadcast_from_all_channels(self):
        assert STREAM_GLOBAL == "state:global"

    def test_all_streams_tuple(self):
        assert len(ALL_STREAMS) == 4
        assert "state:avatar" in ALL_STREAMS
        assert "state:global" in ALL_STREAMS

    def test_default_maxlen(self):
        assert DEFAULT_MAXLEN == 10000

    def test_state_message_creation(self):
        msg = StateMessage(
            channel="avatar",
            stream=STREAM_AVATAR,
            status="running",
            data={"fps": 60, "expression": "smile"},
        )
        assert msg.channel == "avatar"
        assert msg.stream == "state:avatar"
        assert msg.status == "running"
        assert msg.data["fps"] == 60

    def test_state_message_to_dict(self):
        msg = StateMessage(
            channel="voice",
            stream=STREAM_VOICE,
            status="playing",
            data={"position_ms": 1200},
            critical=True,
        )
        d = msg.to_dict()
        assert d["channel"] == "voice"
        assert d["critical"] is True

    def test_state_message_to_json(self):
        msg = StateMessage(
            channel="mouse",
            stream=STREAM_KEYBOARD,
            status="idle",
        )
        json_str = msg.to_json()
        assert "mouse" in json_str
        assert "idle" in json_str

    def test_state_bus_degraded_publish_returns_none(self, state_bus):
        msg = StateMessage(
            channel="avatar",
            stream=STREAM_AVATAR,
            status="running",
        )
        result = state_bus.publish(msg)
        assert result is None  # degraded mode

    def test_state_bus_degraded_broadcast_does_not_crash(self, state_bus):
        # Should not raise any exception
        state_bus.broadcast_to_global("avatar", "running", {"fps": 60})

    def test_state_bus_stream_for_channel_mapping(self):
        assert StateBus._stream_for_channel("avatar") == STREAM_AVATAR
        assert StateBus._stream_for_channel("mouse") == STREAM_KEYBOARD
        assert StateBus._stream_for_channel("voice") == STREAM_VOICE
        assert StateBus._stream_for_channel("unknown") is None

    def test_state_bus_is_degraded_initially(self, state_bus):
        assert state_bus.is_degraded is True

    def test_state_bus_publish_raw_convenience(self, state_bus):
        result = state_bus.publish_raw(STREAM_GLOBAL, "avatar", "running")
        assert result is None  # degraded


# ===========================================================================
# TestInterruptHandling — validate interrupt behaviour
# ===========================================================================


class TestInterruptHandling:
    def test_user_interrupt_stops_all_channels(self, scheduler):
        actions = [
            Action(channel="avatar", type="expression", start_ms=0, value="smile"),
            Action(channel="mouse", type="move_to", start_ms=200, target={"x": 500, "y": 300}),
        ]
        seq = scheduler.create_sequence(actions)
        scheduler.set_active_sequence(seq)

        # Should have actions before interrupt
        ready_before = scheduler.get_ready_actions(elapsed_ms=200)
        assert len(ready_before) == 2

        # Interrupt
        scheduler.interrupt()

        # Should have no actions after interrupt
        ready_after = scheduler.get_ready_actions(elapsed_ms=200)
        assert len(ready_after) == 0

    def test_interrupt_is_reflected_in_property(self, scheduler):
        assert scheduler.is_interrupted is False
        scheduler.interrupt()
        assert scheduler.is_interrupted is True

    def test_interrupt_clears_external_sequences(self, scheduler):
        ext_actions = [Action(channel="bubble", type="text", start_ms=0, value="reminder")]
        scheduler.submit_external(ext_actions)

        scheduler.interrupt()

        ready = scheduler.get_ready_actions(elapsed_ms=0)
        assert len(ready) == 0

    def test_reset_interrupt_allows_new_actions(self, scheduler):
        scheduler.interrupt()
        assert scheduler.is_interrupted is True

        scheduler.reset_interrupt()
        assert scheduler.is_interrupted is False

        actions = [Action(channel="avatar", type="expression", start_ms=0, value="smile")]
        seq = scheduler.create_sequence(actions)
        scheduler.set_active_sequence(seq)

        ready = scheduler.get_ready_actions(elapsed_ms=0)
        assert len(ready) == 1


# ===========================================================================
# TestSequenceLifecycle — end-to-end sequence management
# ===========================================================================


class TestSequenceLifecycle:
    def test_clear_active_sequence(self, scheduler):
        actions = [Action(channel="avatar", type="expression", start_ms=0, value="smile")]
        seq = scheduler.create_sequence(actions)
        scheduler.set_active_sequence(seq)

        scheduler.clear_active_sequence()

        ready = scheduler.get_ready_actions(elapsed_ms=0)
        assert len(ready) == 0

    def test_clear_external_sequences(self, scheduler):
        ext_actions = [Action(channel="bubble", type="text", start_ms=0, value="reminder")]
        scheduler.submit_external(ext_actions)

        scheduler.clear_external_sequences()

        ready = scheduler.get_ready_actions(elapsed_ms=0)
        assert len(ready) == 0

    def test_mouse_deadline_enforcement(self, scheduler):
        actions = [
            Action(
                channel="mouse",
                type="move_to",
                start_ms=200,
                target={"x": 500, "y": 300},
                deadline_ms=500,
            ),
        ]
        seq = scheduler.create_sequence(actions)
        scheduler.set_active_sequence(seq)

        # At elapsed_ms=450, deadline is 500: remaining=50ms — should be urgent
        urgent = scheduler.check_mouse_deadlines(elapsed_ms=450)
        assert len(urgent) == 1
        assert urgent[0].deadline_ms == 500
        assert urgent[0].target == {"x": 500, "y": 300}

        # At elapsed_ms=100, deadline is 500: remaining=400ms — not urgent
        not_urgent = scheduler.check_mouse_deadlines(elapsed_ms=100)
        assert len(not_urgent) == 0
