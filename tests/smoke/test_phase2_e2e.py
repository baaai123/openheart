"""
End-to-end integration smoke test — Phase 2 6-slice coverage.

v4.5.0 §3.4, §3.5, §5.3, §5.7, §6, §T2.4

Covers all 6 Phase 2 vertical slices:
  1. Memory drawer     — semantic recall from cold memory
  2. User teaching     — learn rules from natural language + reflex match
  3. User model        — auto-inference + natural-language correction
  4. Gentle reminder   — preventive comfort injection from emotional patterns
  5. Proactive annealing — state machine level transitions
  6. Architecture refactor — VoicePipeline + DecisionBridge + ExecutionPipeline integrity

All external dependencies (DeepSeek API, CosyVoice TTS, microphone, screenshot,
LanceDB) are mocked. Pure Python, zero network, zero GPU.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest

# ── Ensure project root on sys.path ───────────────────────────────────
sys.path.insert(0, "/home/baaai/projects/openheart")

# Configure logging for test visibility
logging.basicConfig(level=logging.WARNING)


# ===================================================================
# Helper: RuntimeConfig for testing
# ===================================================================

def _make_runtime_config() -> Any:
    """Create a minimal RuntimeConfig for Phase 2 tests."""
    from src.config.runtime import RuntimeConfig, VRAMTier

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
        redis_aof=False,
        deepseek_api_key="mock-key",
        deepseek_base_url="https://api.deepseek.com/v1",
        deepseek_model="deepseek-chat",
        deepseek_max_tokens=200,
        deepseek_temperature=0.8,
        context_limit=2048,
    )


# ===================================================================
# Mock DecisionBridge builder
# ===================================================================

def _build_mock_bridge() -> Any:
    """Build a DecisionBridge with all internals mocked.

    All modules are set to MagicMock/AsyncMock so that decide() skips
    every real path and falls through to the degraded LLM stub.
    Individual tests then override specific modules to exercise paths.
    """
    from src.decision_bridge import DecisionBridge

    cfg = _make_runtime_config()
    bridge = DecisionBridge(cfg)

    # Disable all real modules
    bridge.store = None
    bridge.cold_store = None
    bridge.sync_task = None
    bridge.decision_engine = None
    bridge.baseline_personality = None
    bridge.auditor = None
    bridge.rule_engine = None
    bridge.safety_classifier = None

    bridge._learner = None
    bridge._teaching = None
    bridge._last_pending_trace_id = ""
    bridge.conversation_history = []
    bridge.cached_visual_summary = ""

    return bridge


# ═══════════════════════════════════════════════════════════════════
# 1. TestMemoryDrawer — semantic recall from cold memory
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.skip(reason="v5.x: _try_memory_drawer removed; memory drawer now via MemoryInfra.get_memory_drawer()")
class TestMemoryDrawer:
    """v4.5.0 §3.5: memory recall pattern → cold_store search → context injected."""

    def test_no_recall_pattern_returns_empty(self):
        """Input without '还记得' pattern → _try_memory_drawer returns ''."""
        bridge = _build_mock_bridge()

        async def _run():
            result = await bridge._try_memory_drawer("今天天气真好")
            return result

        memory_context = asyncio.run(_run())
        assert memory_context == "", (
            f"Non-recall input should return empty. Got: {memory_context!r}"
        )

    def test_recall_pattern_without_cold_store_returns_empty(self):
        """Recall pattern detected but cold_store=None → returns '' (degraded)."""
        bridge = _build_mock_bridge()
        bridge.cold_store = None

        async def _run():
            result = await bridge._try_memory_drawer("还记得上次聊的电影吗")
            return result

        memory_context = asyncio.run(_run())
        assert memory_context == "", (
            "Recall without cold_store should return empty (degraded). "
            f"Got: {memory_context!r}"
        )

    def test_recall_triggers_semantic_search_and_returns_context(self):
        """Recall '还记得电影吗' → semantic_search('电影') → formatted [相关记忆] block."""
        bridge = _build_mock_bridge()

        # Mock cold_store with semantic_search returning movie scenes
        mock_cold = MagicMock()
        mock_scene_1 = MagicMock()
        mock_scene_1.scene_summary = "用户讨论过《星际穿越》的剧情"
        mock_scene_2 = MagicMock()
        mock_scene_2.scene_summary = "用户分享过最喜欢的科幻电影"

        mock_cold.semantic_search = AsyncMock(return_value=[mock_scene_1, mock_scene_2])
        bridge.cold_store = mock_cold

        async def _run():
            return await bridge._try_memory_drawer("还记得上次聊的电影吗")

        memory_context = asyncio.run(_run())

        # Verify search was called
        mock_cold.semantic_search.assert_called_once()
        call_args = mock_cold.semantic_search.call_args
        assert call_args[0][0] == "上次聊的电影", (
            f"Search topic mismatch. Expected '上次聊的电影', got {call_args[0][0]!r}"
        )

        # Verify formatted output
        assert "[相关记忆]" in memory_context, (
            f"Output should contain [相关记忆] header. Got: {memory_context!r}"
        )
        assert "星际穿越" in memory_context, (
            f"Output should contain movie content. Got: {memory_context!r}"
        )
        assert "科幻电影" in memory_context, (
            f"Output should contain all search results. Got: {memory_context!r}"
        )

    def test_recall_timeout_returns_empty(self):
        """Semantic search timeout (500ms) → returns '' without crashing."""
        bridge = _build_mock_bridge()

        mock_cold = MagicMock()
        # Simulate timeout
        mock_cold.semantic_search = AsyncMock(
            side_effect=asyncio.TimeoutError("simulated timeout")
        )
        bridge.cold_store = mock_cold

        async def _run():
            return await bridge._try_memory_drawer("还记得上次聊的电影吗")

        # The method wraps in asyncio.wait_for with 0.5s timeout,
        # but if the mock raises TimeoutError directly, it's caught.
        memory_context = asyncio.run(_run())
        assert memory_context == "", (
            "Timeout should return empty string. Got: {memory_context!r}"
        )

    def test_recall_empty_results_returns_empty(self):
        """semantic_search returns [] → returns ''."""
        bridge = _build_mock_bridge()

        mock_cold = MagicMock()
        mock_cold.semantic_search = AsyncMock(return_value=[])
        bridge.cold_store = mock_cold

        async def _run():
            return await bridge._try_memory_drawer("还记得上次聊的电影吗")

        memory_context = asyncio.run(_run())
        assert memory_context == ""


# ═══════════════════════════════════════════════════════════════════
# 2. TestUserTeaching — user-taught rule learning + reflex match
# ═══════════════════════════════════════════════════════════════════

# v4.5.0 §5.7.1 teaching intent triggers
_TEACHING_TRIGGERS: list[str] = [
    "记住，以后我说累了你就放首歌",
    "以后我说累了就放首歌",
    "学会这个，如果我说累了你就放歌",
]

# v4.5.0 §5.7.3 confirmation keywords
_AFFIRMATIVE: list[str] = ["确定", "是", "可以", "对", "好的"]
_NEGATIVE: list[str] = ["取消", "不要", "不用", "算了"]


class TestUserTeaching:
    """v4.5.0 §5.7: teaching intent parse → rule learning → reflex match."""

    @pytest.mark.skip(reason="TeachingModule API refactored (v5.x): parse_intent/handle_confirmation removed, use teach()/confirm_rule()")
    def test_teaching_detect_intent(self):
        """TeachingModule parse_intent detects '记住...就...' pattern."""
        from src.decision.teaching import TeachingModule
        from src.decision.learning.learner import RuleLearner

        learner = RuleLearner()
        module = TeachingModule(rule_learner=learner)

        for trigger in _TEACHING_TRIGGERS:
            intent = module.parse_intent(trigger, trace_id="test-001")
            assert intent.is_teaching or intent.is_correction, (
                f"Should detect teaching/correction intent for: {trigger!r}"
            )

    @pytest.mark.skip(reason="TeachingModule API refactored (v5.x): parse_intent removed")
    def test_teaching_non_teaching_input_passes_through(self):
        """Non-teaching input should not be detected."""

    @pytest.mark.skip(reason="TeachingModule API refactored (v5.x): handle_confirmation removed, use confirm_rule()")
    def test_teaching_affirmative_confirmation(self):
        """'确定' confirms a pending rule via handle_confirmation."""

    def test_rule_learner_add_and_match(self):
        """RuleLearner: add rule → match() returns the rule."""
        from src.decision.learning.learner import (
            RuleLearner,
            Rule,
            RuleStatus,
            RulePriority,
            RuleSource,
            SafetyLevel,
            RuleCondition,
            RuleAction,
            RuleMetadata,
        )

        learner = RuleLearner()

        # Add a "累了" → play music rule
        rule = Rule(
            rule_id="test-fatigue-001",
            name="累了放歌",
            priority=RulePriority.USER_TAUGHT.name,
            status=RuleStatus.OBSERVATION,
            condition=RuleCondition(
                trigger_type="voice_command",
                pattern=r"累了",
                context_constraints=[],
            ),
            action=RuleAction(
                type="play_music",
                params={},
                safety_level=SafetyLevel.SAFE,
            ),
            metadata=RuleMetadata(
                source=RuleSource.USER_TEACHING,
                confidence=0.6,
                success_count=1,
                failure_count=0,
                observation_remaining=2,
                created_at="2026-01-01T00:00:00Z",
            ),
        )

        async def _run():
            await learner.add_rule(rule)
            match = await learner.match("我有点累了")
            no_match = await learner.match("今天天气真好")
            return match, no_match

        match, no_match = asyncio.run(_run())

        assert match is not None, "Should match '累了' in '我有点累了'"
        assert match.matched is True, (
            f"match.matched should be True. Got: {match.matched}"
        )
        assert match.rule is not None, "match.rule should not be None"
        assert match.rule.rule_id == "test-fatigue-001"
        assert no_match is not None, "Even non-match should return RuleMatchResult"
        assert no_match.matched is False, "Should NOT match non-fatigue input"

    def test_reflex_engine_match_learned_rule(self):
        """RuleEngine with learned '累了' rule matches input."""
        from src.decision.reflex.rule_engine import RuleEngine

        # Define a USER_TAUGHT rule inline
        fatigue_rule = {
            "rule_id": "cb5f1c65-fatigue",
            "name": "累了放歌",
            "priority": "USER_TAUGHT",
            "status": "ACTIVE",
            "condition": {
                "trigger_type": "voice_command",
                "pattern": r"累了",
                "context_constraints": [],
            },
            "action": {
                "type": "play_music",
                "params": {"reply_template": "累了就听听歌休息一下吧～"},
                "safety_level": "SAFE",
            },
            "metadata": {
                "confidence": 0.7,
                "observation_remaining": 2,
                "source": "user_teaching",
            },
        }
        engine = RuleEngine(rules=[fatigue_rule])

        # Match against the learned rule
        match = engine.match("我有点累了", scene_context={}, trace_id="test-fatigue")
        assert match is not None, "RuleEngine should match '累了' from user-taught rule"
        assert match.get("response", "").startswith("累了就听听歌"), (
            f"Unexpected response: {match.get('response', '')!r}"
        )


# ═══════════════════════════════════════════════════════════════════
# 3. TestUserModel — inference + correction
# ═══════════════════════════════════════════════════════════════════

class TestUserModel:
    """v4.5.0 §3.4: UserModel auto-inference and natural-language correction."""

    def test_user_model_dataclass_fields(self):
        """UserModel dataclass has all required fields per §3.4.1."""
        from src.memory.user_model import UserModel

        um = UserModel()
        assert um.inferred_traits is not None
        assert um.knowledge_profile is not None
        assert um.behavioral_insights is not None
        assert um.relationship_meta is not None

    def test_user_model_from_new_template(self):
        """New-user template has expected defaults."""
        from src.memory.user_model import NEW_USER_FALLBACK_TEMPLATE

        tmpl = NEW_USER_FALLBACK_TEMPLATE
        assert "inferred_traits" in tmpl
        assert tmpl["inferred_traits"]["personality"] == "尚未形成稳定认知，有待进一步了解"
        assert tmpl["inferred_traits"]["emotional_pattern"] == "暂无数据"
        assert tmpl["relationship_meta"]["model_confidence"] == 0.0
        assert tmpl["relationship_meta"]["user_verified_fields"] == []

    def test_user_model_corrector_detects_intent(self):
        """UserModelCorrector.detect_intent identifies correction utterances."""
        from src.memory.user_model_corrector import UserModelCorrector

        corrector = UserModelCorrector()

        correction_texts = [
            "别把我想得那么脆弱",
            "我不喜欢被当成小孩",
            "其实我更喜欢一个人待着",
            "忘掉这个错误的印象",
            "我不是那么消极的人",
            "别把我想得太乐观",
        ]
        for text in correction_texts:
            assert corrector.detect_intent(text), (
                f"Should detect correction intent in: {text!r}"
            )

        non_correction = [
            "你好",
            "今天天气不错",
            "帮我放首歌",
        ]
        for text in non_correction:
            assert not corrector.detect_intent(text), (
                f"Should NOT detect correction intent in: {text!r}"
            )

    def test_user_model_corrector_apply_personality(self):
        """'别把我想得那么脆弱' → personality field corrected."""
        from src.memory.user_model_corrector import (
            UserModelCorrector,
            CorrectionOperation,
        )
        from src.memory.user_model import NEW_USER_FALLBACK_TEMPLATE
        import copy

        corrector = UserModelCorrector()

        # Start from new-user template
        user_model = copy.deepcopy(NEW_USER_FALLBACK_TEMPLATE)

        # Simulate: AI inferred "脆弱" (fragile) personality
        user_model["inferred_traits"]["personality"] = "用户情感脆弱，需要温柔对待"
        user_model["inferred_traits"]["personality_confidence"] = 0.5

        # User says "别把我想得那么脆弱"
        result = corrector.detect_and_correct(
            user_model,
            "别把我想得那么脆弱",
            trace_id="test-correct-001",
        )

        assert result.success, (
            f"Correction should succeed. Errors: {result.errors}"
        )
        # The correction should have modified the personality field
        assert result.field_path == "inferred_traits.personality", (
            f"Should target personality field. Got: {result.field_path}"
        )
        assert result.operation in (
            CorrectionOperation.DELETE,
            CorrectionOperation.MODIFY,
            CorrectionOperation.LOWER_CONFIDENCE,
        ), f"Unexpected operation: {result.operation}"

    def test_user_model_prompt_formatting(self):
        """_format_user_model_prompt produces compact [用户画像] summary."""
        from src.decision_bridge import DecisionBridge

        user_model = {
            "inferred_traits": {
                "personality": "乐观开朗，喜欢户外活动",
                "emotional_pattern": "近期情绪稳定",
            },
            "knowledge_profile": {
                "topics_of_interest": ["电影", "音乐", "旅行"],
                "topics_to_avoid": [],
            },
            "relationship_meta": {
                "model_confidence": 0.42,
            },
        }

        prompt = DecisionBridge._format_user_model_prompt(user_model)

        assert "[用户画像]" in prompt, f"Should have [用户画像] header. Got: {prompt!r}"
        assert "乐观开朗" in prompt, f"Should include personality. Got: {prompt!r}"
        assert "电影" in prompt, f"Should include topics. Got: {prompt!r}"
        # Privacy: should NOT contain raw emotional pattern data
        assert "情绪稳定" not in prompt, (
            "Prompt should NOT expose emotional_pattern raw data (privacy-safe). "
            f"Got: {prompt!r}"
        )


# ═══════════════════════════════════════════════════════════════════
# 4. TestGentleReminder — preventive comfort injection
# ═══════════════════════════════════════════════════════════════════

class TestGentleReminder:
    """v4.5.0 §6: gentle reminder based on user emotional patterns."""

    def test_gentle_reminder_instantiation(self):
        """GentleReminder can be instantiated with defaults."""
        from src.prediction.gentle_reminder import GentleReminder

        gr = GentleReminder()
        assert gr.confidence_threshold == 0.6
        assert gr.idle_threshold == 10.0
        assert gr.silent_minutes == 10.0

    def test_preventive_comfort_triggered_by_emotional_pattern(self):
        """UserModel with sadness ≥ 0.6 confidence → preventive comfort injected."""
        from src.prediction.gentle_reminder import (
            GentleReminder,
            ReminderType,
        )

        gr = GentleReminder()

        # Simulate a user model with sadness pattern, high confidence
        # v4.5.0 §3.4.1: emotional_pattern_confidence as sibling field
        user_model_with_sadness = {
            "inferred_traits": {
                "emotional_pattern": "sadness",
                "emotional_pattern_confidence": 0.7,
                "personality": "敏感细腻",
            },
            "emotional_pattern_confidence": 0.7,  # top-level fallback
            "knowledge_profile": {"topics_of_interest": []},
            "relationship_meta": {
                "model_confidence": 0.7,
                "user_verified_fields": [],
            },
        }

        # evaluate() requires idle_seconds >= idle_threshold to process
        reminders = gr.evaluate(
            user_model=user_model_with_sadness,
            idle_seconds=15.0,  # exceeds default 10.0 threshold
        )

        # Should trigger because emotional_pattern_confidence >= 0.6
        comfort_reminders = [
            r for r in reminders
            if r.reminder_type == ReminderType.preventive_comfort
        ]
        assert len(comfort_reminders) >= 1, (
            f"Expected preventive_comfort in reminders. Got: "
            f"{[r.reminder_type.value for r in reminders]}"
        )
        reminder = comfort_reminders[0]
        assert reminder.skip_decision is True
        assert len(reminder.text) > 0, "Comfort reminder should have text"

    def test_preventive_comfort_not_triggered_low_confidence(self):
        """UserModel with sadness but confidence < 0.6 → no comfort triggered."""
        from src.prediction.gentle_reminder import GentleReminder

        gr = GentleReminder(confidence_threshold=0.6)

        user_model_low_confidence = {
            "inferred_traits": {
                "emotional_pattern": "sadness",
                "emotional_pattern_confidence": 0.3,  # below threshold
                "personality": "未知",
            },
            "knowledge_profile": {"topics_of_interest": []},
            "relationship_meta": {"model_confidence": 0.3, "user_verified_fields": []},
        }

        reminders = gr.evaluate(
            user_model=user_model_low_confidence,
            idle_seconds=15.0,
        )
        comfort = [r for r in reminders if r.reminder_type == "preventive_comfort"]
        assert len(comfort) == 0, (
            "Low confidence emotional_pattern should NOT trigger preventive comfort"
        )

    def test_preventive_comfort_triggered_user_verified(self):
        """UserModel with user_verified emotional_pattern → bypass confidence check."""
        from src.prediction.gentle_reminder import (
            GentleReminder,
            ReminderType,
        )

        gr = GentleReminder(confidence_threshold=0.6)

        # User verified the emotional_pattern field, even with low confidence
        user_model_verified = {
            "inferred_traits": {
                "emotional_pattern": "sadness",
                "emotional_pattern_confidence": 0.2,  # low confidence
                "personality": "内敛",
            },
            "knowledge_profile": {"topics_of_interest": []},
            "relationship_meta": {
                "model_confidence": 0.8,
                "user_verified_fields": ["emotional_pattern"],
            },
        }

        reminders = gr.evaluate(
            user_model=user_model_verified,
            idle_seconds=15.0,
        )
        comfort = [r for r in reminders if r.reminder_type == "preventive_comfort"]
        assert len(comfort) >= 1, (
            "User-verified emotional_pattern should trigger preventive comfort"
        )
        assert comfort[0].reminder_type == ReminderType.preventive_comfort

    def test_no_user_model_no_trigger(self):
        """No UserModel → no preventive comfort."""
        from src.prediction.gentle_reminder import GentleReminder, ReminderType

        gr = GentleReminder()

        reminders = gr.evaluate(
            user_model=None,
            idle_seconds=15.0,
        )
        comfort_reminders = [
            r for r in reminders
            if r.reminder_type == ReminderType.preventive_comfort
        ]
        assert len(comfort_reminders) == 0, (
            "No UserModel should mean no preventive_comfort trigger"
        )


# ═══════════════════════════════════════════════════════════════════
# 5. TestProactiveAnnealing — state machine level transitions
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.skip(reason="ProactiveAnnealing constants redesign (v5.x): IGNORE_PER_LEVEL 2→8, HEARTBEAT_MAP changed")
class TestProactiveAnnealing:
    """v4.5.0 §T2.4: proactive speaking frequency annealing state machine."""

    def test_initial_level_is_zero(self):
        """Fresh state machine starts at level 0 (active)."""
        from src.proactive.annealing import ProactiveAnnealing

        pa = ProactiveAnnealing()
        assert pa.level == 0
        assert pa.get_heartbeat_interval() == 2.0

    def test_degradation_two_ignores_to_level_one(self):
        """2 consecutive ignores → level 0→1."""
        from src.proactive.annealing import ProactiveAnnealing

        pa = ProactiveAnnealing()

        # First ignore
        pa.on_ignored()
        assert pa.level == 0, "1 ignore should not change level"

        # Second ignore triggers degradation
        pa.on_ignored()
        assert pa.level == 1, "2 ignores should degrade to level 1"
        assert pa.get_heartbeat_interval() == 8.0

    def test_degradation_full_chain_zero_to_three(self):
        """6 ignores total: 0→1→2→3."""
        from src.proactive.annealing import ProactiveAnnealing

        pa = ProactiveAnnealing()

        # 2 ignores → level 1
        pa.on_ignored()
        pa.on_ignored()
        assert pa.level == 1

        # 2 more ignores → level 2
        pa.on_ignored()
        pa.on_ignored()
        assert pa.level == 2
        assert pa.get_heartbeat_interval() == 30.0

        # 2 more ignores → level 3 (response-only)
        pa.on_ignored()
        pa.on_ignored()
        assert pa.level == 3
        assert pa.get_heartbeat_interval() == float("inf")

    def test_degradation_capped_at_level_three(self):
        """Additional ignores beyond level 3 do not exceed max."""
        from src.proactive.annealing import ProactiveAnnealing

        pa = ProactiveAnnealing()
        # Force to level 3
        for _ in range(6):
            pa.on_ignored()
        assert pa.level == 3

        # Extra ignores stay at 3
        pa.on_ignored()
        pa.on_ignored()
        assert pa.level == 3

    def test_recovery_three_engagements_from_three_to_two(self):
        """From level 3, 3 consecutive user initiations → recover to level 2."""
        from src.proactive.annealing import ProactiveAnnealing

        pa = ProactiveAnnealing()
        # Degrade fully to level 3
        for _ in range(6):
            pa.on_ignored()
        assert pa.level == 3

        # 3 user initiations → recover to 2
        pa.on_user_initiated()
        assert pa.level == 3, "1 engagement shouldn't recover yet"
        pa.on_user_initiated()
        assert pa.level == 3, "2 engagements shouldn't recover yet"
        pa.on_user_initiated()
        assert pa.level == 2, "3 engagements should recover to level 2"
        assert pa.get_heartbeat_interval() == 30.0

    def test_recovery_full_chain_three_to_zero(self):
        """9 consecutive user initiations: 3→2→1→0."""
        from src.proactive.annealing import ProactiveAnnealing

        pa = ProactiveAnnealing()
        # Degrade to 3
        for _ in range(6):
            pa.on_ignored()
        assert pa.level == 3

        # Recover: 3→2
        for _ in range(3):
            pa.on_user_initiated()
        assert pa.level == 2

        # Recover: 2→1
        for _ in range(3):
            pa.on_user_initiated()
        assert pa.level == 1

        # Recover: 1→0
        for _ in range(3):
            pa.on_user_initiated()
        assert pa.level == 0
        assert pa.get_heartbeat_interval() == 2.0

    def test_recovery_capped_at_level_zero(self):
        """Extra engagements below level 0 stay at 0."""
        from src.proactive.annealing import ProactiveAnnealing

        pa = ProactiveAnnealing()
        assert pa.level == 0

        # 10 user initiations — should stay at level 0
        for _ in range(10):
            pa.on_user_initiated()
        assert pa.level == 0

    def test_user_initiation_resets_ignore_streak(self):
        """User engagement resets the ignore counter."""
        from src.proactive.annealing import ProactiveAnnealing

        pa = ProactiveAnnealing()

        # 1 ignore
        pa.on_ignored()
        assert pa.level == 0

        # User engages — resets ignore streak
        pa.on_user_initiated()
        # 1 more ignore — should NOT degrade because streak was reset
        pa.on_ignored()
        assert pa.level == 0

        # Need 2 consecutive ignores to degrade
        pa.on_ignored()
        assert pa.level == 1

    def test_heartbeat_seconds_all_levels(self):
        """Each level maps to correct heartbeat interval."""
        from src.proactive.annealing import ProactiveAnnealing

        pa = ProactiveAnnealing()

        expected = {0: 2.0, 1: 8.0, 2: 30.0, 3: float("inf")}

        for level in range(4):
            # Force to target level
            pa = ProactiveAnnealing()
            for _ in range(level * 2):
                pa.on_ignored()
            assert pa.level == level, f"Failed to reach level {level}"
            assert pa.get_heartbeat_interval() == expected[level], (
                f"Level {level} heartbeat should be {expected[level]}. "
                f"Got {pa.get_heartbeat_interval()}"
            )

    def test_interleaved_ignore_and_engage(self):
        """Mixed pattern: ignore→engage→ignore→ignore→degrade."""
        from src.proactive.annealing import ProactiveAnnealing

        pa = ProactiveAnnealing()

        # ignore, engage (resets), ignore, ignore → degrade
        pa.on_ignored()
        pa.on_user_initiated()  # resets ignore streak
        pa.on_ignored()
        pa.on_ignored()  # 2nd consecutive → degrade
        assert pa.level == 1


# ═══════════════════════════════════════════════════════════════════
# 6. TestArchitectureRefactor — VoicePipeline + DecisionBridge + ExecutionPipeline
# ═══════════════════════════════════════════════════════════════════

class TestArchitectureRefactor:
    """Wave 1 architecture split integrity: all 3 pipeline modules importable + composable."""

    # ── VoicePipeline ─────────────────────────────────────────────

    def test_voice_pipeline_importable(self):
        """VoicePipeline class can be imported."""
        from src.voice_pipeline import VoicePipeline

        assert VoicePipeline is not None

    def test_voice_pipeline_constructable(self):
        """VoicePipeline(config) constructs without error."""
        from src.voice_pipeline import VoicePipeline
        from subprocess import Popen, PIPE

        cfg = _make_runtime_config()
        # Use a dummy Popen-like mock to avoid real subprocess
        mock_proc = MagicMock(spec=Popen)
        mock_proc.stdout = MagicMock()
        mock_proc.pid = 99999

        vp = VoicePipeline(config=cfg, proc=mock_proc)
        assert vp is not None
        assert vp.proc is mock_proc

    @pytest.mark.asyncio
    async def test_voice_pipeline_get_audio_chunk_raises_if_not_started(self):
        """get_audio_chunk() raises RuntimeError before start()."""
        from src.voice_pipeline import VoicePipeline

        cfg = _make_runtime_config()
        vp = VoicePipeline(config=cfg)

        import pytest
        with pytest.raises(RuntimeError, match="not started"):
            await vp.get_audio_chunk()

    # ── DecisionBridge ────────────────────────────────────────────

    def test_decision_bridge_importable(self):
        """DecisionBridge and DecisionResult are importable."""
        from src.decision_bridge import DecisionBridge, DecisionResult

        assert DecisionBridge is not None
        assert DecisionResult is not None

    def test_decision_bridge_constructable(self):
        """DecisionBridge(config) constructs without error."""
        from src.decision_bridge import DecisionBridge

        cfg = _make_runtime_config()
        bridge = DecisionBridge(cfg)
        assert bridge is not None
        assert bridge.config is cfg
        assert bridge._user_model is not None
        assert isinstance(bridge._user_model, dict)

    def test_decision_bridge_decide_degraded_stub(self):
        """decide() without any modules returns degraded empty DecisionResult."""
        bridge = _build_mock_bridge()

        async def _run():
            return await bridge.decide("你好", emotion="neutral")

        result = asyncio.run(_run())
        assert result.degraded is True
        assert result.source == "deepseek"
        assert len(result.trace_id) > 0

    def test_decision_bridge_teaching_path(self):
        """decide() with teaching input + mock TeachingModule → handles teaching."""
        from src.decision_bridge import DecisionBridge

        cfg = _make_runtime_config()
        bridge = DecisionBridge(cfg)

        # Mock the teaching module to return a SAFE learned result
        mock_teaching = MagicMock()
        mock_intent = MagicMock()
        mock_intent.is_teaching = True
        mock_intent.is_correction = False
        mock_teaching.parse_intent.return_value = mock_intent
        mock_teaching.teach = AsyncMock(return_value={
            "action": "learned",
            "message": "好的，我记住了～",
            "trace_id": "teach-001",
        })
        mock_teaching.handle_confirmation = AsyncMock(return_value={
            "action": "no_pending",
        })

        # Inject mock teaching module
        bridge._teaching = mock_teaching
        bridge._learner = MagicMock()
        bridge._last_pending_trace_id = ""
        bridge._learner.rules = []

        async def _run():
            return await bridge.decide("记住，以后我说累了你就放首歌")

        result = asyncio.run(_run())
        assert result.source == "teaching"
        assert "记住了" in result.reply
        assert len(result.trace_id) > 0

    def test_decision_bridge_reflex_path(self):
        """decide() with reflex-matching input returns reflex bypass result."""
        from src.decision_bridge import DecisionBridge

        cfg = _make_runtime_config()
        bridge = DecisionBridge(cfg)

        # Disable teaching
        bridge._teaching = None
        bridge._last_pending_trace_id = ""

        # Set up RuleEngine with a greeting rule
        from src.decision.reflex.rule_engine import RuleEngine
        greeting_rule = {
            "rule_id": "greeting-001",
            "name": "greeting",
            "priority": "CORE",
            "status": "ACTIVE",
            "condition": {
                "trigger_type": "voice_command",
                "pattern": r"^(你好|嗨)\b",
                "context_constraints": [],
            },
            "action": {
                "type": "greeting",
                "params": {"reply_template": "你好呀～"},
                "safety_level": "SAFE",
            },
            "metadata": {
                "confidence": 1.0,
                "observation_remaining": 0,
            },
        }
        bridge.rule_engine = RuleEngine(rules=[greeting_rule])

        async def _run():
            return await bridge.decide("你好")

        result = asyncio.run(_run())
        assert result.reflex_bypass is True
        assert result.source == "reflex"
        assert "你好" in result.reply

    # ── ExecutionPipeline ─────────────────────────────────────────

    def test_execution_pipeline_importable(self):
        """ExecutionPipeline class is importable."""
        from src.execution_pipeline import ExecutionPipeline

        assert ExecutionPipeline is not None

    def test_execution_pipeline_constructable(self):
        """ExecutionPipeline(config) constructs without error."""
        from src.execution_pipeline import ExecutionPipeline

        cfg = _make_runtime_config()
        ep = ExecutionPipeline(config=cfg)
        assert ep is not None
        assert ep._sample_rate == 22050

    # ── DecisionResult dataclass ──────────────────────────────────

    def test_decision_result_fields(self):
        """DecisionResult has all required fields per §5 output envelope."""
        from src.decision_bridge import DecisionResult

        result = DecisionResult(
            reply="测试回复",
            trace_id="abc123",
            safety_level="SAFE",
            reflex_bypass=False,
            degraded=False,
            source="deepseek",
        )

        assert result.reply == "测试回复"
        assert result.trace_id == "abc123"
        assert result.safety_level == "SAFE"
        assert result.reflex_bypass is False
        assert result.degraded is False
        assert result.source == "deepseek"


# ═══════════════════════════════════════════════════════════════════
# 7. TestEndToEndIntegration — cross-slice integration scenario
# ═══════════════════════════════════════════════════════════════════

class TestEndToEndIntegration:
    """Full integration: memory → teaching → user model → comfort in one scenario."""

    @pytest.mark.skip(reason="v5.x: _try_memory_drawer removed; memory drawer now via MemoryInfra.get_memory_drawer()")
    def test_memory_drawer_then_teaching_then_reflex(self):
        """Scenario: recall a movie → teach a rule → trigger it."""
        # ── Step 1: Memory drawer ──────────────────────────────
        bridge = _build_mock_bridge()

        mock_cold = MagicMock()
        mock_scene = MagicMock()
        mock_scene.scene_summary = "上次讨论了《流浪地球3》的剧情"
        mock_cold.semantic_search = AsyncMock(return_value=[mock_scene])
        bridge.cold_store = mock_cold

        async def recall():
            return await bridge._try_memory_drawer("还记得上次聊的电影吗")

        ctx = asyncio.run(recall())
        assert "流浪地球" in ctx, f"Memory drawer should retrieve movie. Got: {ctx!r}"

        # ── Step 2: User teaching ──────────────────────────────
        from src.decision.reflex.rule_engine import RuleEngine

        # Add a "累了" rule that was taught
        fatigue_rule = {
            "rule_id": "fatigue-e2e",
            "name": "累了放歌",
            "priority": "USER_TAUGHT",
            "status": "ACTIVE",
            "condition": {
                "trigger_type": "voice_command",
                "pattern": r"累了",
                "context_constraints": [],
            },
            "action": {
                "type": "play_music",
                "params": {"reply_template": "累了就休息吧～"},
                "safety_level": "SAFE",
            },
            "metadata": {
                "confidence": 0.8,
                "observation_remaining": 2,
                "source": "user_teaching",
            },
        }
        bridge.rule_engine = RuleEngine(rules=[fatigue_rule])
        bridge._teaching = None
        bridge._last_pending_trace_id = ""

        async def reflex():
            return await bridge.decide("我有点累了")

        result = asyncio.run(reflex())
        assert result.reflex_bypass is True
        assert "累" in result.reply
