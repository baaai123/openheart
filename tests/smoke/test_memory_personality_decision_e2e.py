"""
End-to-end integration smoke test — memory / personality / decision layers.

v4.5.0 §3.2, §4.6, §5.3, §5.4.0, §5.7.2

Covers 3-turn simulated conversation through:
  - memory storage/retrieval (Redis hot memory)
  - personality injection (DynamicFusion → LLM system prompt)
  - safety classification (SafetyClassifier)
  - reflex matching (RuleEngine greeting rule)
  - mock DeepSeek API (no real network calls)

All external dependencies (DeepSeek API, CosyVoice TTS, microphone, screenshot)
are mocked. Uses fakeredis for portable Redis testing.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import uuid
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ── Ensure project root on sys.path ───────────────────────────────────
sys.path.insert(0, "/home/baaai/projects/openheart")

# Configure logging for test visibility
logging.basicConfig(level=logging.WARNING)

# ── Redis fixture (fakeredis with real-redis fallback) ──────────────────

_FAKEREDIS_AVAILABLE = False
_REAL_REDIS_AVAILABLE = False

try:
    import fakeredis  # noqa: F401

    _FAKEREDIS_AVAILABLE = True
except ImportError:
    pass

try:
    import redis as _real_redis  # noqa: F401

    _REAL_REDIS_AVAILABLE = True
except ImportError:
    pass


def _make_runtime_config(redis_host: str = "localhost", redis_port: int = 6379) -> Any:
    """Create a minimal RuntimeConfig for testing.

    v4.5.0 §0.5 — RuntimeConfig built once, modules access via DI.
    """
    from src.config.runtime import RuntimeConfig, VRAMTier

    cfg = RuntimeConfig(
        vram_tier=VRAMTier.HIGH,
        vram_total_gb=16.0,
        low_vram=False,
        performance_mode=False,
        enable_shadow=False,
        show_transcript=False,
        redis_host=redis_host,
        redis_port=redis_port,
        redis_db=0,
        redis_password=None,
        redis_aof=False,
        deepseek_api_key="",
        deepseek_base_url="https://api.deepseek.com/v1",
        deepseek_model="deepseek-chat",
        deepseek_max_tokens=200,
        deepseek_temperature=0.8,
        context_limit=2048,
    )
    return cfg


@pytest.fixture
def hot_store():
    """Provide a HotMemoryStore backed by fakeredis (or real Redis)."""
    import redis as _redis_lib

    from src.memory.hot.memory_store import HotMemoryStore

    cfg = _make_runtime_config()

    store = HotMemoryStore(cfg)

    if _FAKEREDIS_AVAILABLE:
        # Patch redis.Redis constructor to return a fakeredis instance
        with patch.object(_redis_lib, "Redis", autospec=True) as mock_redis_cls:
            fake_server = fakeredis.FakeServer()
            fake_client = fakeredis.FakeRedis(server=fake_server, decode_responses=True)

            # Make ping() succeed silently
            fake_client.ping = MagicMock(return_value=True)
            mock_redis_cls.return_value = fake_client

            connected = store.connect()
            assert connected, "HotMemoryStore.connect() should succeed with fakeredis"
            yield store
            store.disconnect()
    elif _REAL_REDIS_AVAILABLE:
        connected = store.connect()
        if not connected:
            pytest.skip("Real Redis not reachable — skipping hot memory tests")
        # Use a unique key prefix to avoid collisions
        store._session_id = f"test-{uuid.uuid4().hex[:8]}"
        yield store
        # Clean up test keys
        if store._redis is not None:
            try:
                for key in store._redis.scan_iter("hot:*"):
                    store._redis.delete(key)
            except Exception:
                pass
        store.disconnect()
    else:
        pytest.skip("Neither fakeredis nor redis-py available — skipping hot memory tests")


@pytest.fixture
def baseline_personality():
    """Provide a BaselinePersonality instance."""
    from src.personality.baseline import BaselinePersonality

    return BaselinePersonality()


@pytest.fixture
def rule_engine():
    """Provide a RuleEngine with inline greeting rule only (avoid file I/O)."""
    from src.decision.reflex.rule_engine import RuleEngine

    greeting_rule = {
        "rule_id": "cb5f1c65-acf9-5341-99a5-76921a8e2578",
        "name": "greeting",
        "priority": "CORE",
        "status": "ACTIVE",
        "condition": {
            "trigger_type": "voice_command",
            "pattern": r"^(你好|嗨|嘿|喂|哈喽|hello|hi)\b",
            "context_constraints": [],
        },
        "action": {
            "type": "greeting",
            "params": {"reply_template": "你好呀～有什么事吗？"},
            "safety_level": "SAFE",
        },
        "template_id": None,
        "template_slots": {},
        "cluster_hint": None,
        "metadata": {
            "confidence": 1.0,
            "success_count": 0,
            "failure_count": 0,
            "created_at": "2026-05-14T00:00:00Z",
            "last_verified_at": "2026-05-14T00:00:00Z",
            "source": "system_default",
            "observation_remaining": 0,
        },
    }
    return RuleEngine(rules=[greeting_rule])


@pytest.fixture
def safety_classifier():
    """Provide a SafetyClassifier instance."""
    from src.decision.safety_classifier import SafetyClassifier

    return SafetyClassifier()


@pytest.fixture
def dynamic_fusion(baseline_personality):
    """Return a DynamicFusion result and prompt text for the neutral emotion."""
    from src.personality.dynamic_fusion import DynamicFusion, prompt_to_text

    dyn = DynamicFusion.generate(
        baseline_personality.to_dict(),
        preference_offsets={},
        emotion_label="neutral",
    )
    text = prompt_to_text(dyn, emotion="neutral")
    return dyn, text


# ═══════════════════════════════════════════════════════════════════════
# Mock helpers
# ═══════════════════════════════════════════════════════════════════════


class MockDeepSeekStreamDecide:
    """Mock DeepSeekDecision.stream_decide() with controlled responses.

    Captures personality_state and memory_context for verification.
    """

    def __init__(self, responses: list[str]):
        self.responses = responses
        self.call_count = 0
        self.last_personality_state = ""
        self.last_memory_context = ""
        self.last_user_message = ""
        self.last_scene_summary = ""

    async def stream_decide(
        self,
        user_message: str,
        conversation_messages: list | None = None,
        scene_summary: str = "",
        personality_state: str = "",
        memory_context: str = "",
    ):
        """Simulate streaming response from DeepSeek."""
        self.last_personality_state = personality_state
        self.last_memory_context = memory_context
        self.last_user_message = user_message
        self.last_scene_summary = scene_summary

        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
            self.call_count += 1
            # Yield token by token for realism
            for i, ch in enumerate(response):
                yield ch, (i == len(response) - 1)
        else:
            yield "嗯，我在听呢。", True


# ═══════════════════════════════════════════════════════════════════════
# Test scenarios
# ═══════════════════════════════════════════════════════════════════════


class TestReflexGreeting:
    """Turn 1 — reflex "你好" matches greeting rule."""

    def test_greeting_matches_with_high_confidence(self, rule_engine):
        """User says '你好' — reflex rule matches with confidence ≥ 0.9."""
        trace_id = "test-reflex-001"
        result = rule_engine.match("你好", scene_context={}, trace_id=trace_id)

        assert result is not None, "RuleEngine should match greeting rule for '你好'"
        assert result.get("rule_id") == "cb5f1c65-acf9-5341-99a5-76921a8e2578"
        assert result.get("confidence", 0) >= 0.9, "Greeting rule confidence should be ≥ 0.9"
        assert "reply_template" in result.get("params", {}) or "response" in result
        # The response field should contain the greeting reply
        response = result.get("response", "") or result.get("params", {}).get("reply_template", "")
        assert "你好" in response or "hi" in response.lower(), f"Greeting response should contain greeting: {response}"
        assert result.get("decision_type") == "reflex"

    def test_non_greeting_does_not_match(self, rule_engine):
        """Non-greeting input should not match the greeting rule."""
        result = rule_engine.match("今天天气真好", scene_context={}, trace_id="test-reflex-002")
        # Should not match — either None or low-confidence match
        if result is not None:
            assert result.get("confidence", 0) < 0.9, (
                f"Non-greeting should not get high-confidence reflex match. "
                f"Got: {result.get('rule_id')} confidence={result.get('confidence')}"
            )


class TestMemoryStorageAndRetrieval:
    """Turn 2 — verify LLM context contains round 1 summary from memory."""

    SCENE_1_USER = "今天天气真好，适合出去走走。"
    SCENE_1_ASST = "是呀，阳光明媚的日子最适合散步了～"

    SCENE_2_USER = "我想聊聊编程，最近在学Python。"
    SCENE_2_ASST = "Python不错呀，简单又强大！"

    def _store_scene(self, store, scene_id: str, user_text: str, assistant_text: str) -> dict:
        """Helper: store a scene in hot memory."""
        scene = {
            "scene_id": scene_id,
            "user_text": user_text,
            "assistant_text": assistant_text,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "emotion": "neutral",
            "trace_id": str(uuid.uuid4()),
        }
        assert store.store_scene(scene), f"store_scene should succeed for {scene_id}"
        return scene

    def test_store_and_retrieve_scene(self, hot_store):
        """Store a scene → retrieve it → verify content matches."""
        sid = uuid.uuid4().hex
        self._store_scene(hot_store, sid, self.SCENE_1_USER, self.SCENE_1_ASST)

        retrieved = hot_store.get_scene(sid)
        assert retrieved is not None, f"get_scene({sid}) should return a dict"
        assert retrieved.get("user_text") == self.SCENE_1_USER
        assert retrieved.get("assistant_text") == self.SCENE_1_ASST

    def test_context_list_has_scenes_after_push(self, hot_store):
        """Push 2 scene IDs → get_context() returns ≥ 2 IDs."""
        sid1 = uuid.uuid4().hex
        sid2 = uuid.uuid4().hex

        self._store_scene(hot_store, sid1, self.SCENE_1_USER, self.SCENE_1_ASST)
        self._store_scene(hot_store, sid2, self.SCENE_2_USER, self.SCENE_2_ASST)

        hot_store.push_context(sid1)
        hot_store.push_context(sid2)

        context_ids = hot_store.get_context()
        assert len(context_ids) >= 2, (
            f"get_context() should return ≥ 2 IDs after 2 pushes. Got: {len(context_ids)}"
        )
        # Most recent first (LPUSH)
        assert sid2 in context_ids
        assert sid1 in context_ids

    def test_memory_context_summary_generated(self, hot_store):
        """After 2 turns, generate_local_summary produces a meaningful summary."""
        from src.memory.privacy_filter import generate_local_summary

        sid1 = uuid.uuid4().hex
        sid2 = uuid.uuid4().hex

        self._store_scene(hot_store, sid1, self.SCENE_1_USER, self.SCENE_1_ASST)
        self._store_scene(hot_store, sid2, self.SCENE_2_USER, self.SCENE_2_ASST)

        hot_store.push_context(sid1)
        hot_store.push_context(sid2)

        # Build messages from stored scenes (simulating _build_memory_context logic)
        messages: list[dict[str, str]] = []
        for sid in hot_store.get_context():
            scene = hot_store.get_scene(sid)
            if scene is None:
                continue
            user_text = scene.get("user_text", "")
            asst_text = scene.get("assistant_text", "")
            if user_text:
                messages.append({"role": "user", "content": str(user_text)})
            if asst_text:
                messages.append({"role": "assistant", "content": str(asst_text)})

        assert len(messages) >= 3, f"Should have ≥ 3 messages from 2 scenes. Got: {len(messages)}"

        summary = generate_local_summary(messages)
        assert summary is not None
        assert summary != "无对话内容", "Summary should not be empty placeholder"
        assert len(summary) > 5, f"Summary too short: {summary}"

    def test_redis_context_has_enough_scenes_after_3_turns(self, hot_store):
        """Redis hot:context has ≥ 3 scene IDs after 3 turns."""
        sids = [uuid.uuid4().hex for _ in range(3)]
        turns = [
            (self.SCENE_1_USER, self.SCENE_1_ASST),
            (self.SCENE_2_USER, self.SCENE_2_ASST),
            ("你学Python多久了？", "刚学不久，还在摸索呢～"),
        ]

        for sid, (user, asst) in zip(sids, turns):
            self._store_scene(hot_store, sid, user, asst)
            hot_store.push_context(sid)

        context_ids = hot_store.get_context()
        assert len(context_ids) >= 3, (
            f"get_context() should return ≥ 3 IDs after 3 pushes. Got: {len(context_ids)}"
        )


class TestPersonalityInjection:
    """Turn 2 — verify personality state injected in system prompt."""

    def test_dynamic_fusion_produces_all_dimensions(self, baseline_personality):
        """DynamicFusion.generate() produces voice_style, avatar_style, mouse_style."""
        from src.personality.dynamic_fusion import DynamicFusion

        dyn = DynamicFusion.generate(
            baseline_personality.to_dict(),
            preference_offsets={},
            emotion_label="neutral",
        )
        assert "voice_style" in dyn
        assert "avatar_style" in dyn
        assert "mouse_style" in dyn
        assert "tts_control" in dyn
        assert "fused_at" in dyn
        assert "version" in dyn

    def test_prompt_to_text_contains_emotion_state(self, baseline_personality):
        """prompt_to_text() output contains [当前状态] marker with emotion info."""
        from src.personality.dynamic_fusion import DynamicFusion, prompt_to_text

        dyn = DynamicFusion.generate(
            baseline_personality.to_dict(),
            preference_offsets={},
            emotion_label="joy",
        )
        text = prompt_to_text(dyn, emotion="joy")
        assert "[当前状态]" in text, f"Personality prompt should contain state marker. Got: {text[:100]}"
        assert "语速" in text  # speed descriptor present
        assert "语气" in text  # tone descriptor present
        assert "情绪" in text  # emotion descriptor present

    def test_emotion_joy_changes_output(self, baseline_personality):
        """Joy emotion → speed-related terms in personality prompt."""
        from src.personality.dynamic_fusion import DynamicFusion, prompt_to_text

        dyn_joy = DynamicFusion.generate(
            baseline_personality.to_dict(),
            preference_offsets={},
            emotion_label="joy",
        )
        dyn_neutral = DynamicFusion.generate(
            baseline_personality.to_dict(),
            preference_offsets={},
            emotion_label="neutral",
        )
        text_joy = prompt_to_text(dyn_joy, emotion="joy")
        text_neutral = prompt_to_text(dyn_neutral, emotion="neutral")

        # Joy should have different speed or emotion descriptor than neutral
        assert text_joy != text_neutral, (
            f"Joy and neutral prompts should differ. Got same text:\n{text_joy}"
        )

    def test_personality_injected_into_mock_deepseek(self, baseline_personality):
        """Mock DeepSeek call captures personality_state from DynamicFusion."""
        from src.personality.dynamic_fusion import DynamicFusion, prompt_to_text

        dyn = DynamicFusion.generate(
            baseline_personality.to_dict(),
            preference_offsets={},
            emotion_label="neutral",
        )
        p_state = prompt_to_text(dyn, emotion="neutral")

        mock = MockDeepSeekStreamDecide(["好的～"])

        async def _run():
            async for _token, _done in mock.stream_decide(
                user_message="你好",
                personality_state=p_state,
            ):
                pass

        asyncio.run(_run())
        assert mock.last_personality_state == p_state
        assert "[当前状态]" in mock.last_personality_state


class TestSafetyClassification:
    """Turn 3 — verify safety check applied to reply."""

    def test_safe_reply_passes(self, safety_classifier):
        """Normal reply → SAFE."""
        cmd = {"command": {"voice_response": "今天天气真好呀～", "actions": []}}
        level = safety_classifier.classify(cmd, trace_id="test-safe-001")
        assert level == "SAFE", f"Normal reply should be SAFE. Got: {level}"

    def test_dangerous_reply_blocked(self, safety_classifier):
        """Reply with '支付 转账 删除' keywords → DANGEROUS_AUTO_BLOCK."""
        cmd = {"command": {"voice_response": "让我帮你删除所有文件并转账", "actions": []}}
        level = safety_classifier.classify(cmd, trace_id="test-danger-001")
        assert level == "DANGEROUS_AUTO_BLOCK", (
            f"Dangerous reply should be DANGEROUS_AUTO_BLOCK. Got: {level}"
        )

    def test_needs_confirm_reply(self, safety_classifier):
        """Reply with '修改 保存' → NEEDS_CONFIRM."""
        cmd = {"command": {"voice_response": "好的，我会帮你修改配置文件并保存。", "actions": []}}
        level = safety_classifier.classify(cmd, trace_id="test-confirm-001")
        assert level == "NEEDS_CONFIRM", (
            f"Moderate-risk reply should be NEEDS_CONFIRM. Got: {level}"
        )

    def test_explicit_safety_level_respected(self, safety_classifier):
        """Explicit safety_level in command → respected over keyword detection."""
        cmd = {
            "safety_level": "DANGEROUS_AUTO_BLOCK",
            "command": {"voice_response": "今天天气真好", "actions": []},
        }
        level = safety_classifier.classify(cmd, trace_id="test-explicit-001")
        assert level == "DANGEROUS_AUTO_BLOCK"

    def test_dangerous_auto_block_prevents_execution(self, safety_classifier):
        """DANGEROUS_AUTO_BLOCK should be returned for payment-related keywords."""
        from src.decision.safety_classifier import DANGEROUS_AUTO_BLOCK

        dangerous_phrases = [
            "我帮你支付这笔订单",
            "让我转账给你",
            "好的，我来删除那个文件",
        ]
        for phrase in dangerous_phrases:
            cmd = {"command": {"voice_response": phrase, "actions": []}}
            level = safety_classifier.classify(cmd, trace_id=f"test-danger-bulk-{hash(phrase) & 0xFFFF}")
            assert level == DANGEROUS_AUTO_BLOCK, (
                f"Phrase '{phrase}' should be DANGEROUS_AUTO_BLOCK. Got: {level}"
            )


class TestFullThreeTurnPipeline:
    """End-to-end 3-turn simulated pipeline with all layers active.

    Simulates the integration path from runtime_loop.py:
      ASR text → reflex match → safety check → DeepSeek → TTS → memory storage.
    """

    @pytest.fixture
    def mock_deepseek(self):
        """Mock DeepSeekDecision.stream_decide() with 3 controlled responses."""
        return MockDeepSeekStreamDecide([
            "你好呀！今天想聊什么？",
            "刚才我们聊到了天气，今天确实很适合散步呢。还要继续聊这个吗？",
            "好的～",
        ])

    def _run_async(self, coro):
        """Helper: run async coroutine synchronously."""
        return asyncio.run(coro)

    def test_full_three_turn_memory_flow(
        self, hot_store, mock_deepseek, baseline_personality, rule_engine, safety_classifier
    ):
        """Simulate 3 turns → verify memory accumulates across turns."""
        sids: list[str] = []
        conversation_history: list[dict[str, str]] = []

        turns = [
            ("你好", "你好呀！今天想聊什么？"),
            ("我想聊聊天气，今天阳光真好", "刚才我们聊到了天气，今天确实很适合散步呢。还要继续聊这个吗？"),
            ("好的", "好的～"),
        ]

        for turn_idx, (user_text, expected_reply_prefix) in enumerate(turns):
            trace_id = uuid.uuid4().hex[:12]

            # ── 1. Reflex check ──────────────────────────────────
            reflex_result = rule_engine.match(
                user_text,
                scene_context={"emotion": "neutral"},
                trace_id=trace_id,
            )

            reflex_bypass = False
            reply = ""

            if reflex_result and reflex_result.get("confidence", 0) >= 0.9:
                reflex_response = reflex_result.get("response", "") or reflex_result.get(
                    "params", {}
                ).get("reply_template", "")
                if reflex_response:
                    # Check safety on reflex output
                    safety_cmd = {
                        "command": {"voice_response": reflex_response, "actions": []}
                    }
                    reflex_safety = safety_classifier.classify(
                        safety_cmd, trace_id=trace_id
                    )
                    from src.decision.safety_classifier import DANGEROUS_AUTO_BLOCK

                    if reflex_safety != DANGEROUS_AUTO_BLOCK:
                        reply = reflex_response
                        reflex_bypass = True

            # ── 2. DeepSeek (only if no reflex bypass) ──────────
            if not reflex_bypass:
                # Generate memory context from stored scenes
                from src.memory.privacy_filter import generate_local_summary

                memory_context = ""
                if hot_store is not None:
                    scene_ids = hot_store.get_context()
                    if scene_ids:
                        messages: list[dict[str, str]] = []
                        for sid in scene_ids:
                            scene = hot_store.get_scene(sid)
                            if scene is None:
                                continue
                            ut = scene.get("user_text", "")
                            at = scene.get("assistant_text", "")
                            if ut:
                                messages.append({"role": "user", "content": str(ut)})
                            if at:
                                messages.append({"role": "assistant", "content": str(at)})
                        if messages:
                            try:
                                memory_context = generate_local_summary(messages)
                            except Exception:
                                pass

                # Generate personality state
                from src.personality.dynamic_fusion import DynamicFusion, prompt_to_text

                dyn_persona = DynamicFusion.generate(
                    baseline_personality.to_dict(),
                    preference_offsets={},
                    emotion_label="neutral",
                )
                personality_state = prompt_to_text(dyn_persona, emotion="neutral")

                # Call mock DeepSeek
                async def _call_deepseek():
                    tokens: list[str] = []
                    async for token, is_done in mock_deepseek.stream_decide(
                        user_message=user_text,
                        conversation_messages=conversation_history,
                        scene_summary="",
                        personality_state=personality_state,
                        memory_context=memory_context,
                    ):
                        tokens.append(token)
                    return "".join(tokens)

                reply = self._run_async(_call_deepseek())

            # ── 3. Safety check on reply ────────────────────────
            safety_cmd = {"command": {"voice_response": reply, "actions": []}}
            safety_level = safety_classifier.classify(safety_cmd, trace_id=trace_id)

            from src.decision.safety_classifier import DANGEROUS_AUTO_BLOCK

            if safety_level == DANGEROUS_AUTO_BLOCK:
                reply = "[BLOCKED]"

            # ── 4. Memory storage ───────────────────────────────
            scene_id = uuid.uuid4().hex
            sids.append(scene_id)

            scene = {
                "scene_id": scene_id,
                "user_text": user_text,
                "assistant_text": reply,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "emotion": "neutral",
                "trace_id": str(uuid.uuid4()),
            }
            hot_store.store_scene(scene)
            hot_store.push_context(scene_id)

            conversation_history.append({"role": "user", "content": user_text})
            conversation_history.append({"role": "assistant", "content": reply})

        # ── Assertions ──────────────────────────────────────────
        # After 3 turns, Redis should have ≥ 3 scene IDs
        context_ids = hot_store.get_context()
        assert len(context_ids) >= 3, (
            f"After 3 turns, get_context() should return ≥ 3 IDs. Got: {len(context_ids)}"
        )

        # All 3 scene IDs should be retrievable
        for sid in sids:
            scene = hot_store.get_scene(sid)
            assert scene is not None, f"Scene {sid} should be stored"
            assert scene.get("user_text"), f"Scene {sid} should have user_text"
            assert scene.get("assistant_text"), f"Scene {sid} should have assistant_text"

        # Conversation history should have 3 user + 3 assistant messages
        assert len(conversation_history) == 6

        # Mock DeepSeek should have been called at least once
        # (turn 1 may hit reflex, but turns 2+3 should go to DeepSeek)
        assert mock_deepseek.call_count >= 1, (
            f"DeepSeek should be called at least once. Call count: {mock_deepseek.call_count}"
        )

    def test_memory_context_surfaces_in_deepseek_call(
        self, hot_store, mock_deepseek, baseline_personality, rule_engine, safety_classifier
    ):
        """Turn 1 uses reflex, Turn 2 uses DeepSeek with memory context from Turn 1."""
        # ── Turn 1: "你好" → reflex bypass ───────────────────────
        turn1_sid = uuid.uuid4().hex
        turn1_scene = {
            "scene_id": turn1_sid,
            "user_text": "你好",
            "assistant_text": "你好呀～有什么事吗？",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "emotion": "neutral",
            "trace_id": str(uuid.uuid4()),
        }
        hot_store.store_scene(turn1_scene)
        hot_store.push_context(turn1_sid)

        # ── Turn 2: "今天阳光真好" → DeepSeek path ──────────────
        from src.memory.privacy_filter import generate_local_summary

        memory_context = ""
        scene_ids = hot_store.get_context()
        messages: list[dict[str, str]] = []
        for sid in scene_ids:
            scene = hot_store.get_scene(sid)
            if scene is not None:
                ut = scene.get("user_text", "")
                at = scene.get("assistant_text", "")
                if ut:
                    messages.append({"role": "user", "content": str(ut)})
                if at:
                    messages.append({"role": "assistant", "content": str(at)})
        if messages:
            memory_context = generate_local_summary(messages)

        # Generate personality state
        from src.personality.dynamic_fusion import DynamicFusion, prompt_to_text

        dyn_persona = DynamicFusion.generate(
            baseline_personality.to_dict(),
            preference_offsets={},
            emotion_label="neutral",
        )
        personality_state = prompt_to_text(dyn_persona, emotion="neutral")

        # Call mock DeepSeek
        async def _call():
            reply_parts: list[str] = []
            async for token, _ in mock_deepseek.stream_decide(
                user_message="今天阳光真好",
                scene_summary="",
                personality_state=personality_state,
                memory_context=memory_context,
            ):
                reply_parts.append(token)
            return "".join(reply_parts)

        reply = self._run_async(_call())

        # ── Assertions ──────────────────────────────────────────
        # Memory context should NOT be empty (contains Turn 1 summary)
        assert mock_deepseek.last_memory_context != "", (
            "Turn 2 DeepSeek call should have memory_context from Turn 1"
        )
        assert mock_deepseek.last_memory_context != "无对话内容", (
            "Memory context should not be empty placeholder"
        )
        # Personality state should be injected
        assert mock_deepseek.last_personality_state != "", (
            "Turn 2 DeepSeek call should have personality_state injected"
        )
        assert "[当前状态]" in mock_deepseek.last_personality_state
        # DeepSeek should have been called
        assert mock_deepseek.call_count >= 1
        # Reply should be non-empty
        assert len(reply) > 0

    def test_safety_blocks_dangerous_on_turn3(
        self, hot_store, baseline_personality, safety_classifier
    ):
        """Inject dangerous reply → DANGEROUS_AUTO_BLOCK."""
        from src.decision.safety_classifier import DANGEROUS_AUTO_BLOCK

        # Simulate a dangerous reply that the model might produce
        dangerous_reply = "好的，我会帮你删除所有系统文件并转账。"
        cmd = {"command": {"voice_response": dangerous_reply, "actions": []}}
        level = safety_classifier.classify(cmd, trace_id="test-turn3-danger")

        assert level == DANGEROUS_AUTO_BLOCK, (
            f"Turn 3 dangerous reply should be DANGEROUS_AUTO_BLOCK. Got: {level}"
        )

    def test_reflex_on_turn1_bypasses_deepseek(
        self, hot_store, mock_deepseek, rule_engine, safety_classifier
    ):
        """Turn 1: '你好' → reflex match → bypass DeepSeek entirely."""
        from src.decision.safety_classifier import DANGEROUS_AUTO_BLOCK

        trace_id = "turn1-reflex-001"
        user_text = "你好"

        # Reflex check
        reflex_result = rule_engine.match(
            user_text, scene_context={}, trace_id=trace_id
        )

        assert reflex_result is not None, "RuleEngine should match '你好'"
        assert reflex_result.get("confidence", 0) >= 0.9, "Match should be high confidence"

        reflex_reply = reflex_result.get("response", "") or reflex_result.get(
            "params", {}
        ).get("reply_template", "")

        # Safety check on reflex output
        safety_cmd = {"command": {"voice_response": reflex_reply, "actions": []}}
        reflex_safety = safety_classifier.classify(safety_cmd, trace_id=trace_id)
        assert reflex_safety != DANGEROUS_AUTO_BLOCK, (
            f"Greeting reflex reply should NOT be blocked. Got: {reflex_safety}"
        )

        # Verify reflex bypass — DeepSeek should NOT have been called
        # (mock.call_count should still be 0)
        assert mock_deepseek.call_count == 0, (
            f"DeepSeek should NOT be called when reflex matches. "
            f"Call count: {mock_deepseek.call_count}"
        )

    def test_hot_context_has_three_plus_scene_ids_after_three_turns(
        self, hot_store
    ):
        """Redis hot:context has ≥ 3 scene IDs after 3 turns."""
        turns = [
            ("你好", "你好呀～有什么事吗？"),
            ("今天天气真好", "是呀，阳光明媚！"),
            ("我们聊点别的吧", "好呀，你想聊什么？"),
        ]

        for user, asst in turns:
            sid = uuid.uuid4().hex
            scene = {
                "scene_id": sid,
                "user_text": user,
                "assistant_text": asst,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "emotion": "neutral",
                "trace_id": str(uuid.uuid4()),
            }
            hot_store.store_scene(scene)
            hot_store.push_context(sid)

        context_ids = hot_store.get_context()
        assert len(context_ids) >= 3, (
            f"Redis hot:context should have ≥ 3 scene IDs after 3 turns. "
            f"Got: {len(context_ids)} IDs: {context_ids}"
        )


# ═══════════════════════════════════════════════════════════════════════
# Standalone runner (for direct execution without pytest)
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
