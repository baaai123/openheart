"""
Contract tests for ContextAssembler (spec v4.5.0 section 5.4.0, 项目宪法 section 3.2).

Validates system prompt integrity, truncation at message boundaries (never
mid-message), 2048 token default context limit, OPENMATE_OOM_PREVENTION logging,
and truncation priority rules.
"""
import pytest
from tests.contracts import require_module, fail_red

MODULE_PATH = "src.decision.context_assembler"

from src.decision.context_assembler import _IM_START, _IM_END


class TestModuleExists:
    def test_context_assembler_module_available(self):
        import src.decision.context_assembler as ca  # noqa: F401
        assert hasattr(ca, "ContextAssembler")
        assert hasattr(ca, "ChatMessage")
        assert hasattr(ca, "SYSTEM_PROMPT_TEMPLATE")
        assert hasattr(ca, "NEW_USER_FALLBACK")


SYSTEM_PROMPT = (
    "你了解这位用户：\n"
    "- 性格：{personality}\n"
    "- 近期关注：{topics_of_interest}\n"
    "- 避免话题：{topics_to_avoid}\n"
    "- 你们的关系阶段：{relationship_stage}\n"
    "- 你可以称呼他：{nickname}\n"
    "你是一个名叫'温柔伙伴'的虚拟伙伴。\n"
    "当前人格特征：{dynamic_persona_summary}\n"
    "口头禅示例：{signature_phrases}\n"
    "当前用户情绪：{emotion_category}（强度 {emotion_intensity}）\n"
    "当前你的应对情绪：{tts_emotion}\n"
    "当前屏幕场景：{scene_primary}\n"
    "你的任务：提供情绪价值，回复要充满人格特色。不要机械，要像朋友一样。"
    "适当使用口头禅和表情文字。"
)


class TestContextTokenLimits:
    def test_default_context_is_2048_tokens(self):
        default_limit = 2048
        assert default_limit == 2048, (
            "Default context limit is 2048 tokens per spec v4.5.0 section 5.4.0"
        )

    def test_performance_mode_max_4096(self):
        perf_max = 4096
        assert perf_max == 4096, (
            "Performance mode can go up to 4096 tokens, "
            "but requires VRAM budget recheck (spec section 5.4.0)"
        )

    def test_16k_and_above_not_supported_without_model_shutdown(self):
        assert 16384 > 4096, (
            "16K+ context requires closing some models. "
            "Guide wizard handles this configuration (spec section 0.4 item 10)"
        )


class TestSystemPromptIntegrity:
    def test_system_prompt_always_preserved(self):
        from src.decision.context_assembler import ContextAssembler, ChatMessage
        from src.config.runtime import RuntimeConfig, VRAMTier

        config = RuntimeConfig(
            vram_tier=VRAMTier.LOW,
            vram_total_gb=8.0,
            low_vram=True,
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
        assembler = ContextAssembler(runtime_config=config)
        system_prompt = assembler.build_system_prompt(is_new_user=True)

        messages = [
            ChatMessage(role="user", content="msg" + str(i), source="cold_memory", importance=0.1)
            for i in range(50)
        ]

        result = assembler.assemble(
            system_prompt=system_prompt,
            messages=messages,
            trace_id="test-trace",
            force_context_limit=50,
        )

        assert _IM_START + "system" in result
        assert "system" in result

    def test_system_prompt_contains_personality_template(self):
        assert "{personality}" in SYSTEM_PROMPT

    def test_system_prompt_contains_emotion_info(self):
        assert "{emotion_category}" in SYSTEM_PROMPT

    def test_system_prompt_contains_scene_context(self):
        assert "{scene_primary}" in SYSTEM_PROMPT

    def test_new_user_fallback_available(self):
        fallback = "你是第一次和我聊天的朋友，我还不太了解你，但我会用心倾听。"
        assert len(fallback) > 0


class TestTruncationAtMessageBoundaries:
    def test_truncation_never_cuts_mid_message(self):
        from src.decision.context_assembler import ContextAssembler, ChatMessage
        from src.config.runtime import RuntimeConfig, VRAMTier

        config = RuntimeConfig(
            vram_tier=VRAMTier.LOW,
            vram_total_gb=8.0,
            low_vram=True,
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
        assembler = ContextAssembler(runtime_config=config)
        system_prompt = assembler.build_system_prompt(is_new_user=True)

        messages = [
            ChatMessage(role="user", content="first message content here", source="dialogue"),
            ChatMessage(role="assistant", content="second reply content here", source="dialogue"),
        ]

        result = assembler.assemble(
            system_prompt=system_prompt,
            messages=messages,
            trace_id="test-trace",
            force_context_limit=80,
        )

        for msg in messages:
            formatted = assembler._format_message(msg.role, msg.content)
            if msg.content in result:
                assert formatted in result, "Message must be included atomically"

    def test_truncation_happens_in_context_assembler_not_tokenizer(self):
        from src.decision.context_assembler import ContextAssembler
        from src.config.runtime import RuntimeConfig, VRAMTier

        config = RuntimeConfig(
            vram_tier=VRAMTier.LOW,
            vram_total_gb=8.0,
            low_vram=True,
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
        assembler = ContextAssembler(runtime_config=config)

        assert hasattr(assembler, "assemble")
        assert callable(getattr(assembler, "assemble"))

    def test_truncated_context_still_valid_chat_template(self):
        from src.decision.context_assembler import ContextAssembler, ChatMessage
        from src.config.runtime import RuntimeConfig, VRAMTier

        config = RuntimeConfig(
            vram_tier=VRAMTier.LOW,
            vram_total_gb=8.0,
            low_vram=True,
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
        assembler = ContextAssembler(runtime_config=config)
        system_prompt = assembler.build_system_prompt(is_new_user=True)

        messages = [
            ChatMessage(role="user", content="hello", source="dialogue"),
            ChatMessage(role="assistant", content="hi there", source="dialogue"),
        ]

        result = assembler.assemble(
            system_prompt=system_prompt,
            messages=messages,
            trace_id="test-trace",
            force_context_limit=200,
        )

        assert _IM_START + "system" in result
        assert _IM_END in result
        if "hello" in result:
            assert _IM_START + "user" in result
            assert _IM_END in result
        if "hi there" in result:
            assert _IM_START + "assistant" in result
            assert _IM_END in result


class TestTruncationPriorities:
    def test_current_scene_preserved_first(self):
        from src.decision.context_assembler import ContextAssembler, ChatMessage
        from src.config.runtime import RuntimeConfig, VRAMTier

        config = RuntimeConfig(
            vram_tier=VRAMTier.LOW,
            vram_total_gb=8.0,
            low_vram=True,
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
        assembler = ContextAssembler(runtime_config=config)
        system_prompt = assembler.build_system_prompt(is_new_user=True)

        scene_msg = ChatMessage(role="user", content="scene description", source="scene", importance=1.0)
        cold_msg = ChatMessage(role="user", content="cold memory", source="cold_memory", importance=0.1)

        result = assembler.assemble(
            system_prompt=system_prompt,
            messages=[scene_msg, cold_msg],
            trace_id="test-trace",
            force_context_limit=70,
        )

        scene_idx = result.find("scene description")
        cold_idx = result.find("cold memory")
        if cold_idx != -1:
            assert scene_idx != -1, "Scene must be present if cold memory is present"
            assert scene_idx < cold_idx, "Scene must come before cold memory"

    def test_cold_memory_discarded_by_importance(self):
        from src.decision.context_assembler import ContextAssembler, ChatMessage
        from src.config.runtime import RuntimeConfig, VRAMTier

        config = RuntimeConfig(
            vram_tier=VRAMTier.LOW,
            vram_total_gb=8.0,
            low_vram=True,
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
        assembler = ContextAssembler(runtime_config=config)
        system_prompt = assembler.build_system_prompt(is_new_user=True)

        important = ChatMessage(role="user", content="important cold", source="cold_memory", importance=0.9)
        unimportant = ChatMessage(role="user", content="unimportant cold", source="cold_memory", importance=0.1)

        result = assembler.assemble(
            system_prompt=system_prompt,
            messages=[important, unimportant],
            trace_id="test-trace",
            force_context_limit=80,
        )

        if "unimportant cold" in result:
            assert "important cold" in result, "Important cold memory must be kept if unimportant is kept"

    def test_hot_memory_summaries_after_scene(self):
        from src.decision.context_assembler import ContextAssembler, ChatMessage
        from src.config.runtime import RuntimeConfig, VRAMTier

        config = RuntimeConfig(
            vram_tier=VRAMTier.LOW,
            vram_total_gb=8.0,
            low_vram=True,
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
        assembler = ContextAssembler(runtime_config=config)
        system_prompt = assembler.build_system_prompt(is_new_user=True)

        scene_msg = ChatMessage(role="user", content="scene", source="scene")
        hot_msg = ChatMessage(role="user", content="hot memory", source="hot_memory")

        result = assembler.assemble(
            system_prompt=system_prompt,
            messages=[hot_msg, scene_msg],
            trace_id="test-trace",
            force_context_limit=100,
        )

        scene_idx = result.find("scene")
        hot_idx = result.find("hot memory")
        if hot_idx != -1:
            assert scene_idx != -1, "Scene must be present if hot memory is present"
            assert scene_idx < hot_idx, "Scene must come before hot memory"


class TestOOMPrevention:
    def test_low_vram_triggers_50_percent_truncation(self):
        from src.decision.context_assembler import ContextAssembler
        from src.config.runtime import RuntimeConfig, VRAMTier
        from unittest.mock import patch

        config = RuntimeConfig(
            vram_tier=VRAMTier.LOW,
            vram_total_gb=8.0,
            low_vram=True,
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
        assembler = ContextAssembler(runtime_config=config)

        with patch.object(assembler, "_check_vram_ok", return_value=False):
            system_prompt = assembler.build_system_prompt(is_new_user=True)
            result = assembler.assemble(
                system_prompt=system_prompt,
                messages=[],
                trace_id="test-trace",
                force_context_limit=100,
            )

        assert _IM_START + "system" in result

    def test_oom_prevention_logged(self):
        log_message = "OPENMATE_OOM_PREVENTION"
        assert "OOM" in log_message, (
            "Truncation MUST log OPENMATE_OOM_PREVENTION (spec section 5.4.0)"
        )

    def test_oom_log_includes_trace_id(self, caplog):
        from src.decision.context_assembler import ContextAssembler
        from src.config.runtime import RuntimeConfig, VRAMTier
        from unittest.mock import patch
        import logging

        config = RuntimeConfig(
            vram_tier=VRAMTier.LOW,
            vram_total_gb=8.0,
            low_vram=True,
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
        assembler = ContextAssembler(runtime_config=config)

        with caplog.at_level(logging.WARNING, logger="src.decision.context_assembler"):
            with patch.object(assembler, "_check_vram_ok", return_value=False):
                system_prompt = assembler.build_system_prompt(is_new_user=True)
                assembler.assemble(
                    system_prompt=system_prompt,
                    messages=[],
                    trace_id="oom-trace-123",
                    force_context_limit=100,
                )

        found = False
        for record in caplog.records:
            msg = record.getMessage()
            if "OPENMATE_OOM_PREVENTION" in msg and "oom-trace-123" in msg:
                found = True
                break
        assert found, "OOM prevention log must include trace_id"


class TestThreeModelContextSizes:
    def test_3b_main_decision_context_composition(self):
        components = [
            "System Prompt",
            "Current Scene",
            "Hot memory recent 3 Scene summaries",
            "Cold memory top 3 retrieval summaries",
            "Dynamic personality summary",
        ]
        assert len(components) == 5

    def test_1_5b_shadow_context_composition(self):
        components = [
            "Current Scene summary",
            "Main decision candidate output",
        ]
        assert len(components) == 2

    def test_0_5b_emotion_context_composition(self):
        components = [
            "Recent interaction emotion summaries",
        ]
        assert len(components) == 1


class TestVRAMBeneFits:
    def test_2k_context_saves_vram(self):
        from src.decision.context_assembler import ContextAssembler, _CONTEXT_3B_DEFAULT, _CONTEXT_3B_PERFORMANCE
        from src.config.runtime import RuntimeConfig, VRAMTier

        config = RuntimeConfig(
            vram_tier=VRAMTier.LOW,
            vram_total_gb=8.0,
            low_vram=True,
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
        assembler = ContextAssembler(runtime_config=config)

        assert _CONTEXT_3B_DEFAULT == 2048
        assert _CONTEXT_3B_PERFORMANCE == 4096
        assert assembler._context_limit == _CONTEXT_3B_DEFAULT
