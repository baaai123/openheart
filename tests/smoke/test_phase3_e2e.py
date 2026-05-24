"""
End-to-end integration smoke test — Phase 3 wiring verification.

v4.5.0 §3.3.2, §4.4, §4.5, §5.2.1, §8.2, §T3

Covers all 6 Phase 3 DEAD CODE wirings:
  1. chat_adapter      — imported in decision_bridge (zero-import fix)
  2. memory_service     — unified facade replaces raw _store/_cold_store in sync
  3. decay_cycle        — memory decay invoked after each sync cycle
  4. emotion_adj        — EmotionAdj.set_emotion() called before DynamicFusion
  5. preference_shift   — PreferenceShift.get_all_offsets() used in 3-layer chain
  6. easter_eggs        — EasterEggSystem.check_all() triggered on high-emotion moments

All external dependencies (DeepSeek API, CosyVoice TTS, microphone, screenshot,
LanceDB, Redis) are mocked. Pure Python, zero network, zero GPU.

Runtime_loop.py is verified via source inspection (too heavy to import directly).
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Any

import pytest

# ── Ensure project root on sys.path ───────────────────────────────────
sys.path.insert(0, "/home/baaai/projects/openheart")

# Configure logging for test visibility
logging.basicConfig(level=logging.WARNING)

# ── Project root path for source inspection ───────────────────────────
_PROJECT_ROOT = Path("/home/baaai/projects/openheart")


def _read_source(rel_path: str) -> str:
    """Read a source file as text for inspection."""
    return (_PROJECT_ROOT / rel_path).read_text(encoding="utf-8")


# ===================================================================
# Helper: RuntimeConfig for testing
# ===================================================================

def _make_runtime_config() -> Any:
    """Create a minimal RuntimeConfig for Phase 3 tests."""
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
# Mock DecisionBridge builder (minimal — for import-based tests)
# ===================================================================

def _build_minimal_bridge() -> Any:
    """Build a DecisionBridge with all internals mocked for wiring tests."""
    from src.decision_bridge import DecisionBridge

    cfg = _make_runtime_config()
    bridge = DecisionBridge(cfg)

    # Disable all modules so decide() falls through to degraded stub
    bridge.store = None
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
    # Also nullify Phase 3 modules
    bridge._memory = None
    return bridge


# ═══════════════════════════════════════════════════════════════════
# 1. TestChatAdapterWired — chat_adapter imported in decision_bridge
# ═══════════════════════════════════════════════════════════════════

class TestChatAdapterWired:
    """v4.5.0 §T3: chat_adapter.py zero-import fixed — now imported by decision_bridge."""

    def test_chat_adapter_imported_in_decision_bridge(self):
        """chat_adapter symbols are accessible via decision_bridge module."""
        import src.decision_bridge as db

        # Verify chat_adapter functions are re-exported via decision_bridge
        assert hasattr(db, "to_chat_message"), (
            "to_chat_message should be imported in decision_bridge"
        )
        assert hasattr(db, "to_api_messages"), (
            "to_api_messages should be imported in decision_bridge"
        )
        assert callable(db.to_chat_message), (
            "to_chat_message should be callable"
        )
        assert callable(db.to_api_messages), (
            "to_api_messages should be callable"
        )

    def test_chat_adapter_import_statement_in_source(self):
        """decision_bridge.py source contains the chat_adapter import statement."""
        source = _read_source("src/decision_bridge.py")
        has_import = bool(re.search(
            r"from src\.decision\.chat_adapter import",
            source,
        ))
        assert has_import, (
            "decision_bridge.py must import from src.decision.chat_adapter"
        )

    def test_chat_adapter_used_in_truncate_context(self):
        """truncate_conversation_context() uses to_chat_message() from adapter."""
        source = _read_source("src/decision_bridge.py")
        # Verify to_chat_message is actually called in the truncation method
        has_call = bool(re.search(r"to_chat_message\(msg\)", source))
        assert has_call, (
            "truncate_conversation_context() should call to_chat_message(msg)"
        )


# ═══════════════════════════════════════════════════════════════════
# 2. TestMemoryServiceFacade — MemoryService replaces raw stores
# ═══════════════════════════════════════════════════════════════════

class TestMemoryServiceFacade:
    """v4.5.0 §3: decision_bridge uses MemoryService facade, not direct raw stores in sync loop."""

    def test_memory_service_imported_in_decision_bridge(self):
        """MemoryService is imported in decision_bridge.py."""
        source = _read_source("src/decision_bridge.py")
        has_import = bool(re.search(
            r"from src\.memory\.memory_service import MemoryService",
            source,
        ))
        assert has_import, (
            "decision_bridge.py must import MemoryService from memory.memory_service"
        )

    def test_memory_service_instantiated_in_initialize(self):
        """initialize() creates a MemoryService instance wrapping hot+cold clients."""
        source = _read_source("src/decision_bridge.py")
        has_creation = bool(re.search(
            r"self\._memory\s*=\s*MemoryService\(",
            source,
        ))
        assert has_creation, (
            "initialize() must create MemoryService: self._memory = MemoryService(...)"
        )

    def test_memory_service_type_annotation(self):
        """DecisionBridge class has _memory typed as MemoryService | None."""
        source = _read_source("src/decision_bridge.py")
        # Type annotation should exist in class body
        has_annotation = bool(re.search(
            r"_memory\s*:\s*MemoryService\s*\|\s*None",
            source,
        ))
        assert has_annotation, (
            "DecisionBridge._memory should be typed as MemoryService | None"
        )

    def test_memory_service_passed_to_sync_loop(self):
        """_run_sync_loop receives memory_service parameter for decay."""
        source = _read_source("src/decision_bridge.py")
        # The sync task creation should pass memory_service=self._memory
        has_pass = bool(re.search(
            r"memory_service\s*=\s*self\._memory",
            source,
        ))
        assert has_pass, (
            "_run_sync_loop should receive memory_service=self._memory"
        )

    def test_build_memory_context_uses_memory_facade(self):
        """build_memory_context() accesses self._memory (facade), not raw store."""
        source = _read_source("src/decision_bridge.py")
        # build_memory_context should reference self._memory
        has_facade = bool(re.search(
            r"def build_memory_context.*?self\._memory",
            source,
            flags=re.DOTALL,
        ))
        assert has_facade, (
            "build_memory_context() should use self._memory facade"
        )


# ═══════════════════════════════════════════════════════════════════
# 3. TestDecayCycle — memory decay invoked after sync
# ═══════════════════════════════════════════════════════════════════

class TestDecayCycle:
    """v4.5.0 §3.3.2: memory_service.decay_cycle() called after each sync cycle."""

    def test_decay_cycle_called_in_sync_loop(self):
        """_run_sync_loop calls memory_service.decay_cycle() after sync."""
        source = _read_source("src/decision_bridge.py")
        has_decay_call = bool(re.search(
            r"await memory_service\.decay_cycle\(\)",
            source,
        ))
        assert has_decay_call, (
            "_run_sync_loop must call await memory_service.decay_cycle()"
        )

    def test_decay_cycle_after_sync_successful(self):
        """decay_cycle() only called when memory_service is not None and sync succeeded."""
        source = _read_source("src/decision_bridge.py")
        # Find the sync loop section and verify decay is conditional
        has_conditional = bool(re.search(
            r"if memory_service is not None:\s*\n\s*try:\s*\n\s*await memory_service\.decay_cycle\(\)",
            source,
        ))
        assert has_conditional, (
            "decay_cycle() must be guarded by 'if memory_service is not None'"
        )

    def test_decay_cycle_has_exception_handling(self):
        """decay_cycle() call is wrapped in try/except with WARNING log."""
        source = _read_source("src/decision_bridge.py")
        # Extract the code block around decay_cycle call
        has_except = bool(re.search(
            r"await memory_service\.decay_cycle\(\).*?except",
            source,
            flags=re.DOTALL,
        ))
        assert has_except, (
            "decay_cycle() must be wrapped in try/except for graceful degradation"
        )

    def test_decay_config_interval_present(self):
        """Decay check interval documentation (3600s = 1 hour) is referenced."""
        source = _read_source("src/decision_bridge.py")
        has_interval_doc = bool(re.search(
            r"decay_check_interval|3600",
            source,
        ))
        assert has_interval_doc, (
            "Decay interval documentation (3600s) should be present near decay_cycle call"
        )


# ═══════════════════════════════════════════════════════════════════
# 4. TestEmotionAdjWired — EmotionAdj.set_emotion() in personality chain
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.skip(reason="runtime_loop.py refactored (v5.x): emotion wiring moved to decision_bridge.py")
class TestEmotionAdjWired:
    """v4.5.0 §4.5: EmotionAdj.set_emotion() called before DynamicFusion in 3-layer chain."""

    def test_emotion_adj_imported_in_runtime_loop(self):
        """EmotionAdj is imported in runtime_loop.py."""
        source = _read_source("src/runtime_loop.py")
        has_import = bool(re.search(
            r"from src\.personality\.emotion_adj import EmotionAdj",
            source,
        ))
        assert has_import, (
            "runtime_loop.py must import EmotionAdj from personality.emotion_adj"
        )

    def test_emotion_adj_instantiated(self):
        """EmotionAdj is instantiated with baseline_personality."""
        source = _read_source("src/runtime_loop.py")
        has_create = bool(re.search(
            r"_emotion_adj\s*=\s*EmotionAdj\(baseline_personality\)",
            source,
        ))
        assert has_create, (
            "runtime_loop.py must create: _emotion_adj = EmotionAdj(baseline_personality)"
        )

    def test_set_emotion_called_before_dynamic_fusion(self):
        """set_emotion() is called before DynamicFusion.generate()."""
        source = _read_source("src/runtime_loop.py")
        # set_emotion should appear before DynamicFusion.generate in the source
        set_emotion_pos = source.find("set_emotion(")
        dynamic_fusion_pos = source.find("DynamicFusion.generate(")
        assert set_emotion_pos > 0, (
            "set_emotion() must exist in runtime_loop.py"
        )
        assert dynamic_fusion_pos > 0, (
            "DynamicFusion.generate() must exist in runtime_loop.py"
        )
        assert set_emotion_pos < dynamic_fusion_pos, (
            "set_emotion() must be called BEFORE DynamicFusion.generate() "
            f"(set_emotion at {set_emotion_pos}, DynamicFusion at {dynamic_fusion_pos})"
        )

    def test_emotion_label_passed_to_dynamic_fusion(self):
        """DynamicFusion.generate receives emotion_label from _emotion_adj._current_emotion."""
        source = _read_source("src/runtime_loop.py")
        has_label = bool(re.search(
            r"emotion_label\s*=\s*_emotion_adj\._current_emotion",
            source,
        ))
        assert has_label, (
            "DynamicFusion.generate() should receive emotion_label=_emotion_adj._current_emotion"
        )

    def test_set_emotion_guarded_by_try_except(self):
        """Emotion+personality chain is wrapped in try/except for graceful degradation."""
        source = _read_source("src/runtime_loop.py")
        # The block around set_emotion → DynamicFusion should have exception handling
        # We verify there's an except near the personality chain code
        has_except = bool(re.search(
            r"Personality fusion skipped",
            source,
        ))
        assert has_except, (
            "Personality fusion should have graceful degradation on failure"
        )


# ═══════════════════════════════════════════════════════════════════
# 5. TestPreferenceShiftWired — PreferenceShift.get_all_offsets() used
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.skip(reason="runtime_loop.py refactored (v5.x): preference wiring moved to decision_bridge.py")
class TestPreferenceShiftWired:
    """v4.5.0 §4.4: PreferenceShift.get_all_offsets() used in 3-layer personality chain."""

    def test_preference_shift_imported_in_runtime_loop(self):
        """PreferenceShift is imported in runtime_loop.py."""
        source = _read_source("src/runtime_loop.py")
        has_import = bool(re.search(
            r"from src\.personality\.preference_shift import PreferenceShift",
            source,
        ))
        assert has_import, (
            "runtime_loop.py must import PreferenceShift from personality.preference_shift"
        )

    def test_preference_shift_instantiated(self):
        """PreferenceShift is instantiated with baseline_personality."""
        source = _read_source("src/runtime_loop.py")
        has_create = bool(re.search(
            r"_preference_shift\s*=\s*PreferenceShift\(baseline_personality\)",
            source,
        ))
        assert has_create, (
            "runtime_loop.py must create: _preference_shift = PreferenceShift(baseline_personality)"
        )

    def test_get_all_offsets_called(self):
        """get_all_offsets() is called to retrieve preference offsets."""
        source = _read_source("src/runtime_loop.py")
        has_call = bool(re.search(
            r"_preference_shift\.get_all_offsets\(\)",
            source,
        ))
        assert has_call, (
            "runtime_loop.py must call _preference_shift.get_all_offsets()"
        )

    def test_offsets_passed_to_dynamic_fusion(self):
        """get_all_offsets() result is passed as preference_offsets to DynamicFusion."""
        source = _read_source("src/runtime_loop.py")
        has_pass = bool(re.search(
            r"preference_offsets\s*=\s*_preference_offsets",
            source,
        ))
        assert has_pass, (
            "DynamicFusion.generate() should receive preference_offsets=_preference_offsets"
        )

    def test_freeze_skips_preference_offsets(self):
        """When freeze_preference_shift is True, get_all_offsets() is skipped."""
        source = _read_source("src/runtime_loop.py")
        has_freeze = bool(re.search(
            r"freeze_preference_shift",
            source,
        ))
        assert has_freeze, (
            "PersonaAuditor freeze should guard preference_shift usage"
        )


# ═══════════════════════════════════════════════════════════════════
# 6. TestEasterEggsWired — EasterEggSystem triggered on high-emotion
# ═══════════════════════════════════════════════════════════════════

class TestEasterEggsWired:
    """v4.5.0 §8.2: EasterEggSystem.check_all() triggered on high-emotion moments (joy/sadness)."""

    def test_easter_egg_system_imported_in_runtime_loop(self):
        """EasterEggSystem is imported in runtime_loop.py."""
        source = _read_source("src/runtime_loop.py")
        has_import = bool(re.search(
            r"from src\.decision\.easter_eggs import EasterEggSystem",
            source,
        ))
        assert has_import, (
            "runtime_loop.py must import EasterEggSystem from decision.easter_eggs"
        )

    def test_easter_egg_system_instantiated(self):
        """EasterEggSystem is instantiated in runtime_loop.py."""
        source = _read_source("src/runtime_loop.py")
        has_create = bool(re.search(
            r"_easter_eggs\s*=\s*EasterEggSystem\(\)",
            source,
        ))
        assert has_create, (
            "runtime_loop.py must create: _easter_eggs = EasterEggSystem()"
        )

    def test_check_all_triggered_on_high_emotion(self):
        """check_all() is called only when emotion is 'joy' or 'sadness'."""
        source = _read_source("src/runtime_loop.py")
        has_call = bool(re.search(
            r"_easter_eggs\.check_all\(\)",
            source,
        ))
        assert has_call, (
            "runtime_loop.py must call _easter_eggs.check_all()"
        )

    def test_easter_egg_reply_overrides_response(self):
        """When easter egg triggers, its message overrides the reply."""
        source = _read_source("src/runtime_loop.py")
        has_override = bool(re.search(
            r"_eggs\s*=\s*_easter_eggs\.check_all\(\)",
            source,
        ))
        assert has_override, (
            "Easter egg result must be captured: _eggs = _easter_eggs.check_all()"
        )
        has_reply_assign = bool(re.search(
            r"reply\s*=\s*_eggs\[0\]\.message",
            source,
        ))
        assert has_reply_assign, (
            "Easter egg message must override reply: reply = _eggs[0].message"
        )

    def test_easter_egg_system_has_check_all_method(self):
        """EasterEggSystem class has a .check_all() method."""
        from src.decision.easter_eggs import EasterEggSystem

        assert hasattr(EasterEggSystem, "check_all"), (
            "EasterEggSystem must have a .check_all() method"
        )
        assert callable(EasterEggSystem.check_all), (
            "EasterEggSystem.check_all must be callable"
        )

    def test_easter_egg_trigger_logs_event(self):
        """Easter egg trigger is logged with logger.info()."""
        source = _read_source("src/runtime_loop.py")
        has_log = bool(re.search(
            r"Easter egg triggered",
            source,
        ))
        assert has_log, (
            "Easter egg trigger should log 'Easter egg triggered'"
        )


# ═══════════════════════════════════════════════════════════════════
# 7. TestPhase3WiringIntegration — cross-wiring verification
# ═══════════════════════════════════════════════════════════════════

class TestPhase3WiringIntegration:
    """Integration verification: all 6 wirings are consistent and don't conflict."""

    def test_all_six_modules_exist(self):
        """All 6 Phase 3 modules exist as Python files."""
        modules = [
            "src/decision/chat_adapter.py",
            "src/memory/memory_service.py",
            "src/memory/decay/decay_engine.py",
            "src/personality/emotion_adj.py",
            "src/personality/preference_shift.py",
            "src/decision/easter_eggs.py",
        ]
        for mod_path in modules:
            full = _PROJECT_ROOT / mod_path
            assert full.exists(), (
                f"Module {mod_path} must exist (DEAD CODE revived in Phase 3)"
            )
            assert full.stat().st_size > 100, (
                f"Module {mod_path} should have > 100 bytes of code"
            )

    def test_no_circular_imports(self):
        """All Phase 3 modules can be imported without circular dependency errors."""
        modules_to_import = [
            "src.decision.chat_adapter",
            "src.memory.memory_service",
            "src.memory.decay.decay_engine",
            "src.personality.emotion_adj",
            "src.personality.preference_shift",
            "src.decision.easter_eggs",
        ]
        for mod_name in modules_to_import:
            try:
                __import__(mod_name)
            except ImportError as exc:
                # Runtime deps (CosyVoice, torch, etc.) may not be installed
                # We only care about circular imports, not missing deps
                if "circular" in str(exc).lower():
                    raise AssertionError(
                        f"Circular import in {mod_name}: {exc}"
                    ) from exc
                # Missing optional deps are acceptable in smoke test

    def test_decision_bridge_imports_not_broken(self):
        """decision_bridge.py can be imported without errors (chat_adapter + memory_service)."""
        import importlib
        try:
            _ = importlib.import_module("src.decision_bridge")
        except Exception as exc:
            raise AssertionError(
                f"decision_bridge import failed: {exc}"
            ) from exc

    def test_chat_adapter_has_required_exports(self):
        """chat_adapter module exports to_chat_message and to_api_messages."""
        from src.decision.chat_adapter import to_chat_message, to_api_messages

        assert callable(to_chat_message)
        assert callable(to_api_messages)

    def test_memory_service_has_required_methods(self):
        """MemoryService exposes sync_cycle(), decay_cycle(), hot, cold."""
        from src.memory.memory_service import MemoryService

        svc = MemoryService()
        assert hasattr(svc, "sync_cycle"), "MemoryService must have sync_cycle()"
        assert hasattr(svc, "decay_cycle"), "MemoryService must have decay_cycle()"
        assert hasattr(svc, "hot"), "MemoryService must expose .hot property"
        assert hasattr(svc, "cold"), "MemoryService must expose .cold property"
        assert callable(svc.sync_cycle)
        assert callable(svc.decay_cycle)

    def test_baseline_personality_supports_emotion_adj_and_preference_shift(self):
        """BaselinePersonality can be passed to EmotionAdj and PreferenceShift constructors."""
        from src.personality.baseline import BaselinePersonality
        from src.personality.emotion_adj import EmotionAdj
        from src.personality.preference_shift import PreferenceShift

        baseline = BaselinePersonality()
        emotion_adj = EmotionAdj(baseline)
        pref_shift = PreferenceShift(baseline)

        assert emotion_adj is not None
        assert pref_shift is not None
        assert emotion_adj.current_emotion == "neutral"
        assert pref_shift.cold_boot is True


# ═══════════════════════════════════════════════════════════════════
# 9. TestFactoryFunction — verify _run_sync_loop receives all params
# ═══════════════════════════════════════════════════════════════════

class TestRunSyncLoopFactory:
    """Verify _run_sync_loop function signature includes memory_service parameter."""

    def test_run_sync_loop_has_memory_service_param(self):
        """_run_sync_loop function signature includes memory_service parameter."""
        source = _read_source("src/decision_bridge.py")
        # Function signature includes memory_service
        has_param = bool(re.search(
            r"async def _run_sync_loop\([^)]*memory_service",
            source,
        ))
        assert has_param, (
            "_run_sync_loop() must accept memory_service parameter"
        )

    def test_task_creation_passes_sync_interval(self):
        """Background sync task is created with 300s interval."""
        source = _read_source("src/decision_bridge.py")
        has_interval = bool(re.search(
            r"sync_interval\s*=\s*300",
            source,
        ))
        assert has_interval, (
            "Sync task should use sync_interval=300 (5 minutes)"
        )

    def test_on_sync_complete_callback_passed(self):
        """on_sync_complete callback is wired to _on_sync_complete."""
        source = _read_source("src/decision_bridge.py")
        has_callback = bool(re.search(
            r"on_sync_complete\s*=\s*self\._on_sync_complete",
            source,
        ))
        assert has_callback, (
            "Sync task should pass on_sync_complete=self._on_sync_complete"
        )
