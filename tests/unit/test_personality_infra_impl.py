"""
Unit tests for PersonalityInfraImpl — delegation, degradation, error handling.

v4.5.0 §4.8: tests the concrete adapter layer between main_decision and
BaselinePersonality/PreferenceShift/PersonaAuditor. Uses real PersonalityInfraImpl
with real BaselinePersonality loaded from baseline.json or empty fallback.
"""

from __future__ import annotations

import pytest

from src.personality.baseline import BaselinePersonality
from src.personality.persona_auditor import AuditResult
from src.personality.personality_infra_impl import PersonalityInfraImpl


class TestGetBaseline:
    """PersonalityInfraImpl.get_baseline() — baseline retrieval."""

    def test_get_baseline_returns_baseline_personality(self) -> None:
        """get_baseline returns a BaselinePersonality instance with expected sections
        (v4.5.0 §4.3, §4.8)."""
        impl = PersonalityInfraImpl()
        baseline = impl.get_baseline()
        assert isinstance(baseline, BaselinePersonality)
        sections = baseline.sections()
        assert "voice_style" in sections
        assert "avatar_style" in sections
        assert "mouse_style" in sections
        assert len(baseline.fields("voice_style")) > 0


class TestGetPreferenceOffsets:
    """PersonalityInfraImpl.get_preference_offsets() — user model delegation."""

    def test_get_preference_offsets_none_user_model(self) -> None:
        """get_preference_offsets returns empty dict when user_model is None
        (v4.5.0 §4.4 cold-boot safety)."""
        impl = PersonalityInfraImpl()
        result = impl.get_preference_offsets(user_model=None)
        assert result == {}

    def test_get_preference_offsets_with_user_model(self) -> None:
        """get_preference_offsets returns offset dict when user_model is provided
        (delegates to PreferenceShift.get_all_offsets)."""
        impl = PersonalityInfraImpl()
        result = impl.get_preference_offsets(user_model="any")
        assert isinstance(result, dict)
        assert "voice_style" in result


class TestAudit:
    """PersonalityInfraImpl.audit() — persona auditing delegation."""

    def test_audit_returns_audit_result(self) -> None:
        """audit returns AuditResult with score field (v4.5.0 §4.7, §4.8)."""
        impl = PersonalityInfraImpl()
        baseline = impl.get_baseline()
        result = impl.audit(
            dynamic_persona={"voice_style": {"speed": 1.0}},
            baseline=baseline.to_dict(),
        )
        assert isinstance(result, AuditResult)
        assert isinstance(result.score, int)
        assert result.score == 10


class TestShutdown:
    """PersonalityInfraImpl.shutdown() — lifecycle."""

    @pytest.mark.asyncio
    async def test_shutdown_is_noop(self) -> None:
        """shutdown completes without raising any exception (v4.5.0 §4.8)."""
        impl = PersonalityInfraImpl()
        await impl.shutdown()
