"""Contract tests for UserModelCorrector (spec v4.5.0 §5.7.5)."""
from __future__ import annotations

import pytest

from tests.contracts.conftest import require_module

require_module("src.memory.user_model_corrector", "UserModelCorrector")

from src.memory.user_model_corrector import (  # noqa: E402
    UserModelCorrector,
    CorrectionOperation,
    CorrectionResult,
)


def _make_user_model(
    *,
    personality: str = "偏内向",
    communication_style: str = "喜欢用梗",
    emotional_pattern: str = "工作日下午容易焦虑",
    personality_confidence: float = 0.5,
    emotional_pattern_confidence: float = 0.7,
    model_confidence: float = 0.5,
    user_verified_fields: list[str] | None = None,
) -> dict[str, Any]:
    """Build a minimal user model dict matching spec §3.4.1 schema."""
    return {
        "user_model_id": "test-uuid",
        "version": 3,
        "inferred_traits": {
            "personality": personality,
            "communication_style": communication_style,
            "emotional_pattern": emotional_pattern,
            "personality_confidence": personality_confidence,
            "emotional_pattern_confidence": emotional_pattern_confidence,
        },
        "knowledge_profile": {
            "topics_of_interest": ["科幻电影", "Python"],
            "topics_to_avoid": [],
            "expertise_level": {},
        },
        "behavioral_insights": {
            "active_hours": ["weekday_afternoon"],
            "avg_session_length_min": 45,
            "preferred_interaction_mode": "voice_heavy",
        },
        "key_memories": [
            {
                "memory_id": "mem-001",
                "summary": "工作日下午容易焦虑的记忆",
                "emotional_significance": "high",
                "category": "其他",
            },
            {
                "memory_id": "mem-002",
                "summary": "unrelated memory",
                "emotional_significance": "low",
                "category": "humor",
            },
        ],
        "relationship_meta": {
            "first_interaction_date": "2025-01-01T00:00:00",
            "total_interaction_hours": 120.0,
            "relationship_stage": "familiar",
            "nickname_preference": "小伙伴",
            "model_confidence": model_confidence,
            "user_verified_fields": (
                user_verified_fields if user_verified_fields is not None else []
            ),
        },
    }


class TestUserModelCorrectorExists:
    def test_class_is_importable(self):
        assert UserModelCorrector is not None

    def test_correction_operation_enum_has_all_values(self):
        expected = {"delete", "modify", "lower_confidence"}
        assert set(m.value for m in CorrectionOperation) == expected

    def test_correction_result_dataclass_fields(self):
        r = CorrectionResult(success=True)
        assert r.success is True
        assert r.operation is None
        assert r.field_path == ""


class TestDetectIntent:
    def test_detects_correction_intent(self):
        c = UserModelCorrector()
        assert c.detect_intent("别把我想得那么脆弱") is True
        assert c.detect_intent("我不喜欢被这样标签") is True
        assert c.detect_intent("忘掉这个偏好") is True
        assert c.detect_intent("其实我更喜欢科幻电影") is True
        assert c.detect_intent("我不只是内向") is True

    def test_rejects_normal_text(self):
        c = UserModelCorrector()
        assert c.detect_intent("今天天气不错") is False
        assert c.detect_intent("你好") is False
        assert c.detect_intent("") is False


class TestDeleteOperation:
    def test_delete_resets_field_to_default(self):
        c = UserModelCorrector()
        um = _make_user_model()
        result = c.apply_correction(um, "忘掉我的情绪模式")
        assert result.success is True
        assert result.operation == CorrectionOperation.DELETE
        assert result.field_path == "inferred_traits.emotional_pattern"
        assert um["inferred_traits"]["emotional_pattern"] == "暂无数据"

    def test_delete_zeroes_confidence(self):
        c = UserModelCorrector()
        um = _make_user_model(emotional_pattern_confidence=0.8)
        c.apply_correction(um, "忘掉我的情绪模式")
        assert um["inferred_traits"]["emotional_pattern_confidence"] == 0.0

    def test_delete_removes_related_key_memories(self):
        c = UserModelCorrector()
        um = _make_user_model()
        c.apply_correction(um, "忘掉我的情绪模式")
        summaries = [m["summary"] for m in um["key_memories"]]
        assert "工作日下午容易焦虑的记忆" not in summaries
        assert "unrelated memory" in summaries

    def test_delete_removes_from_user_verified_fields(self):
        c = UserModelCorrector()
        um = _make_user_model(user_verified_fields=["inferred_traits.emotional_pattern"])
        c.apply_correction(um, "忘掉我的情绪模式")
        assert "inferred_traits.emotional_pattern" not in um["relationship_meta"]["user_verified_fields"]

    def test_delete_returns_confirmation_text(self):
        c = UserModelCorrector()
        um = _make_user_model()
        result = c.apply_correction(um, "忘掉我的情绪模式")
        assert "以后不会" in result.confirmation_text

    def test_delete_lowers_model_confidence(self):
        c = UserModelCorrector()
        um = _make_user_model(model_confidence=0.5)
        c.apply_correction(um, "忘掉我的情绪模式")
        assert um["relationship_meta"]["model_confidence"] < 0.5


class TestModifyOperation:
    def test_modify_updates_field_value(self):
        c = UserModelCorrector()
        um = _make_user_model()
        result = c.apply_correction(um, "其实我更喜欢科幻小说")
        assert result.success is True
        assert result.operation == CorrectionOperation.MODIFY
        assert um["knowledge_profile"]["topics_of_interest"] == ["科幻小说"]

    def test_modify_sets_user_verified(self):
        c = UserModelCorrector()
        um = _make_user_model()
        c.apply_correction(um, "其实我更喜欢科幻小说")
        assert "knowledge_profile.topics_of_interest" in um["relationship_meta"]["user_verified_fields"]

    def test_modify_sets_field_confidence_to_one(self):
        c = UserModelCorrector()
        um = _make_user_model()
        c.apply_correction(um, "其实我更喜欢科幻小说")
        assert um["knowledge_profile"].get("topics_of_interest_confidence") == 1.0

    def test_modify_appends_to_user_verified_fields(self):
        c = UserModelCorrector()
        um = _make_user_model(user_verified_fields=["inferred_traits.personality"])
        c.apply_correction(um, "其实我更喜欢科幻小说")
        verified = um["relationship_meta"]["user_verified_fields"]
        assert "inferred_traits.personality" in verified
        assert "knowledge_profile.topics_of_interest" in verified

    def test_modify_returns_confirmation_text(self):
        c = UserModelCorrector()
        um = _make_user_model()
        result = c.apply_correction(um, "其实我更喜欢科幻小说")
        assert "记住了" in result.confirmation_text

    def test_modify_raises_model_confidence(self):
        c = UserModelCorrector()
        um = _make_user_model(model_confidence=0.5)
        c.apply_correction(um, "其实我更喜欢科幻小说")
        assert um["relationship_meta"]["model_confidence"] > 0.5

    def test_modify_personality_field(self):
        c = UserModelCorrector()
        um = _make_user_model()
        result = c.apply_correction(um, "我不只是内向，我其实也挺外向的")
        assert result.success is True
        assert "inferred_traits.personality" in result.field_path


class TestLowerConfidenceOperation:
    def test_lower_confidence_reduces_field_confidence(self):
        c = UserModelCorrector()
        um = _make_user_model(emotional_pattern_confidence=0.9)
        result = c.apply_correction(um, "我不确定我下午总是焦虑")
        assert result.success is True
        assert result.operation == CorrectionOperation.LOWER_CONFIDENCE
        assert result.new_value < 0.9

    def test_lower_confidence_unverifies_when_below_threshold(self):
        c = UserModelCorrector(confidence_threshold=0.6)
        um = _make_user_model(
            emotional_pattern_confidence=0.8,
            user_verified_fields=["inferred_traits.emotional_pattern"],
        )
        c.apply_correction(um, "我不确定我下午总是焦虑")
        assert "inferred_traits.emotional_pattern" not in um["relationship_meta"]["user_verified_fields"]

    def test_lower_confidence_keeps_verified_when_above_threshold(self):
        c = UserModelCorrector(confidence_threshold=0.6)
        um = _make_user_model(
            emotional_pattern_confidence=0.9,
            user_verified_fields=["inferred_traits.emotional_pattern"],
        )
        c.apply_correction(um, "我不确定我下午总是焦虑")
        # confidence was 0.9, reduced by 0.3 -> 0.6, which is >= threshold
        # Wait, 0.9 - 0.3 = 0.6, and threshold is 0.6, so it stays
        assert "inferred_traits.emotional_pattern" in um["relationship_meta"]["user_verified_fields"]

    def test_lower_confidence_lowers_model_confidence(self):
        c = UserModelCorrector()
        um = _make_user_model(model_confidence=0.5)
        c.apply_correction(um, "我不确定我下午总是焦虑")
        assert um["relationship_meta"]["model_confidence"] < 0.5


class TestCorrectionEdgeCases:
    def test_no_intent_returns_failure(self):
        c = UserModelCorrector()
        um = _make_user_model()
        result = c.apply_correction(um, "今天天气不错")
        assert result.success is False

    def test_empty_text_returns_failure(self):
        c = UserModelCorrector()
        um = _make_user_model()
        result = c.apply_correction(um, "")
        assert result.success is False

    def test_result_includes_user_verified_fields(self):
        c = UserModelCorrector()
        um = _make_user_model()
        result = c.apply_correction(um, "其实我更喜欢科幻小说")
        assert isinstance(result.user_verified_fields, list)
        assert "knowledge_profile.topics_of_interest" in result.user_verified_fields

    def test_detect_intent_with_none_returns_false(self):
        c = UserModelCorrector()
        assert c.detect_intent(None) is False  # type: ignore[arg-type]
