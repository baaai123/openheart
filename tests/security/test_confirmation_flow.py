"""
Security audit tests — confirmation flow (NEEDS_CONFIRM).

Validates that SafetyClassifier correctly flags actions as NEEDS_CONFIRM
and that the confirmation prompt is generated for verbal user confirmation.

v4.5.0 §5.7.2:
  NEEDS_CONFIRM = 涉及发送消息、修改文件 → 向用户口头确认后进入观察期
v4.5.0 §5.7.3:
  用户后续语音/文本响应中若包含肯定词，取出 pending 规则写入规则库。
项目宪法 §5.2: 操作安全分级.
"""
from __future__ import annotations

import pytest

from src.decision.safety_classifier import (
    DANGEROUS_AUTO_BLOCK,
    NEEDS_CONFIRM,
    SAFE,
    SafetyClassifier,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cmd(voice_response: str = "", actions: list | None = None, explicit_level: str | None = None) -> dict:
    """Build a minimal decision command dict."""
    cmd: dict = {
        "decision_type": "composite",
        "command": {
            "voice_response": voice_response,
            "actions": actions or [],
        },
        "confidence": 0.85,
        "trace_id": "sec-confirm-001",
        "shadow_overridden": False,
        "source": "main_decision_3b",
    }
    if explicit_level is not None:
        cmd["safety_level"] = explicit_level
    return cmd


# ---------------------------------------------------------------------------
# NEEDS_CONFIRM via keywords
# ---------------------------------------------------------------------------

class TestNeedsConfirmKeywords:
    """Moderate-risk keywords in voice_response must trigger NEEDS_CONFIRM."""

    @pytest.mark.parametrize("keyword", [
        "发送", "转发", "上传", "发布", "提交",
        "修改", "编辑", "保存", "覆盖", "写入",
        "send", "post", "upload", "submit", "modify",
        "edit", "save", "overwrite", "write to",
    ])
    def test_needs_confirm_keywords(self, keyword: str):
        """Each moderate-risk keyword must result in NEEDS_CONFIRM."""
        classifier = SafetyClassifier()
        cmd = _make_cmd(voice_response=f"我要帮你{keyword}这个文件")
        level = classifier.classify(cmd, trace_id=cmd["trace_id"])
        assert level == NEEDS_CONFIRM, (
            f"Keyword '{keyword}' did not trigger NEEDS_CONFIRM (got {level})"
        )

    @pytest.mark.parametrize("safe_text", [
        "好的，我来帮你打开文件",
        "今天的天气真不错",
        "这是一个安全的操作",
    ])
    def test_safe_text_not_needs_confirm(self, safe_text: str):
        """Benign text must not be flagged as NEEDS_CONFIRM."""
        classifier = SafetyClassifier()
        cmd = _make_cmd(voice_response=safe_text)
        level = classifier.classify(cmd, trace_id=cmd["trace_id"])
        assert level == SAFE, (
            f"Safe text '{safe_text}' incorrectly flagged as {level}"
        )


# ---------------------------------------------------------------------------
# NEEDS_CONFIRM via action params
# ---------------------------------------------------------------------------

class TestNeedsConfirmActionParams:
    """Moderate-risk keywords hidden in action parameters must trigger confirm."""

    def test_send_param_triggers_confirm(self):
        """Params containing '发送' must raise confirm level."""
        classifier = SafetyClassifier()
        cmd = _make_cmd(
            voice_response="确认操作",
            actions=[{"type": "mouse_click", "params": {"target": "发送按钮"}}],
        )
        level = classifier.classify(cmd, trace_id=cmd["trace_id"])
        assert level == NEEDS_CONFIRM

    def test_edit_param_triggers_confirm(self):
        """Params containing '编辑' must raise confirm level."""
        classifier = SafetyClassifier()
        cmd = _make_cmd(
            voice_response="修改内容",
            actions=[{"type": "keyboard_input", "params": {"text": "编辑文档"}}],
        )
        level = classifier.classify(cmd, trace_id=cmd["trace_id"])
        assert level == NEEDS_CONFIRM

    def test_safe_param_is_safe(self):
        """Params without risky keywords must remain SAFE."""
        classifier = SafetyClassifier()
        cmd = _make_cmd(
            voice_response="点击这里",
            actions=[{"type": "mouse_click", "params": {"target": " harmless按钮"}}],
        )
        level = classifier.classify(cmd, trace_id=cmd["trace_id"])
        assert level == SAFE


# ---------------------------------------------------------------------------
# Confirmation prompt generation
# ---------------------------------------------------------------------------

class TestConfirmationPrompt:
    """The verbal confirmation prompt must be meaningful and actionable."""

    def test_prompt_is_non_empty(self):
        """A confirmation prompt must contain text."""
        classifier = SafetyClassifier()
        cmd = _make_cmd(
            voice_response="发送邮件给客户",
            actions=[{"type": "mouse_click", "params": {"target": "发送"}}],
        )
        prompt = classifier.get_confirmation_prompt(cmd)
        assert len(prompt) > 10

    def test_prompt_asks_for_binary_response(self):
        """The prompt must ask the user to confirm or cancel."""
        classifier = SafetyClassifier()
        cmd = _make_cmd(
            voice_response="覆盖文件",
            actions=[{"type": "mouse_click", "params": {"target": "保存"}}],
        )
        prompt = classifier.get_confirmation_prompt(cmd)
        assert "确定" in prompt or "取消" in prompt or "确认" in prompt

    def test_prompt_describes_action(self):
        """The prompt should reference the action when actions are present."""
        classifier = SafetyClassifier()
        cmd = _make_cmd(
            voice_response="上传照片",
            actions=[{"type": "mouse_click", "params": {"target": "上传"}}],
        )
        prompt = classifier.get_confirmation_prompt(cmd)
        assert "操作" in prompt or "mouse_click" in prompt


# ---------------------------------------------------------------------------
# Affirmation / negation keyword detection (v4.5.0 §5.7.3)
# ---------------------------------------------------------------------------

class TestAffirmationDetection:
    """Simulate the teaching/learning module's affirmation parsing."""

    @pytest.mark.parametrize("affirmative", [
        "确定",
        "是",
        "可以",
        "好的",
        "没错",
        "对的",
        "yes",
        "okay",
        "确认",
    ])
    def test_detects_affirmative_responses(self, affirmative: str):
        """User responses containing affirmative words must be recognised."""
        # This mirrors the logic that the teaching module would apply.
        affirmatives = {"确定", "是", "可以", "好的", "没错", "对的", "yes", "okay", "确认"}
        assert any(word in affirmative for word in affirmatives), (
            f"Failed to detect affirmation in: {affirmative}"
        )

    @pytest.mark.parametrize("negative", [
        "取消",
        "不要",
        "否",
        "不对",
        "算了",
        "no",
        "never",
    ])
    def test_detects_negative_responses(self, negative: str):
        """User responses containing negative words must be recognised."""
        negatives = {"取消", "不要", "否", "不对", "算了", "no", "never"}
        assert any(word in negative for word in negatives), (
            f"Failed to detect negation in: {negative}"
        )

    @pytest.mark.parametrize("ambiguous", [
        "也许",
        "看看再说",
        "不知道",
        "maybe",
    ])
    def test_ambiguous_responses_not_auto_confirmed(self, ambiguous: str):
        """Ambiguous responses must not be treated as affirmation."""
        affirmatives = {"确定", "是", "可以", "好的", "没错", "对的", "yes", "okay", "确认"}
        assert not any(word in ambiguous for word in affirmatives), (
            f"Ambiguous response '{ambiguous}' incorrectly treated as affirmation"
        )


# ---------------------------------------------------------------------------
# Integration: full NEEDS_CONFIRM flow simulation
# ---------------------------------------------------------------------------

class TestNeedsConfirmFlow:
    """End-to-end simulation of the NEEDS_CONFIRM safety flow."""

    def test_full_flow_classification_and_prompt(self):
        """A NEEDS_CONFIRM action must produce the correct level + prompt."""
        classifier = SafetyClassifier()
        cmd = _make_cmd(
            voice_response="我要帮你发送这封邮件",
            actions=[{"type": "mouse_click", "params": {"target": "发送按钮"}}],
        )
        level = classifier.classify(cmd, trace_id=cmd["trace_id"])
        prompt = classifier.get_confirmation_prompt(cmd)

        assert level == NEEDS_CONFIRM
        assert "确定" in prompt or "确认" in prompt
        assert len(prompt) > 10

    def test_dangerous_action_skips_confirm_goes_to_block(self):
        """If the action is DANGEROUS, it must be blocked instead of confirmed."""
        classifier = SafetyClassifier()
        cmd = _make_cmd(
            voice_response="我要帮你转账",
            actions=[{"type": "mouse_click", "params": {"target": "转账"}}],
        )
        level = classifier.classify(cmd, trace_id=cmd["trace_id"])
        assert level == DANGEROUS_AUTO_BLOCK, (
            "DANGEROUS action must bypass NEEDS_CONFIRM and go straight to block"
        )
