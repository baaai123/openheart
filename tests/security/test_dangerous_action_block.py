"""
Security audit tests — dangerous action auto-blocking.

Validates that SafetyClassifier correctly flags actions as
DANGEROUS_AUTO_BLOCK and that the block warning is generated.

v4.5.0 §5.7.2:
  DANGEROUS_AUTO_BLOCK = 涉及支付、删除数据、系统设置 → 自动阻止
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
        "trace_id": "sec-danger-001",
        "shadow_overridden": False,
        "source": "main_decision_3b",
    }
    if explicit_level is not None:
        cmd["safety_level"] = explicit_level
    return cmd


# ---------------------------------------------------------------------------
# DANGEROUS_AUTO_BLOCK via keywords
# ---------------------------------------------------------------------------

class TestDangerousAutoBlockKeywords:
    """Dangerous keywords in voice_response must trigger auto-block."""

    @pytest.mark.parametrize("keyword", [
        "支付", "付款", "转账", "购买", "下单", "扣款",
        "删除", "清空", "格式化", "销毁", "卸载",
        "系统设置", "注册表", "防火墙", "权限", "管理员",
        "root", "sudo", "rm -rf", "format", "delete all",
        "pay", "payment", "purchase", "transfer", "withdraw",
    ])
    def test_dangerous_keywords_trigger_block(self, keyword: str):
        """Each dangerous keyword must result in DANGEROUS_AUTO_BLOCK."""
        classifier = SafetyClassifier()
        cmd = _make_cmd(voice_response=f"我要帮你{keyword}这个操作")
        level = classifier.classify(cmd, trace_id=cmd["trace_id"])
        assert level == DANGEROUS_AUTO_BLOCK, (
            f"Keyword '{keyword}' did not trigger DANGEROUS_AUTO_BLOCK (got {level})"
        )

    @pytest.mark.parametrize("safe_text", [
        "好的，我来帮你打开文件",
        "今天的天气真不错",
        "让我帮你找找看",
        "这是一个安全的操作",
    ])
    def test_safe_text_does_not_trigger_block(self, safe_text: str):
        """Benign text must not be flagged as dangerous."""
        classifier = SafetyClassifier()
        cmd = _make_cmd(voice_response=safe_text)
        level = classifier.classify(cmd, trace_id=cmd["trace_id"])
        assert level == SAFE, (
            f"Safe text '{safe_text}' incorrectly flagged as {level}"
        )


# ---------------------------------------------------------------------------
# DANGEROUS_AUTO_BLOCK via action params
# ---------------------------------------------------------------------------

class TestDangerousActionParams:
    """Dangerous keywords hidden in action parameters must trigger block."""

    def test_dangerous_param_text_triggers_block(self):
        """Params containing '支付' must raise block level."""
        classifier = SafetyClassifier()
        cmd = _make_cmd(
            voice_response="确认操作",
            actions=[{"type": "mouse_click", "params": {"target": "支付按钮"}}],
        )
        level = classifier.classify(cmd, trace_id=cmd["trace_id"])
        assert level == DANGEROUS_AUTO_BLOCK

    def test_dangerous_param_rm_rf_triggers_block(self):
        """Params containing 'rm -rf' must raise block level."""
        classifier = SafetyClassifier()
        cmd = _make_cmd(
            voice_response="执行命令",
            actions=[{"type": "keyboard_input", "params": {"text": "rm -rf /"}}],
        )
        level = classifier.classify(cmd, trace_id=cmd["trace_id"])
        assert level == DANGEROUS_AUTO_BLOCK

    def test_safe_param_text_is_safe(self):
        """Params without dangerous keywords must remain SAFE."""
        classifier = SafetyClassifier()
        cmd = _make_cmd(
            voice_response="点击这里",
            actions=[{"type": "mouse_click", "params": {"target": "确认按钮"}}],
        )
        level = classifier.classify(cmd, trace_id=cmd["trace_id"])
        assert level == SAFE


# ---------------------------------------------------------------------------
# Explicit safety_level override
# ---------------------------------------------------------------------------

class TestExplicitSafetyLevel:
    """An explicit safety_level in the command must be respected."""

    def test_explicit_dangerous_is_blocked(self):
        """If the command already declares DANGEROUS_AUTO_BLOCK, respect it."""
        classifier = SafetyClassifier()
        cmd = _make_cmd(
            voice_response="安全文本",
            explicit_level=DANGEROUS_AUTO_BLOCK,
        )
        level = classifier.classify(cmd, trace_id=cmd["trace_id"])
        assert level == DANGEROUS_AUTO_BLOCK

    def test_explicit_needs_confirm_is_preserved(self):
        """If the command declares NEEDS_CONFIRM, do not downgrade to SAFE."""
        classifier = SafetyClassifier()
        cmd = _make_cmd(
            voice_response="安全文本",
            explicit_level=NEEDS_CONFIRM,
        )
        level = classifier.classify(cmd, trace_id=cmd["trace_id"])
        assert level == NEEDS_CONFIRM

    def test_explicit_safe_is_preserved(self):
        """If the command declares SAFE, respect it even with dangerous text."""
        classifier = SafetyClassifier()
        cmd = _make_cmd(
            voice_response="我要帮你转账",
            explicit_level=SAFE,
        )
        level = classifier.classify(cmd, trace_id=cmd["trace_id"])
        assert level == SAFE, (
            "Explicit SAFE should override auto-classification (spec §5.7.2 precedence)"
        )


# ---------------------------------------------------------------------------
# Block warning generation
# ---------------------------------------------------------------------------

class TestBlockWarning:
    """The TTS warning returned for blocked actions must be non-empty."""

    def test_block_warning_contains_explanation(self):
        """Warning text must explain why the action was blocked."""
        classifier = SafetyClassifier()
        cmd = _make_cmd(
            voice_response="我要帮你格式化硬盘",
            actions=[{"type": "mouse_click", "params": {"target": "格式化"}}],
        )
        warning = classifier.get_block_warning(cmd)
        assert "危险" in warning or "安全" in warning
        assert len(warning) > 10

    def test_block_warning_includes_action_description(self):
        """Warning should reference the action type when actions are present."""
        classifier = SafetyClassifier()
        cmd = _make_cmd(
            voice_response="删除所有文件",
            actions=[{"type": "keyboard_input", "params": {"text": "delete all"}}],
        )
        warning = classifier.get_block_warning(cmd)
        assert "keyboard_input" in warning or "操作" in warning


# ---------------------------------------------------------------------------
# Precedence: DANGEROUS > NEEDS_CONFIRM > SAFE
# ---------------------------------------------------------------------------

class TestSafetyPrecedence:
    """When multiple signals conflict, the most restrictive level wins."""

    def test_dangerous_text_overrides_safe_action(self):
        """Dangerous voice_response + safe action → DANGEROUS_AUTO_BLOCK."""
        classifier = SafetyClassifier()
        cmd = _make_cmd(
            voice_response="我要帮你支付账单",
            actions=[{"type": "mouse_click", "params": {"target": " harmless"}}],
        )
        level = classifier.classify(cmd, trace_id=cmd["trace_id"])
        assert level == DANGEROUS_AUTO_BLOCK

    def test_needs_confirm_text_and_dangerous_action(self):
        """NEEDS_CONFIRM text + DANGEROUS action → DANGEROUS_AUTO_BLOCK."""
        classifier = SafetyClassifier()
        cmd = _make_cmd(
            voice_response="发送邮件给客户",
            actions=[{"type": "mouse_click", "params": {"target": "支付"}}],
        )
        level = classifier.classify(cmd, trace_id=cmd["trace_id"])
        assert level == DANGEROUS_AUTO_BLOCK

    def test_needs_confirm_text_overrides_safe(self):
        """NEEDS_CONFIRM text + safe action → NEEDS_CONFIRM."""
        classifier = SafetyClassifier()
        cmd = _make_cmd(
            voice_response="发送文件给同事",
            actions=[{"type": "mouse_move", "params": {"x": 100, "y": 200}}],
        )
        level = classifier.classify(cmd, trace_id=cmd["trace_id"])
        assert level == NEEDS_CONFIRM
