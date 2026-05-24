"""
Security audit tests — sensitive information filtering.

Validates that the system correctly detects and handles sensitive data
patterns defined in config/thresholds.yaml:
  - Chinese mobile phone numbers
  - Chinese ID card numbers
  - Password leakage patterns

v4.5.0 §5.1: privacy铁律 — sensitive data defaults to local-only.
项目宪法 §5.1: 隐私铁律.
"""
from __future__ import annotations

import re

import pytest


# ---------------------------------------------------------------------------
# Load the same patterns the production code uses
# ---------------------------------------------------------------------------

def _load_thresholds() -> dict:
    import yaml
    from pathlib import Path
    path = Path(__file__).resolve().parents[2] / "config" / "thresholds.yaml"
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


THRESHOLDS = _load_thresholds()
SAFETY_PATTERNS = THRESHOLDS.get("safety", {}).get("sensitive_patterns", [])
COMPILED_PATTERNS = [re.compile(p) for p in SAFETY_PATTERNS]


# ---------------------------------------------------------------------------
# Phone number tests
# ---------------------------------------------------------------------------

class TestPhoneNumberFilter:
    """Chinese mobile phone number detection (pattern: 1[3-9]\\d{9})."""

    @pytest.mark.parametrize("phone", [
        "13800138000",
        "15012345678",
        "19988889999",
        "13600000000",
        "18765432109",
    ])
    def test_detects_valid_phone_numbers(self, phone: str):
        """Standard 11-digit Chinese mobile numbers must be caught."""
        assert any(p.search(phone) for p in COMPILED_PATTERNS), (
            f"Phone number {phone} was not detected by any sensitive pattern"
        )

    @pytest.mark.parametrize("not_phone", [
        "12345678901",   # starts with 1 but second digit < 3
        "1380013800",    # 10 digits
        "100000000000",  # 12 digits but second digit = 0 (no 1[3-9] substring)
        "abcdefghijk",   # letters
        "138-0013-8000", # with dashes (pattern expects continuous digits)
    ])
    def test_ignores_invalid_phone_numbers(self, not_phone: str):
        """Non-conforming strings must not trigger the phone pattern."""
        assert not any(p.search(not_phone) for p in COMPILED_PATTERNS), (
            f"False positive: {not_phone} was incorrectly flagged as sensitive"
        )

    def test_detects_phone_in_sentence(self):
        """Phone numbers embedded in natural language must be detected."""
        sentence = "我的手机号是13812345678，记得联系我哦"
        assert any(p.search(sentence) for p in COMPILED_PATTERNS)

    def test_detects_multiple_phones(self):
        """Multiple phone numbers in one string must all be detected."""
        text = "联系人：13811111111 和 15022222222"
        matches = []
        for pat in COMPILED_PATTERNS:
            matches.extend(pat.findall(text))
        assert len(matches) >= 2, f"Expected ≥2 phone matches, got {len(matches)}"


# ---------------------------------------------------------------------------
# ID card number tests
# ---------------------------------------------------------------------------

class TestIdCardFilter:
    """Chinese national ID card detection (pattern: \\d{17}[\\dXx])."""

    @pytest.mark.parametrize("id_card", [
        "110101199001011234",
        "31010119850215321X",
        "44010619780808123x",
    ])
    def test_detects_valid_id_cards(self, id_card: str):
        """18-digit ID cards (ending digit or X/x) must be caught."""
        assert any(p.search(id_card) for p in COMPILED_PATTERNS), (
            f"ID card {id_card} was not detected"
        )

    @pytest.mark.parametrize("not_id", [
        "12345678901234567",   # 17 digits — too short for ID pattern
        "1234567890123456AAA", # 16 digits + letters — no 17-digit+digit/X sequence
        "1101011990010112Y",   # 16 digits + Y — no 17-digit prefix followed by digit/X
        "abcdefghijklmnopqr",  # letters only
    ])
    def test_ignores_invalid_id_cards(self, not_id: str):
        """Non-conforming strings must not trigger the ID pattern."""
        assert not any(p.search(not_id) for p in COMPILED_PATTERNS), (
            f"False positive: {not_id} was incorrectly flagged as sensitive"
        )

    def test_detects_id_in_sentence(self):
        """ID numbers embedded in sentences must be detected."""
        sentence = "身份证号是310101199001011234，请核对"
        assert any(p.search(sentence) for p in COMPILED_PATTERNS)


# ---------------------------------------------------------------------------
# Password leak tests
# ---------------------------------------------------------------------------

class TestPasswordLeakFilter:
    """Password leakage detection (pattern: (password|passwd|pwd)\\s*[:=]\\s*\\S+)."""

    @pytest.mark.parametrize("leak", [
        "password: secret123",
        "passwd=root",
        "pwd: mypassword",
        "password:   123456",
        "pwd=admin",
    ])
    def test_detects_password_leakage(self, leak: str):
        """Strings exposing passwords via common keywords must be caught."""
        assert any(p.search(leak) for p in COMPILED_PATTERNS), (
            f"Password leak not detected: {leak}"
        )

    @pytest.mark.parametrize("not_leak", [
        "password",           # no value
        "pwd ",               # no value after space
        "the password field", # not an assignment
        "passwd",             # standalone keyword
    ])
    def test_ignores_non_leak_patterns(self, not_leak: str):
        """Standalone keywords without values must not trigger."""
        assert not any(p.search(not_leak) for p in COMPILED_PATTERNS), (
            f"False positive: {not_leak} incorrectly flagged as password leak"
        )

    def test_detects_password_in_sentence(self):
        """Password leaks embedded in prose must be detected."""
        sentence = "我的账户密码是 password: supersecret ，别告诉别人"
        assert any(p.search(sentence) for p in COMPILED_PATTERNS)


# ---------------------------------------------------------------------------
# Cross-pattern negative tests
# ---------------------------------------------------------------------------

class TestNoFalsePositives:
    """Ensure benign text does not trigger any sensitive pattern."""

    @pytest.mark.parametrize("benign", [
        "你好，今天天气不错",
        "Let us meet at the park around 3 PM",
        "The quick brown fox jumps over 13 lazy dogs",
        "项目代号是 Alpha-7，没有敏感信息",
        "10000000000",  # 11 digits but not a valid phone (second digit = 0)
    ])
    def test_benign_text_not_flagged(self, benign: str):
        """Everyday harmless text must pass cleanly."""
        assert not any(p.search(benign) for p in COMPILED_PATTERNS), (
            f"False positive on benign text: {benign}"
        )
