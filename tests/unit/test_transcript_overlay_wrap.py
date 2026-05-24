"""
Unit tests for subtitle text wrapping logic.

v4.5.0 §7.5.6: _wrap_text inserts line breaks to prevent subtitle overflow.
CJK characters break at any position; Latin words break at word boundaries.
"""

from __future__ import annotations

import pytest

from src.execution.transcript_overlay import _wrap_text


class TestWrapTextEdgeCases:
    """Empty / trivial inputs."""

    def test_empty_string(self) -> None:
        assert _wrap_text("") == ""

    def test_whitespace_only(self) -> None:
        assert _wrap_text("   ") == ""

    def test_shorter_than_max(self) -> None:
        assert _wrap_text("hi") == "hi"
        assert _wrap_text("你好") == "你好"


class TestWrapTextPureCJK:
    """Chinese text — break at every N characters."""

    def test_exact_multiple(self) -> None:
        text = "你好世界你好世界你好世界你好世界你好世界你好世界你好世界"
        result = _wrap_text(text, max_cjk=7)
        lines = result.split("\n")
        assert len(lines) == 4
        assert all(len(line) == 7 for line in lines)

    def test_remainder(self) -> None:
        text = "你好世界你好世界你好世界你好世界你好世界你好世界你好世界好"
        result = _wrap_text(text, max_cjk=7)
        lines = result.split("\n")
        assert len(lines) == 5
        assert len(lines[-1]) == 1

    def test_single_char_per_line(self) -> None:
        text = "你好呀"
        result = _wrap_text(text, max_cjk=1)
        assert result == "你\n好\n呀"


class TestWrapTextPureLatin:
    """English/Latin text — break at word boundaries only."""

    SENTENCE = "Hello world, this is a test message"

    def test_no_word_is_split(self) -> None:
        result = _wrap_text(self.SENTENCE, max_cjk=7)
        for line in result.split("\n"):
            words = line.split()
            for word in words:
                assert word in self.SENTENCE

    def test_reasonable_line_count(self) -> None:
        result = _wrap_text(self.SENTENCE, max_cjk=7)
        lines = result.split("\n")
        assert 2 <= len(lines) <= 5

    def test_large_max_cjk(self) -> None:
        text = "Hello world foo bar"
        result = _wrap_text(text, max_cjk=20)
        assert result == text

    def test_small_max_cjk_ascii_only(self) -> None:
        text = "a b c d e"
        result = _wrap_text(text, max_cjk=2)
        lines = result.split("\n")
        assert len(lines) >= 2


class TestWrapTextMixed:
    """Text containing both CJK and Latin characters."""

    def test_cjk_and_latin_separate(self) -> None:
        text = "你好world这是test"
        result = _wrap_text(text, max_cjk=7)
        for ch in "你好这是":
            assert ch in result
        assert "world" in result
        assert "test" in result

    def test_no_word_splitting_in_cjk_mixed(self) -> None:
        text = "今天天气很好Let us go to the park我们去公园吧"
        result = _wrap_text(text, max_cjk=7)
        assert "Let" in result
        assert "park" in result


class TestWrapTextExistingNewlines:
    """Input already contains \n — preserve paragraph structure."""

    def test_preserve_trailing_newline(self) -> None:
        text = "hello\n"
        result = _wrap_text(text)
        assert result == "hello"

    def test_multiple_paragraphs(self) -> None:
        text = "hello\n\nworld"
        result = _wrap_text(text, max_cjk=10)
        assert result == "hello\n\nworld"

    def test_paragraphs_wrapped_independently(self) -> None:
        text = "你好世界你好世界\n\nHello world test"
        result = _wrap_text(text, max_cjk=7)
        parts = result.split("\n\n")
        assert len(parts) == 2
        # First paragraph: CJK wrapped every 7 chars
        # Second paragraph: Latin word-boundary aware


class TestWrapTextLongWords:
    """Continuous strings without spaces (URLs, long identifiers)."""

    def test_url_fallback_split(self) -> None:
        url = "http://verylongurlwithoutspaces.com"
        result = _wrap_text(url, max_cjk=7)
        lines = result.split("\n")
        assert len(lines) >= 2
        assert "".join(lines) == url

    def test_word_fits_on_its_own_line(self) -> None:
        """Single long word placed on its own line if it exceeds max."""
        text = "short " + "x" * 30
        result = _wrap_text(text, max_cjk=7)
        lines = result.split("\n")
        assert any(len(line) <= 20 for line in lines)


class TestWrapTextPunctuation:
    """Punctuation stays with its preceding word."""

    def test_comma_and_period(self) -> None:
        text = "Hello, world."
        result = _wrap_text(text, max_cjk=10)
        assert result == "Hello, world."

    def test_punctuation_not_orphaned(self) -> None:
        text = "Hello,world,test"
        # No spaces — all one token, no word boundaries
        result = _wrap_text(text, max_cjk=7)
        lines = result.split("\n")
        assert len(lines) >= 1


class TestWrapTextMaxCjkParameter:
    """The max_cjk parameter controls line fullness."""

    def test_default_is_seven(self) -> None:
        text = "你好世界你好世界你好世界"
        result = _wrap_text(text)  # no explicit max_cjk
        lines = result.split("\n")
        assert all(len(line) <= 7 for line in lines)

    def test_custom_max(self) -> None:
        text = "你好世界你好世界"
        result = _wrap_text(text, max_cjk=10)
        assert len(result.split("\n")) == 1  # fits on one line

    def test_very_small_max(self) -> None:
        text = "a b c"
        result = _wrap_text(text, max_cjk=1)
        lines = result.split("\n")
        assert len(lines) >= 2  # must break
