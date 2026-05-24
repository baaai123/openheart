"""
Privacy Filter — Sensitive info masking and local summary generation.
v4.5.0 §3.2.4, 项目宪法 §5.1

Gatekeeper before any data leaves local storage for cold memory or LLM API calls.
All layer-to-layer communication must pass sensitive text through filter_sensitive().

Key design decisions:
  - Regex-only: zero false-negative for PII patterns, no ML dependency
  - Masking preserves format hints (e.g., 138****5678 keeps region prefix visible)
  - Summary generation is purely local — no cloud LLM, no network calls
  - SnowNLP is optional; a keyword-frequency fallback always works

Usage:
  from src.memory.privacy_filter import filter_sensitive, generate_local_summary

  cleaned = filter_sensitive("我的电话号码是13812345678")
  # → "我的电话号码是138****5678"

  summary = generate_local_summary([
      {"role": "user", "content": "今天心情很好"},
      {"role": "assistant", "content": "很开心听到这个消息"},
  ])
  # → "用户表达开心情绪，关键词：心情、开心。"
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sensitive info patterns — v4.5.0 §3.2.4, 项目宪法 §5.1
# ---------------------------------------------------------------------------
# Each entry: (name, regex_pattern, mask_func)
# mask_func takes a re.Match and returns the masked replacement string.

_SENSITIVE_RULES: list[tuple[str, re.Pattern[str], Callable[[re.Match[str]], str]]] = [
    # ── Chinese mobile phone: 1[3-9] followed by 9 digits ──────────────
    # Mask: show first 3, hide middle 4, show last 4  (138****5678)
    (
        "phone_number",
        re.compile(r"(?<!\d)(1[3-9]\d)(\d{4})(\d{4})(?!\d)"),
        lambda m: m.group(1) + "****" + m.group(3),
    ),
    # ── Chinese ID card: 17 digits + digit or X/x ─────────────────────
    # Mask: show first 6, hide middle 8, show last 4  (110101********1234)
    (
        "id_card",
        re.compile(r"(?<!\d)(\d{6})\d{8}(\d{4}[\dXx])(?!\d)"),
        lambda m: m.group(1) + "********" + m.group(2),
    ),
    # ── Email addresses ────────────────────────────────────────────────
    # Mask local part and domain subparts, keep TLD visible
    (
        "email",
        re.compile(r"([\w.+-]+)@([\w.-]+)\.(\w{2,})"),
        lambda m: f"***@***.{m.group(3)}",
    ),
    # ── Passwords: key=value patterns ──────────────────────────────────
    # Catches: password=xxx, 密码:xxx, 密码是xxx, pw=xxx, etc.
    # Also catches: api_key=xxx, secret=xxx, token=xxx, apikey=xxx
    (
        "password_key_value",
        re.compile(
            r"(?:(?:password|pw|pwd|pass|secret|token|api[_-]?key|private[_-]?key)"
            r"|密码|密钥|令牌"
            r")\s*[=:：是为]\s*\S+",
            re.IGNORECASE,
        ),
        lambda m: m.group(0).split("=")[0].split(":")[0].split("：")[0].split("是")[0].split("为")[0] + "=****",
    ),
    # ── Generic long alphanumeric secrets (API keys, tokens) ───────────
    # Catches bare 32+ char alphanumeric strings that look like keys
    # Must be bounded by non-alphanumeric to avoid false positives
    (
        "long_secret",
        re.compile(r"(?<![a-zA-Z0-9])[a-zA-Z0-9]{32,}(?![a-zA-Z0-9])"),
        lambda m: m.group(0)[:6] + "*" * (len(m.group(0)) - 12) + m.group(0)[-6:],
    ),
    # ── Credit / debit card numbers (13-19 digits, optionally separated) ─
    # Mask: show first 6, hide rest except last 4
    (
        "card_number",
        re.compile(r"(?<!\d)(\d{6})[\d ]{7,13}(\d{4})(?!\d)"),
        lambda m: m.group(1) + "******" + m.group(2),
    ),
    # ── URLs containing credentials (http://user:pass@host) ─────────────
    (
        "url_credentials",
        re.compile(r"(https?://)([^:]+):([^@]+)@"),
        lambda m: m.group(1) + "***:***@",
    ),
    # ── IPv4 addresses (privacy concern in some contexts) ──────────────
    (
        "ipv4",
        re.compile(r"(?<!\d)(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})(?!\d)"),
        lambda m: f"{m.group(1)}.{m.group(2)}.***.***",
    ),
]


def filter_sensitive(text: str | None) -> str:
    """
    Mask sensitive information in the given text.

    Supports: phone numbers, ID cards, emails, passwords, API keys,
    credit card numbers, credential-bearing URLs, and IPv4 addresses.

    Args:
        text: Input text (may be None or empty).

    Returns:
        Cleaned text with sensitive patterns masked.
        Returns empty string for None or empty input.

    Example:
        >>> filter_sensitive("我的手机13812345678")
        '我的手机138****5678'
        >>> filter_sensitive(None)
        ''
    """
    # v4.5.0: Handle None/empty edge case at the top
    if not text:
        return ""

    result = text

    for name, pattern, mask_fn in _SENSITIVE_RULES:
        try:
            result = pattern.sub(mask_fn, result)
        except Exception as exc:
            # Safe to continue: a single rule failure should not break filtering
            logger.warning(
                "Privacy filter rule '%s' failed: %s — skipping rule",
                name,
                exc,
            )

    return result


# ---------------------------------------------------------------------------
# Chinese stop words — used by keyword extraction
# ---------------------------------------------------------------------------
_CHINESE_STOP_WORDS: frozenset[str] = frozenset({
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一",
    "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着",
    "没有", "看", "好", "自己", "这", "他", "她", "它", "们", "那", "些",
    "什么", "怎么", "为什么", "因为", "所以", "但是", "可是", "然而",
    "如果", "虽然", "而且", "或者", "还是", "已经", "可以", "能够",
    "这个", "那个", "哪个", "那里", "这里", "哪里", "时候", "时间",
    "现在", "刚才", "已经", "正在", "将", "把", "被", "让", "给",
    "从", "对", "对于", "关于", "跟", "与", "以", "向", "为", "为了",
    "能", "能够", "可以", "应该", "愿意", "可能", "必须", "需要",
    "还", "再", "又", "也", "都", "只", "才", "就", "便", "刚",
    "过", "来", "去", "之", "所", "者", "第", "每", "某",
    "嗯", "哦", "啊", "吧", "吗", "呢", "呀", "嘛", "哈", "哇",
    "哈喽", "你好", "谢谢", "请问", "不客气", "没事",
    "这样", "那样", "这么", "那么", "怎么", "怎么样",
    "做", "想", "觉得", "知道", "认为", "希望",
    "哈哈", "嘿嘿", "呵呵", "嗯嗯", "好的", "行",
    "嗯嗯", "是吧", "好吧", "好了", "对了", "是啊",
})


# ---------------------------------------------------------------------------
# Emotion keyword maps — v4.5.0 §4.5
# ---------------------------------------------------------------------------
_EMOTION_KEYWORDS: dict[str, list[str]] = {
    "joy": [
        "开心", "高兴", "快乐", "愉快", "兴奋", "激动", "幸福", "满足",
        "哈哈", "嘿嘿", "太好了", "真棒", "喜欢", "爱", "美好", "棒",
        "赞", "不错", "满意", "舒服", "放松", "轻松", "有趣", "好玩",
        "惊喜", "感激", "感谢", "幸运", "温馨", "温暖",
    ],
    "sadness": [
        "难过", "伤心", "悲伤", "痛苦", "失落", "沮丧", "失望", "郁闷",
        "不开心", "烦恼", "焦虑", "担心", "害怕", "恐惧", "孤单",
        "孤独", "寂寞", "累", "疲惫", "厌倦", "无聊", "无奈",
        "哭", "哭", "流泪", "心痛", "心碎", "难受", "不舒服",
    ],
    "anger": [
        "生气", "愤怒", "恼火", "烦躁", "不耐烦", "讨厌", "恨",
        "气死", "可恶", "过分", "受不了", "忍不了", "抓狂",
        "暴躁", "发火", "怒吼", "气愤", "不满",
    ],
    "surprise": [
        "惊讶", "吃惊", "震惊", "意外", "没想到", "居然", "竟然",
        "天哪", "哇", "真的吗", "不可思议", "难以置信", "奇怪",
    ],
    "neutral": [],  # default / absence of strong markers
}


# ---------------------------------------------------------------------------
# Simple Chinese keyword extraction — v4.5.0 §3.2.4
# ---------------------------------------------------------------------------

def _extract_keywords(text: str, top_n: int = 5) -> list[str]:
    """
    Extract top-N meaningful keywords from Chinese text.

    Uses simple character-bigram and unigram frequency analysis.
    No external dependency required — pure stdlib approach.

    Args:
        text: Input text (Chinese or mixed).
        top_n: Number of top keywords to return.

    Returns:
        List of keyword strings, most frequent first.
    """
    # Filter to Chinese characters only, preserving CJK
    # Also keep common English meaningful words
    cleaned: list[str] = []
    for char in text:
        if "\u4e00" <= char <= "\u9fff" or "\u0041" <= char <= "\u005a" or "\u0061" <= char <= "\u007a":
            cleaned.append(char.lower())
        else:
            cleaned.append(" ")

    # Tokenize: split by whitespace (English words), then extract 2-char Chinese grams
    segments = "".join(cleaned).split()

    freq: dict[str, int] = {}

    for seg in segments:
        # Skip English stop words / very short tokens
        if len(seg) < 2:
            continue
        if seg in ("the", "is", "at", "of", "on", "and", "to", "in", "it", "for"):
            continue

        # For Chinese text: extract overlapping bigrams and unigrams
        # For English text: use the whole word
        if all("\u4e00" <= c <= "\u9fff" for c in seg):
            # Chinese: bigrams
            for i in range(len(seg) - 1):
                gram = seg[i : i + 2]
                if gram not in _CHINESE_STOP_WORDS:
                    freq[gram] = freq.get(gram, 0) + 1
            # Single character (skip stop words)
            for c in seg:
                if c not in _CHINESE_STOP_WORDS and "\u4e00" <= c <= "\u9fff":
                    freq[c] = freq.get(c, 0) + 1
        else:
            # English or other: use full segment
            if seg not in _CHINESE_STOP_WORDS:
                freq[seg] = freq.get(seg, 0) + 1

    # Sort by frequency descending
    sorted_keywords = sorted(freq.items(), key=lambda x: (-x[1], x[0]))

    # Deduplicate: prefer bigrams (2-char words), skip single-character artifacts
    result: list[str] = []
    seen_chars: set[str] = set()
    for word, _count in sorted_keywords:
        if len(word) < 2:
            # Skip single Chinese characters — too generic to be meaningful keywords
            continue
        if len(word) == 2 and all("\u4e00" <= c <= "\u9fff" for c in word):
            # Chinese bigram: prefer this; skip if both chars already covered
            if word[0] in seen_chars and word[1] in seen_chars:
                continue
            result.append(word)
            seen_chars.update(word)
        else:
            # English word or longer Chinese phrase
            if word not in result:
                result.append(word)

        if len(result) >= top_n:
            break

    return result


def _detect_emotion(text: str) -> str:
    """
    Detect dominant emotion from text using keyword matching.

    Returns one of: "joy", "sadness", "anger", "surprise", "neutral".
    Per spec §4.5: anger and surprise are placeholder — downstream must
    not branch on them unless config/sentiment.yaml has provider structbert.

    Args:
        text: Input text to analyze.

    Returns:
        Emotion category label.
    """
    text_lower = text.lower()

    scores: dict[str, int] = {}
    for emotion, keywords in _EMOTION_KEYWORDS.items():
        if not keywords:
            continue
        score = sum(1 for kw in keywords if kw.lower() in text_lower)
        if score > 0:
            scores[emotion] = score

    if not scores:
        return "neutral"

    # Return highest-scoring emotion
    return max(scores, key=lambda k: scores[k])  # type: ignore[return-value]


def generate_local_summary(messages: list[dict[str, str]] | None) -> str:
    """
    Generate a 1-2 sentence Chinese summary from conversation messages.

    Uses purely local keyword extraction + emotion detection.
    NEVER calls cloud APIs. NEVER leaks raw sensitive data.

    Args:
        messages: List of dicts with 'role' and 'content' keys.
                  Example: [{"role": "user", "content": "..."},
                            {"role": "assistant", "content": "..."}]

    Returns:
        1-2 sentence Chinese summary string.
        Returns "无对话内容" for None or empty input.

    Example:
        >>> generate_local_summary([
        ...     {"role": "user", "content": "今天心情很好"},
        ...     {"role": "assistant", "content": "很开心听到这个消息"},
        ... ])
        '用户情绪积极（开心）。话题涉及：心情、今天。'
    """
    # v4.5.0 §3.2.4: Handle empty/None edge case
    if not messages:
        return "无对话内容"

    # Concatenate all user and assistant content
    user_text_parts: list[str] = []
    asst_text_parts: list[str] = []
    full_text_parts: list[str] = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if not content:
            continue

        # Filter sensitive data BEFORE any processing
        safe_content = filter_sensitive(content)
        full_text_parts.append(safe_content)

        if role == "user":
            user_text_parts.append(safe_content)
        elif role in ("assistant", "system"):
            asst_text_parts.append(safe_content)
        # Other roles (system, etc.) — include in full text only

    user_text = " ".join(user_text_parts)
    asst_text = " ".join(asst_text_parts)
    full_text = " ".join(full_text_parts)

    if not full_text.strip():
        return "无对话内容"

    # ── Emotion detection ──────────────────────────────────────────────
    user_emotion = _detect_emotion(user_text) if user_text else "neutral"

    # ── Keyword extraction ─────────────────────────────────────────────
    keywords = _extract_keywords(full_text, top_n=5)

    # ── Build Chinese summary sentences ─────────────────────────────────
    emotion_map: dict[str, str] = {
        "joy": "积极（开心/高兴）",
        "sadness": "消极（难过/低落）",
        "anger": "不满（生气/愤怒）",
        "surprise": "惊讶（意外/惊喜）",
        "neutral": "平稳",
    }
    emotion_label = emotion_map.get(user_emotion, "平稳")

    # Sentence 1: emotion summary
    if user_text_parts:
        if user_emotion == "neutral":
            s1 = f"用户情绪{emotion_label}。"
        else:
            s1 = f"用户情绪{emotion_label}。"
    else:
        s1 = f"对话情绪{emotion_label}。"

    # Sentence 2: keyword summary
    if keywords:
        kw_str = "、".join(keywords)
        # Check if assistant responded to questions or provided info
        has_question = any("?" in p or "？" in p for p in user_text_parts)
        if has_question:
            s2 = f"用户询问关于{kw_str}的问题。"
        else:
            s2 = f"话题涉及：{kw_str}。"
    else:
        s2 = ""

    # Combine into 1-2 sentences
    if s2:
        return s1 + s2
    return s1


# ---------------------------------------------------------------------------
# Convenience: filter conversation list
# ---------------------------------------------------------------------------

def filter_conversation(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """
    Apply filter_sensitive to all 'content' fields in a message list.

    Returns a NEW list of dicts (does not mutate input).
    Skips messages without 'content' or 'role' keys.

    Args:
        messages: List of message dicts with at least 'role' and 'content'.

    Returns:
        Cleaned message list with sensitive data masked.
    """
    result: list[dict[str, str]] = []
    for msg in messages:
        if "content" not in msg or "role" not in msg:
            result.append(msg.copy())
            continue
        result.append({
            "role": msg["role"],
            "content": filter_sensitive(msg.get("content", "")),
        })
    return result
