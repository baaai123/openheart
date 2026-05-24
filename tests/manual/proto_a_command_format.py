#!/usr/bin/env python3
"""
Prototype A: LLM command format extraction from dialogue.

Validates that DeepSeek API can stably output natural language operation commands
in a structured ``{{action:target}}`` format from conversational user input.

Plan: .sisyphus/plans/proto-four-validate.md — Prototype A
v4.5.0 — throwaway prototype, do NOT commit to production path.

Usage:
    conda run -n cv311 python tests/manual/proto_a_command_format.py
"""

from __future__ import annotations

import logging
import os
import re
import sys
from typing import Any

import openai

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Command format constants (match plan specification)
# ---------------------------------------------------------------------------

VALID_ACTIONS: tuple[str, ...] = ("click", "type", "move", "right", "double")

# Regex to extract {{action:target}} from LLM response.
# Captures action name and target text (non-greedy, allows Chinese/emoji).
_CMD_RE = re.compile(r"\{\{(click|type|move|right|double):([^}]+?)\}\}")


# ---------------------------------------------------------------------------
# System prompt — personality + command output rules
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "你是雪奈(Yukina)，一个桌面AI助手。你的任务是理解用户的自然语言请求，"
    "并将其转换为桌面操作命令。\n\n"
    "## 命令输出规则\n"
    "当用户需要你执行桌面操作时，你必须在回复中使用以下格式：\n"
    "- 点击某个元素：{{click:目标名称}}\n"
    "- 输入文字：{{type:要输入的文字}}\n"
    "- 移动/拖拽某物：{{move:来源 → 目标}}\n"
    "- 右键菜单：{{right:目标名称}}\n"
    "- 双击打开：{{double:目标名称}}\n\n"
    "示例：\n"
    "- 用户说\"打开浏览器\" → 回复：\"好的，帮你打开浏览器 {{double:浏览器}}\"\n"
    "- 用户说\"搜索Python教程\" → 回复：\"马上搜索 {{type:Python教程}}\"\n\n"
    "## 重要规则\n"
    "1. 如果用户的请求涉及桌面操作，你必须输出至少一个{{action:target}}命令。\n"
    "2. 命令格式必须精确，使用双花括号{{}}。\n"
    "3. 你可以同时输出自然语言回复和命令，命令放在最后。\n"
    "4. 如果用户没有请求任何操作，正常聊天即可，不需要输出命令。"
)


# ---------------------------------------------------------------------------
# API client — mirrors deepseek_client.py pattern
# ---------------------------------------------------------------------------

class DeepSeekChat:
    """Minimal synchronous DeepSeek chat client for prototype testing.

    Mirrors the API call pattern from ``src/decision/deepseek_client.py``
    (AsyncOpenAI → synchronous OpenAI for standalone script).
    """

    def __init__(self) -> None:
        api_key: str = os.environ.get("DEEPSEEK_API_KEY", "")
        if not api_key:
            logger.warning("DEEPSEEK_API_KEY not set — API calls will fail")
        self._client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com/v1",
        )
        self._model: str = "deepseek-v4-flash"

    def chat(self, system_prompt: str, user_message: str) -> str:
        """Send a single-turn chat and return the assistant's text response.

        Args:
            system_prompt: System-level instructions (persona + command rules).
            user_message: The user's conversational input.

        Returns:
            The assistant's response text. Returns "[ERROR: ...]" on failure.
        """
        try:
            # Mirrors deepseek_client.py _call_api(): temperature=0.8, max_tokens=256,
            # thinking mode disabled for lower latency.  v4.5.0 §5.4
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.8,
                max_tokens=256,
                extra_body={"thinking": {"type": "disabled"}},
            )
            choice = response.choices[0] if response.choices else None
            if choice is None or choice.message is None or choice.message.content is None:
                return "[ERROR: empty API response]"
            return choice.message.content
        except Exception as exc:
            # Catches network errors, API errors, auth failures.
            # Safe: returns a string the caller can inspect.
            logger.warning("DeepSeek API call failed: %s", exc)
            return f"[ERROR: {exc}]"


# ---------------------------------------------------------------------------
# Command extraction
# ---------------------------------------------------------------------------

def extract_commands(text: str) -> list[tuple[str, str]]:
    """Extract all ``{{action:target}}`` commands from LLM response text.

    Args:
        text: The raw LLM response string.

    Returns:
        List of ``(action, target)`` tuples found.
    """
    return _CMD_RE.findall(text)


def has_command(text: str) -> bool:
    """Return True if the text contains at least one valid command format."""
    return bool(_CMD_RE.search(text))


# ---------------------------------------------------------------------------
# Main: run 5 test inputs
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    test_inputs: list[str] = [
        "帮忙搜一下DeepSeek",
        "打开那个",
        "把文件拖过去",
        "关掉这个窗口",
        "最大化",
    ]

    client = DeepSeekChat()
    results: list[dict[str, Any]] = []
    passed: int = 0

    for i, user_input in enumerate(test_inputs, 1):
        print(f"{'─' * 60}")
        print(f"Test {i}/5 — Input: {user_input}")
        print(f"{'─' * 60}")

        response = client.chat(SYSTEM_PROMPT, user_input)
        commands = extract_commands(response)
        ok = len(commands) > 0

        if ok:
            passed += 1
            for action, target in commands:
                print(f"  ✓ Found command: {{{{ {action}:{target} }}}}")
        else:
            print(f"  ✗ No command format found in response")

        print(f"  LLM Response: {response}")
        print()

        results.append({
            "input": user_input,
            "response": response,
            "commands": commands,
            "has_command": ok,
        })

    # Summary
    print(f"{'═' * 60}")
    print(f"SUMMARY: {passed}/{len(test_inputs)} responses contained command format")
    print(f"{'═' * 60}")
    for r in results:
        status = "✓" if r["has_command"] else "✗"
        cmds = ", ".join(f"{{{{{a}:{t}}}}}" for a, t in r["commands"]) if r["commands"] else "—"
        print(f"  {status} \"{r['input']}\" → {cmds}")

    rc = 0 if passed == len(test_inputs) else 1
    sys.exit(rc)


if __name__ == "__main__":
    main()
