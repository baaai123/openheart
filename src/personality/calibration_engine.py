# v4.5.0 §4.6 — CalibrationEngine: independent personality evaluator
# Holds its own DeepSeekDecision instance with an EVALUATOR system prompt
# (NOT nahida's persona) to objectively measure style distance.
#
# Degradation: if the API is unavailable, returns neutral defaults.
# regression_damping is enforced by score-based rules per §4.6.

from __future__ import annotations

import json
import logging
import re
from typing import Any

from src.decision.deepseek_client import DeepSeekDecision

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Evaluator system prompt — 人格评估专家 (not 雪奈)
# ------------------------------------------------------------------

EVALUATOR_SYSTEM_PROMPT = (
    "你是一个客观的人格评估专家。你的任务是比较两段文本的风格一致性，"
    "判断后者是否偏离了前者的表达风格。"
    "输出格式：{\"score\": <1-10的整数>, "
    "\"deviation\": \"<偏离描述或'无偏离'>\", "
    "\"correction_hint\": \"<若偏离，一句修正建议，不超过20字；若无偏离则为空字符串>\", "
    "\"regression_damping\": <0.0-0.3的浮点数，建议的人格回拉阻尼值>}"
    "只输出JSON，不要输出其他内容。"
)


# ------------------------------------------------------------------
# Regression damping lookup table (v4.5.0 §4.6)
# ------------------------------------------------------------------

def _compute_regression_damping(score: int) -> float:
    """Enforce score-based regression_damping rules.

    v4.5.0 §4.6:
      score >= 8  → 0.0   (no correction needed)
      score 5-7   → 0.1   (mild regression)
      score < 5   → 0.2-0.3 (strong regression — use 0.25 as default)
    """
    if score >= 8:
        return 0.0
    elif score >= 5:
        return 0.1
    else:
        return 0.25


# ------------------------------------------------------------------
# CalibrationEngine
# ------------------------------------------------------------------

class CalibrationEngine:
    """Independent evaluator for personality calibration.

    Uses a separate ``DeepSeekDecision`` instance with an **evaluator**
    system prompt (NOT nahida's persona) to objectively measure
    stylistic distance between the baseline persona and the current reply.

    The evaluator returns a structured JSON result, which this class
    parses and enforces with score-based ``regression_damping`` rules.
    """

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "https://api.deepseek.com/v1",
        model: str = "deepseek-v4-flash",
    ) -> None:
        """Initialise the calibration engine.

        Args:
            api_key: DeepSeek API key.  If empty, ``evaluate()`` returns
                a neutral fallback result immediately.
            base_url: OpenAI-compatible API endpoint.
            model: Model identifier.
        """
        self._evaluator: DeepSeekDecision = DeepSeekDecision(
            api_key=api_key,
            base_url=base_url,
            model=model,
            system_prompt=EVALUATOR_SYSTEM_PROMPT,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def evaluate(
        self,
        baseline_persona: str,
        current_reply: str,
    ) -> dict[str, Any]:
        """Evaluate style distance between baseline persona and current reply.

        Args:
            baseline_persona: Human-readable description of the baseline
                personality (e.g. from ``BaselinePersonality.description``
                or the assembled system prompt).
            current_reply: The generated reply text to evaluate.

        Returns:
            Dict with keys:
                - ``score`` (int): Consistency score 1-10.
                - ``deviation`` (str): Deviation description or ``"无偏离"``.
                - ``correction_hint`` (str): Correction hint, max 20 chars.
                - ``regression_damping`` (float): Enforced by score rules
                  (0.0, 0.1, or 0.25).

            On API failure or parse error, returns a neutral fallback:
            ``{"score": 5, "deviation": "评估失败", "correction_hint": "",
               "regression_damping": 0.1}``.
        """
        # Build the calibration prompt — pass it as a single user message
        # so the evaluator model sees it naturally.  Using conversation_messages
        # avoids the hard-coded "请继续对话" suffix in _build_messages because
        # the last message role is "user".
        prompt: str = (
            f"基线人格描述：{baseline_persona}\n\n"
            f"当前回复：{current_reply}\n\n"
            f"请评估当前回复的风格一致性，返回JSON格式结果。"
        )

        try:
            result: dict[str, Any] = await self._evaluator.decide(
                conversation_messages=[{"role": "user", "content": prompt}],
            )
        except Exception:
            # Catches any unexpected error during the API call path.
            # Safe: returns a neutral fallback so calibration doesn't block
            # the personality pipeline.
            logger.warning(
                "CalibrationEngine.evaluate API call failed — returning neutral fallback.",
                exc_info=True,
            )
            return self._neutral_fallback()

        # Check for degraded response
        command: dict[str, Any] = result.get("command", {})
        if command.get("degraded", False):
            logger.warning(
                "CalibrationEngine.evaluate received degraded response — returning neutral fallback."
            )
            return self._neutral_fallback()

        response_text: str = command.get("voice_response", "")
        if not response_text:
            logger.warning(
                "CalibrationEngine.evaluate got empty voice_response — returning neutral fallback."
            )
            return self._neutral_fallback()

        return self._parse_response(response_text)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_response(response_text: str) -> dict[str, Any]:
        """Parse JSON from LLM response, handling markdown code blocks.

        Args:
            response_text: Raw text from the evaluator model.

        Returns:
            Parsed evaluation dict with enforced ``regression_damping``.
            On parse failure, returns neutral fallback.
        """
        # Try to extract JSON from the response — handle markdown code blocks
        # and pure JSON strings.
        json_str: str = response_text.strip()

        # v4.5.0 §4.6 — strip markdown code fences (```json ... ```)
        code_fence_pattern = re.compile(
            r"```(?:json)?\s*\n?(.*?)\n?```", re.DOTALL
        )
        match = code_fence_pattern.search(json_str)
        if match:
            json_str = match.group(1).strip()

        # If the response still doesn't look like JSON, try to find the
        # first '{' ... last '}' span.
        if not json_str.startswith("{"):
            start = json_str.find("{")
            end = json_str.rfind("}")
            if start != -1 and end != -1 and end > start:
                json_str = json_str[start : end + 1]

        # Parse
        try:
            parsed: dict[str, Any] = json.loads(json_str)
        except (json.JSONDecodeError, TypeError):
            # Catches malformed JSON — safe to fall back.
            logger.warning(
                "CalibrationEngine._parse_response failed to parse JSON from evaluator response: %r",
                response_text[:200],
            )
            return CalibrationEngine._neutral_fallback()

        # Enforce score-based regression_damping (v4.5.0 §4.6)
        score: int = int(parsed.get("score", 5))
        parsed["score"] = max(1, min(10, score))  # clamp 1-10
        parsed["regression_damping"] = _compute_regression_damping(parsed["score"])

        # Ensure all expected keys exist
        parsed.setdefault("deviation", "无偏离")
        parsed.setdefault("correction_hint", "")

        return parsed

    @staticmethod
    def _neutral_fallback() -> dict[str, Any]:
        """Return a neutral fallback evaluation when API is unavailable.

        v4.5.0 §4.6 — score=5 indicates no strong opinion; damping=0.1
        provides mild regression pull.
        """
        return {
            "score": 5,
            "deviation": "评估失败",
            "correction_hint": "",
            "regression_damping": 0.1,
        }
