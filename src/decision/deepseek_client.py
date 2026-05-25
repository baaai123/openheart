# v4.5.0 §5.4 — DeepSeek cloud fallback decision client
# Degradation matrix: local 3B → 1.5B → API fallback (项目宪法 §4.3)

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# System prompt — modular sections loaded from config/prompt_modules.json
# v4.5.0 §0.5 — config-driven, DI-friendly
# ------------------------------------------------------------------

import json
from pathlib import Path

_PROMPT_CONFIG_PATH = Path(__file__).parents[2] / "config" / "prompt_modules.json"
_UI_SETTINGS_PATH = Path(__file__).parents[2] / "config" / "ui_settings.json"
_ENDPOINTS_PATH = Path(__file__).parents[2] / "config" / "endpoints.yaml"


def _resolve_model_from_config() -> str | None:
    """Read model name from config/ui_settings.json → fallback config/endpoints.yaml.

    Returns the model string or None if neither config has a value.
    """
    # Try ui_settings.json first
    try:
        with open(_UI_SETTINGS_PATH, "r", encoding="utf-8") as f:
            ui: dict = json.load(f)
        model = ui.get("model", "") or ""
        if model:
            logger.info("model resolved from ui_settings.json: %s", model)
            return model
    except Exception as exc:
        logger.debug("ui_settings.json not readable for model query: %s", exc)

    # Fallback to endpoints.yaml
    try:
        import yaml
        with open(_ENDPOINTS_PATH, "r", encoding="utf-8") as f:
            ep: dict = yaml.safe_load(f) or {}
        model = (ep.get("deepseek", {}) or {}).get("model", "") or ""
        if model:
            logger.info("model resolved from endpoints.yaml: %s", model)
            return model
    except ImportError:
        logger.debug("PyYAML not available — skipping endpoints.yaml model lookup")
    except Exception as exc:
        logger.debug("endpoints.yaml not readable for model query: %s", exc)

    logger.info("model not configured in ui_settings.json or endpoints.yaml — using default")
    return None

def _load_prompt_modules() -> dict:
    try:
        with open(_PROMPT_CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def build_system_prompt(persona_override: str | None = None) -> str:
    """Assemble system prompt from config/prompt_modules.json.

    Args:
        persona_override: If provided, replaces the persona section from
            prompt_modules.json. Used by live config reload to inject the
            custom persona from server_config.json without losing other
            instruction modules (output_format, rules, capabilities, teaching).
    """
    pm = _load_prompt_modules()
    if persona_override is not None:
        pm["persona"] = persona_override
    parts = []
    if pm.get("persona"):
        parts.append(pm["persona"])
    if pm.get("output_format"):
        parts.append(pm["output_format"])
    rules = pm.get("reply_rules", {})
    for v in rules.values():
        if v:
            parts.append(v)
    caps = pm.get("capabilities", {})
    for v in caps.values():
        if v:
            parts.append(v)
    return " ".join(parts)


# Keep DEFAULT_SYSTEM_PROMPT for backwards compatibility
DEFAULT_SYSTEM_PROMPT = build_system_prompt()

SAFE = "SAFE"
NEEDS_CONFIRM = "NEEDS_CONFIRM"
DANGEROUS_AUTO_BLOCK = "DANGEROUS_AUTO_BLOCK"

_VALID_SAFETY_LEVELS = {SAFE, NEEDS_CONFIRM, DANGEROUS_AUTO_BLOCK}


class DeepSeekDecision:
    """Async DeepSeek API decision client using OpenAI-compatible SDK.

    Attributes:
        api_key: DeepSeek API key.
        base_url: API base URL (e.g. ``https://api.deepseek.com/v1``).
        model: Model identifier (e.g. ``deepseek-chat``).
        system_prompt: System prompt for the chat model.
    """

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "https://api.deepseek.com/v1",
        model: str = "deepseek-chat",
        system_prompt: str | None = None,
    ) -> None:
        """Initialise the DeepSeek decision client.

        Args:
            api_key: DeepSeek API key. If empty, all ``decide()`` calls will
                immediately return a degraded result.
            base_url: OpenAI-compatible API endpoint. Defaults to DeepSeek's API.
            model: Model name. Defaults to ``deepseek-chat``.
            system_prompt: Override for the default 雪奈 system prompt.
        """
        self.api_key: str = api_key
        self.base_url: str = base_url
        # Resolve model: explicit arg > ui_settings.json > endpoints.yaml > hardcoded default
        if model == "deepseek-chat":
            resolved = _resolve_model_from_config()
            if resolved:
                model = resolved
        self.model: str = model
        self.system_prompt: str = system_prompt or DEFAULT_SYSTEM_PROMPT
        self._client: Any = None  # lazy-import + init in decide()
        self._stream_lock = asyncio.Lock()
        logger.info(
            "DeepSeekDecision initialized (model=%s, base_url=%s)",
            self.model, self.base_url,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def decide(
        self,
        scene_summary: str = "",
        conversation_messages: list[dict[str, str]] | None = None,
        emotion_category: str = "neutral",
        personality_summary: str = "",
    ) -> dict[str, Any]:
        """Call DeepSeek API and return a structured decision.

        Args:
            scene_summary: Free-form text describing the current visual scene
                (from SyncVisionQuery / PerceptionBus, spec §1.x).
            conversation_messages: Chat history as a list of ``{"role": ..., "content": ...}``
                dicts. ``role`` must be ``"user"`` or ``"assistant"``.
            emotion_category: Detected user emotion — one of ``joy``, ``sadness``,
                ``neutral``, ``anger``, ``surprise`` (v4.5.0 §2.4). Module only uses it
                to bias the system instruction; final response emotion is up to the model.
            personality_summary: Current personality state summary (from
                ``personality/`` sub-system, spec §4.x).

        Returns:
            Decision dict matching the contract format (see module docstring).
            On error, returns a safe degraded fallback with ``confidence=0.3``
            and ``degraded=True`` in command metadata.
        """
        trace_id: str = str(uuid.uuid4())
        messages: list[dict[str, str]] = self._build_messages(
            scene_summary=scene_summary,
            conversation_messages=conversation_messages,
            emotion_category=emotion_category,
            personality_summary=personality_summary,
        )

        # If no API key configured, return degraded immediately.
        # Catches missing config — safe: caller gets a graceful degraded path.
        if not self.api_key:
            logger.warning(
                "DeepSeek API key not configured — returning degraded result. trace_id=%s",
                trace_id,
            )
            return self._degraded_response(trace_id, reason="no_api_key")

        try:
            response_text: str = await self._call_api(messages)
        except Exception as exc:
            # Covers network errors, timeouts, API errors, and SDK exceptions.
            # Safe: returns a degraded response so the caller can fall through
            # to other decision channels.
            logger.warning(
                "DeepSeek API call failed — returning degraded result. trace_id=%s error=%s",
                trace_id,
                exc,
            )
            return self._degraded_response(trace_id, reason=f"api_error: {exc}")

        return {
            "decision_type": "voice_response",
            "command": {
                "voice_response": response_text.strip(),
                "actions": [],
                "degraded": False,
            },
            "confidence": 0.8,
            "safety_level": SAFE,
            "trace_id": trace_id,
            "shadow_overridden": False,
            "source": "deepseek_api",
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_messages(
        self,
        scene_summary: str,
        conversation_messages: list[dict[str, str]] | None,
        emotion_category: str,
        personality_summary: str,
    ) -> list[dict[str, str]]:
        """Build the message list for the API call.

        Structure:
          1. System prompt (with emotion/personality/scene injected).
          2. Conversation history (if any).
          3. Final user turn instructing the model to respond.
        """
        system_content: str = self.system_prompt

        # Inject context into system prompt so the model has scene awareness
        # and emotional context.  v4.5.0 §5.4.1
        context_parts: list[str] = []
        if scene_summary:
            context_parts.append(f"当前场景: {scene_summary}")
        if emotion_category and emotion_category != "neutral":
            context_parts.append(f"用户情绪: {emotion_category}")
        if personality_summary:
            context_parts.append(f"你的状态: {personality_summary}")
        if context_parts:
            system_content = system_content + "\n\n" + "\n".join(context_parts)

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_content},
        ]

        # Append conversation history — assume caller has already truncated
        # to a reasonable window.  ContextAssembler (v4.5.0 §5.4.0) is the
        # canonical truncation point, but this client does NOT re-truncate.
        if conversation_messages:
            messages.extend(conversation_messages)

        # If the last message is from "user", the model will respond to it.
        # If not, we add an explicit user turn.
        if not messages or messages[-1].get("role") != "user":
            messages.append({"role": "user", "content": "请继续对话"})

        return messages

    async def _call_api(self, messages: list[dict[str, str]]) -> str:
        """Execute the API call via the OpenAI SDK with tool-calling loop.

        Supports up to 2 rounds of tool calls (v5.x spec). After the model returns
        ``tool_calls``, this method executes ``_execute_query_visual_tool()``, feeds
        the result back as a ``tool`` role message, and makes a second API call.
        If the model still requests tools after 2 rounds, a final call without
        tools is made to force a text response.

        Args:
            messages: The message list prepared by ``_build_messages``.

        Returns:
            The model's response text.

        Raises:
            Various exceptions from the OpenAI/httpx stack on network or API
            errors. Caller is responsible for catching and degrading.
        """
        # Lazy import — openai is an optional dependency for cloud fallback.
        # This keeps the decision layer loadable without the SDK installed.
        from openai import AsyncOpenAI  # v4.5.0 — optional cloud dependency

        client: AsyncOpenAI
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
        client = self._client

        # v5.x: Register visual memory query tool + executor for tool-calling loop
        from src.memory.query_tools import (
            QUERY_VISUAL_TOOL_DEFINITION,
            _execute_query_visual_tool,
        )

        MAX_TOOL_ROUNDS: int = 2  # v5.x — max 2 rounds per decision call

        for round_num in range(MAX_TOOL_ROUNDS):
            _t0 = time.monotonic()
            response = await client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.8,
                max_tokens=256,
                extra_body={"thinking": {"type": "disabled"}},
                tools=[QUERY_VISUAL_TOOL_DEFINITION],
                tool_choice="auto",
            )
            _dt = time.monotonic() - _t0
            print(f"[PERF-LLM] {_dt:.1f}s", flush=True)

            choice = response.choices[0] if response.choices else None
            if choice is None or choice.message is None:
                logger.warning(
                    "DeepSeek API returned empty response — using fallback text."
                )
                return "..."  # neutral fallback for empty API response

            # No tool_calls — return the text response directly
            if not choice.message.tool_calls:
                if choice.message.content is None:
                    logger.warning(
                        "DeepSeek API returned empty content — using fallback text."
                    )
                    return "..."  # neutral fallback for empty response
                return choice.message.content

            # Tool calls detected — execute each tool and prepare for next round.
            # v5.x: Tool pipeline — execute, feed result back for second call.
            n_tools: int = len(choice.message.tool_calls)
            logger.info(
                "DeepSeek requested %d tool call(s) — executing round %d of %d.",
                n_tools,
                round_num + 1,
                MAX_TOOL_ROUNDS,
            )

            # Append assistant message (with tool_calls) to messages so the
            # API sees the conversation history including the tool request.
            assistant_msg: dict[str, Any] = choice.message.model_dump(
                exclude_unset=True
            )
            messages.append(assistant_msg)  # type: ignore[arg-type]

            # Execute each tool call and append results as tool messages
            for tc in choice.message.tool_calls:
                # v5.x: Safe — _execute_query_visual_tool catches internal
                # errors and returns an error string, so this never raises.
                tool_result: str = await _execute_query_visual_tool(
                    {
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        }
                    }
                )
                messages.append({  # type: ignore[arg-type]
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_result,
                })

        # After MAX_TOOL_ROUNDS tool-loop iterations, make one final call
        # WITHOUT tools to force the model to produce a text response.
        # v5.x: Final synthesis call — no tools, model must respond with text.
        _t0 = time.monotonic()
        response = await client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.8,
            max_tokens=256,
            extra_body={"thinking": {"type": "disabled"}},
        )
        _dt = time.monotonic() - _t0
        print(f"[PERF-LLM] {_dt:.1f}s (final)", flush=True)

        choice = response.choices[0] if response.choices else None
        if choice is None or choice.message is None or choice.message.content is None:
            logger.warning(
                "DeepSeek API returned empty response — using fallback text."
            )
            return "..."  # neutral fallback for empty API response

        return choice.message.content

    def _degraded_response(
        self,
        trace_id: str,
        reason: str = "",
    ) -> dict[str, Any]:
        """Build a safe degraded fallback decision.

        Args:
            trace_id: Correlation ID for the original request.
            reason: Short description of why degradation occurred (logged).

        Returns:
            A decision dict with low confidence and degraded=True.
        """
        if reason:
            logger.warning(
                "DeepSeekDecision degraded. trace_id=%s reason=%s",
                trace_id,
                reason,
            )
        return {
            "decision_type": "voice_response",
            "command": {
                "voice_response": "",
                "actions": [],
                "degraded": True,
            },
            "confidence": 0.3,
            "safety_level": SAFE,
            "trace_id": trace_id,
            "shadow_overridden": False,
            "source": "deepseek_api",
        }

    async def stream_decide(self, user_message: str, conversation_messages: list, scene_summary: str = "", personality_state: str = "", memory_context: str = "", comfort_injection: str = "", spatial_context: str = ""):
        """Stream tokens from DeepSeek API. Yields (token_text, is_complete).

        v4.5.0 §4.6 — personality_state is injected as an additional system message
        so the model can adjust its tone to match the current dynamic personality.

        v4.5.0 §3.2.4 / §5.1 — memory_context replaces raw conversation_history in the
        cloud API request. Raw history is NEVER sent to DeepSeek; only the locally-generated
        privacy-safe summary is injected as a [历史记忆] system message.

        v4.5.0 §5.7.5 / §6.3 — comfort_injection is appended as a system message
        wrapping GentleReminder output.  Silent skip when empty.

        v4.5.0 §T2.5 / §6 — spatial_context is injected as a [空间布局] system message
        providing UI element spatial clustering info.  Silent skip when empty.
        """
        # ── API key validation (matching decide() pattern at line 186) ─
        if not self.api_key or not self.api_key.strip().startswith("sk-"):
            logger.warning(
                "DeepSeek API key not configured or invalid — returning degraded result."
            )
            yield "嗯，我在听呢。", True
            return
        # ── base_url validation — guard against config corruption ──────
        _cleaned_url = self.base_url.strip()
        if not _cleaned_url.startswith("http"):
            _cleaned_url = "https://api.deepseek.com/v1"
            logger.warning(
                "DeepSeek base_url invalid (%r) — falling back to default",
                self.base_url,
            )
        if _cleaned_url != self.base_url:
            self.base_url = _cleaned_url
            self._client = None  # Reset so new client uses corrected URL

        async with self._stream_lock:
            messages = [{"role": "system", "content": self.system_prompt}]
            if personality_state:
                messages.append({"role": "system", "content": personality_state})
            if comfort_injection:
                messages.append({"role": "system", "content": comfort_injection})
            if spatial_context:
                messages.append({"role": "system", "content": spatial_context})
            # Include full conversation history + memory context for coherent dialogue.
            if conversation_messages:
                messages.extend(conversation_messages)
            if memory_context:
                messages.append({"role": "system", "content": memory_context})
            if scene_summary:
                user_message = f"{scene_summary}\n\n用户说: {user_message}"
            messages.append({"role": "user", "content": user_message})
            if self._client is None:
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                )
            try:
                _t0 = time.monotonic()
                print(f"[LLM-CTX] msgs={len(messages) if messages else 0}", flush=True)
                stream = await self._client.chat.completions.create(
                    model=self.model, messages=messages, stream=True,
                    stream_options={"include_usage": True},
                    extra_body={"thinking": {"type": "disabled"}},
                )
                _dt = time.monotonic() - _t0
                print(f"[PERF-LLM] stream_decide: {_dt:.1f}s", flush=True)
            except Exception as e:
                import logging; logging.warning(f"DeepSeek stream error: {e}")
                yield "嗯，我在听呢。", True
                return
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    yield token, False
            yield "", True
