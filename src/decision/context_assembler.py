"""
ContextAssembler — assemble and truncate decision-layer context.  v4.5.0 §5.4.0

The ContextAssembler is the SOLE place where context truncation happens for the
decision layer. It enforces:

  - System prompt is ALWAYS preserved (never truncated).
  - Truncation occurs at message boundaries — a single message is never cut in half.
  - Default context limit: 2048 tokens (4096 in performance mode).
  - When VRAM drops below 1.0 GB, context is truncated to 50 % of current length
    and the event is logged as OPENMATE_OOM_PREVENTION.
  - Priority: System Prompt > Current Scene > Recent dialogue turns (hot memory
    summaries) > Cold memory summaries (discarded from tail by importance).

项目宪法 §1.2, §3.2: 禁止在 tokenization 阶段直接截断原始 token 序列。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from src.config.runtime import RuntimeConfig  # v4.5.0 §0.5

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — v4.5.0 §5.4.0
# ---------------------------------------------------------------------------

# Qwen2.5 chat template markers — preserved as boundaries during truncation.
_IM_START = "<|im_start|>"
_IM_END = "<|im_end|>"

# Role labels used within the chat template.
_ROLE_SYSTEM = "system"
_ROLE_USER = "user"
_ROLE_ASSISTANT = "assistant"

# Context limits — v4.5.0 §5.4.0 table
_CONTEXT_3B_DEFAULT = 2048
_CONTEXT_3B_PERFORMANCE = 4096
_CONTEXT_1_5B = 1024
_CONTEXT_0_5B = 512

# OOM prevention threshold (GB) — v4.5.0 §5.4.0
_OOM_THRESHOLD_GB = 1.0
# Truncation ratio when OOM prevention triggers
_OOM_TRUNCATION_RATIO = 0.5


# ---------------------------------------------------------------------------
# Data class for a single message in the chat template
# ---------------------------------------------------------------------------


@dataclass
class ChatMessage:
    """A single message in the Qwen2.5 chat template format.

    v4.5.0 §5.4.0: Context is assembled as a sequence of role-tagged messages
    following the Qwen2.5 chat_template structure.
    """

    role: str  # "system" | "user" | "assistant"
    content: str
    importance: float = 1.0  # higher = retain longer during truncation
    source: str = ""  # "system_prompt" | "scene" | "hot_memory" | "cold_memory" | "dialogue"


# ---------------------------------------------------------------------------
# System prompt template — v4.5.0 §5.4
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_TEMPLATE = (
    "你了解这位用户：\n"
    "- 性格：{personality}\n"
    "- 近期关注：{topics_of_interest}\n"
    "- 避免话题：{topics_to_avoid}\n"
    "- 你们的关系阶段：{relationship_stage}\n"
    "- 你可以称呼他：{nickname}\n"
    "你是一个名叫'温柔伙伴'的虚拟伙伴。\n"
    "当前人格特征：{dynamic_persona_summary}\n"
    "口头禅示例：{signature_phrases}\n"
    "当前用户情绪：{emotion_category}（强度 {emotion_intensity}）\n"
    "当前你的应对情绪：{tts_emotion}\n"
    "当前屏幕场景：{scene_primary}\n"
    "你的任务：提供情绪价值，回复要充满人格特色。不要机械，要像朋友一样。"
    "适当使用口头禅和表情文字。"
)

# New-user fallback values — v4.5.0 §5.4
NEW_USER_FALLBACK = {
    "personality": "我正在慢慢了解你",
    "topics_of_interest": "目前还在发现中",
    "topics_to_avoid": "",
    "relationship_stage": "初次相识",
    "nickname": "",
    "new_user_fallback": "你是第一次和我聊天的朋友，我还不太了解你，但我会用心倾听。",
}


# ---------------------------------------------------------------------------
# ContextAssembler
# ---------------------------------------------------------------------------


class ContextAssembler:
    """Assemble and truncate the decision model's input context.

    v4.5.0 §5.4.0:
      Truncation is performed here, at the message level, NOT in the tokenizer.
      The system prompt is always kept intact. Messages are never split mid-way.

    项目宪法 §1.2:
      禁止在 tokenization 阶段直接截断原始 token 序列。
      禁止在模块内部直接读取环境变量。

    项目宪法 §3.2:
      截断必须由 ContextAssembler 在高层上下文组装阶段完成。
    """

    def __init__(self, runtime_config: RuntimeConfig) -> None:
        """
        Args:
            runtime_config: Immutable runtime config (DI pattern — never
                reads os.environ directly per 项目宪法 §3.3).
        """
        self._config: RuntimeConfig = runtime_config
        self._context_limit: int = (
            _CONTEXT_3B_PERFORMANCE if runtime_config.performance_mode
            else _CONTEXT_3B_DEFAULT
        )
        # Tokenizer reference — set via inject_tokenizer() after model load.
        self._tokenizer: Any = None
        self._tokenizer_name: str = ""

    # ------------------------------------------------------------------
    # Tokenizer injection
    # ------------------------------------------------------------------

    def inject_tokenizer(self, tokenizer: Any, name: str = "") -> None:
        """Inject the model's tokenizer for accurate token counting.

        Called by the decision engine after the model is initialized (e.g. DeepSeek client).
        The tokenizer is used ONLY for counting — never for truncation
        (truncation happens here, at the message level).

        Args:
            tokenizer: A HuggingFace-style tokenizer with a
                ``__call__`` / ``encode`` method that accepts a string
                and returns token counts (via ``return_tensors`` or
                ``return_length``).
            name: Human-readable name for logging.
        """
        self._tokenizer = tokenizer
        self._tokenizer_name = name or getattr(tokenizer, "name_or_path", "unknown")

    # ------------------------------------------------------------------
    # Dynamic context limit adjustment — v4.5.0 §12.1
    # ------------------------------------------------------------------

    def set_context_limit(self, limit: int) -> None:
        """Dynamically adjust the context limit after construction.

        v4.5.0 §12.1: Used by Orchestrator to reduce context when
        voice mode runs on the LOW VRAM tier (< 12 GB), preventing
        CUDA OOM by reducing from 2048 to 1024 tokens.

        Args:
            limit: New context token limit (must be > 0).

        Raises:
            ValueError: if limit is <= 0.
        """
        if limit <= 0:
            raise ValueError(
                f"set_context_limit: limit must be > 0, got {limit}"
            )
        old_limit: int = self._context_limit
        self._context_limit = limit
        logger.info(
            "ContextAssembler: context limit adjusted %d → %d (voice mode OOM prevention)",
            old_limit, limit,
        )

    # ------------------------------------------------------------------
    # Token counting helper
    # ------------------------------------------------------------------

    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string using the injected tokenizer.

        If no tokenizer is available, falls back to a rough character-based
        estimate (Chinese: ~1 token/char, English: ~1 token/4 chars).

        v4.5.0 §5.4.0: token counting must use the actual model tokenizer
        when available, not a heuristic.

        Args:
            text: The text to count tokens for.

        Returns:
            Estimated or exact token count.
        """
        if self._tokenizer is not None:
            try:
                # Attempt HuggingFace tokenizer interface.
                encoded = self._tokenizer(text, return_length=True, add_special_tokens=False)
                if isinstance(encoded, dict):
                    return int(encoded.get("length", [len(text)])[0])
                # Some tokenizers return a BatchEncoding with input_ids.
                if hasattr(encoded, "input_ids"):
                    return len(encoded.input_ids[0])  # type: ignore[reportUnknownArgumentType]
            except Exception:
                # Expected: tokenizer interface mismatch (e.g. older API).
                # Fall back to heuristic below — not a crash-worthy error.
                logger.debug(
                    "Tokenizer count failed for tokenizer=%s, falling back to heuristic.",
                    self._tokenizer_name,
                )

        # Fallback heuristic: rough token estimate.
        # Chinese characters ≈ 1 token each, ASCII ≈ 0.25 tokens each.
        cjk_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff' or '\u3000' <= c <= '\u303f')
        ascii_chars = len(text) - cjk_chars
        return cjk_chars + max(1, ascii_chars // 4)

    # ------------------------------------------------------------------
    # Message formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _format_message(role: str, content: str) -> str:
        """Format a single message in Qwen2.5 chat template style.

        Args:
            role: One of "system", "user", "assistant".
            content: The message text content.

        Returns:
            A string like ``<|im_start|>role\ncontent<|im_end|>\n``.
        """
        return f"{_IM_START}{role}\n{content}{_IM_END}\n"

    # ------------------------------------------------------------------
    # System prompt construction
    # ------------------------------------------------------------------

    @classmethod
    def build_system_prompt(
        cls,
        *,
        personality: str = "",
        topics_of_interest: str = "",
        topics_to_avoid: str = "",
        relationship_stage: str = "",
        nickname: str = "",
        dynamic_persona_summary: str = "",
        signature_phrases: str = "",
        emotion_category: str = "neutral",
        emotion_intensity: float = 0.5,
        tts_emotion: str = "neutral",
        scene_primary: str = "other",
        is_new_user: bool = False,
    ) -> str:
        """Build the system prompt from template with variable substitution.

        v4.5.0 §5.4: when UserModel is empty or version < 2 (is_new_user=True),
        user-related fields use NEW_USER_FALLBACK values instead of real data.

        Args:
            personality: User personality description.
            topics_of_interest: Recent topics user cares about.
            topics_to_avoid: Topics to avoid in conversation.
            relationship_stage: Current relationship stage.
            nickname: User's preferred nickname.
            dynamic_persona_summary: Summary of current dynamic personality.
            signature_phrases: Representative catchphrases.
            emotion_category: Perceived user emotion (joy/sadness/neutral).
            emotion_intensity: Emotion intensity (0.0 to 1.0).
            tts_emotion: System's response emotion.
            scene_primary: Primary scene classification.
            is_new_user: If True, use fallback values for user fields.

        Returns:
            Formatted system prompt string.
        """
        if is_new_user:
            personality = NEW_USER_FALLBACK["personality"]
            topics_of_interest = NEW_USER_FALLBACK["topics_of_interest"]
            topics_to_avoid = NEW_USER_FALLBACK["topics_to_avoid"]
            relationship_stage = NEW_USER_FALLBACK["relationship_stage"]
            nickname = NEW_USER_FALLBACK["nickname"]

        return SYSTEM_PROMPT_TEMPLATE.format(
            personality=personality,
            topics_of_interest=topics_of_interest,
            topics_to_avoid=topics_to_avoid,
            relationship_stage=relationship_stage,
            nickname=nickname,
            dynamic_persona_summary=dynamic_persona_summary,
            signature_phrases=signature_phrases,
            emotion_category=emotion_category,
            emotion_intensity=emotion_intensity,
            tts_emotion=tts_emotion,
            scene_primary=scene_primary,
        )

    # ------------------------------------------------------------------
    # Message priority sorting — v4.5.0 §5.4.0
    # ------------------------------------------------------------------

    @staticmethod
    def _sort_messages_for_truncation(
        messages: list[ChatMessage],
    ) -> list[ChatMessage]:
        """Sort messages for truncation: most important first.

        v4.5.0 §5.4.0 truncation priority:
          1. Current Scene (keep first)
          2. Recent dialogue turns / hot memory summaries
          3. Cold memory summaries (discard from tail by importance)

        System messages are ALWAYS kept and not passed here (they are handled
        separately by the assembler).

        Args:
            messages: Non-system messages to sort.

        Returns:
            Sorted copy — most-retainable first, least-retainable last.
        """
        source_priority: dict[str, int] = {
            "scene": 100,
            "dialogue": 90,
            "hot_memory": 80,
            "cold_memory": 40,
            "": 0,
        }

        def sort_key(msg: ChatMessage) -> tuple[int, float]:
            src_pri = source_priority.get(msg.source, 0)
            return (-src_pri, -msg.importance)

        return sorted(messages, key=sort_key)

    # ------------------------------------------------------------------
    # Token-level check helper (GPU VRAM check)
    # ------------------------------------------------------------------

    def _check_vram_ok(self) -> bool:
        """Check if remaining GPU VRAM is above the OOM prevention threshold.

        v4.5.0 §5.4.0: When torch.cuda.mem_get_info() shows < 1.0 GB free,
        trigger context truncation to 50 %.

        Returns:
            True if VRAM is sufficient, False if OOM prevention should trigger.
        """
        # Attempt to import torch lazily — only needed for VRAM check.
        try:
            import torch  # type: ignore[import-untyped]  # pyright: ignore[reportMissingImports]
        except ImportError:
            # No CUDA / torch available (e.g. test environment) — assume OK.
            return True

        if not torch.cuda.is_available():  # pyright: ignore[reportUnknownMemberType]
            return True

        try:
            # mem_get_info returns (free, total) in bytes.
            free_bytes, _total_bytes = torch.cuda.mem_get_info()  # pyright: ignore[reportUnknownMemberType]
            free_gb = free_bytes / (1024 ** 3)
            return free_gb >= _OOM_THRESHOLD_GB
        except Exception:
            # Expected: CUDA error or driver issue — cannot check VRAM.
            # Conservative: assume OK to avoid false truncation.
            return True

    # ------------------------------------------------------------------
    # Main assembly and truncation logic
    # ------------------------------------------------------------------

    def assemble(
        self,
        *,
        system_prompt: str,
        messages: list[ChatMessage],
        trace_id: str = "",
        force_context_limit: int | None = None,
    ) -> str:
        """Assemble the full context string with truncation at message boundaries.

        v4.5.0 §5.4.0: This is the core method. It:
          1. Always keeps the system prompt intact.
          2. Checks VRAM; if < 1.0 GB free, reduces context to 50 % of limit.
          3. Sorts non-system messages by retention priority.
          4. Accumulates messages (each wrapped in chat template format) until
             the token budget is exhausted — messages are added atomically
             (never split mid-message).
          5. Logs OPENMATE_OOM_PREVENTION if truncation occurred.

        项目宪法 §3.2 rule 5: 截断发生时记录 OPENMATE_OOM_PREVENTION 日志.

        Args:
            system_prompt: The fully formatted system prompt (always preserved).
            messages: All non-system messages, each tagged with source and importance.
            trace_id: Trace ID for logging (项目宪法 §1.3 compliance).
            force_context_limit: Override the context limit (for testing only).

        Returns:
            The full assembled and truncated context string, ready for model.generate().
        """
        effective_limit = force_context_limit if force_context_limit is not None else self._context_limit

        # --- Check VRAM for OOM prevention ---
        vram_ok = self._check_vram_ok()
        oom_truncation_applied = False

        if not vram_ok:
            # v4.5.0 §5.4.0: OOM prevention — truncate to 50 % of current limit.
            effective_limit = max(1, int(effective_limit * _OOM_TRUNCATION_RATIO))
            oom_truncation_applied = True
            logger.warning(
                ("OPENMATE_OOM_PREVENTION: VRAM below %.1f GB threshold, "
                 "context limit reduced from %d to %d tokens. trace_id=%s"),
                _OOM_THRESHOLD_GB,
                self._context_limit,
                effective_limit,
                trace_id or "unknown",
            )

        # --- Step 1: System prompt (always intact) ---
        system_msg = self._format_message("system", system_prompt)
        system_tokens = self.count_tokens(system_msg)
        remaining_budget = effective_limit - system_tokens

        # v4.5.0 §5.4.0 rule 2: system prompt must be preserved even if it
        # alone exceeds the budget. In that case, log a warning but proceed
        # with just the system prompt (no other messages can fit).
        if remaining_budget <= 0:
            logger.warning(
                ("System prompt alone (%d tokens) exceeds context limit "
                 "(%d tokens). No additional messages can be included. "
                 "trace_id=%s"),
                system_tokens,
                effective_limit,
                trace_id or "unknown",
            )
            return system_msg

        # --- Step 2: Sort messages by retention priority ---
        sorted_messages = self._sort_messages_for_truncation(list(messages))

        # --- Step 3: Accumulate messages atomically ---
        assembled = [system_msg]
        tokens_used = system_tokens
        messages_included = 0
        messages_skipped = 0

        for msg in sorted_messages:
            formatted = self._format_message(msg.role, msg.content)
            msg_tokens = self.count_tokens(formatted)

            # v4.5.0 §5.4.0 rule 1: 禁止在单条message中间截断 — atomic inclusion.
            if tokens_used + msg_tokens <= remaining_budget:
                tokens_used += msg_tokens
                assembled.append(formatted)
                messages_included += 1
            else:
                messages_skipped += 1

        # --- Step 4: Log truncation results ---
        if messages_skipped > 0:
            logger.info(
                ("ContextAssembler truncated %d messages to fit within %d token "
                 "budget (used %d/%d tokens, %d messages included). trace_id=%s"),
                messages_skipped,
                effective_limit,
                tokens_used,
                effective_limit,
                messages_included,
                trace_id or "unknown",
            )

        if oom_truncation_applied:
            # v4.5.0 §5.4.0 rule 5: OPENMATE_OOM_PREVENTION must be logged.
            # (Already logged above; duplicate guard here for clarity.)
            logger.warning(
                ("OPENMATE_OOM_PREVENTION: context truncated from %d to %d "
                 "tokens. trace_id=%s"),
                self._context_limit,
                effective_limit,
                trace_id or "unknown",
            )

        return "".join(assembled)

    # ------------------------------------------------------------------
    # Convenience: message creation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def scene_message(content: str) -> ChatMessage:
        """Create a ChatMessage tagged as 'scene' (highest retention priority).

        v4.5.0 §5.4.0: current Scene is always in the context.
        """
        return ChatMessage(role="user", content=content, source="scene", importance=1.0)

    @staticmethod
    def dialogue_message(content: str, importance: float = 0.9) -> ChatMessage:
        """Create a ChatMessage tagged as 'dialogue'.

        Args:
            content: The dialogue turn text.
            importance: Higher = retained longer (default 0.9 for recent dialogue).
        """
        return ChatMessage(role="user", content=content, source="dialogue", importance=importance)

    @staticmethod
    def hot_memory_message(content: str, importance: float = 0.7) -> ChatMessage:
        """Create a ChatMessage tagged as 'hot_memory'.

        v4.5.0 §5.4.0: hot memory recent 3 Scene summaries.
        """
        return ChatMessage(role="user", content=content, source="hot_memory", importance=importance)

    @staticmethod
    def cold_memory_message(content: str, importance: float = 0.5) -> ChatMessage:
        """Create a ChatMessage tagged as 'cold_memory'.

        v4.5.0 §5.4.0: cold memory top 3 retrieval summaries.
        """
        return ChatMessage(role="user", content=content, source="cold_memory", importance=importance)

    @staticmethod
    def persona_message(content: str) -> ChatMessage:
        """Create a ChatMessage tagged as 'persona' with medium-high importance."""
        return ChatMessage(role="user", content=content, source="hot_memory", importance=0.8)

    @staticmethod
    def user_model_message(content: str) -> ChatMessage:
        """Create a ChatMessage with user model summary info."""
        return ChatMessage(role="user", content=content, source="hot_memory", importance=0.75)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def context_limit(self) -> int:
        """The current effective context limit (2048 default, 4096 performance)."""
        return self._context_limit

    @property
    def has_tokenizer(self) -> bool:
        """Whether a tokenizer has been injected for accurate token counting."""
        return self._tokenizer is not None
