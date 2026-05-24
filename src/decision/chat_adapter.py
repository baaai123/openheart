"""
chat_adapter — lightweight bidirectional format adapter.  v4.5.0

Bridges between runtime_loop's ``[{"role": ..., "content": ...}]`` dict format and
``ChatMessage`` dataclass objects, supporting ISO8601 timestamps and bulk conversion.

Conversions supported:
  - dict -> ChatMessage (to_chat_message)
  - ChatMessage -> dict (to_api_message)
  - list[ChatMessage] -> list[dict] (to_api_messages)

Task 6: Blocks Tasks 8, 13 (memory integration and ContextAssembler pipeline).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from collections.abc import Mapping


@dataclass
class ChatMessage:
    """A single chat message with ISO8601 timestamp support.

    This is the adapter's own dataclass, distinct from
    ``src.decision.context_assembler.ChatMessage``. It adds a ``timestamp``
    field that is auto-populated when converting from dict format.

    Fields:
        role: Message role — ``"system"``, ``"user"``, or ``"assistant"``.
        content: Message text content.
        timestamp: ISO8601 formatted datetime string. Auto-generated from
            ``datetime.now(timezone.utc)`` if missing during conversion.
    """

    role: str
    content: str
    timestamp: str = ""


_REQUIRED_ROLES = frozenset({"system", "user", "assistant"})
_ISO_FMT = "%Y-%m-%dT%H:%M:%S.%fZ"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime(_ISO_FMT)


def to_chat_message(d: Mapping[str, object]) -> ChatMessage:
    """Convert a ``{"role": ..., "content": ...}`` dict to ``ChatMessage``.

    Args:
        d: A dictionary with at least ``role`` and ``content`` keys.

    Returns:
        A ``ChatMessage`` instance.

    Raises:
        ValueError: If ``role`` is missing, empty, or not one of
            ``system``/``user``/``assistant``.
        ValueError: If ``content`` is missing or ``None``.

    Notes:
        - If ``d`` contains a ``"timestamp"`` key, it is passed through.
        - Otherwise, a new ISO8601 timestamp is auto-generated.
        - Extra keys in ``d`` are silently ignored.
    """
    role = d.get("role")
    if not role or not isinstance(role, str):
        roles_str = ", ".join(sorted(_REQUIRED_ROLES))
        raise ValueError(f"Missing or invalid 'role': expected one of {roles_str}, got {role!r}")
    role_norm = role.strip().lower()
    if role_norm not in _REQUIRED_ROLES:
        raise ValueError(
            f"Invalid role {role!r}: must be one of {sorted(_REQUIRED_ROLES)}"
        )

    if "content" not in d or d["content"] is None:
        raise ValueError("Missing or None 'content' in message dict")
    content = d["content"]
    if not isinstance(content, str):
        raise ValueError(f"'content' must be a string, got {type(content).__name__}")

    timestamp = d.get("timestamp", "")
    if not timestamp or not isinstance(timestamp, str):
        timestamp = _now_iso()

    return ChatMessage(role=role_norm, content=content, timestamp=timestamp)


def to_api_message(c: ChatMessage) -> dict[str, str]:
    """Convert a ``ChatMessage`` to an OpenAI-compatible dict.

    Args:
        c: A ``ChatMessage`` instance.

    Returns:
        A dict with ``"role"`` and ``"content"`` keys (OpenAI-compatible).
        The ``timestamp`` field is **not** included in the output dict
        (it is not part of the OpenAI API schema).
    """
    return {"role": c.role, "content": c.content}


def to_api_messages(messages: list[ChatMessage]) -> list[dict[str, str]]:
    """Bulk-convert a list of ``ChatMessage`` to OpenAI-compatible dicts.

    Args:
        messages: List of ``ChatMessage`` instances.

    Returns:
        List of ``{"role": ..., "content": ...}`` dicts.
    """
    result: list[dict[str, str]] = []
    for msg in messages:
        result.append(to_api_message(msg))
    return result
