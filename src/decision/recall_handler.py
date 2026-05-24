"""
{{recall}} tag handler — parse LLM response for memory retrieval tags (Task 9).

v4.5.0 §5 — Parses ``{{recall:type keywords}}`` patterns from LLM output,
searches VisualMemoryStore by type, returns formatted results, and updates
SharedContext for next-turn context injection.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from src.memory.cold.visual_schema import MEMORY_TYPES, VisualMemoryRecord
from src.memory.cold.visual_store import VisualMemoryStore

logger = logging.getLogger(__name__)

# v4.5.0 §5 — Task 9: recall tag pattern, used for both parse and strip
_RECALL_RE = re.compile(r"\{\{recall:(\w+)\s+([^}]+)\}\}")


def parse_recall_tags(text: str) -> list[dict[str, str]]:
    """Extract all ``{{recall:type keywords}}`` tags from ``text``.

    Args:
        text: LLM response text possibly containing ``{{recall:...}}`` tags.

    Returns:
        List of dicts, each with ``memory_type`` (str) and ``query`` (str).
        Empty list if no tags found.
    """
    queries: list[dict[str, str]] = []
    for match in _RECALL_RE.finditer(text):
        memory_type = match.group(1)
        query = match.group(2).strip()
        if not query:
            continue
        queries.append({"memory_type": memory_type, "query": query})
    return queries


def strip_recall_tags(text: str) -> str:
    """Remove all ``{{recall:...}}`` tags from ``text``.

    Used to clean LLM output before TTS so the tags are not spoken aloud.

    Args:
        text: Text containing ``{{recall:...}}`` tags.

    Returns:
        Text with all recall tags removed.
    """
    return _RECALL_RE.sub("", text)


async def handle_recall_tags(
    text: str,
    visual_store: Optional[VisualMemoryStore],
    top_k: int = 3,
) -> str:
    """Search ``visual_store`` for each ``{{recall:type keywords}}`` tag.

    For each valid tag, calls ``visual_store.search_by_type(type, query, top_k)``
    and collects results.  Silently skips unknown ``memory_type`` values (with
    a WARNING log).  Gracefully degrades if ``visual_store`` is ``None``.

    Args:
        text: LLM response text containing ``{{recall:...}}`` tags.
        visual_store: VisualMemoryStore instance, or ``None`` (degrades
            gracefully with WARNING log).
        top_k: Maximum results **per query**.

    Returns:
        Formatted results string::

            [系统：检索到 N 条相关记忆：
            1. {content}
            2. {content}]

        Empty string if no tags, no store, or no results.

    Side effects:
        Sets ``SharedContext[NS_DECISION]["last_recall"]`` with
        ``{"type": str, "query": str, "count": int}``.
    """
    from src.memory.shared_context import NS_DECISION, SharedContext

    queries = parse_recall_tags(text)
    if not queries:
        return ""

    if visual_store is None:
        logger.warning(
            "VisualMemoryStore not initialized — {{recall}} tags ignored",
        )
        return ""

    all_results: list[tuple[str, str, VisualMemoryRecord]] = []

    for q in queries:
        memory_type = q["memory_type"]
        query = q["query"]

        if memory_type not in MEMORY_TYPES:
            logger.warning(
                "Unknown memory_type '%s' in {{recall}} tag — "
                "valid: %s. Skipping.",
                memory_type,
                MEMORY_TYPES,
            )
            continue

        if not query:
            continue

        try:
            # v4.5.0 §5 — search_by_type is async; returns list[VisualMemoryRecord]
            records = await visual_store.search_by_type(
                memory_type, query, top_k=top_k,
            )
        except Exception:
            # try/except safe: logs full traceback at WARNING, never crashes
            logger.warning(
                "visual_store.search_by_type failed "
                "(type=%s query='%s') — skipping tag",
                memory_type,
                query,
                exc_info=True,
            )
            continue

        for record in records:
            all_results.append((memory_type, query, record))

    if not all_results:
        return ""

    # ── Record last recall in SharedContext ────────────────────────
    last_type, last_query, _ = all_results[-1]
    shared_ctx = SharedContext.get_instance()
    try:
        shared_ctx.set(
            NS_DECISION,
            "last_recall",
            {"type": last_type, "query": last_query, "count": len(all_results)},
        )
    except Exception:
        # try/except safe: SharedContext set failure is non-fatal
        logger.warning("SharedContext.set(last_recall) failed", exc_info=True)

    # ── Format results ─────────────────────────────────────────────
    lines: list[str] = []
    for i, (_, _, record) in enumerate(all_results, start=1):
        lines.append(f"{i}. {record.content_text}")

    return f"[系统：检索到 {len(lines)} 条相关记忆：\n" + "\n".join(lines) + "]"
