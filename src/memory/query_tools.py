"""v5.x insight-memory-joint: LLM-facing visual memory query tool.

Provides ``query_visual_memory`` — a tiered retrieval function that the LLM
can call to answer spatial/visual questions about the current scene and
accumulated entity-graph patterns.

Tiers:
  1  Compact ~25-token summary from SharedContext (current frame only).
  2  ~150-token spatial context combining SharedContext + EntityGraph patterns.
  3  Full structured records from RetrievalGate across cold/deep tiers.

v4.5.0 §5 (SharedContext), §6.6 (EntityGraph), §5.x (RetrievalGate).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from src.insight.entity_graph import EntityGraph
from src.memory.retrieval_gate import RetrievalGate, get_global_gate
from src.memory.shared_context import NS_PERCEPTION, SharedContext
from src.memory.tier_types import TierLevel

logger = logging.getLogger(__name__)

_CONCEPTUAL_TIER_TO_TIERLEVELS: dict[int, list[TierLevel]] = {
    1: [TierLevel.HOT],
    2: [TierLevel.HOT, TierLevel.WARM],
    3: [TierLevel.COLD, TierLevel.DEEP],
}

def _estimate_tokens(text: str) -> int:
    """Rough token count: ~4 chars/word for Latin, ~2 chars for CJK.

    Not linguistically precise — used only for truncation budgeting so
    we stay within the 25 / 150 token targets.
    """
    if not text:
        return 0
    # Count CJK characters (Unicode range U+4E00–U+9FFF)
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    latin = len(text) - cjk
    # English ~4 chars/token, CJK ~2 chars/token
    return max(1, (latin // 4) + (cjk // 2))


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate *text* to approximately *max_tokens* tokens, appending '…' if cut."""
    if _estimate_tokens(text) <= max_tokens:
        return text
    # Binary search for the right character cutoff
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _estimate_tokens(text[:mid]) <= max_tokens:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo] + "…"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def query_visual_memory(
    query: str,
    tier: int = 2,
    *,
    gate: Optional[RetrievalGate] = None,
    entity_graph: Optional[EntityGraph] = None,
) -> dict[str, Any]:
    """Query visual/spatial memory across three tier levels.

    Conceptual tiers are internally mapped to storage TierLevel values:
      Tier 1 → HOT (session-level, ~25 tokens)
      Tier 2 → HOT + WARM (~150 tokens)
      Tier 3 → COLD + DEEP (full archive)

    Args:
        query: Natural-language question about the visual scene.
        tier: Retrieval depth (1, 2, or 3). Default 2.
        gate: Optional explicit RetrievalGate. Resolved via module-level
              singleton (get_global_gate) when None.
        entity_graph: Optional EntityGraph for spatial pattern queries.

    Returns:
        dict with keys: tier (int), summary (str), details (list[dict]).
    """
    resolved_gate = gate if gate is not None else get_global_gate()

    ctx = SharedContext.get_instance()

    tier = max(1, min(tier, 3))

    if tier == 1:
        visual_summary: str = ctx.get(NS_PERCEPTION, "visual_summary", "")
        if not visual_summary:
            frame_degraded = ctx.get(NS_PERCEPTION, "visual_frame_degraded", None)
            if frame_degraded is not None:
                visual_summary = (
                    "Visual frame is currently degraded — "
                    "no detailed scene description available."
                )
            else:
                visual_summary = "No visual frame data available yet."

        compact = _truncate_to_tokens(visual_summary, 25)
        return {
            "tier": 1,
            "summary": compact,
            "details": [],
        }

    if tier == 2:
        visual_summary = ctx.get(NS_PERCEPTION, "visual_summary", "")
        frame_degraded = ctx.get(NS_PERCEPTION, "visual_frame_degraded", False)

        pattern_details: list[dict[str, Any]] = []
        if entity_graph is not None:
            try:
                patterns = entity_graph.detect_patterns(min_occurrences=2)
                pattern_details = patterns[:5]
            except Exception:
                logger.warning(
                    "EntityGraph.detect_patterns failed (degraded) — "
                    "spatial patterns unavailable for query '%s'",
                    query[:80],
                    exc_info=True,
                )

        # Compose the ~150-token summary
        parts: list[str] = []
        if visual_summary:
            parts.append(f"Current scene: {visual_summary}")
        if frame_degraded:
            parts.append("[Visual frame is degraded — accuracy may be reduced.]")
        if pattern_details:
            pattern_lines = []
            for p in pattern_details:
                pattern_lines.append(
                    f"  {p['source']} --[{p['relation']}]--> {p['target']} "
                    f"(seen {p['count']} times)"
                )
            parts.append(
                "Recurring spatial relationships:\n" + "\n".join(pattern_lines)
            )
        if not parts:
            parts.append(
                "No visual frame or spatial pattern data available for the current context."
            )

        combined = "\n".join(parts)
        summary = _truncate_to_tokens(combined, 150)

        return {
            "tier": 2,
            "summary": summary,
            "details": pattern_details,
        }

    # --- Tier 3: full structured records from RetrievalGate -----------
    if tier == 3:
        visual_summary = ctx.get(NS_PERCEPTION, "visual_summary", "") or ""
        frame_degraded = ctx.get(NS_PERCEPTION, "visual_frame_degraded", False)

        # Collect EntityGraph patterns as supplementary detail
        pattern_details: list[dict[str, Any]] = []
        if entity_graph is not None:
            try:
                patterns = entity_graph.detect_patterns(min_occurrences=1)
                pattern_details = patterns
            except Exception:
                logger.warning(
                    "EntityGraph.detect_patterns failed (degraded) for Tier-3 query",
                    exc_info=True,
                )

        records: list[dict[str, Any]] = []
        degraded = False
        degraded_reason: list[str] = []

        if resolved_gate is not None:
            tier_levels = _CONCEPTUAL_TIER_TO_TIERLEVELS.get(
                tier, [TierLevel.COLD, TierLevel.DEEP]
            )
            try:
                tier_results = resolved_gate.query(
                    query_text=query,
                    tiers=tier_levels,
                    limit=10,
                )
                for rec in tier_results:
                    serialized: dict[str, Any] = {
                        "record_id": rec.record_id,
                        "tier": rec.tier.name if rec.tier else "unknown",
                        "importance": rec.importance,
                        "recency": rec.recency,
                        "tags": rec.tags,
                    }
                    if rec.payload is not None:
                        serialized["payload"] = rec.payload
                    records.append(serialized)
            except Exception:
                logger.warning(
                    "RetrievalGate.query failed for Tier-3 query '%s' (degraded)",
                    query[:80],
                    exc_info=True,
                )
                degraded = True
                degraded_reason.append("gate_query_failed")

        if not records:
            if resolved_gate is None:
                degraded = True
                degraded_reason.append("no_gate_available")
            else:
                degraded = True
                degraded_reason.append("stores_empty_or_no_match")

        # Build summary with degradation awareness
        prefix_parts: list[str] = []
        if visual_summary:
            prefix_parts.append(f"Current scene: {visual_summary}")
        if frame_degraded:
            prefix_parts.append("[Visual frame is degraded.]")
        record_count = len(records)
        if record_count > 0:
            prefix_parts.append(
                f"Retrieved {record_count} historical record(s) from long-term memory."
            )
            if degraded:
                prefix_parts.append(
                    "[Warning: some storage tiers degraded — results may be incomplete.]"
                )
        else:
            if degraded:
                prefix_parts.append(
                    "Long-term memory backends unavailable — "
                    "showing current scene context only (degraded to tier 2 level)."
                )
                if pattern_details:
                    prefix_parts.append(
                        "Recurring spatial patterns from current session are available "
                        "as supplementary context."
                    )
            else:
                prefix_parts.append(
                    "No matching historical records found in long-term memory."
                )
        summary = "\n".join(prefix_parts) if prefix_parts else (
            "No visual context available."
        )

        # Merge pattern details into the details list
        details: list[dict[str, Any]] = []
        if pattern_details:
            details.append({"type": "entity_patterns", "patterns": pattern_details})
        if records:
            details.append({"type": "tiered_records", "records": records})
        if degraded:
            details.append({"type": "degradation_info", "degraded": True, "reason": degraded_reason})

        return {
            "tier": 3,
            "summary": summary,
            "details": details,
            "degraded": degraded,
        }

    # Should never reach here (tier is clamped), but defensive fallback
    logger.warning("query_visual_memory: unhandled tier=%d, falling back to tier 1", tier)
    return await query_visual_memory(
        query, tier=1, gate=gate, entity_graph=entity_graph
    )


# ---------------------------------------------------------------------------
# OpenAI function calling schema for query_visual
# v4.5.0 §T2.5 — LLM tool pipeline for visual memory retrieval
# ---------------------------------------------------------------------------

QUERY_VISUAL_TOOL_DEFINITION: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "query_visual",
        "description": (
            "Query visual memory for screen context, spatial layout, or historical "
            "patterns. Use this when you need information about the user's current "
            "screen state (window titles, UI elements, OCR text, scene labels) or "
            "when the user asks about something they've seen before. "
            "Tier 1 = recent snapshot (~25 tokens), Tier 2 = current + spatial "
            "patterns (~150 tokens, default), Tier 3 = full long-term records."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Natural language query. Examples: 'what is on screen?', "
                        "'close button location', 'VS Code layout', 'browser tabs'."
                    ),
                },
                "tier": {
                    "type": "integer",
                    "description": (
                        "Retrieval depth: 1 = recent only (~25 tokens), "
                        "2 = recent + spatial patterns (~150 tokens, default), "
                        "3 = full long-term records."
                    ),
                    "minimum": 1,
                    "maximum": 3,
                },
            },
            "required": ["query"],
        },
    },
}


async def _execute_query_visual_tool(
    tool_call: dict[str, Any],
) -> str:
    """Execute a query_visual tool call and return formatted result string.

    Called by the LLM tool pipeline when DeepSeek returns a tool_call for
    ``query_visual``. Formats the dict result from query_visual_memory()
    into a string suitable for the `content` field of a 'tool' role message.

    Args:
        tool_call: OpenAI-format tool_call dict with ``function.name`` and
                   ``function.arguments`` (JSON string).

    Returns:
        Formatted result string for injection as a tool message.
    """
    import json as _json

    func_name = tool_call.get("function", {}).get("name", "")
    if func_name != "query_visual":
        return f"Unknown tool: {func_name}"

    try:
        args = _json.loads(tool_call.get("function", {}).get("arguments", "{}"))
    except _json.JSONDecodeError:
        return "[query_visual] Error: invalid JSON arguments."

    query_text = args.get("query", "")
    tier_val = int(args.get("tier", 2))

    if not query_text:
        return "[query_visual] Error: missing required 'query' parameter."

    try:
        result = await query_visual_memory(query=query_text, tier=tier_val)
    except Exception as exc:
        logger.warning(
            "query_visual tool execution failed: %s. degraded=true",
            exc,
        )
        return f"[query_visual] Error: {exc}"

    summary = result.get("summary", "") if isinstance(result, dict) else str(result)
    details = result.get("details", []) if isinstance(result, dict) else []

    lines: list[str] = [summary]
    if details:
        lines.append("\nSupporting records:")
        for d in details[:5]:
            lines.append(f"  - {d}")
    return "\n".join(lines)
