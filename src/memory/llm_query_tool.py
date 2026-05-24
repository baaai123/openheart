"""LLM-facing memory query tool — structured search across all memory tiers.

``query_memory`` is the primary entry-point for LLM-driven memory retrieval.
It searches across all five tiers (HOT→DEEP) via RetrievalGate, then filters
by type and time range before returning privacy-cleaned results.

``format_results_for_llm`` renders the result dict into compact natural-language
text suitable for injection into an LLM context window.

v4.5.0 §3.5 (MemoryService.get_memory_drawer),
       §5.1 (MemorySnapshot.to_prompt_text).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from src.memory.privacy_filter import filter_sensitive
from src.memory.retrieval_gate import RetrievalGate, get_global_gate
from src.memory.tier_types import TierLevel, TieredRecord

logger = logging.getLogger(__name__)

_EPOCH = datetime(2000, 1, 1, tzinfo=timezone.utc)

_TIME_RANGE_ALIASES: dict[str, timedelta] = {
    "last_1h": timedelta(hours=1),
    "last_24h": timedelta(hours=24),
    "last_7d": timedelta(days=7),
}

_VALID_MEMORY_TYPES: frozenset[str] = frozenset({
    "visual", "emotion", "fact", "action", "conversation", "all",
})

_MEMORY_TYPE_TRANSLATIONS: dict[str, str] = {
    "EMOTION": "emotion",
    "FACT": "fact",
    "ACTION": "action",
}


def _parse_time_range(time_range: str | None) -> tuple[datetime, datetime] | None:
    """Parse a time_range string into (start, end) datetime bounds.

    Returns None when the range covers all time (no filtering needed).
    """
    if time_range is None:
        return None
    time_range = time_range.strip().lower()

    alias = _TIME_RANGE_ALIASES.get(time_range)
    if alias is not None:
        now = datetime.now(timezone.utc)
        return (now - alias, now)

    parts = time_range.split("/")
    if len(parts) == 2:
        try:
            start = datetime.strptime(parts[0].strip(), "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            )
            end = datetime.strptime(parts[1].strip(), "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            )
            return (start, end + timedelta(days=1))
        except ValueError:
            logger.warning(
                "query_memory: invalid date range '%s' — ignoring time filter",
                time_range,
            )
            return None

    logger.warning(
        "query_memory: unrecognised time_range '%s' — ignoring time filter",
        time_range,
    )
    return None


def _build_gate_from_service(memory_service: Any) -> RetrievalGate:
    """Populate a RetrievalGate from an injected MemoryService.

    Maps MemoryService's hot/cold clients into the gate's ``_stores`` dict
    so ``gate.query()`` can reach all available storage backends.
    """
    stores: dict[str, Any] = {}
    try:
        if hasattr(memory_service, "_hot") and memory_service._hot is not None:
            stores["hot"] = memory_service._hot
        if hasattr(memory_service, "_cold") and memory_service._cold is not None:
            stores["cold"] = memory_service._cold
    except Exception:
        logger.warning(
            "Failed to introspect memory_service stores — using empty gate"
        )
    return RetrievalGate(stores=stores, timeout_ms=500)


def _record_timestamp(record: TieredRecord) -> datetime:
    """Extract a best-effort datetime from a TieredRecord's payload."""
    ts = record.recency
    if isinstance(ts, (int, float)) and ts > 0:
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    payload = record.payload
    if isinstance(payload, dict):
        ts_str = payload.get("timestamp") or payload.get("created_at")
        if ts_str:
            try:
                return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass
        ts_val = payload.get("recency")
        if isinstance(ts_val, (int, float)) and ts_val > 0:
            return datetime.fromtimestamp(ts_val, tz=timezone.utc)
    return _EPOCH


def _matches_memory_type(record: TieredRecord, memory_type: str) -> bool:
    """Check whether a TieredRecord matches the requested memory_type."""
    if memory_type in (None, "all"):
        return True

    payload = record.payload
    if isinstance(payload, dict):
        rtype = payload.get("memory_type", "")
        translated = _MEMORY_TYPE_TRANSLATIONS.get(rtype, rtype.lower())
        if translated == memory_type:
            return True
        tags_lower = {t.lower() for t in record.tags}
        if memory_type in tags_lower:
            return True
        scene_class = payload.get("scene_class", "").lower()
        if scene_class == memory_type:
            return True
    return False


async def query_memory(
    query_text: str,
    memory_type: str | None = None,
    time_range: str | None = None,
    top_k: int = 10,
    memory_service: Any | None = None,
) -> dict[str, Any]:
    """Search all memory tiers and return filtered, privacy-cleaned results.

    Args:
        query_text: Natural-language search query.
        memory_type: Filter to a specific memory category.
            One of ``"visual"``, ``"emotion"``, ``"fact"``, ``"action"``,
            ``"conversation"``, ``"all"``, or ``None`` (no filter).
        time_range: Narrow results to a time window.
            ``"last_1h"``, ``"last_24h"``, ``"last_7d"``, or a
            ``"YYYY-MM-DD/YYYY-MM-DD"`` date range.
        top_k: Maximum results to return (default 10).
        memory_service: Optional MemoryService instance providing hot/cold
            store clients.  When ``None``, results will be empty.

    Returns:
        dict with:
          - ``results`` (list[dict]): Filtered, privacy-cleaned records.
          - ``metadata`` (dict): ``total_found``, ``tiers_searched``, ``degraded``.
    """
    degraded = False
    tiers_searched: list[str] = []
    results: list[dict[str, Any]] = []

    if memory_type is not None and memory_type not in _VALID_MEMORY_TYPES:
        logger.warning(
            "query_memory: unknown memory_type '%s' — treating as 'all'",
            memory_type,
        )
        memory_type = None

    time_bounds = _parse_time_range(time_range)

    gate: RetrievalGate | None = None
    if memory_service is not None:
        try:
            gate = _build_gate_from_service(memory_service)
        except Exception:
            logger.warning(
                "query_memory: failed to build RetrievalGate from memory_service (degraded)",
                exc_info=True,
            )
            degraded = True

    if gate is None:
        gate = get_global_gate()

    if gate is None:
        return {
            "results": [],
            "metadata": {
                "total_found": 0,
                "tiers_searched": [],
                "degraded": degraded,
            },
        }

    tiers = list(TierLevel)
    try:
        raw_results = gate.query(
            query_text=query_text,
            tiers=tiers,
            limit=top_k * 3,
            timeout_ms=500,
        )
        tiers_searched = [t.name for t in tiers]
    except Exception:
        logger.warning(
            "query_memory: RetrievalGate.query failed (degraded) for '%s'",
            query_text[:80],
            exc_info=True,
        )
        return {
            "results": [],
            "metadata": {
                "total_found": 0,
                "tiers_searched": [],
                "degraded": True,
            },
        }

    for record in raw_results:
        if not _matches_memory_type(record, memory_type or "all"):
            continue

        if time_bounds is not None:
            ts = _record_timestamp(record)
            if ts < time_bounds[0] or ts > time_bounds[1]:
                continue

        payload = record.payload
        if isinstance(payload, dict):
            summary = payload.get("summary") or payload.get("scene_summary") or ""
        else:
            summary = str(payload) if payload is not None else ""

        summary = filter_sensitive(summary)

        results.append({
            "type": (
                _MEMORY_TYPE_TRANSLATIONS.get(
                    payload.get("memory_type", ""),
                    payload.get("memory_type", "").lower(),
                )
                if isinstance(payload, dict)
                else "unknown"
            ),
            "summary": summary,
            "similarity": round(record.importance, 2),
            "record_id": record.record_id,
            "tier": record.tier.name,
            "importance": round(record.importance, 2),
            "tags": record.tags,
        })

    results.sort(key=lambda r: r["similarity"], reverse=True)
    results = results[:top_k]

    return {
        "results": results,
        "metadata": {
            "total_found": len(results),
            "tiers_searched": tiers_searched,
            "degraded": degraded,
        },
    }


def format_results_for_llm(results: dict[str, Any]) -> str:
    """Render a ``query_memory`` result dict as compact LLM-friendly text.

    Format:
        [emotion] User felt joyful about progress (similarity: 0.85)
        [fact] User mentioned Python 3.12 features (similarity: 0.72)
    """
    result_list: list[dict[str, Any]] = results.get("results", [])
    if not result_list:
        return "[记忆检索] 未找到相关记忆。"

    lines: list[str] = ["[记忆检索] 找到以下相关记忆："]
    for r in result_list:
        rtype = r.get("type", "unknown")
        summary = r.get("summary", "")
        sim = r.get("similarity", 0)
        if summary:
            lines.append(f"[{rtype}] {summary} (similarity: {sim:.2f})")
        else:
            lines.append(f"[{rtype}] (无内容) (similarity: {sim:.2f})")

    metadata = results.get("metadata", {})
    degraded = metadata.get("degraded", False)
    if degraded:
        lines.append("(注意: 部分记忆存储不可用，结果可能不完整)")

    return "\n".join(lines)
