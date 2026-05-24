"""v5.x insight-memory-joint: ReflectionEngine — cross-tier pattern discovery."""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from src.infra.tracing import TraceManager, trace_span
from src.insight.entity_graph import EntityGraph
from src.memory.retrieval_gate import RetrievalGate
from src.memory.tier_types import TierLevel, TieredRecord

logger = logging.getLogger(__name__)


class ReflectionEngine:
    """Background task for cross-tier pattern discovery.
    
    Scans T1 (Warm) for recent topics → expands to T3 (Cold) →
    connects via EntityGraph → stores insights in T4 (Deep).
    """

    def __init__(
        self,
        retrieval_gate: RetrievalGate,
        entity_graph: EntityGraph,
        interval_seconds: float = 5.0,
    ) -> None:
        self._gate = retrieval_gate
        self._graph = entity_graph
        self._interval = interval_seconds
        self._running = False

    async def run(self, stop_event: Optional[asyncio.Event] = None) -> None:
        """Main reflection loop."""
        async with TraceManager(layer="memory", component="reflection_engine", operation="run"):
            self._running = True
            logger.info("ReflectionEngine started (interval=%.1fs)", self._interval)
            while self._running:
                try:
                    await self._reflect_cycle()
                except Exception as e:
                    logger.warning("ReflectionEngine cycle failed: %s", e)
                await asyncio.sleep(self._interval)
                if stop_event and stop_event.is_set():
                    self._running = False

    @trace_span(layer="memory", component="reflection_engine", operation="reflect_cycle")
    async def _reflect_cycle(self) -> None:
        """One reflection cycle."""
        # Scan T1 (Warm) for recent topics
        warm_records = self._gate.query("", tiers=[TierLevel.WARM], limit=5)
        
        # Detect spatial patterns from EntityGraph
        patterns = self._graph.detect_patterns(min_occurrences=3)
        
        if patterns:
            logger.info("ReflectionEngine: detected %d patterns", len(patterns))
            for p in patterns[:3]:
                insight = TieredRecord(
                    record_id=f"reflection_{p['source']}_{p['target']}",
                    tier=TierLevel.DEEP,
                    importance=0.7,
                    recency=0.0,
                    access_count=1,
                    tags=["insight", "spatial_pattern"],
                    payload={
                        "type": "spatial_pattern",
                        "source": p["source"],
                        "target": p["target"],
                        "relation": p["relation"],
                        "count": p["count"],
                    },
                )
                self._gate.write_record(insight)
        elif warm_records:
            logger.debug("ReflectionEngine: %d warm records, no patterns yet", len(warm_records))
