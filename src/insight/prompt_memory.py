"""v5.x insight-memory-joint: PromptMemory — concept-to-reference mapping for SAVPE."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import yaml

from src.insight.types import PromptRef
from src.memory.retrieval_gate import RetrievalGate
from src.memory.tier_types import TierLevel, TieredRecord

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = "config/insight.yaml"


class PromptMemory:
    """Persistent store of visual concept references for YOLOE SAVPE inference.
    
    Dual storage:
      T1 (Warm, in-memory dict): active session concepts, fast recall
      T3 (Cold, RetrievalGate/LanceDB): full archive, persistent
    
    VPE embeddings: calculated at remember() time, cached for 24h.
    """

    def __init__(
        self,
        retrieval_gate: Optional[RetrievalGate] = None,
        config_path: str = _DEFAULT_CONFIG_PATH,
    ) -> None:
        self._gate = retrieval_gate
        self._config = self._load_config(config_path)
        pm_cfg = self._config.get("prompt_memory", {})
        self._t1_cache_size = int(pm_cfg.get("t1_cache_size", 200))
        self._max_recall_concepts = int(pm_cfg.get("max_recall_concepts", 200))
        self._vpe_recalc_hours = int(pm_cfg.get("vpe_recalc_hours", 24))
        self._vpe_max_age_hours = int(pm_cfg.get("vpe_max_age_hours", 72))
        # T1: in-memory cache (keyed by prompt_id)
        self._t1_cache: dict[str, PromptRef] = {}

    @staticmethod
    def _load_config(path: str) -> dict:
        try:
            with open(path, "r") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            logger.warning("PromptMemory: cannot load config from %s", path)
            return {}

    # ------------------------------------------------------------------
    # Remember (write)
    # ------------------------------------------------------------------

    def remember(
        self,
        name: str,
        crop: np.ndarray,
        context_tags: list[str],
        confidence: float = 0.5,
    ) -> str:
        """Store a new concept reference. Returns prompt_id."""
        # v5.x: Dedup — skip if similar name exists
        for existing in self._t1_cache.values():
            if self._name_similar(name, existing.name):
                existing.last_seen = datetime.now(timezone.utc)
                existing.confidence = max(existing.confidence, confidence)
                return existing.id

        prompt_id = uuid.uuid4().hex[:12]
        now = datetime.now(timezone.utc)

        ref = PromptRef(
            id=prompt_id,
            name=name,
            crop=crop,
            vpe_embedding=None,  # computed lazily by ConceptClassifier
            context_tags=context_tags,
            confidence=confidence,
            last_seen=now,
            # vpe never computed yet → expires now (triggers compute on first recall)
            vpe_expires_at=now,
        )

        # Write to T1 cache
        if len(self._t1_cache) >= self._t1_cache_size:
            # Evict oldest by last_seen
            oldest = min(self._t1_cache.values(), key=lambda r: r.last_seen)
            del self._t1_cache[oldest.id]
        self._t1_cache[prompt_id] = ref

        # Write to T3 (LanceDB via RetrievalGate)
        if self._gate:
            record = TieredRecord(
                record_id=prompt_id,
                tier=TierLevel.COLD,
                importance=float(confidence),
                recency=now.timestamp(),
                access_count=0,
                tags=context_tags,
                payload={"name": name, "confidence": confidence},
            )
            self._gate.write_record(record)

        logger.debug("PromptMemory: remembered concept '%s' (id=%s, tags=%s)", name, prompt_id, context_tags)
        return prompt_id

    # ------------------------------------------------------------------
    # Recall (read)
    # ------------------------------------------------------------------

    def recall(
        self, context_tags: list[str], limit: Optional[int] = None
    ) -> list[PromptRef]:
        """Retrieve concepts matching context tags (strict match).

        Args:
            context_tags: Tags to filter concepts by (intersection match).
            limit: Max concepts to return. None uses self._max_recall_concepts
                   (default 200 from config). Set to 0 for unlimited (no slice).
        """
        # Resolve limit: None → config value, <=0 → unlimited
        if limit is None:
            limit = self._max_recall_concepts
        elif limit <= 0:
            limit = None  # unlimited

        results: list[PromptRef] = []

        # Check T1 cache first
        for ref in self._t1_cache.values():
            if self._tags_match(ref.context_tags, context_tags):
                results.append(ref)

        # T3 fallback via RetrievalGate
        if self._gate and (limit is None or len(results) < limit):
            try:
                t3_limit = 10_000 if limit is None else limit - len(results)
                t3_results = self._gate.query(
                    query_text=" ".join(context_tags),
                    tiers=[TierLevel.COLD],
                    limit=t3_limit,
                )
                for r in t3_results:
                    payload = r.payload or {}
                    results.append(PromptRef(
                        id=r.record_id,
                        name=payload.get("name", "unknown"),
                        context_tags=r.tags,
                        confidence=float(payload.get("confidence", 0.5)),
                    ))
            except Exception:
                logger.warning("PromptMemory T3 recall failed", exc_info=True)

        # Validate VPE cache freshness
        for ref in results:
            now = datetime.now(timezone.utc)
            age_hours = (now - ref.vpe_expires_at).total_seconds() / 3600
            if age_hours > self._vpe_max_age_hours:
                ref.vpe_embedding = None  # expired → needs recompute

        return results if limit is None else results[:limit]

    @staticmethod
    def _tags_match(ref_tags: list[str], query_tags: list[str]) -> bool:
        """Strict match: at least one tag must be shared."""
        if not query_tags:
            return True
        return bool(set(ref_tags) & set(query_tags))

    # ------------------------------------------------------------------
    # Forget (delete)
    # ------------------------------------------------------------------

    def forget(self, prompt_id: str) -> bool:
        """Remove a concept from T1 and T3. Returns True if found."""
        found = False
        if prompt_id in self._t1_cache:
            del self._t1_cache[prompt_id]
            found = True
        # T3 forget not implemented for simplicity (LanceDB delete is async)
        return found

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _name_similar(a: str, b: str) -> bool:
        """Check if two concept names are similar enough to dedup."""
        if a == b:
            return True
        # Basic overlap: if one contains the other or shares key words
        a_set = set(a)
        b_set = set(b)
        overlap = len(a_set & b_set)
        shorter = min(len(a_set), len(b_set))
        return shorter > 0 and overlap / shorter > 0.5  # >50% character overlap

    def list_concepts(self) -> list[str]:
        """Return all known concept names (T1 only for speed)."""
        return list({ref.name for ref in self._t1_cache.values()})

    def get_vpe(self, name: str) -> Optional[np.ndarray]:
        """Get cached VPE embedding for a concept name. Returns None if not found/expired."""
        now = datetime.now(timezone.utc)
        for ref in self._t1_cache.values():
            if ref.name == name:
                age_hours = (now - ref.vpe_expires_at).total_seconds() / 3600
                if age_hours > self._vpe_recalc_hours:
                    return None  # expired → trigger recompute
                return ref.vpe_embedding
        return None

    def set_vpe(self, prompt_id: str, vpe: np.ndarray) -> None:
        """Store a computed VPE embedding for a PromptRef."""
        if prompt_id in self._t1_cache:
            self._t1_cache[prompt_id].vpe_embedding = vpe
            self._t1_cache[prompt_id].vpe_expires_at = datetime.now(timezone.utc)
