"""
VisualMemoryStore — LanceDB-backed classified visual memory.

v5.x visual-memory-design, Option A: separate store class, shared LanceDB
directory with ColdMemoryStore (data/cold_memory), separate table (visual_memory).

Patterns borrowed from ColdMemoryStore (src/memory/cold/memory_store.py):
  - lancedb.connect_async for DB handle
  - sentence-transformers lazy-load for embeddings
  - zero-vector fallback on embedding failure
  - WARNING-level logging with trace context

Table schema: visual_schema.py / VisualMemoryRecord
  memory_id | timestamp | memory_type | content_text | source_window |
  tags | embedding[512] | meta_json

Indices: scalar on memory_type, source_window, timestamp; vector IVF_PQ on embedding.
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any

import numpy as np

from src.memory.cold.visual_schema import (
    EMBEDDING_DIM,
    MEMORY_TIERS,
    MEMORY_TYPES,
    TABLE_NAME,
    VisualMemoryRecord,
    _seed_record,
)

logger = logging.getLogger(__name__)

_DEFAULT_DB_PATH = "data/cold_memory"
_DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"


class VisualMemoryStore:
    """Async LanceDB store for classified visual memory.

    Shared LanceDB directory with ColdMemoryStore. Separate table to avoid
    schema collision. Optimized for type-filtered vector search — the
    {{recall:type keywords}} retrieval pattern.
    """

    def __init__(
        self,
        db_path: str = _DEFAULT_DB_PATH,
        embedding_model_name: str = _DEFAULT_EMBEDDING_MODEL,
        embedding_dim: int = EMBEDDING_DIM,
    ) -> None:
        self._db_path: str = db_path
        self._embedding_model_name: str = embedding_model_name
        self._embedding_dim: int = embedding_dim

        self._db: Any = None
        self._table: Any = None
        self._embedding_model: Any = None
        self._ready: bool = False

        logger.info(
            "VisualMemoryStore configured: db_path=%s table=%s embedding=%s",
            db_path, TABLE_NAME, embedding_model_name,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Connect to LanceDB, create table + scalar indices if needed."""
        try:
            import lancedb  # noqa: F401
        except ImportError:
            raise ImportError(
                "lancedb is not installed. VisualMemoryStore requires lancedb>=0.6."
                " Install with: pip install lancedb"
            )

        os.makedirs(self._db_path, exist_ok=True)

        import lancedb
        self._db = await lancedb.connect_async(self._db_path)

        table_names = await self._db.table_names()
        if TABLE_NAME not in table_names:
            logger.info("Creating LanceDB table '%s' in %s", TABLE_NAME, self._db_path)
            self._table = await self._db.create_table(
                TABLE_NAME,
                data=[_seed_record()],
            )
            await self._create_scalar_indices()
        else:
            self._table = await self._db.open_table(TABLE_NAME)
            logger.info("Opened existing LanceDB table '%s'", TABLE_NAME)

        self._ready = True
        logger.info("VisualMemoryStore initialized (table=%s)", TABLE_NAME)

    async def _create_scalar_indices(self) -> None:
        """Create scalar indices for the type-filtered search pattern."""
        if self._table is None:
            return
        for col in ("memory_type", "source_window", "timestamp", "tier"):
            try:
                await self._table.create_scalar_index(col)
                logger.debug("Created scalar index on '%s'", col)
            except Exception:
                # Index may already exist from a prior partial creation
                logger.debug("Scalar index on '%s' already exists or not needed", col)

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    async def _ensure_embedding_model(self) -> None:
        """Lazy-load the Sentence-BERT embedding model."""
        if self._embedding_model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer

            self._embedding_model = SentenceTransformer(
                self._embedding_model_name,
                device="cpu",
            )
            logger.info(
                "Loaded embedding model: %s (dim=%d)",
                self._embedding_model_name, self._embedding_dim,
            )
        except Exception as e:
            logger.warning(
                "Embedding model %s failed to load: %s. Zero-vector fallback enabled.",
                self._embedding_model_name, e,
            )
            self._embedding_model = None

    async def _generate_embedding(self, text: str) -> list[float]:
        """Generate embedding vector. Falls back to zero-vector."""
        await self._ensure_embedding_model()

        if self._embedding_model is None:
            return [0.0] * self._embedding_dim

        if not text or not text.strip():
            return [0.0] * self._embedding_dim

        try:
            embedding = self._embedding_model.encode(
                text,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return embedding.tolist()
        except Exception as e:
            logger.warning(
                "Embedding generation failed for text len=%d: %s. Zero-vector fallback.",
                len(text), e,
            )
            return [0.0] * self._embedding_dim

    # ------------------------------------------------------------------
    # Insert
    # ------------------------------------------------------------------

    async def _ensure_initialized(self) -> None:
        if self._ready and self._table is not None:
            return
        await self.initialize()

    async def insert_batch(self, records: list[VisualMemoryRecord]) -> list[str]:
        """Insert a batch of visual memory records with embedding generation.

        Args:
            records: VisualMemoryRecord instances. embedding field is
                     auto-populated from content_text if empty.

        Returns:
            List of memory_id strings that were successfully inserted.
        """
        await self._ensure_initialized()

        if not records:
            return []

        rows: list[dict[str, object]] = []
        memory_ids: list[str] = []

        for rec in records:
            # Auto-generate embedding if not provided
            if not rec.embedding or all(v == 0.0 for v in rec.embedding):
                rec.embedding = await self._generate_embedding(rec.content_text)

            # Auto-generate memory_id if empty
            if not rec.memory_id or rec.memory_id == "_init_":
                rec.memory_id = str(uuid.uuid4())

            # v5.x: enforce valid tier and defaults for 5-tier fields
            if rec.tier not in MEMORY_TIERS:
                logger.warning(
                    "Invalid tier '%s' — defaulting to 'core'. Valid: %s",
                    rec.tier, MEMORY_TIERS,
                )
                rec.tier = "core"

            rows.append(rec.to_dict())
            memory_ids.append(rec.memory_id)

        try:
            await self._table.add(rows)
            logger.info("Inserted %d visual memory records", len(rows))
            return memory_ids
        except Exception as e:
            logger.error(
                "Failed to insert %d visual memory records: %s",
                len(rows), e,
            )
            raise

    # ------------------------------------------------------------------
    # Search — the {{recall:type keywords}} core
    # ------------------------------------------------------------------

    async def search_by_type(
        self,
        memory_type: str,
        query: str,
        top_k: int = 5,
    ) -> list[VisualMemoryRecord]:
        """Type-filtered semantic search.

        This is the engine behind {{recall:ui close button}} — pre-filters
        by memory_type, then vector-searches the content_text embedding.

        Args:
            memory_type: One of "window", "ui", "ocr", "scene", "spatial".
            query: Natural language search query (e.g. "close button").
            top_k: Max results to return.

        Returns:
            Sorted list of VisualMemoryRecord (by similarity, descending).
        """
        if not self._ready or self._table is None:
            logger.warning("VisualMemoryStore not initialized — returning empty")
            return []

        if memory_type not in MEMORY_TYPES:
            logger.warning(
                "Unknown memory_type '%s' — valid: %s. Returning empty.",
                memory_type, MEMORY_TYPES,
            )
            return []

        query_embedding = await self._generate_embedding(query)

        try:
            q = await self._table.search(
                np.array(query_embedding, dtype=np.float32),
                vector_column_name="embedding",
            )
            results = await (
                q.where(f"memory_type = '{memory_type}'")
                .limit(top_k)
                .to_list()
            )
        except Exception as e:
            logger.warning(
                "LanceDB search failed (type=%s query='%s'): %s",
                memory_type, query, e,
            )
            return []

        records = [VisualMemoryRecord.from_lancedb_row(row) for row in results]
        return records

    async def search_by_type_and_window(
        self,
        memory_type: str,
        source_window: str,
        query: str,
        top_k: int = 5,
    ) -> list[VisualMemoryRecord]:
        """Type + window filtered semantic search.

        Scoped to a specific window (e.g., "only in VS Code").

        Args:
            memory_type: One of MEMORY_TYPES.
            source_window: Window title to scope the search.
            query: Natural language query.
            top_k: Max results.
        """
        if not self._ready or self._table is None:
            logger.warning("VisualMemoryStore not initialized — returning empty")
            return []

        if memory_type not in MEMORY_TYPES:
            logger.warning("Unknown memory_type '%s' — returning empty", memory_type)
            return []

        query_embedding = await self._generate_embedding(query)

        # LanceDB where-clause: single-quoted string literals
        safe_window = source_window.replace("'", "''")
        safe_type = memory_type.replace("'", "''")

        try:
            q = await self._table.search(
                np.array(query_embedding, dtype=np.float32),
                vector_column_name="embedding",
            )
            results = await (
                q.where(
                    f"memory_type = '{safe_type}' "
                    f"AND source_window = '{safe_window}'"
                )
                .limit(top_k)
                .to_list()
            )
        except Exception as e:
            logger.warning(
                "LanceDB scoped search failed (type=%s window='%s'): %s",
                memory_type, source_window, e,
            )
            return []

        return [VisualMemoryRecord.from_lancedb_row(row) for row in results]

    # ------------------------------------------------------------------
    # v5.x Tier-based search & access tracking
    # ------------------------------------------------------------------

    async def search_by_tier(
        self,
        tier: str,
        query: str,
        top_k: int = 10,
    ) -> list[VisualMemoryRecord]:
        """Tier-filtered semantic search.

        Retrieves records from a specific tier (hot/warm/core/cold/deep)
        ranked by embedding similarity. Automatically increments access_count.

        Args:
            tier: One of MEMORY_TIERS.
            query: Natural language search query.
            top_k: Max results to return.
        """
        if not self._ready or self._table is None:
            logger.warning("VisualMemoryStore not initialized — returning empty")
            return []

        if tier not in MEMORY_TIERS:
            logger.warning("Unknown tier '%s' — valid: %s", tier, MEMORY_TIERS)
            return []

        query_embedding = await self._generate_embedding(query)

        try:
            safe_tier = tier.replace("'", "''")
            q = await self._table.search(
                np.array(query_embedding, dtype=np.float32),
                vector_column_name="embedding",
            )
            results = await (
                q.where(f"tier = '{safe_tier}'")
                .limit(top_k)
                .to_list()
            )
        except Exception as e:
            logger.warning(
                "LanceDB tier search failed (tier=%s query='%s'): %s",
                tier, query, e,
            )
            return []

        records = [VisualMemoryRecord.from_lancedb_row(row) for row in results]

        # v5.x: track access for retrieved records
        for rec in records:
            await self._record_access(rec.memory_id)

        return records

    async def search_by_tier_and_type(
        self,
        tier: str,
        memory_type: str,
        query: str,
        top_k: int = 10,
    ) -> list[VisualMemoryRecord]:
        """Tier + type filtered semantic search.

        The most precise retrieval: by tier, by memory_type, by embedding.

        Args:
            tier: One of MEMORY_TIERS.
            memory_type: One of MEMORY_TYPES.
            query: Natural language search query.
            top_k: Max results.
        """
        if not self._ready or self._table is None:
            logger.warning("VisualMemoryStore not initialized — returning empty")
            return []

        if tier not in MEMORY_TIERS:
            logger.warning("Unknown tier '%s' — valid: %s", tier, MEMORY_TIERS)
            return []

        if memory_type not in MEMORY_TYPES:
            logger.warning("Unknown memory_type '%s' — valid: %s", memory_type, MEMORY_TYPES)
            return []

        query_embedding = await self._generate_embedding(query)

        safe_tier = tier.replace("'", "''")
        safe_type = memory_type.replace("'", "''")

        try:
            q = await self._table.search(
                np.array(query_embedding, dtype=np.float32),
                vector_column_name="embedding",
            )
            results = await (
                q.where(
                    f"tier = '{safe_tier}' "
                    f"AND memory_type = '{safe_type}'"
                )
                .limit(top_k)
                .to_list()
            )
        except Exception as e:
            logger.warning(
                "LanceDB tier+type search failed (tier=%s type=%s query='%s'): %s",
                tier, memory_type, query, e,
            )
            return []

        records = [VisualMemoryRecord.from_lancedb_row(row) for row in results]

        for rec in records:
            await self._record_access(rec.memory_id)

        return records

    async def _record_access(self, memory_id: str) -> None:
        """Increment access_count for a record (idempotent)."""
        if not self._ready or self._table is None:
            return
        try:
            safe_id = memory_id.replace("'", "''")
            await self._table.update(
                where=f"memory_id = '{safe_id}'",
                values={"access_count": "access_count + 1"},
            )
        except Exception as e:
            logger.warning(
                "Failed to record access for memory_id=%s: %s",
                memory_id, e,
            )

    async def update_memory_tier(
        self,
        memory_id: str,
        new_tier: str,
    ) -> bool:
        """Promote or demote a memory record to a new tier.

        Args:
            memory_id: Target record.
            new_tier: One of MEMORY_TIERS.

        Returns:
            True if updated, False if not found or invalid tier.
        """
        if not self._ready or self._table is None:
            return False

        if new_tier not in MEMORY_TIERS:
            logger.warning("Invalid tier '%s' — valid: %s", new_tier, MEMORY_TIERS)
            return False

        try:
            safe_id = memory_id.replace("'", "''")
            safe_tier = new_tier.replace("'", "''")
            await self._table.update(
                where=f"memory_id = '{safe_id}'",
                values={"tier": safe_tier},
            )
            logger.info("Promoted memory_id=%s to tier=%s", memory_id, new_tier)
            return True
        except Exception as e:
            logger.warning(
                "Failed to update tier for memory_id=%s: %s",
                memory_id, e,
            )
            return False

    async def update_importance(
        self,
        memory_id: str,
        new_score: float,
    ) -> bool:
        """Update importance_score for a memory record.

        Args:
            memory_id: Target record.
            new_score: New importance score (0.0-1.0 clamped).

        Returns:
            True if updated successfully.
        """
        if not self._ready or self._table is None:
            return False

        clamped = max(0.0, min(1.0, new_score))
        try:
            safe_id = memory_id.replace("'", "''")
            await self._table.update(
                where=f"memory_id = '{safe_id}'",
                values={"importance_score": clamped},
            )
            return True
        except Exception as e:
            logger.warning(
                "Failed to update importance for memory_id=%s: %s",
                memory_id, e,
            )
            return False

    async def evaluate_decay(
        self,
        tier: str,
        current_time: str,
    ) -> int:
        """Evaluate decay for all records in a tier.

        Updates decay_timestamp to current_time for all matching records.
        Downstream decay_engine handles the actual score computation.

        Args:
            tier: Tier to evaluate.
            current_time: ISO8601 timestamp to set as decay_timestamp.

        Returns:
            Number of records updated.
        """
        if not self._ready or self._table is None:
            return 0

        if tier not in MEMORY_TIERS:
            logger.warning("Invalid tier '%s' for decay evaluation", tier)
            return 0

        try:
            safe_tier = tier.replace("'", "''")
            safe_time = current_time.replace("'", "''")
            result = await self._table.update(
                where=f"tier = '{safe_tier}'",
                values={"decay_timestamp": safe_time},
            )
            count = int(result) if isinstance(result, (int, float)) else 0
            if count:
                logger.info(
                    "Decay evaluated for %d records in tier=%s",
                    count, tier,
                )
            return count
        except Exception as e:
            logger.warning(
                "Decay evaluation failed for tier=%s: %s",
                tier, e,
            )
            return 0

    async def get_by_tier(
        self,
        tier: str,
        limit: int = 100,
    ) -> list[VisualMemoryRecord]:
        """Non-vector lookup: get all records in a tier sorted by importance.

        Args:
            tier: Target tier.
            limit: Max records to return.

        Returns:
            List of VisualMemoryRecord sorted by importance_score descending.
        """
        if not self._ready or self._table is None:
            return []

        if tier not in MEMORY_TIERS:
            return []

        try:
            safe_tier = tier.replace("'", "''")
            q = await self._table.search(
                np.array([0.0] * self._embedding_dim, dtype=np.float32),
                vector_column_name="embedding",
            )
            rows = await (
                q.where(f"tier = '{safe_tier}'")
                .limit(limit)
                .to_list()
            )
        except Exception as e:
            logger.warning("get_by_tier failed (tier=%s): %s", tier, e)
            return []

        records = [VisualMemoryRecord.from_lancedb_row(row) for row in rows]
        records.sort(key=lambda r: r.importance_score, reverse=True)
        return records

    async def count_by_tier(self, tier: str) -> int:
        """Return record count for a specific tier."""
        if not self._ready or self._table is None:
            return 0

        if tier not in MEMORY_TIERS:
            return 0

        try:
            safe_tier = tier.replace("'", "''")
            result = await self._table.to_lance().count_rows(
                filter=f"tier = '{safe_tier}'"
            )
            return int(result)
        except Exception:
            return 0

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    async def delete_older_than(self, before_iso: str) -> int:
        """Purge records older than a timestamp. Returns count deleted."""
        if not self._ready or self._table is None:
            logger.warning("VisualMemoryStore not initialized")
            return 0

        try:
            result = await self._table.delete(f"timestamp < '{before_iso}'")
            count = int(result) if isinstance(result, (int, float)) else 0
            if count:
                logger.info("Purged %d visual memory records older than %s", count, before_iso)
            return count
        except Exception as e:
            logger.warning("Purge failed: %s", e)
            return 0

    async def count(self) -> int:
        """Return total record count."""
        if not self._ready or self._table is None:
            return 0

        try:
            rows = await self._table.to_lance().count_rows()
            return int(rows)
        except Exception:
            return 0
