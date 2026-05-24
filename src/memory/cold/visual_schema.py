"""
LanceDB schema for classified visual memory — v5.x visual-memory-design

Design goals:
- ONE flat table (visual_memory), no joins, no nested docs.
- Fine-grained records: each window title, UI element, OCR text line, scene
  label, or spatial observation gets its own row.
- Type-filtered semantic search via {{recall:type keywords}} retrieval tags.
- Consistent with existing cold_memory_vectors embedding (bge-small-zh-v1.5, 512-dim).

Data flow:
  Perception pipeline → classify by type → insert into visual_memory table
  ContextAssembler/LLM loop → {{recall:ui button}} → LanceDB vector search → inject

v4.5.0 §3.3: Cold memory schema constraints (flat, vector-indexed, time-aware).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Table name — separate from cold_memory_vectors to avoid schema collision
# ---------------------------------------------------------------------------
TABLE_NAME: str = "visual_memory"

# ---------------------------------------------------------------------------
# memory_type enum — the 5 classification categories
# ---------------------------------------------------------------------------
MEMORY_TYPES: tuple[str, ...] = (
    "window",   # Window title, class_name, app name
    "ui",       # UI element: button, textbox, checkbox, dropdown, etc.
    "ocr",      # OCR text line or block
    "scene",    # CLIP scene classification tag (code_editor, webpage, ...)
    "spatial",  # Window position, size, Z-order, mouse proximity, attention score
)

# ---------------------------------------------------------------------------
# memory_tier enum — 5-tier architecture for memory lifecycle management
# v5.x memory-as-connective-tissue design:
#   hot   — last 30 seconds, highest recency weight
#   warm  — last 24 hours, moderate recency weight
#   core  — important/valuable, retained indefinitely
#   cold  — semantic/background, retained at lower priority
#   deep  — identity-forming, permanent unless explicitly removed
# ---------------------------------------------------------------------------
MEMORY_TIERS: tuple[str, ...] = (
    "hot",
    "warm",
    "core",
    "cold",
    "deep",
)

# ---------------------------------------------------------------------------
# Embedding config — matches cold_memory_vectors (bge-small-zh-v1.5)
# ---------------------------------------------------------------------------
EMBEDDING_DIM: int = 512

# ---------------------------------------------------------------------------
# LanceDB scalar index columns
# ---------------------------------------------------------------------------
SCALAR_INDEX_COLUMNS: tuple[str, ...] = (
    "memory_type",
    "source_window",
    "timestamp",
    "tier",  # v5.x: tier-filtered retrieval for 5-tier architecture
)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class VisualMemoryRecord:
    """A single row in the visual_memory LanceDB table.

    Flat by design. No nested objects — LanceDB scalar indices work best
    on flat columns. Extra metadata (bbox, confidence, state, etc.) goes
    into a JSON string for rare debug queries, not indexed.

    v5.x additions: tier, importance_score, access_count, decay_timestamp
    support the 5-tier memory-as-connective-tissue architecture.
    """

    memory_id: str           # UUID4 — primary lookup key
    timestamp: str           # ISO8601 — when the observation was made
    memory_type: str         # One of MEMORY_TYPES
    content_text: str        # Human-readable content (window title, OCR text, etc.)
    source_window: str       # Window title this data came from ("" for full-screen)
    tags: list[str]          # Free-form tags (e.g., ["button", "enabled", "top-right"])
    embedding: list[float]   # float32[512] — bge-small-zh-v1.5 vector
    meta_json: str = "{}"    # Optional JSON blob (bbox, confidence, state, etc.)

    # ── v5.x 5-tier fields (backward-compatible defaults) ──
    tier: str = "core"                # One of MEMORY_TIERS
    importance_score: float = 0.5     # Composite recency×relevance×importance
    access_count: int = 0             # Times this record was retrieved
    decay_timestamp: str = ""         # ISO8601 of last decay evaluation ("" = never)

    def to_dict(self) -> dict[str, object]:
        """Convert to dict for LanceDB insert. LanceDB expects plain dicts."""
        return {
            "memory_id": self.memory_id,
            "timestamp": self.timestamp,
            "memory_type": self.memory_type,
            "content_text": self.content_text,
            "source_window": self.source_window,
            "tags": self.tags,
            "embedding": self.embedding,
            "meta_json": self.meta_json,
            # v5.x 5-tier fields
            "tier": self.tier,
            "importance_score": self.importance_score,
            "access_count": self.access_count,
            "decay_timestamp": self.decay_timestamp,
        }

    @classmethod
    def from_lancedb_row(cls, row: dict[str, object]) -> "VisualMemoryRecord":
        """Reconstruct from a LanceDB query result row.

        Backward-compatible: missing v5.x tier columns default to "core"/0/"".
        """
        raw_tags: Any = row.get("tags")
        raw_emb: Any = row.get("embedding")
        raw_importance: Any = row.get("importance_score", 0.5)
        raw_access: Any = row.get("access_count", 0)
        return cls(
            memory_id=str(row.get("memory_id", "")),
            timestamp=str(row.get("timestamp", "")),
            memory_type=str(row.get("memory_type", "scene")),
            content_text=str(row.get("content_text", "")),
            source_window=str(row.get("source_window", "")),
            tags=list(raw_tags) if isinstance(raw_tags, list) else [],
            embedding=list(raw_emb) if isinstance(raw_emb, list) else [0.0] * EMBEDDING_DIM,
            meta_json=str(row.get("meta_json", "{}")),
            # v5.x fields — default gracefully when absent from existing rows
            tier=str(row.get("tier", "core")),
            importance_score=float(raw_importance),
            access_count=int(raw_access),
            decay_timestamp=str(row.get("decay_timestamp", "")),
        )


# ---------------------------------------------------------------------------
# Seed record for table creation (LanceDB 0.30+ needs data)
# ---------------------------------------------------------------------------
def _seed_record() -> dict[str, object]:
    """Return a single valid record used to bootstrap the table schema."""
    return VisualMemoryRecord(
        memory_id="_init_",
        timestamp="1970-01-01T00:00:00Z",
        memory_type="scene",
        content_text="",
        source_window="",
        tags=[],
        embedding=[0.0] * EMBEDDING_DIM,
        meta_json="{}",
        tier="core",
        importance_score=0.0,
        access_count=0,
        decay_timestamp="",
    ).to_dict()


# ---------------------------------------------------------------------------
# Table creation SQL (for documentation; actual creation is via LanceDB Python API)
# ---------------------------------------------------------------------------
# LanceDB does NOT use SQL DDL. The equivalent create_table call:
#
#   table = await db.create_table(
#       "visual_memory",
#       data=[_seed_record()],
#   )
#   await table.create_scalar_index("memory_type")
#   await table.create_scalar_index("source_window")
#   await table.create_scalar_index("timestamp")
#
# Vector index is created separately after enough data exists:
#   await table.create_index(
#       metric="cosine",
#       num_partitions=256,
#       num_sub_vectors=64,
#   )
# ---------------------------------------------------------------------------


# v5.x insight-memory-joint: Prompt memory schemas
# ===================================================


class PromptRefSchema:
    """LanceDB table schema for SAVPE visual concept references."""

    TABLE_NAME = "prompt_refs"

    @classmethod
    def schema(cls):
        """Return LanceDB schema dict."""
        import pyarrow as pa
        return pa.schema([
            pa.field("prompt_id", pa.string()),
            pa.field("name", pa.string()),
            pa.field("crop_blob", pa.binary()),
            pa.field("vpe_blob", pa.binary()),
            pa.field("context_tags", pa.string()),  # JSON-serialized list
            pa.field("confidence", pa.float32()),
            pa.field("last_seen", pa.string()),  # ISO timestamp
            pa.field("vpe_expires_at", pa.string()),
            pa.field("tier", pa.string()),
        ])

    @classmethod
    def create_table(cls, db):
        """Create the prompt_refs table in LanceDB. Returns table handle."""
        if cls.TABLE_NAME in db.table_names():
            return db.open_table(cls.TABLE_NAME)
        return db.create_table(cls.TABLE_NAME, schema=cls.schema())


class VisualFrameSchema:
    """LanceDB table schema for structured visual frame outputs."""

    TABLE_NAME = "visual_frames"

    @classmethod
    def schema(cls):
        """Return LanceDB schema dict."""
        import pyarrow as pa
        return pa.schema([
            pa.field("frame_id", pa.string()),
            pa.field("timestamp", pa.string()),  # ISO timestamp
            pa.field("window_title", pa.string()),
            pa.field("app_name", pa.string()),
            pa.field("scene_category", pa.string()),
            pa.field("concepts_json", pa.string()),  # JSON-serialized list[dict]
            pa.field("ocr_texts_json", pa.string()),
            pa.field("spatial_edges_json", pa.string()),
            pa.field("degraded", pa.bool_()),
        ])

    @classmethod
    def create_table(cls, db):
        """Create the visual_frames table in LanceDB. Returns table handle."""
        if cls.TABLE_NAME in db.table_names():
            return db.open_table(cls.TABLE_NAME)
        return db.create_table(cls.TABLE_NAME, schema=cls.schema())
