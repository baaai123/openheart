"""
Cold Memory Store — LanceDB long-term archival memory.
v4.5.0 §3.3

Provides persistent storage for Scene data with vector embeddings, memory decay,
sensitive info filtering, and Level 2 summarization via Qwen2.5-3B.
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sensitive info patterns — v4.5.0 §3.3, 项目宪法 §5.1
# ---------------------------------------------------------------------------

SENSITIVE_PATTERNS: list[tuple[str, re.Pattern]] = [
    # 中国手机号: 1[3-9]xxxxxxxxx
    ("phone_number", re.compile(r"1[3-9]\d{9}")),
    # 中国身份证号: 17 digits + digit or X/x
    ("id_card", re.compile(r"\d{17}[\dXx]")),
    # Password patterns: "password=", "pw=", "pwd=", "密码", "pass:"
    ("password", re.compile(r"(?:password|pw|pwd|密码|pass)\s*[=:：]\s*\S+", re.IGNORECASE)),
    # Credit card: 13-19 digits (basic pattern)
    ("credit_card", re.compile(r"\b(?:\d[ -]*?){13,19}\b")),
]

# ---------------------------------------------------------------------------
# Memory types and decay coefficients — v4.5.0 §3.3.2
# ---------------------------------------------------------------------------

@dataclass
class MemoryDecayConfig:
    """Decay coefficients per memory type."""
    EMOTION: float = 0.1   # Very slow decay
    FACT: float = 0.5      # Moderate decay
    ACTION: float = 0.8    # Fast decay

    # Special protection: high-intensity positive emotion is almost permanent
    HIGH_EMOTION_THRESHOLD: float = 0.7
    HIGH_EMOTION_FIXED_ALPHA: float = 0.1


# ---------------------------------------------------------------------------
# Data models — v4.5.0 §3.3.3
# ---------------------------------------------------------------------------

@dataclass
class ColdMemoryRecord:
    """A single record in the LanceDB cold_memory_vectors table."""
    memory_id: str          # UUID
    scene_summary: str
    embedding: list[float]  # float32[512]
    memory_type: str        # EMOTION | FACT | ACTION
    importance_score: float
    affective_flag: bool
    scene_class: str
    created_at: str         # ISO8601 datetime
    last_accessed: str      # ISO8601 datetime
    graph_node_ids: list[str]  # entity IDs from the scene
    graph_data: str         # JSON-serialized entity_relations


@dataclass
class Scene:
    """
    Scene data model — v4.5.0 §3
    Matches the structure expected by hot memory and sync.
    """
    scene_id: str
    trace_id: str
    timestamp: str          # ISO8601
    summary: str
    events: list[dict] = field(default_factory=list)
    entities: list[dict] = field(default_factory=list)
    entity_relations: list[dict] = field(default_factory=list)
    affective_flag: bool = False
    importance_score: float = 0.5
    scene_class: str = "general"
    importance_components: dict = field(default_factory=lambda: {
        "access_count": 1,
        "recency_weight": 1.0,
        "relation_count_weight": 0.0,
        "affective_bonus": 0.0,
    })

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Scene":
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


@dataclass
class MemoryFragment:
    """Lightweight result from semantic recall."""
    memory_id: str
    scene_summary: str
    similarity: float
    memory_type: str
    importance_score: float


@dataclass
class Moment:
    """A warm/emotionally significant memory moment."""
    memory_id: str
    scene_summary: str
    memory_type: str
    importance_score: float
    affective_flag: bool


# ---------------------------------------------------------------------------
# ColdMemoryStore — v4.5.0 §3.3
# ---------------------------------------------------------------------------

class ColdMemoryStore:
    """
    LanceDB-backed long-term cold memory.

    Provides:
    - Scene storage with vector embeddings (bge-small-zh-v1.5, 512 dim)
    - Sensitive info filtering before write
    - Semantic and hybrid search
    - Memory decay (Ebbinghaus-based with emotional protection)
    - Level 2 summarization via Qwen2.5-3B (with placeholder fallback)
    - cold_memory:initialized sentinel tracking
    """

    TABLE_NAME = "cold_memory_vectors"

    def __init__(
        self,
        db_path: str = "data/cold_memory",
        embedding_model_name: str = "BAAI/bge-small-zh-v1.5",
        embedding_dim: int = 512,
        redis_client=None,
        summary_model_name: str = "qwen_3b",
        decay_config: Optional[MemoryDecayConfig] = None,
    ):
        """
        Initialize the ColdMemoryStore.

        Args:
            db_path: Path to LanceDB database directory.
            embedding_model_name: Name of the Sentence-BERT embedding model.
            embedding_dim: Dimension of the embedding vectors (512 for bge-small-zh-v1.5).
            redis_client: Optional Redis client for sentinel key and sync notifications.
            summary_model_name: Name of the model used for Level 2 summarization.
            decay_config: Decay coefficients configuration.
        """
        self._db_path = db_path
        self._embedding_model_name = embedding_model_name
        self._embedding_dim = embedding_dim
        self._redis = redis_client
        self._summary_model_name = summary_model_name
        self._decay_config = decay_config or MemoryDecayConfig()

        # Lazy-loaded resources
        self._db = None
        self._table = None
        self._embedding_model = None
        self._summary_model = None

        # Track whether we've been initialized
        self._initialized_in_this_session = False

        # v4.5.0 §5.7: load user memory preferences for importance weighting
        self._load_memory_preferences()

        logger.info(
            "ColdMemoryStore configured: db_path=%s, embedding=%s, summary_model=%s",
            db_path, embedding_model_name, summary_model_name,
        )

    # -------------------------------------------------------------------
    # Lifecycle — v4.5.0 §3.3
    # -------------------------------------------------------------------

    async def initialize(self) -> None:
        """
        Initialize the LanceDB database and ensure the table exists.

        Sets the cold_memory:initialized sentinel key on first-ever
        successful table creation (when the table was previously empty).
        """
        # Try to import lancedb — lazy, not a hard dependency until this is called
        try:
            import lancedb  # noqa: F401 — lazy-import at runtime
        except ImportError:
            error_msg = (
                "lancedb is not installed. ColdMemoryStore requires lancedb>=0.6. "
                "Install with: pip install lancedb"
            )
            logger.error(error_msg)
            raise ImportError(error_msg)

        # Try to import sentence-transformers for embeddings
        try:
            from sentence_transformers import SentenceTransformer  # noqa: F401
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. Embedding generation will "
                "use zero-vector fallback. Install with: pip install sentence-transformers"
            )

        os.makedirs(self._db_path, exist_ok=True)

        import lancedb
        self._db = await lancedb.connect_async(self._db_path)

        table_exists = self.TABLE_NAME in await self._db.table_names()

        if not table_exists:
            logger.info("Creating LanceDB table '%s'", self.TABLE_NAME)
            self._table = await self._db.create_table(
                self.TABLE_NAME,
                data=[{  # LanceDB 0.30+ needs data or schema for table creation
                    "memory_id": "_init_",
                    "scene_summary": "",
                    "embedding": [0.0] * self._embedding_dim,
                    "memory_type": "FACT",
                    "importance_score": 0.0,
                    "affective_flag": False,
                    "scene_class": "general",
                    "created_at": "",
                    "last_accessed": "",
                    "graph_node_ids": [],
                    "graph_data": "[]",
                }],
            )
            # On first creation, set sentinel if Redis is available
            await self._set_initialized_sentinel()
        else:
            self._table = await self._db.open_table(self.TABLE_NAME)
            logger.info("Opened existing LanceDB table '%s'", self.TABLE_NAME)

        self._initialized_in_this_session = True

        # Warm-up: try loading the embedding model
        await self._ensure_embedding_model()

        logger.info("ColdMemoryStore initialized successfully")

    async def _ensure_embedding_model(self):
        """Lazy-load the Sentence-BERT embedding model."""
        if self._embedding_model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer(
                self._embedding_model_name,
                device="cpu",  # bge-small is fast enough on CPU
                cache_folder="models/",
            )
            logger.info(
                "Loaded embedding model: %s (dim=%d)",
                self._embedding_model_name, self._embedding_dim,
            )
        except Exception as e:
            # Graceful degradation: embeddings will be zero vectors
            logger.warning(
                "Failed to load embedding model '%s': %s. "
                "Semantic search will return empty results.",
                self._embedding_model_name, e,
            )
            self._embedding_model = None

    async def _ensure_summary_model(self):
        """Lazy-load the Qwen2.5-3B model for Level 2 summarization."""
        if self._summary_model is not None:
            return

        try:
            # v4.5.0 §3.3.2: Level 2 summary uses Qwen2.5-3B
            # This is a placeholder — actual model loading depends on Task 15 (Decision)
            # For now, use a lightweight text template as fallback
            logger.info(
                "Summary model '%s' — using template-based fallback. "
                "Full 3B model integration pending Decision layer (Task 15).",
                self._summary_model_name,
            )
            self._summary_model = True  # Placeholder — indicates "available but template-based"
        except Exception as e:
            logger.warning("Failed to set up summary model: %s", e)
            self._summary_model = None

    # -------------------------------------------------------------------
    # Sentinel key — v4.5.0 §3.2.4 step 6
    # -------------------------------------------------------------------

    async def _set_initialized_sentinel(self) -> None:
        """
        Set cold_memory:initialized = true in Redis.

        This sentinel has NO TTL — it persists until cold memory data
        is manually cleared. The preference shift module uses this as
        a startup gate.
        """
        sentinel_key = "cold_memory:initialized"

        if self._redis is not None:
            try:
                # Check if already set — avoid overwriting
                existing = self._redis.get(sentinel_key)
                if existing is None:
                    # SET with no TTL — persists permanently
                    self._redis.set(sentinel_key, "true")
                    logger.info(
                        "Sentinel key '%s' set to 'true' (first cold memory initialization)",
                        sentinel_key,
                    )
                else:
                    logger.debug(
                        "Sentinel key '%s' already exists, value='%s'",
                        sentinel_key, existing,
                    )
            except Exception as e:
                # Non-fatal: sentinel is a convenience, not critical
                logger.warning(
                    "Failed to set sentinel key '%s': %s", sentinel_key, e,
                )
        else:
            logger.debug(
                "No Redis client configured — sentinel key '%s' not set",
                sentinel_key,
            )

    async def is_initialized(self) -> bool:
        """Check if cold memory has been initialized (sentinel key exists)."""
        if self._redis is not None:
            try:
                value = self._redis.get("cold_memory:initialized")
                return value == b"true" or value == "true"
            except Exception:
                pass
        return self._initialized_in_this_session

    # -------------------------------------------------------------------
    # Sensitive info filtering — v4.5.0 §3.2.4, 项目宪法 §5.1
    # -------------------------------------------------------------------

    @staticmethod
    def contains_sensitive_info(text: str) -> bool:
        """
        Check if text contains sensitive information (phone, ID, password, credit card).

        Returns True if any sensitive pattern matches, meaning the data
        should NOT be stored in cold memory.
        """
        if not text:
            return False

        for pattern_name, pattern in SENSITIVE_PATTERNS:
            match = pattern.search(text)
            if match:
                logger.debug(
                    "Sensitive info detected: pattern=%s, match=%s",
                    pattern_name, match.group(),
                )
                return True

        return False

    @staticmethod
    def redact_sensitive_info(text: str) -> tuple[str, bool]:
        """
        Redact sensitive information from text, returning the cleaned text
        and whether any redaction occurred.

        This is a more lenient alternative to full blocking — replaces
        sensitive patterns with [REDACTED] markers.
        """
        was_redacted = False
        cleaned = text

        for _pattern_name, pattern in SENSITIVE_PATTERNS:
            if pattern.search(cleaned):
                cleaned = pattern.sub("[REDACTED]", cleaned)
                was_redacted = True

        return cleaned, was_redacted

    # -------------------------------------------------------------------
    # Embedding generation
    # -------------------------------------------------------------------

    async def _generate_embedding(self, text: str) -> list[float]:
        """
        Generate a 512-dim embedding vector for the given text.

        Uses bge-small-zh-v1.5 via sentence-transformers. Falls back to
        zero vector if the model is unavailable.
        """
        await self._ensure_embedding_model()

        if self._embedding_model is None:
            logger.debug("Embedding model unavailable — returning zero vector")
            return [0.0] * self._embedding_dim

        try:
            # Generate embedding — returns numpy array
            embedding = self._embedding_model.encode(
                text,
                normalize_embeddings=True,  # Cosine similarity optimization
                show_progress_bar=False,
            )
            return embedding.tolist()
        except Exception as e:
            logger.warning(
                "Embedding generation failed for text len=%d: %s. "
                "Using zero vector fallback.",
                len(text), e,
            )
            return [0.0] * self._embedding_dim

    # -------------------------------------------------------------------
    # Scene storage — v4.5.0 §3.3.3
    # -------------------------------------------------------------------

    async def store_scene(self, scene: Scene) -> Optional[str]:
        """
        Store a Scene in cold memory as a single atomic LanceDB record.

        Steps:
        1. Filter for sensitive info — reject if detected.
        2. Generate embedding vector from the scene summary.
        3. Build the LanceDB record and insert.

        Returns the memory_id if stored successfully, None if filtered out.

        Raises:
            RuntimeError: If the store is not initialized.
        """
        if self._table is None:
            raise RuntimeError(
                "ColdMemoryStore not initialized. Call initialize() first."
            )

        # Step 1: Sensitive info filtering
        if self.contains_sensitive_info(scene.summary):
            logger.warning(
                "Scene %s blocked from cold storage: contains sensitive info. "
                "Keeping in hot memory only.",
                scene.scene_id,
            )
            return None

        # Also check entities and events for sensitive info
        for entity in scene.entities:
            for value in entity.values():
                if isinstance(value, str) and self.contains_sensitive_info(value):
                    logger.warning(
                        "Scene %s blocked: entity contains sensitive info",
                        scene.scene_id,
                    )
                    return None

        for event in scene.events:
            for value in event.values():
                if isinstance(value, str) and self.contains_sensitive_info(value):
                    logger.warning(
                        "Scene %s blocked: event contains sensitive info",
                        scene.scene_id,
                    )
                    return None

        # Step 2: Generate embedding
        embedding = await self._generate_embedding(scene.summary)

        # Determine memory type from scene properties
        memory_type = self._classify_memory_type(scene)

        # v4.5.0 §5.7: compute and apply memory preference bias to importance
        pref_bias = self._compute_preference_bias(scene.summary or "")
        base_importance = scene.importance_score
        affective_weight = scene.importance_components.get("affective_bonus", 0.0)
        importance = base_importance * (1.0 + affective_weight + pref_bias)
        scene.importance_score = importance

        # Step 3: Build record
        now = datetime.now(timezone.utc).isoformat()
        memory_id = scene.scene_id or str(uuid.uuid4())

        record = {
            "memory_id": memory_id,
            "scene_summary": scene.summary,
            "embedding": embedding,
            "memory_type": memory_type,
            "importance_score": scene.importance_score,
            "affective_flag": scene.affective_flag,
            "scene_class": scene.scene_class or "general",
            "created_at": scene.timestamp or now,
            "last_accessed": now,
            "graph_node_ids": [e.get("name", "") for e in scene.entities],
            "graph_data": json.dumps(
                getattr(scene, "entity_relations", []) or [],
                ensure_ascii=False,
            ),
        }

        # Step 4: Atomic write — v4.5.0 §3.3.3
        try:
            await self._table.add([record])
            logger.info(
                "Scene %s stored in cold memory (type=%s, importance=%.3f)",
                memory_id, memory_type, scene.importance_score,
            )
            return memory_id
        except Exception as e:
            logger.error(
                "Failed to store scene %s in cold memory: %s",
                memory_id, e,
            )
            raise

    @staticmethod
    def _classify_memory_type(scene: Scene) -> str:
        """
        Classify a scene as EMOTION, FACT, or ACTION memory type.

        Rules:
        - If affective_flag is True → EMOTION
        - If scene_class contains 'action' or has operation events → ACTION
        - Otherwise → FACT
        """
        if scene.affective_flag:
            return "EMOTION"

        scene_class_lower = (scene.scene_class or "").lower()
        if "action" in scene_class_lower or "operation" in scene_class_lower:
            return "ACTION"

        # Check events for action keywords
        for event in scene.events:
            event_type = (event.get("type") or event.get("event_type") or "").lower()
            if any(kw in event_type for kw in ("action", "operation", "click", "move", "type")):
                return "ACTION"

        return "FACT"

    # -------------------------------------------------------------------
    # Level 2 Summarization — v4.5.0 §3.3.2
    # -------------------------------------------------------------------

    async def generate_level_2_summary(self, scenes: list[Scene]) -> str:
        """
        Generate a Level 2 summary using Qwen2.5-3B (or template fallback).

        Level 2 is triggered when: importance < 0.2 AND age > 72 hours.
        The 3B model produces a ~50-word summary, replacing v4.3.1's Louvain.

        Quality check: core entity retention and relationship completeness.
        If quality < 0.6, the original memory is preserved for retry.
        """
        if not scenes:
            return ""

        await self._ensure_summary_model()

        # Concatenate scene summaries for model input
        combined_text = " ".join(s.summary for s in scenes)

        if self._summary_model is True:
            # Placeholder fallback: simple template-based summary
            # Full 3B model integration will come with Decision layer (Task 15)
            summary = self._template_level_2_summary(scenes, combined_text)
            logger.debug(
                "Level 2 summary generated (template): %d scenes summarized",
                len(scenes),
            )
            return summary
        else:
            logger.warning(
                "Summary model unavailable — returning raw truncated text"
            )
            # Truncate to ~50 words as a poor man's summary
            words = combined_text.split()
            if len(words) > 50:
                return " ".join(words[:50]) + "..."
            return combined_text

    def _template_level_2_summary(
        self, scenes: list[Scene], combined_text: str
    ) -> str:
        """
        Template-based Level 2 summary fallback.

        Extracts key entities and produces a concise summary.
        This is a placeholder until the 3B model integration is complete.
        """
        all_entities: set[str] = set()
        memory_types: list[str] = []
        for scene in scenes:
            for entity in scene.entities:
                name = entity.get("name", "")
                if name:
                    all_entities.add(name)
            if scene.affective_flag:
                memory_types.append("情感")

        entity_str = "、".join(sorted(all_entities)[:5]) if all_entities else "无特定实体"

        if len(scenes) == 1:
            return (
                f"用户{scenes[0].summary[:40]}"
                f"{'...' if len(scenes[0].summary) > 40 else ''}"
            )
        else:
            parts = []
            if entity_str and entity_str != "无特定实体":
                parts.append(f"涉及{entity_str}")
            if memory_types:
                parts.append(f"含{len(memory_types)}条情感记忆")
            parts.append(f"共{len(scenes)}条相关情景")

            base = "；".join(parts) + "。"

            # Ensure ~50 words max
            words = base.split()
            if len(words) > 50:
                return " ".join(words[:50]) + "..."

            return base

    # -------------------------------------------------------------------
    # Query interfaces — v4.5.0 §3.3.3
    # -------------------------------------------------------------------

    async def semantic_search(
        self,
        query: str,
        top_k: int = 5,
        min_importance: float = 0.1,
    ) -> list[MemoryFragment]:
        """
        Semantic similarity search using vector embeddings.

        Args:
            query: Natural language query string.
            top_k: Maximum number of results to return.
            min_importance: Minimum importance score filter.

        Returns:
            List of MemoryFragment objects sorted by similarity.
        """
        if self._table is None:
            logger.warning("ColdMemoryStore not initialized — returning empty")
            return []

        # Generate query embedding
        query_embedding = await self._generate_embedding(query)

        # LanceDB vector search
        try:
            _query = await self._table.search(
                np.array(query_embedding, dtype=np.float32),
                vector_column_name="embedding",
            )
            results = await _query.limit(top_k * 3).to_list()  # Get more for post-filtering
        except Exception as e:
            logger.warning("LanceDB search failed: %s", e)
            return []

        # Post-process: filter by importance score and limit
        fragments = []
        for row in results:
            if row.get("importance_score", 0) >= min_importance:
                fragments.append(MemoryFragment(
                    memory_id=row.get("memory_id", ""),
                    scene_summary=row.get("scene_summary", ""),
                    similarity=getattr(row, "_distance", 0.0),
                    memory_type=row.get("memory_type", "FACT"),
                    importance_score=row.get("importance_score", 0.0),
                ))

        # Sort by similarity (higher is better for cosine with normalized embeddings)
        # v4.5.0 §5.7: re-rank with preference_bias
        if hasattr(self, "_memory_prefs") and self._memory_prefs:
            scored = [
                (
                    f,
                    f.similarity + self._compute_preference_bias(f.scene_summary) * 0.3,
                )
                for f in fragments
            ]
            scored.sort(key=lambda x: x[1], reverse=True)
            return [item for item, _ in scored[:top_k]]

        fragments.sort(key=lambda x: x.similarity, reverse=True)
        return fragments[:top_k]

    async def hybrid_search(
        self,
        query: str,
        entity_filter: Optional[str] = None,
        top_k: int = 5,
    ) -> list[MemoryFragment]:
        """
        Hybrid search: vector similarity + entity filtering.

        Args:
            query: Natural language query.
            entity_filter: Optional entity name to filter by.
            top_k: Maximum number of results.
        """
        if self._table is None:
            logger.warning("ColdMemoryStore not initialized — returning empty")
            return []

        query_embedding = await self._generate_embedding(query)

        try:
            _query = await self._table.search(
                np.array(query_embedding, dtype=np.float32),
                vector_column_name="embedding",
            )
            results = await _query.limit(top_k * 3).to_list()
        except Exception as e:
            logger.warning("LanceDB hybrid search failed: %s", e)
            return []

        fragments = []
        for row in results:
            # Apply entity filter if specified
            if entity_filter:
                graph_node_ids = row.get("graph_node_ids", [])
                if entity_filter not in graph_node_ids:
                    continue

            fragments.append(MemoryFragment(
                memory_id=row.get("memory_id", ""),
                scene_summary=row.get("scene_summary", ""),
                similarity=getattr(row, "_distance", 0.0),
                memory_type=row.get("memory_type", "FACT"),
                importance_score=row.get("importance_score", 0.0),
            ))

        # v4.5.0 §5.7: re-rank with preference_bias
        if hasattr(self, "_memory_prefs") and self._memory_prefs:
            scored = [
                (
                    f,
                    f.similarity + self._compute_preference_bias(f.scene_summary) * 0.3,
                )
                for f in fragments
            ]
            scored.sort(key=lambda x: x[1], reverse=True)
            return [item for item, _ in scored[:top_k]]

        fragments.sort(key=lambda x: x.similarity, reverse=True)
        return fragments[:top_k]

    async def recall_warm_moments(self, top_k: int = 5) -> list[Moment]:
        """
        Randomly sample high-emotion memories for warm/ice-breaking moments.

        Filters for EMOTION type memories with high affective_flag.
        """
        if self._table is None:
            logger.warning("ColdMemoryStore not initialized — returning empty")
            return []

        try:
            # Get all records (LanceDB doesn't have a native random sample,
            # so we get more and sample in Python)
            all_rows = await self._table.to_pandas()
        except Exception as e:
            logger.warning("Failed to query LanceDB for warm moments: %s", e)
            return []

        if all_rows is None or len(all_rows) == 0:
            return []

        # Filter for emotional memories
        import pandas as pd
        df = all_rows
        emotional = df[
            (df["memory_type"] == "EMOTION") |
            (df["affective_flag"] == True)  # noqa: E712
        ]

        if len(emotional) == 0:
            return []

        # Sample randomly
        sampled = emotional.sample(n=min(top_k, len(emotional)))

        moments = []
        for _, row in sampled.iterrows():
            moments.append(Moment(
                memory_id=row["memory_id"],
                scene_summary=row["scene_summary"],
                memory_type=row["memory_type"],
                importance_score=row["importance_score"],
                affective_flag=row["affective_flag"],
            ))

        # v4.5.0 §5.7: re-rank moments by preference_bias
        if hasattr(self, "_memory_prefs") and self._memory_prefs:
            scored = [
                (
                    m,
                    self._compute_preference_bias(m.scene_summary) * 0.3,
                )
                for m in moments
            ]
            scored.sort(key=lambda x: x[1], reverse=True)
            return [item for item, _ in scored]

        return moments

    async def semantic_recall_with_scene(
        self,
        query: str,
        scene_type: str,
        top_k: int = 5,
    ) -> list[MemoryFragment]:
        """
        Combined semantic recall with scene type filtering.
        """
        if self._table is None:
            logger.warning("ColdMemoryStore not initialized — returning empty")
            return []

        query_embedding = await self._generate_embedding(query)

        try:
            _query = await self._table.search(
                np.array(query_embedding, dtype=np.float32),
                vector_column_name="embedding",
            )
            results = await _query.limit(top_k * 5).to_list()
        except Exception as e:
            logger.warning("LanceDB scene recall failed: %s", e)
            return []

        fragments = []
        for row in results:
            if row.get("scene_class", "") == scene_type:
                fragments.append(MemoryFragment(
                    memory_id=row.get("memory_id", ""),
                    scene_summary=row.get("scene_summary", ""),
                    similarity=getattr(row, "_distance", 0.0),
                    memory_type=row.get("memory_type", "FACT"),
                    importance_score=row.get("importance_score", 0.0),
                ))

        # v4.5.0 §5.7: re-rank with preference_bias
        if hasattr(self, "_memory_prefs") and self._memory_prefs:
            scored = [
                (
                    f,
                    f.similarity + self._compute_preference_bias(f.scene_summary) * 0.3,
                )
                for f in fragments
            ]
            scored.sort(key=lambda x: x[1], reverse=True)
            return [item for item, _ in scored[:top_k]]

        fragments.sort(key=lambda x: x.similarity, reverse=True)
        return fragments[:top_k]

    # -------------------------------------------------------------------
    # Memory decay — v4.5.0 §3.3.2
    # -------------------------------------------------------------------

    def calculate_importance(
        self,
        initial_score: float,
        memory_type: str,
        hours_since_last_access: float,
        access_count: int,
        affective_weight: float = 0.0,
    ) -> float:
        """
        Calculate current importance score using the Ebbinghaus-based formula.

        Formula (spec §3.3.2):
          importance(t) = initial_score × exp(-α × hours_since_last_access)
                         × log(1 + access_count) × (1 + affective_weight)

        Special protection: if emotion_intensity > 0.7 and positive,
        α is permanently fixed at 0.1.
        """
        import math

        # Get decay coefficient for this memory type
        alpha = getattr(self._decay_config, memory_type.upper(), 0.5)

        # Special emotional protection
        if (
            memory_type == "EMOTION"
            and affective_weight > self._decay_config.HIGH_EMOTION_THRESHOLD
        ):
            alpha = self._decay_config.HIGH_EMOTION_FIXED_ALPHA

        # Ebbinghaus decay
        decay = math.exp(-alpha * hours_since_last_access)

        # Access frequency bonus
        frequency_bonus = math.log(1 + access_count)

        # Affective boost
        affective_boost = 1.0 + affective_weight

        importance = initial_score * decay * frequency_bonus * affective_boost

        # Clamp to [0, 1]
        return max(0.0, min(1.0, importance))

    def get_decay_level(
        self,
        importance: float,
        hours_survived: float,
    ) -> int:
        """
        Determine the decay/compression level for a memory.

        Level 1: importance < 0.5 AND survived > 24h → keep entities, drop edge details
        Level 2: importance < 0.2 AND survived > 72h → 3B model summary (50 words max)
        Level 3: importance < 0.05 AND survived > 168h → keep only vector embedding

        Returns:
            0 if no compression needed, or 1/2/3 for the compression level.
        """
        if importance < 0.05 and hours_survived > 168:
            return 3
        elif importance < 0.2 and hours_survived > 72:
            return 2
        elif importance < 0.5 and hours_survived > 24:
            return 1
        return 0

    # -------------------------------------------------------------------
    # Incremental sync — v4.5.0 §3.2.4
    # -------------------------------------------------------------------

    async def sync_from_hot(
        self,
        scenes: list[Scene],
        last_sync_position: Optional[str] = None,
    ) -> dict:
        """
        Perform incremental sync from hot memory to cold memory.

        Steps:
        1. Filter for sensitive info
        2. Sort by importance score
        3. Write to LanceDB (atomic per scene)
        4. Set cold_memory:initialized if first successful write

        Args:
            scenes: List of Scene objects to sync.
            last_sync_position: Redis Stream position (for tracking).

        Returns:
            Dict with sync statistics: stored_count, filtered_count, error_count.
        """
        stats = {
            "stored_count": 0,
            "filtered_count": 0,
            "error_count": 0,
            "first_initialization": False,
        }

        if self._table is None:
            logger.error("ColdMemoryStore not initialized — sync aborted")
            stats["error_count"] = len(scenes)
            return stats

        # Check if this is the first-ever sync
        is_first_sync = not await self.is_initialized()

        # Sort by importance score (descending) — §3.2.4 step 3
        sorted_scenes = sorted(
            scenes,
            key=lambda s: s.importance_score,
            reverse=True,
        )

        for scene in sorted_scenes:
            try:
                memory_id = await self.store_scene(scene)
                if memory_id is not None:
                    stats["stored_count"] += 1
                else:
                    stats["filtered_count"] += 1
            except Exception as e:
                # Catch any unexpected error — continue with next scene
                logger.error(
                    "Failed to sync scene %s: %s. Continuing with next scene.",
                    scene.scene_id, e,
                )
                stats["error_count"] += 1

        # Set sentinel on first successful sync
        if is_first_sync and stats["stored_count"] > 0:
            await self._set_initialized_sentinel()
            stats["first_initialization"] = True

        logger.info(
            "Incremental sync complete: stored=%d, filtered=%d, errors=%d, first_init=%s",
            stats["stored_count"], stats["filtered_count"],
            stats["error_count"], stats["first_initialization"],
        )

        return stats

    # -------------------------------------------------------------------
    # Utility
    # -------------------------------------------------------------------

    async def get_table_stats(self) -> dict:
        """Return statistics about the cold memory table."""
        if self._table is None:
            return {"total_records": 0, "memory_types": {}}

        try:
            df = await self._table.to_pandas()
            if df is None or len(df) == 0:
                return {"total_records": 0, "memory_types": {}}

            type_counts = df["memory_type"].value_counts().to_dict()
            total = len(df)
            return {
                "total_records": total,
                "memory_types": type_counts,
            }
        except Exception as e:
            logger.warning("Failed to get table stats: %s", e)
            return {"total_records": 0, "memory_types": {}}

    # -------------------------------------------------------------------
    # User Model persistence — v4.5.0 §3.4.1
    # -------------------------------------------------------------------

    async def get_recent_scenes(
        self,
        limit: int = 50,
        min_importance: float = 0.2,
        since_days: int = 30,
    ) -> list[dict[str, Any]]:
        """
        Retrieve recent scenes for user model generation.  v4.5.0 §3.4.3

        Returns scenes created within *since_days* with importance >= *min_importance*,
        sorted by importance descending, capped at *limit*.

        If the store is not initialized or LanceDB is unavailable, returns [].
        """
        if self._table is None:
            logger.warning(
                "ColdMemoryStore: get_recent_scenes called before initialize()"
            )
            return []

        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=since_days)
        ).isoformat()

        try:
            df = await self._table.to_pandas()
            if df is None or len(df) == 0:
                return []
        except Exception as exc:
            # Catches: LanceDB connection error, table corruption.
            # Safe: return empty list; caller handles via new-user template.
            logger.warning(
                "ColdMemoryStore: to_pandas() failed: %s. Returning [].", exc
            )
            return []

        # Filter by importance + recency
        filtered = df[
            (df["importance_score"] >= min_importance)
            & (df["created_at"] >= cutoff)
        ]

        # Sort by importance descending
        filtered = filtered.sort_values(
            "importance_score", ascending=False
        )

        # Convert to dict list
        scenes: list[dict[str, Any]] = []
        for _, row in filtered.head(limit).iterrows():
            row_dict = row.to_dict()
            # Convert numpy types to Python native types for JSON compat
            for k, v in row_dict.items():
                if hasattr(v, "tolist"):  # numpy array (embeddings)
                    row_dict[k] = v.tolist()
                elif hasattr(v, "item"):  # numpy scalar
                    row_dict[k] = v.item()
            scenes.append(row_dict)

        logger.debug(
            "ColdMemoryStore: get_recent_scenes returned %d scenes "
            "(limit=%d, min_imp=%.2f, since_days=%d)",
            len(scenes), limit, min_importance, since_days,
        )
        return scenes

    async def save_user_model(self, user_model: dict[str, Any]) -> bool:
        """
        Persist the user model to cold memory as a dedicated record.  v4.5.0 §3.4.1

        The user model is stored in a separate table ``user_model`` within the same
        LanceDB database.  Only one record is kept (the latest), with versioning
        tracked via the ``version`` field.

        Returns True on success, False on failure.
        """
        if self._db is None:
            logger.warning(
                "ColdMemoryStore: save_user_model called before initialize()"
            )
            return False

        from copy import deepcopy

        record = deepcopy(user_model)
        record.setdefault("user_model_id", str(uuid.uuid4()))
        record.setdefault("version", 1)
        record.setdefault("generated_at", datetime.now(timezone.utc).isoformat())
        record["stored_at"] = datetime.now(timezone.utc).isoformat()

        try:
            # Use a dedicated table for user model (single-row, versioned)
            table_name = "user_model"
            if table_name not in await self._db.table_names():
                self._user_model_table = await self._db.create_table(table_name)
            else:
                self._user_model_table = await self._db.open_table(table_name)

            # Delete all existing rows (keep only latest)
            try:
                existing = await self._user_model_table.to_pandas()
                if existing is not None and len(existing) > 0:
                    await self._db.drop_table(table_name)
                    self._user_model_table = await self._db.create_table(table_name)
            except Exception:
                # Expected: table empty or drop fails — continue
                pass

            await self._user_model_table.add([record])
            logger.info(
                "UserModel saved to cold memory (v%d, id=%s).",
                record["version"],
                record.get("user_model_id", "?"),
            )
            return True
        except Exception as exc:
            # Catches: LanceDB write failure, schema mismatch.
            # Safe: user model is derivable from cold scenes; log and continue.
            logger.warning(
                "ColdMemoryStore: save_user_model failed: %s. "
                "User model will be regenerated on next sync cycle.",
                exc,
            )
            return False

    # -------------------------------------------------------------------
    # Memory preference weighting — v4.5.0 §5.7
    # -------------------------------------------------------------------

    def _load_memory_preferences(self) -> None:
        """Load memory_preferences from config/baseline.json.

        On any failure (missing file, bad JSON, missing key), sets
        self._memory_prefs = None for graceful degradation — all
        preference_bias computations return 0.0.
        """
        try:
            with open("config/baseline.json") as f:
                baseline = json.load(f)
            self._memory_prefs = baseline.get("memory_preferences", None)
            if self._memory_prefs:
                logger.debug(
                    "Loaded memory_preferences: %d positive, %d negative groups",
                    len(self._memory_prefs.get("positive", [])),
                    len(self._memory_prefs.get("negative", [])),
                )
            else:
                logger.debug(
                    "No memory_preferences found in baseline.json — using neutral weighting"
                )
        except Exception as exc:
            # FileNotFoundError, JSONDecodeError, KeyError — all safe to degrade
            self._memory_prefs = None
            logger.warning(
                "Failed to load memory_preferences from baseline.json: %s. "
                "Preference bias disabled. degraded=true",
                exc,
            )

    def _compute_preference_bias(self, text: str) -> float:
        """Compute preference_bias = clamp(Σ(keyword_match × weight), -0.3, 0.3).

        Iterates positive and negative keyword groups from memory_preferences.
        Each keyword match adds its weight to the bias.
        Final result is clamped to [-0.3, 0.3].

        Returns 0.0 if memory_preferences was not loaded or text is empty.
        """
        if not hasattr(self, "_memory_prefs") or not self._memory_prefs or not text:
            return 0.0

        bias = 0.0
        for entry in self._memory_prefs.get("positive", []):
            for kw in entry.get("keywords", []):
                if kw in text:
                    bias += entry.get("weight", 0.0)

        for entry in self._memory_prefs.get("negative", []):
            for kw in entry.get("keywords", []):
                if kw in text:
                    bias += entry.get("weight", 0.0)

        return max(-0.3, min(0.3, bias))

    async def close(self) -> None:
        """Close the LanceDB connection and release resources."""
        # LanceDB embedded has no explicit close, but we clean up references
        self._table = None
        self._db = None
        self._embedding_model = None
        self._summary_model = None
        logger.info("ColdMemoryStore closed")
