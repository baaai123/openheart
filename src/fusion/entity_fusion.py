"""Cross-modal entity fusion — v4.5.0 §2.5

Aligns entities extracted from audio (speech) events with those from
visual snapshots using bge-small-zh-v1.5 embeddings and cosine similarity.
Supports deictic reference boosting for reduced alignment thresholds.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Constants — v4.5.0 §2.5
# --------------------------------------------------------------------------

DEFAULT_ALIGN_THRESHOLD: float = 0.75
DEICTIC_BOOST_REDUCTION: float = 0.10  # threshold lowered by 0.1 when deictic

# --------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------


@dataclass
class EntityObject:
    """A single extracted entity with text, label, and optional embedding."""

    text: str
    label: str
    embedding: list[float] | None = None


@dataclass
class AlignedPair:
    """A matched audio-visual entity pair."""

    audio_entity: EntityObject
    visual_entity: EntityObject
    similarity: float
    deictic_boosted: bool = False


@dataclass
class EntityFusionResult:
    """Output of cross-modal entity fusion — v4.5.0 §2.5"""

    aligned_pairs: list[AlignedPair] = field(default_factory=list)
    unmatched_audio_entities: list[EntityObject] = field(default_factory=list)
    unmatched_visual_entities: list[EntityObject] = field(default_factory=list)


# --------------------------------------------------------------------------
# spaCy NER — lazy-loaded
# --------------------------------------------------------------------------

def _get_nlp():
    """Lazily load spaCy model with NER capability."""
    try:
        import spacy
        for model_name in ("zh_core_web_trf", "zh_core_web_sm", "en_core_web_sm"):
            try:
                nlp = spacy.load(model_name)
                logger.info(
                    "fusion.entity_fusion: loaded spaCy NER model %s",
                    model_name,
                )
                return nlp
            except OSError:
                continue
        logger.warning(
            "fusion.entity_fusion: no spaCy NER model available, "
            "entity extraction degraded"
        )
        return None
    except ImportError:
        logger.warning(
            "fusion.entity_fusion: spaCy not installed, "
            "NER extraction degraded"
        )
        return None


def _extract_entities_spacy(text: str) -> list[tuple[str, str]]:
    """Extract named entities from text using spaCy NER.

    Returns list of (entity_text, entity_label) tuples.
    Recognized labels: PERSON, ORG, LOC, PRODUCT, EVENT, GPE, FAC, etc.
    """
    if not text or not text.strip():
        return []

    nlp = _get_nlp()
    if nlp is None:
        return []

    try:
        doc = nlp(text[:1024])  # Safety limit
        entities: list[tuple[str, str]] = []
        for ent in doc.ents:
            entities.append((ent.text, ent.label_))
        return entities
    except Exception as exc:
        # spaCy NER failure — safe to continue with empty entities
        logger.warning(
            "fusion.entity_fusion: spaCy NER failed for text=%r: %s",
            text[:80],
            exc,
        )
        return []


def _extract_visual_entities(
    vision_snapshot: dict[str, Any],
) -> list[tuple[str, str]]:
    """Extract entities from a vision snapshot's objects and text content.

    Returns list of (entity_text, entity_label) tuples.
    """
    entities: list[tuple[str, str]] = []

    # Objects with labels
    for obj in vision_snapshot.get("objects", []):
        label = obj.get("label") or obj.get("class_name")
        if label:
            entities.append((str(label), "OBJECT"))

    # Text content (OCR results)
    for txt_item in vision_snapshot.get("text_content", []):
        content = txt_item.get("text") or txt_item.get("content")
        if content:
            # Also run NER on OCR text for richer entities
            ner_results = _extract_entities_spacy(str(content))
            if ner_results:
                entities.extend(ner_results)
            else:
                entities.append((str(content), "TEXT"))

    return entities


# --------------------------------------------------------------------------
# Embedding computation
# --------------------------------------------------------------------------

def _get_embedder():
    """Lazily load Sentence-BERT for entity embedding."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("bge-small-zh-v1.5")
        return model
    except Exception:
        logger.warning(
            "fusion.entity_fusion: sentence-transformers not available, "
            "entity alignment will use string matching fallback"
        )
        return None


def _compute_embeddings(
    texts: list[str],
) -> list[list[float]]:
    """Compute embeddings for a list of entity texts.

    Returns list of embedding vectors. Falls back to None placeholders
    on failure.
    """
    if not texts:
        return []

    model = _get_embedder()
    if model is None:
        return []  # Caller should fall back to string matching

    try:
        import numpy as np
        embeddings = model.encode(texts, normalize_embeddings=True)
        return [emb.tolist() for emb in embeddings]
    except Exception as exc:
        # Embedding computation failed — fallback to string matching
        logger.warning(
            "fusion.entity_fusion: embedding computation failed: %s", exc
        )
        return []


def _cosine_similarity(
    emb_a: list[float], emb_b: list[float]
) -> float:
    """Compute cosine similarity between two normalized embedding vectors."""
    if not emb_a or not emb_b:
        return 0.0
    try:
        import numpy as np
        a = np.asarray(emb_a, dtype=np.float32)
        b = np.asarray(emb_b, dtype=np.float32)
        # Assume already normalized; clip for numerical safety
        sim = float(np.dot(a, b))
        return max(-1.0, min(1.0, sim))
    except Exception:
        return 0.0


def _string_jaccard(text_a: str, text_b: str) -> float:
    """Fallback: Jaccard similarity on character bigrams."""
    if not text_a or not text_b:
        return 0.0

    def bigrams(s: str) -> set[str]:
        return {s[i : i + 2] for i in range(len(s) - 1)} if len(s) >= 2 else {s}

    set_a = bigrams(text_a.lower())
    set_b = bigrams(text_b.lower())

    if not set_a or not set_b:
        return 0.0

    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


# --------------------------------------------------------------------------
# Deictic reference detection
# --------------------------------------------------------------------------

def _detect_deictic_signal(
    events: list[dict[str, Any]],
) -> bool:
    """Check if any event in the window carries a deictic reference signal.

    Deictic references come from perception layer (e.g., "this button",
    "that icon"). The perception bus may set a `deictic_reference` flag
    in metadata.
    """
    for evt in events:
        metadata = evt.get("metadata", {})
        if metadata.get("deictic_reference") is True:
            return True
        # Also check for deictic words in audio text
        audio_text = evt.get("payload", {}).get("audio", {}).get("text", "")
        deictic_words = {"这个", "那个", "this", "that", "这", "那", "these", "those"}
        if any(w in str(audio_text) for w in deictic_words):
            return True
    return False


# --------------------------------------------------------------------------
# EntityFusionEngine
# --------------------------------------------------------------------------


class EntityFusionEngine:
    """Cross-modal entity alignment engine.

    Parameters
    ----------
    align_threshold: float
        Cosine similarity threshold for entity alignment (default 0.75).
    deictic_boost_reduction: float
        Amount to reduce alignment threshold when deictic signal detected
        (default 0.10, resulting in 0.65).

    v4.5.0 §2.5
    """

    def __init__(
        self,
        align_threshold: float = DEFAULT_ALIGN_THRESHOLD,
        deictic_boost_reduction: float = DEICTIC_BOOST_REDUCTION,
    ) -> None:
        self.align_threshold = align_threshold
        self.deictic_boost_reduction = deictic_boost_reduction

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def fuse(
        self,
        classified_events: Any,
        window_events: list[dict[str, Any]] | None = None,
    ) -> EntityFusionResult:
        """Align entities across audio (speech) and visual modalities.

        Parameters
        ----------
        classified_events:
            Output from EventClassifier.classify().
        window_events:
            Raw window events for deictic signal detection.

        Returns
        -------
        EntityFusionResult with aligned pairs and unmatched entities.
        """
        # Extract audio entities from primary and secondary events
        audio_texts: list[str] = []
        primary = getattr(classified_events, "primary_event", None)
        if primary and isinstance(primary, dict):
            audio_texts.append(primary.get("text", ""))
        for sec in getattr(classified_events, "secondary_events", []):
            if isinstance(sec, dict):
                audio_texts.append(sec.get("text", ""))

        # Extract visual entities from vision snapshots
        visual_entities: list[tuple[str, str]] = []
        for evt in (window_events or []):
            payload = evt.get("payload", {})
            if payload.get("type") == "vision_snapshot":
                vs = payload.get("vision_snapshot", {})
                visual_entities.extend(_extract_visual_entities(vs))

        # NER on audio texts
        audio_entities: list[tuple[str, str]] = []
        for text in audio_texts:
            audio_entities.extend(_extract_entities_spacy(text))

        if not audio_entities and not visual_entities:
            return EntityFusionResult()

        # Determine effective threshold (deictic boost)
        has_deictic = _detect_deictic_signal(window_events or [])
        effective_threshold = (
            self.align_threshold - self.deictic_boost_reduction
            if has_deictic
            else self.align_threshold
        )

        # Compute embeddings for all entity texts
        all_audio_texts = [e[0] for e in audio_entities]
        all_visual_texts = [e[0] for e in visual_entities]

        audio_embeddings = _compute_embeddings(all_audio_texts)
        visual_embeddings = _compute_embeddings(all_visual_texts)

        # Build EntityObjects
        audio_objs = [
            EntityObject(
                text=audio_entities[i][0],
                label=audio_entities[i][1],
                embedding=audio_embeddings[i] if i < len(audio_embeddings) else None,
            )
            for i in range(len(audio_entities))
        ]
        visual_objs = [
            EntityObject(
                text=visual_entities[j][0],
                label=visual_entities[j][1],
                embedding=visual_embeddings[j] if j < len(visual_embeddings) else None,
            )
            for j in range(len(visual_entities))
        ]

        # Pairwise similarity matching
        matched_audio_indices: set[int] = set()
        matched_visual_indices: set[int] = set()
        aligned_pairs: list[AlignedPair] = []

        for i, a_obj in enumerate(audio_objs):
            best_sim = 0.0
            best_j = -1

            for j, v_obj in enumerate(visual_objs):
                if j in matched_visual_indices:
                    continue

                if a_obj.embedding and v_obj.embedding:
                    sim = _cosine_similarity(a_obj.embedding, v_obj.embedding)
                else:
                    sim = _string_jaccard(a_obj.text, v_obj.text)

                if sim > best_sim:
                    best_sim = sim
                    best_j = j

            if best_j >= 0 and best_sim >= effective_threshold:
                aligned_pairs.append(
                    AlignedPair(
                        audio_entity=a_obj,
                        visual_entity=visual_objs[best_j],
                        similarity=best_sim,
                        deictic_boosted=has_deictic,
                    )
                )
                matched_audio_indices.add(i)
                matched_visual_indices.add(best_j)

        # Collect unmatched
        unmatched_audio = [
            a_obj
            for i, a_obj in enumerate(audio_objs)
            if i not in matched_audio_indices
        ]
        unmatched_visual = [
            v_obj
            for j, v_obj in enumerate(visual_objs)
            if j not in matched_visual_indices
        ]

        logger.debug(
            "fusion.entity_fusion: %d aligned pairs, %d unmatched audio, "
            "%d unmatched visual (threshold=%.2f, deictic=%s)",
            len(aligned_pairs),
            len(unmatched_audio),
            len(unmatched_visual),
            effective_threshold,
            has_deictic,
        )

        return EntityFusionResult(
            aligned_pairs=aligned_pairs,
            unmatched_audio_entities=unmatched_audio,
            unmatched_visual_entities=unmatched_visual,
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        """Reset internal state. Currently stateless, but reserved for
        session-level entity tracking in future versions."""
        pass
