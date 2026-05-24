"""Scene synthesis — v4.5.0 §2.6

Composes the final Scene output from classified events and cross-modal
entity alignments. Builds entity relation graphs via spaCy, computes
semantic fusion representations via Sentence-BERT, and emits the
structured Scene message envelope for downstream consumption.
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Data structures — v4.5.0 §2.6 Scene
# --------------------------------------------------------------------------


@dataclass
class EntityRelation:
    """Subject-predicate-object triple — v4.5.0 §2.6"""

    subject: str
    predicate: str
    object: str


@dataclass
class EmotionSnapshot:
    """Emotion state at the time of scene synthesis — v4.5.0 §2.6"""

    category: str  # joy | sadness | neutral
    intensity: float


@dataclass
class SceneClass:
    """Scene classification — v4.5.0 §2.6"""

    primary: str
    confidence: float


@dataclass
class ProvenanceSource:
    """Traceability for a component of the scene — v4.5.0 §2.6"""

    fragment_id: str
    timestamp: str


@dataclass
class UserModelSnapshot:
    """Snapshot of user model at scene time — v4.5.0 §2.6"""

    inferred_personality: str
    current_concerns: list[str]
    relationship_stage: str  # new | familiar | close


@dataclass
class ScenePayload:
    """The payload of a Scene output — v4.5.0 §2.6"""

    summary: str
    primary_event: str
    secondary_events: list[str] = field(default_factory=list)
    entity_relations: list[EntityRelation] = field(default_factory=list)
    aligned_entities: list[dict[str, Any]] = field(default_factory=list)
    emotion_snapshot: EmotionSnapshot = field(
        default_factory=lambda: EmotionSnapshot(category="neutral", intensity=0.0)
    )
    scene_class: SceneClass = field(
        default_factory=lambda: SceneClass(primary="unknown", confidence=0.0)
    )
    confidence: float = 0.0
    provenance: dict[str, Any] = field(default_factory=dict)
    user_model_snapshot: dict[str, Any] = field(default_factory=dict)


@dataclass
class SceneMetadata:
    """Metadata envelope for Scene output — v4.5.0 §2.6"""

    confidence: float = 0.0
    latency_ms: float = 0.0
    degraded: bool = False
    affective_flag: bool = False
    user_model_version: int = 0


def _build_scene_envelope(
    scene_id: str,
    trace_id: str,
    version: int,
    payload: ScenePayload,
    metadata: SceneMetadata,
) -> dict[str, Any]:
    """Build the full Scene message envelope per spec §0.3 and §2.6."""
    return {
        "scene_id": scene_id,
        "trace_id": trace_id,
        "source_layer": "fusion",
        "source_component": "scene_synthesis",
        "timestamp": _iso_now(),
        "version": version,
        "payload_type": "scene",
        "payload": {
            "summary": payload.summary,
            "primary_event": payload.primary_event,
            "secondary_events": payload.secondary_events,
            "entity_relations": [
                {"subject": r.subject, "predicate": r.predicate, "object": r.object}
                for r in payload.entity_relations
            ],
            "aligned_entities": payload.aligned_entities,
            "emotion_snapshot": {
                "category": payload.emotion_snapshot.category,
                "intensity": payload.emotion_snapshot.intensity,
            },
            "scene_class": {
                "primary": payload.scene_class.primary,
                "confidence": payload.scene_class.confidence,
            },
            "confidence": payload.confidence,
            "provenance": payload.provenance,
            "user_model_snapshot": payload.user_model_snapshot,
        },
        "metadata": {
            "confidence": metadata.confidence,
            "latency_ms": metadata.latency_ms,
            "degraded": metadata.degraded,
            "affective_flag": metadata.affective_flag,
            "user_model_version": metadata.user_model_version,
        },
    }


def _iso_now() -> str:
    """Return current UTC time as ISO 8601 string."""
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


# --------------------------------------------------------------------------
# spaCy helpers for relation extraction
# --------------------------------------------------------------------------

def _get_nlp():
    """Lazily load spaCy model."""
    try:
        import spacy
        for model_name in ("zh_core_web_trf", "zh_core_web_sm", "en_core_web_sm"):
            try:
                nlp = spacy.load(model_name)
                logger.info(
                    "fusion.scene_synthesis: loaded spaCy model %s", model_name
                )
                return nlp
            except OSError:
                continue
        logger.warning(
            "fusion.scene_synthesis: no spaCy model found, "
            "relation extraction degraded"
        )
        return None
    except ImportError:
        logger.warning(
            "fusion.scene_synthesis: spaCy not installed, "
            "relation extraction degraded"
        )
        return None


def _extract_triples(text: str) -> list[tuple[str, str, str]]:
    """Extract subject-predicate-object triples from text using spaCy.

    Returns list of (subject, predicate, object) tuples.
    """
    if not text or not text.strip():
        return []

    nlp = _get_nlp()
    if nlp is None:
        return []

    triples: list[tuple[str, str, str]] = []
    try:
        doc = nlp(text[:1024])
        for token in doc:
            if token.dep_ == "ROOT" and token.pos_ in ("VERB", "ADJ"):
                subj_text = ""
                obj_text = ""
                for child in token.children:
                    if child.dep_ in ("nsubj", "nsubjpass"):
                        # Collect subject subtree
                        subjects = [t for t in child.subtree]
                        subj_text = " ".join(t.text for t in subjects)
                    elif child.dep_ in ("dobj", "obj", "pobj"):
                        # Collect object subtree
                        objects = [t for t in child.subtree]
                        obj_text = " ".join(t.text for t in objects)

                if subj_text and token.text:
                    triples.append((subj_text, token.lemma_, obj_text or ""))
    except Exception as exc:
        # spaCy parse failure — safe to continue with empty triples
        logger.warning(
            "fusion.scene_synthesis: triple extraction failed for text=%r: %s",
            text[:80],
            exc,
        )

    return triples


# --------------------------------------------------------------------------
# Sentence-BERT semantic fusion
# --------------------------------------------------------------------------

def _get_embedder():
    """Lazily load Sentence-BERT for semantic fusion."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("bge-small-zh-v1.5")
        return model
    except ImportError:
        logger.warning(
            "fusion.scene_synthesis: sentence-transformers not available, "
            "semantic fusion degraded"
        )
        return None


def _semantic_fusion_score(texts: list[str]) -> float:
    """Compute pairwise cosine similarity mean across texts as fusion score.

    A higher score means stronger semantic coherence across events.
    """
    if len(texts) < 2:
        return 1.0  # Single text is coherent with itself

    model = _get_embedder()
    if model is None:
        return 0.5  # Neutral fallback

    try:
        import numpy as np
        embeddings = model.encode(texts, normalize_embeddings=True)
        sims = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = float(np.dot(embeddings[i], embeddings[j]))
                sims.append(max(-1.0, min(1.0, sim)))
        return float(np.mean(sims)) if sims else 0.0
    except Exception as exc:
        logger.warning(
            "fusion.scene_synthesis: semantic fusion failed: %s", exc
        )
        return 0.5


# --------------------------------------------------------------------------
# SceneSynthesizer
# --------------------------------------------------------------------------


class SceneSynthesizer:
    """Compose classified events and entity alignments into a Scene.

    v4.5.0 §2.6
    """

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def synthesize(
        self,
        trace_id: str,
        version: int,
        classified_events: Any,
        entity_fusion_result: Any,
        window_events: list[dict[str, Any]] | None = None,
        user_model_version: int = 0,
        start_time: float | None = None,
    ) -> dict[str, Any]:
        """Synthesize a complete Scene from classified events and entities.

        Parameters
        ----------
        trace_id:
            Trace ID for the current interaction chain.
        version:
            Monotonic version number.
        classified_events:
            Output from EventClassifier.classify().
        entity_fusion_result:
            Output from EntityFusionEngine.fuse().
        window_events:
            Raw window events for provenance and emotion extraction.
        user_model_version:
            Current user model version number.
        start_time:
            Processing start time for latency calculation.

        Returns
        -------
        dict representing the full Scene message envelope (§2.6).
        """
        begin = start_time if start_time is not None else time.time()

        # Extract data from classified events
        primary = getattr(classified_events, "primary_event", None)
        primary_text = primary.get("text", "") if isinstance(primary, dict) else ""
        secondary_events = getattr(classified_events, "secondary_events", [])
        secondary_texts = [
            s.get("text", "") if isinstance(s, dict) else str(s)
            for s in secondary_events
        ]

        # Build entity relations from primary event text
        entity_relations: list[EntityRelation] = []
        if primary_text:
            triples = _extract_triples(primary_text)
            for subj, pred, obj in triples:
                entity_relations.append(
                    EntityRelation(subject=subj, predicate=pred, object=obj)
                )

        # Build aligned entities from fusion result
        aligned_pairs = getattr(entity_fusion_result, "aligned_pairs", [])
        aligned_entities: list[dict[str, Any]] = []
        for pair in aligned_pairs:
            aligned_entities.append({
                "audio_entity": {
                    "text": pair.audio_entity.text if hasattr(pair, "audio_entity") else "",
                    "label": pair.audio_entity.label if hasattr(pair, "audio_entity") else "",
                },
                "visual_entity": {
                    "text": pair.visual_entity.text if hasattr(pair, "visual_entity") else "",
                    "label": pair.visual_entity.label if hasattr(pair, "visual_entity") else "",
                },
                "similarity": pair.similarity if hasattr(pair, "similarity") else 0.0,
                "deictic_boosted": pair.deictic_boosted if hasattr(pair, "deictic_boosted") else False,
            })

        # Extract emotion snapshot from primary or window events
        emotion_snapshot = self._extract_emotion(primary, window_events)

        # Determine affective flag
        affective_flag = self._check_affective(
            primary, secondary_events, window_events
        )

        # Determine scene class from vision snapshots
        scene_class = self._extract_scene_class(window_events)

        # Compute semantic fusion score across all event texts
        all_texts = [primary_text] + secondary_texts
        all_texts = [t for t in all_texts if t.strip()]
        fusion_score = _semantic_fusion_score(all_texts)

        # Build confidence as weighted combination
        primary_score = (
            primary.get("score", 0.5) if isinstance(primary, dict) else 0.5
        )
        confidence = (
            0.5 * fusion_score
            + 0.3 * primary_score
            + 0.2 * emotion_snapshot.intensity
        )
        confidence = min(1.0, max(0.0, confidence))

        # Build provenance
        provenance = self._build_provenance(primary, window_events)

        # Build user model snapshot
        user_model_snapshot = self._build_user_model_snapshot(classified_events)

        # Generate natural language summary
        summary = self._generate_summary(
            primary_text, secondary_texts, entity_relations, emotion_snapshot
        )

        # Assemble payload
        payload = ScenePayload(
            summary=summary,
            primary_event=primary_text,
            secondary_events=secondary_texts,
            entity_relations=entity_relations,
            aligned_entities=aligned_entities,
            emotion_snapshot=emotion_snapshot,
            scene_class=scene_class,
            confidence=confidence,
            provenance=provenance,
            user_model_snapshot=user_model_snapshot,
        )

        # Compute latency
        latency_ms = (time.time() - begin) * 1000.0

        # Build metadata
        metadata = SceneMetadata(
            confidence=confidence,
            latency_ms=latency_ms,
            degraded=False,  # Will be set by pipeline if any component degraded
            affective_flag=affective_flag,
            user_model_version=user_model_version,
        )

        # Check for degradation signals
        if any(
            e.get("metadata", {}).get("degraded", False) is True
            for e in (window_events or [])
        ):
            metadata.degraded = True

        # Build final envelope
        result = _build_scene_envelope(
            scene_id=str(uuid.uuid4()),
            trace_id=trace_id,
            version=version,
            payload=payload,
            metadata=metadata,
        )

        logger.debug(
            "fusion.scene_synthesis: synthesized scene %s (confidence=%.2f, "
            "latency=%.1fms, emotion=%s, relations=%d)",
            result["scene_id"],
            confidence,
            latency_ms,
            emotion_snapshot.category,
            len(entity_relations),
        )

        return result

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_emotion(
        primary: dict[str, Any] | None,
        window_events: list[dict[str, Any]] | None,
    ) -> EmotionSnapshot:
        """Extract emotion snapshot from primary event or window events."""
        # Try primary event first
        if isinstance(primary, dict):
            emotion_data = primary.get("emotion") or primary.get("emotion_snapshot")
            if emotion_data:
                return EmotionSnapshot(
                    category=emotion_data.get("category", "neutral"),
                    intensity=emotion_data.get("intensity", 0.0),
                )

        # Fall back to first window event with emotion
        for evt in (window_events or []):
            metadata = evt.get("metadata", {})
            emotion = metadata.get("emotion", {})
            if emotion.get("category"):
                return EmotionSnapshot(
                    category=emotion["category"],
                    intensity=emotion.get("intensity", 0.0),
                )

        return EmotionSnapshot(category="neutral", intensity=0.0)

    @staticmethod
    def _check_affective(
        primary: dict[str, Any] | None,
        secondary_events: list[dict[str, Any]],
        window_events: list[dict[str, Any]] | None,
    ) -> bool:
        """Check if the scene is affectively significant."""
        if isinstance(primary, dict) and primary.get("affective") is True:
            return True

        for sec in secondary_events:
            if isinstance(sec, dict) and sec.get("affective") is True:
                return True

        for evt in (window_events or []):
            metadata = evt.get("metadata", {})
            if metadata.get("affective_flag") is True:
                return True

        return False

    @staticmethod
    def _extract_scene_class(
        window_events: list[dict[str, Any]] | None,
    ) -> SceneClass:
        """Extract scene class from vision snapshot metadata."""
        for evt in (window_events or []):
            payload = evt.get("payload", {})
            if payload.get("type") == "vision_snapshot":
                vs = payload.get("vision_snapshot", {})
                scene = vs.get("scene_class", {})
                if scene.get("primary"):
                    return SceneClass(
                        primary=scene["primary"],
                        confidence=scene.get("confidence", 0.0),
                    )
            metadata = evt.get("metadata", {})
            scene_ctx = metadata.get("scene_context", {})
            if scene_ctx.get("primary_type"):
                return SceneClass(
                    primary=scene_ctx["primary_type"],
                    confidence=scene_ctx.get("confidence", 0.0),
                )

        return SceneClass(primary="unknown", confidence=0.0)

    @staticmethod
    def _build_provenance(
        primary: dict[str, Any] | None,
        window_events: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        """Build provenance tracking for the scene."""
        provenance: dict[str, Any] = {
            "primary_source": {},
            "secondary_sources": [],
            "alignment_scores": [],
        }

        if isinstance(primary, dict):
            provenance["primary_source"] = {
                "fragment_id": str(uuid.uuid4()),
                "timestamp": _iso_now(),
            }

        for evt in (window_events or []):
            ts = evt.get("timestamp", _iso_now())
            provenance["secondary_sources"].append({
                "roi_id": str(uuid.uuid4()),
                "timestamp": ts,
            })

        return provenance

    @staticmethod
    def _build_user_model_snapshot(classified_events: Any) -> dict[str, Any]:
        """Build a minimal user model snapshot from classified events.

        Full user model is maintained by the memory layer; this captures
        a lightweight snapshot for the scene.
        """
        primary = getattr(classified_events, "primary_event", None)
        has_primary = isinstance(primary, dict) and primary.get("text")
        has_secondary = bool(getattr(classified_events, "secondary_events", []))

        interaction_complexity = "simple"
        if has_primary and has_secondary:
            interaction_complexity = "complex"
        elif has_primary:
            interaction_complexity = "moderate"

        return {
            "inferred_personality": "",
            "current_concerns": [],
            "relationship_stage": "familiar",
            "interaction_complexity": interaction_complexity,
        }

    @staticmethod
    def _generate_summary(
        primary_text: str,
        secondary_texts: list[str],
        entity_relations: list[EntityRelation],
        emotion_snapshot: EmotionSnapshot,
    ) -> str:
        """Generate a natural-language summary of the scene.

        Uses template-based generation. In production, this would use
        the 3B model for richer descriptions.
        """
        parts: list[str] = []

        if primary_text:
            parts.append(primary_text)

        if secondary_texts:
            visual_summary = "; ".join(secondary_texts[:3])
            if visual_summary:
                parts.append(f"上下文: {visual_summary}")

        if entity_relations:
            relation_strs = [
                f"{r.subject}{r.predicate}{r.object}" if r.object else f"{r.subject}{r.predicate}"
                for r in entity_relations[:2]
            ]
            if relation_strs:
                parts.append("关系: " + ", ".join(relation_strs))

        if emotion_snapshot.category != "neutral" or emotion_snapshot.intensity > 0.1:
            emo_str = f"情绪: {emotion_snapshot.category}"
            if emotion_snapshot.intensity > 0.3:
                emo_str += f" (强度 {emotion_snapshot.intensity:.1f})"
            parts.append(emo_str)

        if not parts:
            return "空情景"

        return " | ".join(parts)

    # ------------------------------------------------------------------ #
    # Reset
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        """Reset internal state (stateless, reserved for future use)."""
        pass
