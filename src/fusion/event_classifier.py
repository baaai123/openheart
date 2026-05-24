"""Event classifier with emotion-weighted scoring — v4.5.0 §2.4

Classifies events from a time window into primary, secondary, fragment,
and ambient categories. Uses spaCy dependency parsing for semantic
completeness scoring with emotion boost weighting.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Default scoring weights — v4.5.0 §2.4
# --------------------------------------------------------------------------

DEFAULT_W1 = 0.25  # has_subject
DEFAULT_W2 = 0.25  # has_predicate
DEFAULT_W3 = 0.15  # has_object
DEFAULT_W4 = 0.15  # sentence_length_factor
DEFAULT_W5 = 0.20  # information_density
DEFAULT_W6 = 0.30  # emotion intensity weight

# Emotion multipliers — v4.5.0 §2.4
EMOTION_MULTIPLIER_POSITIVE = 1.2
EMOTION_MULTIPLIER_NEGATIVE = 1.4

# Classification thresholds
PRIMARY_SCORE_THRESHOLD = 0.6
EMOTION_INTENSITY_THRESHOLD = 0.7
CONTEXT_MERGE_SIMILARITY = 0.85

# Number of recent primary events to track for context-aware merging
CONTEXT_WINDOW_SIZE = 3

# --------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------


@dataclass
class ClassifiedEvents:
    """Output of event classification — v4.5.0 §2.4"""

    primary_event: dict[str, Any] | None
    secondary_events: list[dict[str, Any]] = field(default_factory=list)
    fragment_events: list[dict[str, Any]] = field(default_factory=list)
    ambient_events: list[dict[str, Any]] = field(default_factory=list)


# --------------------------------------------------------------------------
# spaCy helpers — lazy-loaded to avoid import-time failures
# --------------------------------------------------------------------------

def _get_nlp():
    """Lazily load spaCy model. Returns None if unavailable."""
    try:
        import spacy
        # Try Chinese model first (primary target), fall back to English
        for model_name in ("zh_core_web_trf", "zh_core_web_sm", "en_core_web_sm"):
            try:
                nlp = spacy.load(model_name)
                logger.info(
                    "fusion.event_classifier: loaded spaCy model %s", model_name
                )
                return nlp
            except OSError:
                continue
        logger.warning(
            "fusion.event_classifier: no spaCy model found, "
            "semantic completeness scoring will degrade"
        )
        return None
    except ImportError:
        logger.warning(
            "fusion.event_classifier: spaCy not installed, "
            "classification degraded"
        )
        return None


def _has_subject(doc) -> bool:
    """Check spaCy dependency parse for a nominal subject (nsubj)."""
    try:
        return any(
            token.dep_ in ("nsubj", "nsubjpass", "csubj")
            for token in doc
        )
    except Exception:
        return False


def _has_predicate(doc) -> bool:
    """Check spaCy dependency parse for a root verb."""
    try:
        for token in doc:
            if token.dep_ == "ROOT" and token.pos_ in (
                "VERB", "AUX", "ADJ"
            ):
                return True
        return False
    except Exception:
        return False


def _has_object(doc) -> bool:
    """Check spaCy dependency parse for a direct object (dobj)."""
    try:
        return any(
            token.dep_ in ("dobj", "obj", "pobj", "iobj")
            for token in doc
        )
    except Exception:
        return False


def _information_density(text: str) -> float:
    """Estimate information density as ratio of content words to total.

    Content words: nouns, verbs, adjectives, adverbs.
    Returns a value in [0, 1], clamped.
    """
    if not text or not text.strip():
        return 0.0

    nlp = _get_nlp()
    if nlp is None:
        return 0.5  # Neutral default when NLP unavailable

    try:
        doc = nlp(text[:512])  # Safety limit for long texts
        if len(doc) == 0:
            return 0.0
        content_tags = {"NOUN", "VERB", "ADJ", "ADV", "PROPN"}
        content_count = sum(1 for t in doc if t.pos_ in content_tags)
        return min(1.0, content_count / len(doc))
    except Exception:
        # Catch-all for unexpected NLP errors — degrade gracefully
        return 0.5


# --------------------------------------------------------------------------
# Event classifier
# --------------------------------------------------------------------------


class EventClassifier:
    """Classify windowed events by semantic completeness + emotion weight.

    Parameters
    ----------
    w1..w6: float
        Scoring weights (see spec §2.4).
    primary_threshold: float
        Minimum score for primary event classification (default 0.6).
    emotion_intensity_threshold: float
        Intensity threshold for promoting emotional fragments (default 0.7).
    context_window_size: int
        Number of recent primary events for context-aware merging (default 3).

    v4.5.0 §2.4
    """

    def __init__(
        self,
        w1: float = DEFAULT_W1,
        w2: float = DEFAULT_W2,
        w3: float = DEFAULT_W3,
        w4: float = DEFAULT_W4,
        w5: float = DEFAULT_W5,
        w6: float = DEFAULT_W6,
        primary_threshold: float = PRIMARY_SCORE_THRESHOLD,
        emotion_intensity_threshold: float = EMOTION_INTENSITY_THRESHOLD,
        context_window_size: int = CONTEXT_WINDOW_SIZE,
    ) -> None:
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4
        self.w5 = w5
        self.w6 = w6
        self.primary_threshold = primary_threshold
        self.emotion_intensity_threshold = emotion_intensity_threshold
        self.context_window_size = context_window_size

        # Context tracking: recent primary event texts for merging
        self._recent_primary_texts: list[str] = []

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def classify(
        self,
        events: list[dict[str, Any]],
        window_affective: bool = False,
    ) -> ClassifiedEvents:
        """Classify a list of perception events from a time window.

        Returns a ClassifiedEvents with primary, secondary, fragment,
        and ambient categories.

        v4.5.0 §2.4: classification rules
        """
        if not events:
            return ClassifiedEvents(primary_event=None)

        audio_events: list[dict[str, Any]] = []
        visual_events: list[dict[str, Any]] = []

        for evt in events:
            ptype = evt.get("payload", {}).get("type", "")
            if ptype == "audio_event":
                audio_events.append(evt)
            elif ptype == "vision_snapshot":
                visual_events.append(evt)

        # Separate audio events with and without text
        text_events = [
            e
            for e in audio_events
            if e.get("payload", {}).get("audio", {}).get("text", "").strip()
        ]
        silent_events = [e for e in audio_events if e not in text_events]

        # Score all text-bearing audio events
        scored_events: list[tuple[dict[str, Any], float]] = []
        for evt in text_events:
            text = evt.get("payload", {}).get("audio", {}).get("text", "")
            metadata = evt.get("metadata", {})
            emotion = metadata.get("emotion", {})
            intensity = emotion.get("intensity", 0.0)
            category = emotion.get("category", "neutral")
            affective = metadata.get("affective_flag", False)

            score = self._compute_score(text, intensity, category, affective)
            scored_events.append((evt, score))

        # Sort by score descending for primary selection
        scored_events.sort(key=lambda x: x[1], reverse=True)

        # Determine primary event
        primary: dict[str, Any] | None = None
        fragments: list[dict[str, Any]] = []
        secondary: list[dict[str, Any]] = []
        ambient: list[dict[str, Any]] = list(silent_events)

        remaining_scored: list[tuple[dict[str, Any], float]] = []

        for evt, score in scored_events:
            if primary is None and score >= self.primary_threshold:
                primary = self._check_context_merge(evt, score)
            elif score < self.primary_threshold:
                intensity_val = (
                    evt.get("metadata", {})
                    .get("emotion", {})
                    .get("intensity", 0.0)
                )
                if intensity_val >= self.emotion_intensity_threshold:
                    # Promote emotional fragment to key event (primary)
                    scored_serializable = {
                        "text": evt.get("payload", {}).get("audio", {}).get("text", ""),
                        "source": "audio",
                        "score": score,
                        "affective": evt.get("metadata", {}).get("affective_flag", False),
                        "_event": evt,
                    }
                    if primary is None:
                        primary = scored_serializable
                    else:
                        fragments.append(scored_serializable)
                else:
                    scored_serializable = {
                        "text": evt.get("payload", {}).get("audio", {}).get("text", ""),
                        "source": "audio",
                        "score": score,
                        "affective": evt.get("metadata", {}).get("affective_flag", False),
                    }
                    fragments.append(scored_serializable)
            else:
                remaining_scored.append((evt, score))

        # Classify visual events as secondary or ambient
        primary_text = (
            primary.get("text", "")
            if isinstance(primary, dict) and "text" in primary
            else ""
        )
        for vevt in visual_events:
            if self._has_deictic_association(vevt, primary_text):
                vs = vevt.get("payload", {}).get("vision_snapshot", {})
                secondary.append({
                    "text": vs.get("scene_class", {}).get("primary", "visual"),
                    "source": "visual",
                    "linked_entity": self._extract_primary_entity(vevt),
                })
            else:
                ambient.append({
                    "text": vevt.get("payload", {}).get("vision_snapshot", {}).get("scene_class", {}).get("primary", "visual"),
                    "source": "visual",
                })

        # Track primary for context-aware merging
        if primary and isinstance(primary, dict):
            primary_event_data = primary.get("_event", primary)
            primary_event_text = primary.get("text", "")
            if primary_event_text:
                self._recent_primary_texts.append(primary_event_text)
                if len(self._recent_primary_texts) > self.context_window_size:
                    self._recent_primary_texts.pop(0)

        # Clean up internal _event references before returning
        result_primary: dict[str, Any] | None = None
        if primary is not None:
            result_primary = {
                k: v for k, v in primary.items() if k != "_event"
            }

        return ClassifiedEvents(
            primary_event=result_primary,
            secondary_events=secondary,
            fragment_events=fragments,
            ambient_events=ambient,
        )

    # ------------------------------------------------------------------ #
    # Scoring
    # ------------------------------------------------------------------ #

    def _compute_score(
        self,
        text: str,
        emotion_intensity: float,
        emotion_category: str,
        affective: bool,
    ) -> float:
        """Compute semantic completeness score with emotion bonus.

        S = S_base + w6 * (emotion_intensity * emotion_multiplier)

        When spaCy is unavailable, uses a degraded heuristic based on
        text length and content word ratio from simple token counting.

        v4.5.0 §2.4
        """
        nlp = _get_nlp()

        if nlp is not None and text.strip():
            return self._compute_score_spacy(text, emotion_intensity, emotion_category)

        # Degraded path: spaCy unavailable — use simple text heuristics
        return self._compute_score_degraded(text, emotion_intensity, emotion_category)

    def _compute_score_spacy(
        self,
        text: str,
        emotion_intensity: float,
        emotion_category: str,
    ) -> float:
        """Full spaCy-based scoring (primary path)."""
        nlp = _get_nlp()
        has_subj = False
        has_pred = False
        has_obj = False
        sent_len_factor = 0.0
        info_density = 0.5

        try:
            doc = nlp(text[:512])
            has_subj = _has_subject(doc)
            has_pred = _has_predicate(doc)
            has_obj = _has_object(doc)
            sent_len_factor = min(1.0, len(doc) / 20.0)
            info_density = _information_density(text)
        except Exception as exc:
            logger.warning(
                "fusion.event_classifier: spaCy parse failed for text=%r: %s",
                text[:80],
                exc,
            )

        s_base = (
            self.w1 * float(has_subj)
            + self.w2 * float(has_pred)
            + self.w3 * float(has_obj)
            + self.w4 * sent_len_factor
            + self.w5 * info_density
        )
        return self._apply_emotion_bonus(s_base, emotion_intensity, emotion_category)

    def _compute_score_degraded(
        self,
        text: str,
        emotion_intensity: float,
        emotion_category: str,
    ) -> float:
        """Degraded scoring when spaCy is unavailable.

        Uses text length and simple word counting as fallback heuristics.
        Logs at WARNING with degraded flag for observability.
        """
        stripped = text.strip() if text else ""
        if not stripped:
            return 0.0

        char_len = len(stripped)

        # Length factor: longer text → higher score
        sent_len_factor = min(1.0, char_len / 20.0)

        # Simple word count for information density estimate
        words = stripped.split()
        word_count = len(words)
        info_density = min(1.0, word_count / 8.0)

        # Subject heuristic: Chinese pronouns + topic markers + English pronouns
        chinese_subj = {
            "我", "你", "他", "她", "它", "我们", "你们", "他们", "她们",
            "用户", "系统", "程序", "电脑", "手机", "机器",
        }
        has_possible_subject = any(
            s in stripped for s in chinese_subj
        ) or any(
            w.lower() in {"i", "you", "he", "she", "it", "we", "they", "user"}
            for w in words
        )

        # Predicate heuristic: common Chinese/English verbs
        chinese_verbs = {
            "点击", "打开", "关闭", "运行", "启动", "输入", "查看", "搜索",
            "是", "有", "说", "问", "做", "去", "看", "听", "写", "读",
            "编译", "调试", "下载", "上传", "保存", "删除", "修改",
        }
        has_possible_predicate = any(
            v in stripped for v in chinese_verbs
        ) or word_count >= 3

        s_base = (
            self.w1 * float(has_possible_subject)
            + self.w2 * float(has_possible_predicate)
            + self.w3 * 0.0  # Object detection is too complex for simple heuristic
            + self.w4 * sent_len_factor
            + self.w5 * info_density
        )

        return self._apply_emotion_bonus(s_base, emotion_intensity, emotion_category)

    def _apply_emotion_bonus(
        self,
        s_base: float,
        emotion_intensity: float,
        emotion_category: str,
    ) -> float:
        """Apply emotion multiplier bonus to the base score."""
        if emotion_category in ("sadness",):
            multiplier = EMOTION_MULTIPLIER_NEGATIVE
        elif emotion_category in ("joy",):
            multiplier = EMOTION_MULTIPLIER_POSITIVE
        else:
            multiplier = 1.0

        emotion_bonus = self.w6 * (emotion_intensity * multiplier)
        return s_base + emotion_bonus

    # ------------------------------------------------------------------ #
    # Context-aware merging
    # ------------------------------------------------------------------ #

    def _check_context_merge(
        self, event: dict[str, Any], score: float
    ) -> dict[str, Any]:
        """Check if this event should be merged with a recent primary.

        If the event text is semantically similar to the most recent
        primary (>0.85 Sentence-BERT cosine similarity), merge them.

        v4.5.0 §2.4: context-aware merging
        """
        text = event.get("payload", {}).get("audio", {}).get("text", "")

        if not text or not self._recent_primary_texts:
            return self._event_to_dict(event, score)

        prev_text = self._recent_primary_texts[-1]

        try:
            similarity = self._compute_similarity(text, prev_text)
            if similarity > CONTEXT_MERGE_SIMILARITY:
                # Merge: combine texts
                merged_text = f"{prev_text} {text}"
                return {
                    "text": merged_text,
                    "source": event.get("payload", {}).get(
                        "type", "audio_event"
                    ).replace("audio_event", "audio"),
                    "score": score,
                    "affective": event.get("metadata", {}).get(
                        "affective_flag", False
                    ),
                }
        except Exception as exc:
            # Sentence-BERT may be unavailable — merge check failure is safe
            logger.warning(
                "fusion.event_classifier: similarity check failed: %s", exc
            )

        return self._event_to_dict(event, score)

    @staticmethod
    def _event_to_dict(
        event: dict[str, Any], score: float
    ) -> dict[str, Any]:
        """Convert a raw perception event to the classified output format."""
        return {
            "text": event.get("payload", {}).get("audio", {}).get("text", ""),
            "source": "audio",
            "score": score,
            "affective": event.get("metadata", {}).get("affective_flag", False),
        }

    # ------------------------------------------------------------------ #
    # Visual event association
    # ------------------------------------------------------------------ #

    @staticmethod
    def _has_deictic_association(
        visual_event: dict[str, Any], primary_text: str
    ) -> bool:
        """Check if a visual event has deictic/reference association.

        A visual event is associated with the primary if:
        - The vision snapshot references entities that overlap with primary text
        - OR a deictic_reference signal exists in the visual event metadata

        Simplified heuristic: check for text content overlap.
        """
        if not primary_text:
            return False

        vs = visual_event.get("payload", {}).get("vision_snapshot", {})
        text_items = vs.get("text_content", [])

        for item in text_items:
            content = item.get("text") or item.get("content") or ""
            # Simple substring overlap as lightweight association check
            if content and any(
                word in primary_text
                for word in content.split()
                if len(word) >= 2
            ):
                return True

        # Check for explicit deictic reference
        metadata = visual_event.get("metadata", {})
        if metadata.get("deictic_reference"):
            return True

        return False

    @staticmethod
    def _extract_primary_entity(
        visual_event: dict[str, Any],
    ) -> str:
        """Extract a primary entity name from a visual event."""
        vs = visual_event.get("payload", {}).get("vision_snapshot", {})
        objects = vs.get("objects", [])
        if objects:
            first_obj = objects[0]
            return first_obj.get("label") or first_obj.get(
                "class_name", "unknown"
            )
        return "unknown"

    # ------------------------------------------------------------------ #
    # Embedding similarity (lazy-loaded)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _compute_similarity(text_a: str, text_b: str) -> float:
        """Compute cosine similarity between two texts using Sentence-BERT.

        Falls back to Jaccard word overlap if Sentence-BERT unavailable.
        """
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("bge-small-zh-v1.5")
            import numpy as np
            embeds = model.encode([text_a, text_b], normalize_embeddings=True)
            similarity = float(np.dot(embeds[0], embeds[1]))
            return similarity
        except (ImportError, Exception):
            # Fallback: simple Jaccard word overlap
            set_a = set(text_a.split())
            set_b = set(text_b.split())
            if not set_a or not set_b:
                return 0.0
            intersection = len(set_a & set_b)
            union = len(set_a | set_b)
            return intersection / union if union > 0 else 0.0

    # ------------------------------------------------------------------ #
    # Reset
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        """Clear context tracking (e.g. on session restart)."""
        self._recent_primary_texts.clear()
