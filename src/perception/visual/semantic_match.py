"""
Semantic matching for mouse action targets via bge-small-zh-v1.5 embeddings.

v4.5.0 §7.4.2: Replace substring matching with embedding-based semantic matching
for mouse action targets. LLM may emit imprecise target descriptions (e.g.,
"回收站图标") while OCR reads shorter variants ("回收站"). Embedding similarity
with a cosine threshold of 0.6 bridges this gap.

Lazy-loads the SentenceTransformer model on first use. Falls back to
bidirectional substring matching if the embedding model is unavailable.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-loaded embedding model (global singleton)
# ---------------------------------------------------------------------------

_model = None


def _get_model():
    """Lazily load bge-small-zh-v1.5 SentenceTransformer model.

    Returns the model instance or None if sentence-transformers is not
    installed or model loading fails.  The caller (find_best_match) handles
    the None case by falling through to substring matching.
    """
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415
            _model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
        except Exception:
            logger.warning(
                "semantic_match: sentence-transformers / bge-small-zh-v1.5 "
                "unavailable — falling back to substring matching"
            )
            _model = False  # Sentinel: tried and failed
    # _model may be False on failed load → return None
    return _model if _model is not False else None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def find_best_match(
    target: str,
    candidates: list[str],
) -> tuple[str, float] | None:
    """Find the best semantic match for *target* among *candidates*.

    Args:
        target: The action target string from the LLM (e.g. "回收站图标").
        candidates: Visible text/UI candidates from the visual snapshot.

    Returns:
        ``(best_candidate, similarity_score)`` if the best match exceeds the
        0.6 cosine-similarity threshold, otherwise ``None``.

        Substring fallback returns ``(candidate, 1.0)`` to signal a literal
        (non-embedding) match.

    Raises:
        Nothing — all exceptions are caught and fall through to substring.
    """
    # v4.5.0 §7.4.2: cosine threshold for semantic matching
    SEMANTIC_THRESHOLD: float = 0.6

    if not candidates or not target:
        return None

    # ── Primary path: embedding-based semantic matching ──
    try:
        model = _get_model()
        if model is not None:
            target_vec = model.encode([target])[0]
            cand_vecs = model.encode(candidates)
            # model.similarity returns a tensor of shape (1, N)
            similarities = model.similarity(target_vec, cand_vecs)[0]
            best_idx = similarities.argmax().item()
            best_score = similarities[best_idx].item()
            if best_score > SEMANTIC_THRESHOLD:
                logger.debug(
                    "semantic_match: '%s' → '%s' (score=%.3f)",
                    target, candidates[best_idx], best_score,
                )
                return (candidates[best_idx], best_score)
            logger.debug(
                "semantic_match: no candidate above threshold for '%s' "
                "(best=%.3f)",
                target, best_score,
            )
    except Exception:
        # Safe: embedding failures degrade to substring — no action blocked
        logger.warning(
            "semantic_match: embedding lookup failed for target='%s' — "
            "falling back to substring", target,
        )

    # ── Fallback: bidirectional substring matching ──
    for c in candidates:
        if target in c or c in target:
            logger.debug("semantic_match: substring fallback '%s' → '%s'", target, c)
            return (c, 1.0)

    # ── Last resort: character overlap (prefer shorter candidates)
    best_match = None
    best_score = 0.0
    for c in candidates:
        overlap = sum(1 for ch in target if ch in c)
        score = overlap / max(len(target), 1)
        if score >= 0.4 and score > best_score:
            best_score = score
            best_match = c
        elif score == best_score and best_match and len(c) < len(best_match):
            best_match = c  # prefer shorter candidate at equal score
    if best_match:
        logger.debug("semantic_match: char-overlap '%s' → '%s' (score=%.2f)", target, best_match, best_score)
        return (best_match, best_score)

    return None
