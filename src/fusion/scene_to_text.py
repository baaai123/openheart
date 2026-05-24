"""Convert fused scene data to natural Chinese text for LLM prompt injection.

v4.5.0 §2.7: 将融合后的 Scene 结果压缩为 ~80-120 token 中文描述，
注入 DeepSeek 决策提示，使模型感知当前多模态场景上下文。

Accepts:
    - FusionResult (from fusion_pipeline)
    - dict scene envelope (full §0.3 message envelope)
    - dict payload (the "payload" sub-dict of the envelope)
    - ScenePayload dataclass instance
"""

from __future__ import annotations

import logging
from typing import Any

from src.fusion.scene_synthesis import ScenePayload

logger = logging.getLogger(__name__)


def scene_to_text(scene_result: Any) -> str:
    """Convert fusion Scene result to natural Chinese text for LLM prompt.

    Parameters
    ----------
    scene_result:
        One of:
        - ``FusionResult`` (from ``fusion_pipeline.py``)
        - ``dict`` scene envelope (full §0.3 message envelope with ``"payload"`` key)
        - ``dict`` payload (the ``"payload"`` sub-dict)
        - ``ScenePayload`` dataclass instance
        - ``None``

    Returns
    -------
    str:
        Concise Chinese sentence describing the multi-modal scene.
        Returns ``""`` when the input is empty, ``None``, or fully degraded.

    Examples
    --------
    >>> scene_to_text(None)
    ''
    >>> scene_to_text({"payload": {"summary": "用户在使用手机", "primary_event": "浏览网页", ...}})
    '用户在使用手机，用户正在: 浏览网页。'
    """
    if scene_result is None:
        return ""

    # --- Unwrap FusionResult → dict scene envelope ---
    if _is_fusion_result(scene_result):
        if scene_result.degraded and scene_result.scene is None:
            return ""
        payload = _extract_payload_from_envelope(scene_result.scene)
        aligned = scene_result.aligned_entities
    else:
        payload = _resolve_to_payload(scene_result)
        aligned = None

    if payload is None:
        return ""

    # --- Normalise to dict for uniform access ---
    payload_dict = _payload_to_dict(payload)

    parts: list[str] = []

    # 1. Summary (top-line scene description, §2.6)
    summary = payload_dict.get("summary", "")
    if summary:
        parts.append(summary)

    # 2. Primary event
    primary = payload_dict.get("primary_event", "")
    if primary:
        parts.append(f"用户正在: {primary}")

    # 3. Secondary events (up to 2)
    secondary = payload_dict.get("secondary_events", [])
    if secondary:
        for ev in secondary[:2]:
            parts.append(f"同时: {ev}")

    # 4. Entity relations (subject-predicate-object triples, §2.6)
    relations = payload_dict.get("entity_relations", [])
    if relations:
        for r in relations[:2]:
            desc = _relation_description(r)
            if desc:
                parts.append(desc)

    # 5. Aligned entities (cross-modal, up to 2)
    aligned_payload = aligned or payload_dict.get("aligned_entities", [])
    if aligned_payload:
        for ae in aligned_payload[:2]:
            desc = _aligned_description(ae)
            if desc:
                parts.append(desc)

    # 6. Emotion snapshot
    emotion = payload_dict.get("emotion_snapshot", {})
    if emotion and isinstance(emotion, dict):
        cat = emotion.get("category", "")
        intensity = emotion.get("intensity", 0.0)
        if cat and cat != "neutral" and intensity > 0.3:
            emoji_map = {"joy": "愉悦", "sadness": "低落", "anger": "生气", "surprise": "惊讶"}
            label = emoji_map.get(cat, cat)
            parts.append(f"情绪{label}({intensity:.1f})")

    if not parts:
        return ""

    return "，".join(parts) + "。"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_fusion_result(obj: Any) -> bool:
    return hasattr(obj, "scene") and hasattr(obj, "degraded") and hasattr(obj, "aligned_entities")


def _extract_payload_from_envelope(envelope: Any) -> dict[str, Any] | None:
    if envelope is None:
        return None
    if isinstance(envelope, dict):
        payload = envelope.get("payload")
        if isinstance(payload, dict):
            return payload
        # Maybe the envelope IS the payload
        if "summary" in envelope or "primary_event" in envelope:
            return envelope
    return None


def _resolve_to_payload(obj: Any) -> dict[str, Any] | ScenePayload | None:
    if obj is None:
        return None
    if isinstance(obj, ScenePayload):
        return obj
    if isinstance(obj, dict):
        if "payload" in obj and isinstance(obj["payload"], dict):
            return obj["payload"]
        if "summary" in obj or "primary_event" in obj:
            return obj
    return None


def _payload_to_dict(payload: Any) -> dict[str, Any]:
    if isinstance(payload, ScenePayload):
        return {
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
        }
    if isinstance(payload, dict):
        return payload
    return {}


def _relation_description(r: Any) -> str:
    if isinstance(r, dict):
        sub = r.get("subject", "")
        pred = r.get("predicate", "")
        obj = r.get("object", "")
        if sub and pred and obj:
            return f"{sub}{pred}{obj}"
        if sub and obj:
            return f"{sub}相关{obj}"
    return ""


def _aligned_description(ae: Any) -> str:
    if isinstance(ae, dict):
        desc = ae.get("description") or ae.get("label") or ae.get("name") or ae.get("text")
        if not desc:
            audio_desc = ae.get("audio", "")
            visual_desc = ae.get("visual", "")
            if audio_desc and visual_desc:
                return f"音频「{audio_desc}」对应画面「{visual_desc}」"
        if desc:
            return str(desc)
    return ""
