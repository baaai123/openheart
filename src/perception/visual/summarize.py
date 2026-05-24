"""
Compact structured text summary of a VisionSnapshot for LLM prompt injection.

v4.5.0 §1.3.5: 将视觉快照压缩为 ~50-80 token 文本摘要，注入 DeepSeek 决策提示
"""

from src.perception.visual.types import VisionSnapshot


def summarize_for_llm(snapshot: VisionSnapshot) -> str:
    """
    Convert a VisionSnapshot into a compact structured summary (≤500 chars).

    Format::

        [视觉] 场景: <scene> | 物体: <labels> | 文字: "<snippets>"

    Lanes (high→low priority):
        1. Scene primary class (Lane 4 TinyCLIP-ViT)
        2. Top-3 objects by confidence (Lane 1 YOLOE)
        3. Top-3 OCR snippets by bbox area, truncated 15 chars (Lane 3 EasyOCR)

    Returns:
        - ``""``  — all-degraded (no scene, no objects, no text)
        - ``[视觉] 场景: <x> | (无显著内容)`` — scene exists but nothing else
        - ``[视觉] ...`` — normal structured summary

    Truncation (applied when >500 chars):
        1. Drop text section first
        2. Hard-truncate to 500 chars (last resort)

    v4.5.0 §1.3.5
    """
    scene = snapshot.scene_class
    objects = snapshot.objects
    texts = snapshot.text_content

    # --- All-degraded check: nothing at all → "" ---
    if scene is None and not objects and not texts:
        return ""

    # --- Empty-but-scene-exists guard ---
    if not objects and not texts and scene is not None:
        primary = scene.primary if scene.primary else "未知"
        return f"[视觉] 场景: {primary} | (无显著内容)"

    # --- Scene (Lane 4 TinyCLIP-ViT) ---
    primary = scene.primary if scene and scene.primary else "未知"
    parts = [f"场景: {primary}"]

    # --- Objects (Lane 1 YOLOE): top-3 by confidence descending ---
    if objects:
        top_labels = [o.label for o in sorted(objects, key=lambda o: o.confidence, reverse=True)[:3]]
        parts.append(f"物体: {', '.join(top_labels)}")

    # --- Text (Lane 3 EasyOCR): top-3 by bbox area descending, truncated 15 chars ---
    if texts:
        top_texts = sorted(texts, key=lambda t: t.bbox.area(), reverse=True)[:3]
        snippets = ['"{}"'.format(t.content[:15]) for t in top_texts]
        parts.append(f"文字: {' '.join(snippets)}")

    summary = f"[视觉] " + " | ".join(parts)

    # --- Truncation ---
    if len(summary) > 500:
        text_marker = " | 文字: "
        if text_marker in summary:
            summary = summary[: summary.index(text_marker)]
    if len(summary) > 500:
        summary = summary[:497] + "..."

    return summary
