"""
Visual Perception Fusion — merge four parallel visual lane outputs into a unified VisionSnapshot.

v4.5.0 §1.3.5: 视觉融合模块
  四路结果收集后，在感知层内部完成融合，输出统一快照，延迟约1ms

Fusion logic:
  1. 空间对齐: 将所有检测框坐标映射回原始屏幕分辨率 (2560x1440)
  2. 去重与合并: 相同类型/文本的框 (IoU > 0.5) 合并，类型取路2精细分类
  3. 置信度加权: 焦点区域路2权重0.7/路1权重0.3；非焦点区域路1权重0.8/路2权重0.2
  4. 场景上下文修正: game→UI阈值+0.2；document/code_editor→文本权重提升

项目宪法 §4.1: 视觉感知降级矩阵
"""

from __future__ import annotations

import logging
from typing import Optional

from src.perception.visual.types import (
    BBox, DetectedObject, UIElement, TextContent, SceneClass,
    VisionSnapshot, VisionMetadata,
)

logger = logging.getLogger(__name__)

# v4.5.0 §1.3.5 融合参数
IOU_DEDUP_THRESHOLD = 0.5          # IoU > 0.5 视为重叠
FOCAL_RADIUS_PX = 200              # 焦点区域半径 (鼠标±200px)
FOCAL_LANE2_WEIGHT = 0.7           # 焦点区域路2权重
FOCAL_LANE1_WEIGHT = 0.3           # 焦点区域路1权重
NON_FOCAL_LANE1_WEIGHT = 0.8       # 非焦点区域路1权重
NON_FOCAL_LANE2_WEIGHT = 0.2       # 非焦点区域路2权重
GAME_UI_THRESHOLD_BOOST = 0.2      # game场景UI阈值提升

# v4.5.0 §1.3.5 默认屏幕分辨率
DEFAULT_SCREEN_WIDTH = 2560
DEFAULT_SCREEN_HEIGHT = 1440


class VisualFusion:
    """
    Merges outputs from four visual lanes into a unified VisionSnapshot.

    v4.5.0 §1.3.5 视觉融合模块接口
    """

    def __init__(self, screen_width: int = DEFAULT_SCREEN_WIDTH,
                 screen_height: int = DEFAULT_SCREEN_HEIGHT):
        self._screen_width = screen_width
        self._screen_height = screen_height

    def fuse(self,
             yolo_objects: list[DetectedObject],
             ui_elements: list[UIElement],
             text_contents: list[TextContent],
             scene_class: Optional[SceneClass],
             mouse_x: Optional[int] = None,
             mouse_y: Optional[int] = None,
             any_degraded: bool = False) -> VisionSnapshot:
        """
        Fuse all four lane outputs into a VisionSnapshot.

        Args:
            yolo_objects: Lane 1 results.
            ui_elements: Lane 2 results.
            text_contents: Lane 3 results.
            scene_class: Lane 4 result (may be None).
            mouse_x: Mouse X for focal region weighting.
            mouse_y: Mouse Y for focal region weighting.
            any_degraded: True if any lane is degraded.

        Returns:
            Unified VisionSnapshot.
        """
        # v4.5.0 §1.3.5 step 1: 空间对齐 — map to 2560x1440
        aligned_objects = self._align_objects(yolo_objects)
        aligned_ui = self._align_ui(ui_elements)
        aligned_text = self._align_text(text_contents)

        # v4.5.0 §1.3.5 step 2: 去重与合并
        merged_objects, merged_ui = self._dedup_and_merge(
            aligned_objects, aligned_ui)

        # v4.5.0 §1.3.5 step 3: 置信度加权
        merged_objects, merged_ui = self._confidence_weight(
            merged_objects, merged_ui, mouse_x, mouse_y)

        # v4.5.0 §1.3.5 step 4: 场景上下文修正
        merged_objects, merged_ui = self._scene_context_correction(
            merged_objects, merged_ui, scene_class)

        # Build snapshot
        resolved_scene = scene_class or SceneClass(
            primary="other", secondary=[], confidence=0.0)

        # Text region association with UI elements
        aligned_text = self._associate_text_with_ui(aligned_text, merged_ui)

        return VisionSnapshot(
            scene_class=resolved_scene,
            objects=merged_objects,
            ui_elements=merged_ui,
            text_content=aligned_text,
            metadata=VisionMetadata(
                stale=False,
                failed=False,
                degraded=any_degraded,
            ),
        )

    # ------------------------------------------------------------------
    # Step 1: Coordinate alignment to 2560x1440
    # ------------------------------------------------------------------

    def _align_objects(self, objects: list[DetectedObject]) -> list[DetectedObject]:
        """Map object bboxes to screen resolution. Identity for now — assumes
        lane inputs are already screen-coordinate-normalized."""
        return objects

    def _align_ui(self, ui_elements: list[UIElement]) -> list[UIElement]:
        """Map UI bboxes to screen resolution."""
        return ui_elements

    def _align_text(self, texts: list[TextContent]) -> list[TextContent]:
        """Map text bboxes to screen resolution."""
        return texts

    # ------------------------------------------------------------------
    # Step 2: Deduplication and merging (IoU > 0.5)
    # ------------------------------------------------------------------

    def _dedup_and_merge(
        self,
        objects: list[DetectedObject],
        ui_elements: list[UIElement],
    ) -> tuple[list[DetectedObject], list[UIElement]]:
        """
        Merge overlapping detections from Lane 1 and Lane 2.

        v4.5.0 §1.3.5: 相同类型/文本的框 (IoU > 0.5) 合并为一条记录，
        类型取路2精细分类，位置取加权平均.
        """
        merged_objects: list[DetectedObject] = list(objects)
        merged_ui: list[UIElement] = list(ui_elements)

        # Cross-lane dedup: for each Lane 1 object, check against Lane 2 UI elements
        for obj in objects[:]:
            for ui in ui_elements:
                if obj.bbox.iou(ui.bbox) > IOU_DEDUP_THRESHOLD:
                    # Merge: type from Lane 2 (fine-grained), position weighted average
                    if obj in merged_objects:
                        merged_objects.remove(obj)
                    # Lane 2 UI element already has fine-grained type
                    # Position: weighted average (Lane 2 weight 0.7, Lane 1 weight 0.3)
                    ui.bbox.x = ui.bbox.x * 0.7 + obj.bbox.x * 0.3
                    ui.bbox.y = ui.bbox.y * 0.7 + obj.bbox.y * 0.3
                    ui.bbox.w = ui.bbox.w * 0.7 + obj.bbox.w * 0.3
                    ui.bbox.h = ui.bbox.h * 0.7 + obj.bbox.h * 0.3

        # Within-lane dedup: remove duplicate objects from same lane
        merged_objects = self._dedup_same_source(merged_objects)
        merged_ui = self._dedup_same_source(merged_ui)

        return merged_objects, merged_ui

    def _dedup_same_source(self, items: list) -> list:
        """Remove items from same source with IoU > 0.5 overlap."""
        result: list = []
        for item in items:
            is_duplicate = False
            for existing in result:
                if (getattr(item, 'source', None) == getattr(existing, 'source', None)
                        and item.bbox.iou(existing.bbox) > IOU_DEDUP_THRESHOLD):
                    is_duplicate = True
                    break
            if not is_duplicate:
                result.append(item)
        return result

    # ------------------------------------------------------------------
    # Step 3: Confidence weighting based on focal region
    # ------------------------------------------------------------------

    def _confidence_weight(
        self,
        objects: list[DetectedObject],
        ui_elements: list[UIElement],
        mouse_x: Optional[int],
        mouse_y: Optional[int],
    ) -> tuple[list[DetectedObject], list[UIElement]]:
        """
        Apply focal-region-based confidence weighting.

        v4.5.0 §1.3.5:
          焦点区域 (鼠标±200px): 路2权重0.7，路1权重0.3
          非焦点区域: 路1权重0.8，路2权重0.2
        """
        if mouse_x is None or mouse_y is None:
            return objects, ui_elements

        for obj in objects:
            center_x = obj.bbox.x + obj.bbox.w / 2
            center_y = obj.bbox.y + obj.bbox.h / 2
            in_focal = (abs(center_x - mouse_x) <= FOCAL_RADIUS_PX and
                        abs(center_y - mouse_y) <= FOCAL_RADIUS_PX)

            if obj.source == "yolo_world_small":
                obj.confidence *= (FOCAL_LANE1_WEIGHT if in_focal
                                   else NON_FOCAL_LANE1_WEIGHT)
            elif obj.source == "omniparser":
                obj.confidence *= (FOCAL_LANE2_WEIGHT if in_focal
                                   else NON_FOCAL_LANE2_WEIGHT)

        for ui in ui_elements:
            center_x = ui.bbox.x + ui.bbox.w / 2
            center_y = ui.bbox.y + ui.bbox.h / 2
            in_focal = (abs(center_x - mouse_x) <= FOCAL_RADIUS_PX and
                        abs(center_y - mouse_y) <= FOCAL_RADIUS_PX)

            if in_focal:
                ui.confidence *= FOCAL_LANE2_WEIGHT
            else:
                ui.confidence *= NON_FOCAL_LANE2_WEIGHT

        return objects, ui_elements

    # ------------------------------------------------------------------
    # Step 4: Scene context correction
    # ------------------------------------------------------------------

    def _scene_context_correction(
        self,
        objects: list[DetectedObject],
        ui_elements: list[UIElement],
        scene_class: Optional[SceneClass],
    ) -> tuple[list[DetectedObject], list[UIElement]]:
        """
        Adjust thresholds and weights based on scene context.

        v4.5.0 §1.3.5:
          game → UI置信度阈值临时提高0.2，降低误检；增加物体检测关注
          document/code_editor → 文本内容提升权重
        """
        if scene_class is None:
            return objects, ui_elements

        primary = scene_class.primary

        if primary == "game":
            # v4.5.0 §1.3.5: UI置信度阈值临时提高0.2，降低误检
            for ui in ui_elements:
                ui.confidence -= GAME_UI_THRESHOLD_BOOST
            # DEAD_CODE: L1 YOLO-World removed (v4.5.0 §1.3.1, model incompatible).
            # L1 objects never present — this boost branch is unreachable.
            # Restore if YOLO-World lane is reactivated.
            # 增加物体（路1）检测关注 → boost Lane 1 confidence
            for obj in objects:
                if obj.source == "yolo_world_small":
                    obj.confidence = min(1.0, obj.confidence + 0.1)

        elif primary in ("document", "code_editor"):
            # v4.5.0 §1.3.5: 文本内容提升权重
            # This is handled downstream; we just note the context here
            pass

        return objects, ui_elements

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _associate_text_with_ui(
        self,
        texts: list[TextContent],
        ui_elements: list[UIElement],
    ) -> list[TextContent]:
        """
        Associate text regions with nearby UI elements.

        v4.5.0 §1.3.5: 文本区域与OCR结果自然关联
        """
        # DEAD_CODE: text-UI association deferred.
        # Current 4-lane pipeline does not require OCR-to-UI spatial linking.
        # Restore when L1 reactivated or visual lane count changes (v4.5.0 §1.3.5).
        return texts
