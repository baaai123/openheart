"""
VisualOrchestrator — top-level coordinator for the multi-lane visual pipeline.

Owns and manages the lifecycle of VisualPipeline, WindowAttentionPipeline,
QwenVLLane, FusionPipeline, SyncVisionQuery, and the background poller task.
This is the single entry point for runtime_loop to start/stop visual perception.

# v5.x §VisualOrchestrator
"""

from __future__ import annotations

import asyncio
import logging
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional

from src.insight.region_proposer import RegionProposer
from src.insight.concept_classifier import ConceptClassifier
from src.insight.prompt_memory import PromptMemory
from src.insight.prompt_learner import PromptLearner
from src.perception.visual.ocr_pipeline import OCRPipeline
from src.perception.visual.spatial_graph import SpatialGraphBuilder
from src.perception.visual.visual_frame import VisualFrameFormatter

if TYPE_CHECKING:
    from src.memory.cold.visual_store import VisualMemoryStore

from .snapshot_types import WindowAttentionSnapshot, WindowMeta, LLMContext
from .window_attention import WindowAttentionPipeline
from .screenshot import capture_screenshot
from .mouse_capture import get_mouse_position
from .qwen_vl_lane import QwenVLLane

from src.config.runtime import VRAMTier
from src.infra.tracing import TraceManager, sync_trace_span, trace_span
from src.memory.shared_context import SharedContext, NS_PERCEPTION

logger = logging.getLogger(__name__)


def _dict_to_window_meta(d: dict[str, Any]) -> WindowMeta:
    """Convert a raw window dict from process_frame() into typed WindowMeta.

    v5.x: dict→WindowMeta bridge for typed snapshot construction.
    """
    bounds = d.get("bounds")
    return WindowMeta(
        title=d.get("title", ""),
        app=d.get("app", ""),
        primary=d.get("primary", ""),
        tags=d.get("tags", []),
        bounds=bounds,
        left=d.get("left", 0.0),
        top=d.get("top", 0.0),
        z=d.get("z", 0),
        attention_score=d.get("attention_score", 0.0),
        change_score=d.get("change_score", 0.0),
        crop=d.get("crop"),
        ui=d.get("ui", []),
        text=d.get("text", []),
        vlm_description=d.get("vlm_description"),
    )


class VisualOrchestrator:
    """Coordinates all visual perception lanes and exposes a unified snapshot.

    Owns VisualPipeline (4-lane), WindowAttentionPipeline, QwenVLLane,
    FusionPipeline, SyncVisionQuery. Runs a background poller task that
    captures screenshots, runs window-level attention + VLM, and updates
    the latest typed WindowAttentionSnapshot.

    # v5.x §VisualOrchestrator — full implementation
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, config, log_callback=None) -> None:
        self._config = config
        self._log = log_callback or print

        # ── Thread pool for blocking GPU inference ──
        self._visual_pool = ThreadPoolExecutor(
            max_workers=3, thread_name_prefix="visual"
        )

        # ── WindowAttention pipeline (v5.x) ──
        self._window_pipeline = WindowAttentionPipeline()

        # ── Fusion pipeline (non-fatal) ──
        self._fusion = None
        try:
            from src.fusion.fusion_pipeline import FusionPipeline

            self._fusion = FusionPipeline()
            self._log("[融合] FusionPipeline 已初始化")
        except Exception as e:
            logger.warning("FusionPipeline init failed — non-fatal: %s", e)

        # ── SyncVisionQuery (non-fatal) ──
        self._sync_query = None
        try:
            from src.perception.sync_vision_query import SyncVisionQuery

            self._sync_query = SyncVisionQuery()
            self._log("[视觉查询] SyncVisionQuery 已初始化")
        except Exception as e:
            logger.warning("SyncVisionQuery init failed — non-fatal: %s", e)

        # ── Internal async state ──
        self._poller_task: Optional[asyncio.Task] = None
        self._latest: Optional[WindowAttentionSnapshot] = None
        self._stop_event = asyncio.Event()

        # ── Runtime dependencies (injected via set_* methods) ──
        self._execution = None   # ExecutionPipeline
        self._session = None     # SessionState

        # ── Poller mutable state ──
        self._last_mouse_x: Optional[int] = None
        self._last_mouse_y: Optional[int] = None
        self._visual_last_capture: float = 0.0
        self._l2d_description: str = ""
        self._last_window_title: str = ""  # v5.x: track window changes to invalidate stale VLM queue

        # ── Visual memory store (LanceDB, lazy-init) ──
        self._visual_store: Optional[VisualMemoryStore] = None
        self._visual_store_warned: bool = False  # v5.x: one-time warning on init failure

        # ── New visual pipeline (v5.x insight-memory-joint) ──
        self._region_proposer: Any = None
        self._concept_classifier: Any = None
        self._ocr_pipeline: Any = None
        self._prompt_memory: Any = None
        self._prompt_learner: Any = None
        self._spatial_builder = SpatialGraphBuilder()
        self._frame_formatter = VisualFrameFormatter()

        # ── Memory-layer wiring (v5.x: EntityGraph + RetrievalGate) ──
        self._entity_graph = None   # injected via set_entity_graph()
        self._retrieval_gate = None  # injected via set_retrieval_gate()

        try:
            self._region_proposer = RegionProposer()
            self._prompt_memory = PromptMemory()
            # v5.x: Lazy-load ConceptClassifier — only when PromptMemory has concepts (saves ~1.2GB at startup)
            self._concept_classifier = None
            self._ocr_pipeline = OCRPipeline()
            # v5.x: VLM lazy-loaded on first PromptLearner use (saves ~4GB VRAM at startup)
            self._qwen_vl: Any = None
            self._prompt_learner = PromptLearner(self._prompt_memory, vlm_lane=self._qwen_vl)
            self._log("[视觉v5] RegionProposer + OCR + PromptLearner ready (ConceptClassifier lazy)")
        except Exception as e:
            self._log(f"[视觉v5] Init degraded: {e}")

    # ------------------------------------------------------------------
    # Runtime dependency injection
    # ------------------------------------------------------------------

    def set_execution(self, execution) -> None:
        """Inject ExecutionPipeline reference for is_speaking() checks.

        v5.x: orchestrator needs this guard so VLM inference is skipped
        during TTS playback to avoid GPU contention.
        """
        self._execution = execution

    def set_session(self, session) -> None:
        """Inject SessionState reference for cached_visual_summary writes.

        v5.x: orchestrator writes assembled LLM context to session
        so downstream decision/heartbeat can pick it up.
        """
        self._session = session

    def set_entity_graph(self, graph) -> None:
        """Inject EntityGraph for spatial frame persistence (v5.x)."""
        self._entity_graph = graph

    def set_retrieval_gate(self, gate) -> None:
        """Inject RetrievalGate for tiered memory writes (v5.x)."""
        self._retrieval_gate = gate
        # Wire PromptMemory to RetrievalGate for long-term concept storage
        if self._prompt_memory:
            self._prompt_memory._gate = gate
        if self._prompt_memory:
            self._prompt_memory._gate = gate
            print("[记忆] PromptMemory → RetrievalGate wired for T3 cold storage", flush=True)

    # ------------------------------------------------------------------
    # Visual memory store (LanceDB, lazy-init)
    # ------------------------------------------------------------------

    async def _init_visual_store(self) -> None:
        """Lazy-initialize VisualMemoryStore for scene/window memory persistence.

        Loads only if lancedb is available. Logs WARNING once on failure.
        v5.x §10: visual memory persistence — non-fatal, best-effort.
        """
        if self._visual_store_warned:
            return
        self._visual_store_warned = True
        try:
            from src.memory.cold.visual_store import VisualMemoryStore

            self._visual_store = VisualMemoryStore()
            await self._visual_store.initialize()
            logger.info("VisualMemoryStore initialized for poll-loop writes")
        except ImportError:
            logger.warning(
                "lancedb not available — visual memory writes disabled"
            )
            self._visual_store = None
        except Exception as e:
            logger.warning(
                "VisualMemoryStore init failed — visual memory writes disabled: %s",
                e,
            )
            self._visual_store = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the background window-attention poller task.

        v5.x: creates asyncio.Task for _poller() and clears stop event.
        Safe to call multiple times (idempotent — no-op if already running).
        """
        if self._poller_task is not None and not self._poller_task.done():
            return
        # v5.x: Initialize YOLOE before poller starts if not already available
        if self._region_proposer and not self._region_proposer.available:
            self._log("[视觉] 视觉v5 管道不可用")
        self._stop_event.clear()
        self._poller_task = asyncio.create_task(self._poller())

    async def stop(self) -> None:
        """Gracefully stop the poller task and shut down the thread pool."""
        self._stop_event.set()
        if self._poller_task is not None:
            self._poller_task.cancel()
            try:
                await self._poller_task
            except asyncio.CancelledError:
                pass
            self._poller_task = None
        self._visual_pool.shutdown(wait=True)

    async def pause(self) -> None:
        """Pause poller without destroying thread pool (for heartbeat TTS)."""
        self._stop_event.set()
        if self._poller_task is not None:
            self._poller_task.cancel()
            try:
                await self._poller_task
            except asyncio.CancelledError:
                pass
            self._poller_task = None

    async def resume(self) -> None:
        """Resume poller after pause (thread pool still alive)."""
        self._stop_event.clear()
        self._poller_task = asyncio.create_task(self._poller())

    # ------------------------------------------------------------------
    # Background poller — migrated from runtime_loop.py:407-524
    # ------------------------------------------------------------------

    @trace_span(layer="perception", component="visual_orchestrator", operation="_poller")
    async def _poller(self) -> None:
        """Window-level attention polling loop (3s interval, background)."""
        _cycle_count = 0
        while not self._stop_event.is_set():
            await asyncio.sleep(3.0)  # v5.x: 3s window-attention cycle
            print(f"[POLLER] cycle #{_cycle_count} start", flush=True)
            _cycle_count += 1
            _t0 = time.monotonic()
            # [VRAM TRACE] Baseline at cycle start
            try:
                import torch as _vt0
                _vfree0, _vtotal0 = _vt0.cuda.mem_get_info()
                print(f"[VRAM][{_cycle_count}] CYCLE START: free={_vfree0/1e9:.2f}GB used={(_vtotal0-_vfree0)/1e9:.2f}GB", flush=True)
                del _vt0
            except Exception:
                pass

            # v5.x: Skip visual during TTS to avoid GPU contention
            if self._execution and self._execution.is_speaking():
                self._log(
                    f"[视觉] 轮次 #{_cycle_count} 跳过 (TTS活跃)",
                    flush=True,
                )
                continue

            # ── Full path (not rendering) ──
            _is_speaking = self._execution and self._execution.is_speaking()
            # v5.x: VRAM guard — skip visual if GPU memory critically low
            try:
                import torch
                torch.cuda.empty_cache()
                vram_free, _ = torch.cuda.mem_get_info()
                if vram_free < 0.2 * 1024**3:
                    logger.warning("[VRAM] Skipping visual cycle: %.1f GB free", vram_free/1e9)
                    await asyncio.sleep(1.0)
                    continue
            except Exception:
                pass
            self._log(
                f"[视觉] 轮次 #{_cycle_count} 全路径 (is_speaking={_is_speaking})",
                flush=True,
            )
            try:
                _t0 = time.monotonic()
                # ── Span: capture_screenshot + mouse ──
                async with TraceManager(
                    layer="perception", component="visual_orchestrator",
                    operation="capture_inputs",
                ):
                    frame = await asyncio.get_running_loop().run_in_executor(
                        self._visual_pool, capture_screenshot
                    )
                    raw_mouse = get_mouse_position()
                    self._last_mouse_x, self._last_mouse_y = (
                        raw_mouse if raw_mouse else (None, None)
                    )
                    mouse_xy: tuple[int, int] = raw_mouse if raw_mouse else (0, 0)

                # ── Span: process_frame ──
                async with TraceManager(
                    layer="perception", component="window_attention",
                    operation="process_frame",
                ):
                    result = await self._window_pipeline.process_frame(frame, mouse_xy)
                self._log(f"[PERF-VISUAL] windows: {time.monotonic()-_t0:.2f}s", flush=True)

                top_windows = result.get("top_windows", [])
                # v5.x: Exclude L2D avatar window
                top_windows = [tw for tw in top_windows if "l2d" not in str(tw.get("title", "")).lower()]

                l2d_crop = None  # v5.x: always defined for snapshot; unused when top_windows empty

                # ── v5.x Dual-track visual pipeline ──
                # Track 1: Analysis (synchronous, <500ms)
                _yoloe_concepts: list[dict] = []
                _yoloe_scene = None
                _yoloe_app_name = "unknown"
                _spatial_graph = None

                if self._region_proposer and self._region_proposer.available:
                    self._log("[视觉v5] Pipeline active — RegionProposer running", flush=True)
                    try:
                        # v5.x: Full window crop + mouse-distance sort
                        yoloe_input = top_windows[0].get("crop") if top_windows and top_windows[0].get("crop") is not None else frame
                        self._mouse_offset = (0, 0)  # no offset needed

                        # ── Span: RegionProposer.propose ──
                        async with TraceManager(
                            layer="perception", component="region_proposer",
                            operation="propose",
                        ):
                            import torch as _torch
                            _free_before, _total = _torch.cuda.mem_get_info()
                            print(f"[VRAM][{_cycle_count}] BEFORE RegionProposer.propose: free={_free_before/1e9:.2f}GB / total={_total/1e9:.2f}GB (used={(_total-_free_before)/1e9:.2f}GB)", flush=True)

                            bboxes = await asyncio.get_running_loop().run_in_executor(
                                self._visual_pool, self._region_proposer.propose, yoloe_input
                            )

                        # [VRAM TRACE] After RegionProposer.propose
                        _free_after, _ = _torch.cuda.mem_get_info()
                        print(f"[VRAM][{_cycle_count}] AFTER RegionProposer.propose: free={_free_after/1e9:.2f}GB (delta={(_free_before-_free_after)/1e9:.3f}GB)", flush=True)
                        del _torch
                        context_tags = self._get_context_tags(top_windows)
                        # v5.x: Cold start — label bboxes as "unknown" until VLM learns concepts
                        if self._concept_classifier is None:
                            if self._prompt_memory and self._prompt_memory.list_concepts():
                                # [VRAM TRACE] Before ConceptClassifier creation (lazy-load)
                                import torch as _torch2
                                _free_b4, _total2 = _torch2.cuda.mem_get_info()
                                print(f"[VRAM][{_cycle_count}] BEFORE ConceptClassifier.__init__: free={_free_b4/1e9:.2f}GB (used={(_total2-_free_b4)/1e9:.2f}GB)", flush=True)
                                del _torch2

                                self._concept_classifier = ConceptClassifier(self._prompt_memory)
                                self._prompt_learner.set_compute_vpe(self._concept_classifier.compute_vpe)

                                # [VRAM TRACE] After ConceptClassifier creation
                                import torch as _torch2b
                                _free_af, _total2b = _torch2b.cuda.mem_get_info()
                                print(f"[VRAM][{_cycle_count}] AFTER ConceptClassifier.__init__: free={_free_af/1e9:.2f}GB (delta={(_free_b4-_free_af)/1e9:.3f}GB, used={(_total2b-_free_af)/1e9:.2f}GB)", flush=True)
                                del _torch2b

                                self._log("[视觉v5] ConceptClassifier lazy-loaded (PromptMemory has concepts)", flush=True)
                            else:
                                # Cold start: create "unknown" concepts from bboxes
                                _yoloe_concepts = [
                                    {"name": "unknown", "bbox": b, "confidence": 0.1, "source": "yoloe-coldstart"}
                                    for b in bboxes[:50]
                                ]
                                self._log(f"[视觉v5] Cold start: {len(_yoloe_concepts)} bboxes → 'unknown' concepts", flush=True)
                        if self._concept_classifier is not None:
                            # ── Span: ConceptClassifier.classify_with_bboxes ──
                            async with TraceManager(
                                layer="perception", component="concept_classifier",
                                operation="classify_with_bboxes",
                            ):
                                import torch as _torch3
                                _free_b4_cc, _total3 = _torch3.cuda.mem_get_info()
                                print(f"[VRAM][{_cycle_count}] BEFORE ConceptClassifier.classify_with_bboxes: free={_free_b4_cc/1e9:.2f}GB", flush=True)
                                del _torch3

                                _yoloe_concepts = await asyncio.get_running_loop().run_in_executor(
                                    self._visual_pool,
                                    lambda: self._concept_classifier.classify_with_bboxes(yoloe_input, bboxes, context_tags)
                                ) if self._concept_classifier.available else []

                            # [VRAM TRACE] After ConceptClassifier.classify_with_bboxes
                            import torch as _torch3b
                            _free_af_cc, _ = _torch3b.cuda.mem_get_info()
                            print(f"[VRAM][{_cycle_count}] AFTER ConceptClassifier.classify_with_bboxes: free={_free_af_cc/1e9:.2f}GB (delta={(_free_b4_cc-_free_af_cc)/1e9:.3f}GB)", flush=True)
                            del _torch3b
                        # v5.x: Sort concepts by mouse distance, keep nearest N
                        _mx, _my = mouse_xy
                        def _dist(c):
                            b = c.get("bbox")
                            if b and len(b)==4:
                                cx = (b[0]+b[2])/2; cy = (b[1]+b[3])/2
                                return ((cx-_mx)**2 + (cy-_my)**2)**0.5
                            return 9999
                        _yoloe_concepts.sort(key=_dist)
                        # v5.x: no cap — process all detected concepts
                        _yoloe_app_name = str(context_tags[1] if len(context_tags) > 1 else context_tags[0])[:15]
                        # v5.x: Offset bboxes from window-local → full-frame BEFORE VLM cropping
                        if yoloe_input is not frame and top_windows:
                            win_left = int(top_windows[0].get("left", 0))
                            win_top = int(top_windows[0].get("top", 0))
                            for _c in _yoloe_concepts:
                                _b = _c.get("bbox")
                                if _b and len(_b) == 4:
                                    _c["bbox"] = [_b[0]+win_left, _b[1]+win_top, _b[2]+win_left, _b[3]+win_top]
                        # Collect icon_labels from classified concepts for window context
                        _icon_labels = []
                        for _c in _yoloe_concepts:
                            _name = _c.get("name", "")
                            if _name and _name != "unknown":
                                _icon_labels.append(_name)
                        _icon_labels = list(dict.fromkeys(_icon_labels))  # v5.x: no cap — use all unique labels
                        if top_windows:
                            top_windows[0]["icon_labels"] = list(_icon_labels)

                        # ── Span: OCRPipeline.scan_full ──
                        async with TraceManager(
                            layer="perception", component="ocr_pipeline",
                            operation="scan_full",
                        ):
                            _ocr_results = await asyncio.get_running_loop().run_in_executor(
                                self._visual_pool,
                                self._ocr_pipeline.scan_full, yoloe_input
                            ) if self._ocr_pipeline and self._ocr_pipeline.available else []

                        # ── Span: post-pipeline (build + entity_graph + VisualFrame + write + summary) ──
                        async with TraceManager(
                            layer="perception", component="visual_orchestrator",
                            operation="post_pipeline",
                        ):
                            if _yoloe_concepts:
                                from src.perception.visual.snapshot_types import VisualConcept
                                concepts = [
                                    VisualConcept(name=c["name"], confidence=c["confidence"], source=c.get("source", ""))
                                    for c in _yoloe_concepts
                                ]
                                _spatial_graph = self._spatial_builder.build(
                                    concepts, (yoloe_input.shape[1], yoloe_input.shape[0])
                                )

                                # v5.x: Write spatial frame to EntityGraph
                                if hasattr(self, '_entity_graph') and self._entity_graph and _spatial_graph:
                                    try:
                                        self._entity_graph.add_spatial_frame(_spatial_graph, time.time())
                                    except Exception:
                                        pass

                            from src.perception.visual.snapshot_types import VisualFrame
                            _frame = VisualFrame(
                                concepts=[VisualConcept(name=c["name"], confidence=c["confidence"], source=c.get("source", "")) for c in _yoloe_concepts],
                                ocr_texts=_ocr_results,
                                spatial_graph=_spatial_graph,
                                window_title=str(top_windows[0].get("title", "")) if top_windows else "",
                                app_name=str(top_windows[0].get("app", "")) if top_windows else "",
                                degraded=False,
                                timestamp=time.time(),
                            )

                            # v5.x: Write VisualFrame to RetrievalGate (T3 Cold)
                            if hasattr(self, '_retrieval_gate') and self._retrieval_gate:
                                try:
                                    from src.memory.visual_frame_integration import frame_to_tiered_record
                                    record = frame_to_tiered_record(_frame, f"vf_{int(time.time()*1000)}")
                                    self._retrieval_gate.write_record(record)
                                except Exception:
                                    pass

                            tier1 = self._frame_formatter.to_tier1_summary(_frame)
                            tier2 = self._frame_formatter.to_tier2_context(_frame)
                            SharedContext.get_instance().set(
                                NS_PERCEPTION, "spatial_context", tier2,
                            )
                        _yoloe_scene = tier1
                    except Exception as e:
                        logger.warning(f"v5.x visual pipeline: {e}")

                # v5.x: Window change detection — clear stale VLM queue when context shifts
                if top_windows and self._prompt_learner:
                    current_title = str(top_windows[0].get("title", ""))
                    if current_title and current_title != self._last_window_title:
                        old_title = self._last_window_title or "(first frame)"
                        self._last_window_title = current_title
                        self._prompt_learner.clear_queue()
                        self._log(f"[视觉v5] Window change: '{old_title}' → '{current_title[:40]}', cleared VLM queue", flush=True)

                # Track 2: Learning (async) — low-conf concepts → VLM queue
                if self._prompt_learner and _yoloe_concepts:
                    try:
                        print(f"[LEARN-DIAG] _yoloe_concepts={len(_yoloe_concepts)} items, checking low conf...", flush=True)
                        low_conf = [
                            c for c in _yoloe_concepts
                            if c.get("confidence", 0) < 0.5
                        ]
                        if low_conf and top_windows:
                            tags = self._get_context_tags(top_windows)
                            import numpy as np
                            for c in low_conf[:3]:
                                bbox = c.get("bbox")
                                # v5.x: bboxes are in yoloe_input (crop) space → offset to full-frame space before cropping from `frame`
                                if bbox and len(bbox) == 4 and isinstance(frame, np.ndarray):
                                    x1, y1, x2, y2 = [int(v) for v in bbox]
                                    # v5.x: If yoloe_input is a window crop, add window position offset to get full-frame coords
                                    if yoloe_input is not frame and top_windows:
                                        win_left = int(top_windows[0].get("left", 0))
                                        win_top = int(top_windows[0].get("top", 0))
                                        x1 += win_left; y1 += win_top
                                        x2 += win_left; y2 += win_top
                                    h_img, w_img = frame.shape[:2]
                                    # Verify bbox coords are within frame bounds; clamp safely
                                    x1 = max(0, min(w_img, x1)); y1 = max(0, min(h_img, y1))
                                    x2 = max(0, min(w_img, x2)); y2 = max(0, min(h_img, y2))
                                    if x2 > x1 + 10 and y2 > y1 + 10:
                                        crop_img = frame[y1:y2, x1:x2].copy()
                                        self._prompt_learner.enqueue(
                                            [(crop_img, c.get("name", "unknown"), c.get("confidence", 0.2))],
                                            tags
                                        )
                            # Fire async learning (non-blocking)
                            if self._prompt_learner.queue_size > 0:
                                asyncio.create_task(self._prompt_learner.learn_from_queue())
                                self._log(f"[VLM-LEARN] queued {len(low_conf[:3])} low-conf concepts (queue={self._prompt_learner.queue_size})")
                    except Exception as e:
                        logger.warning(f"PromptLearner enqueue failed: {e}")

                if top_windows:
                    top1 = top_windows[0]

                    self._log("[视觉] ctx-assembly...", flush=True)
                    # ── Span: assemble_llm_context ──
                    async with TraceManager(
                        layer="perception", component="window_attention",
                        operation="assemble_llm_context",
                    ):
                        ctx = self._window_pipeline.assemble_llm_context(top1)
                    ctx_text = str(ctx.get("text", ""))
                    ctx_vlm = str(ctx.get("vlm_description", ""))
                    # v5.x: Build summary from YOLOE concepts (primary) or window_attention (fallback)
                    if _yoloe_concepts:
                        concept_names = [c["name"] for c in _yoloe_concepts]
                        unique_names = list(dict.fromkeys(concept_names))
                        print(f"[DIAG-YOLOE] names={unique_names} n={len(_yoloe_concepts)}", flush=True)
                        if len(unique_names) == 1 and unique_names[0] == "unknown":
                            yolo_scene = f"检测到{len(_yoloe_concepts)}个区域, VLM学习中"
                        else:
                            yolo_scene = ", ".join(unique_names)[:80]  # v5.x: no cap — use all unique names, extended width
                        # v5.x: Add OCR texts to context — audit P1: [:6]×[:40] (was [:3]×[:20])
                        ocr_texts = [t.text for t in _ocr_results if t.text]  # v5.x audit P2: no limit on count or char length; scanning window crop so terminal-noise filter not needed
                        logger.debug("OCR raw detections: %d, passed filter: %d", len(_ocr_results), len(ocr_texts))
                        ocr_suffix = f" | OCR: {'; '.join(ocr_texts)}" if ocr_texts else ""
                        cached = f"[app: {_yoloe_app_name}] [视觉: {yolo_scene}{ocr_suffix}]"
                        # v5.x: Free CUDA cache after YOLOE to prevent VRAM leak
                        try:
                            import torch
                            torch.cuda.empty_cache()
                        except: pass
                    else:
                        _app_label = str(top1.get("app", top1.get("title", "")))[:15]
                        _scene_label = str(top1.get("primary", ""))[:15]
                        cached = f"[app: {_app_label}] [scene: {_scene_label}]"
                    # ── Span: cached_visual_summary + SharedContext.set ──
                    async with TraceManager(
                        layer="perception", component="visual_orchestrator",
                        operation="write_summary",
                    ):
                        if self._session:
                            self._session.cached_visual_summary = cached

                        SharedContext.get_instance().set(
                            NS_PERCEPTION, "visual_summary", cached,
                        )
                    # v5.x: Show what LLM actually gets
                    self._log(f"[视觉→LLM] cached_visual_summary = {cached}", flush=True)
                    # v5.x: L2D description injected via comfort_injection (system prompt)
                    # — handled in runtime_loop via _l2d_description attribute
                    # v5.x: Log cycle result — always show VLM + scene status
                    vlm_status = (
                        "VLM:cached"
                        if top1.get("vlm_description")
                        else "VLM:skipped"
                        if _is_speaking
                        else "VLM:ok"
                    )
                    self._log(
                        f"[视觉] {len(result.get('windows', []))}窗, top-1={top1.get('title', '?')[:30]} (场景={top1.get('primary', '?')[:20]}, 面积={top1.get('bounds').w * top1.get('bounds').h if top1.get('bounds') else 0}, 分数={top1.get('attention_score', 0.0):.2f}, {vlm_status})",
                        flush=True,
                    )
                    _parts: list[str] = []
                    _parts.append(f"[视觉输出] 共{len(top_windows)}窗")
                    for i, tw in enumerate(top_windows[:3]):
                        _v = "VLM" if tw.get("vlm_description") else "-"
                        _b = tw.get("bounds")
                        _pos_info = (
                            f"@({int(_b.x)},{int(_b.y)})" if _b else ""
                        )
                        _parts.append(
                            f"  #{i + 1} [{tw.get('title', '?')[:25]}] "
                            f"场景={tw.get('primary', '?')[:15]} "
                            f"面积={tw.get('bounds').w * tw.get('bounds').h if tw.get('bounds') else 0} "
                            f"分数={tw.get('attention_score', 0):.2f} "
                            f"变化={tw.get('change_score', 0):.1f} "
                            f"{_pos_info} "
                            f"{_v}"
                        )
                    self._log("\n".join(_parts), flush=True)
                    # v5.x: Show assembled LLM context (what LLM actually receives)
                    _ctx_parts: list[str] = []
                    if ctx.get("text"):
                        _ctx_parts.append(f"  text: {ctx['text'][:80]}")
                    if ctx.get("scene"):
                        _ctx_parts.append(f"  scene: {ctx['scene']}")
                    if ctx.get("position"):
                        _ctx_parts.append(f"  position: {ctx['position']}")
                    if ctx.get("vlm_description"):
                        _ctx_parts.append(
                            f"  vlm: {ctx['vlm_description']}"
                        )
                    if ctx.get("matched"):
                        for m in ctx["matched"]:
                            _ctx_parts.append(f"  match: {m}")
                    if _ctx_parts:
                        _vlm_note = (
                            "VLM:empty"
                            if not ctx.get("vlm_description")
                            else "VLM:present"
                        )
                        _ctx_str = "\n".join(_ctx_parts)
                        self._log(
                            f"[视觉→LLM 上下文] {_vlm_note}\n{_ctx_str}",
                            flush=True,
                        )
                    # ── Build typed snapshot ──
                    top_metas = [_dict_to_window_meta(tw) for tw in top_windows]
                    top_meta = top_metas[0] if top_metas else None
                    self._latest = WindowAttentionSnapshot(
                        top_window=top_meta,
                        top_windows=top_metas,
                        l2d_crop=l2d_crop,
                        scene_label=top1.get("primary", ""),
                        llm_context=LLMContext(
                            text=ctx_text,
                            scene=str(ctx.get("scene", "")),
                            position=str(ctx.get("position", "")),
                            vlm_description=ctx_vlm,
                            matched=ctx.get("matched", []) if isinstance(ctx.get("matched"), list) else [],
                        ),
                        heartbeat_context=ctx_text + "\n" + ctx_vlm,
                        timestamp=time.time(),
                        cycle=_cycle_count,
                    )

                    # ── Span: visual store write (LanceDB) ──
                    async with TraceManager(
                        layer="perception", component="visual_orchestrator",
                        operation="visual_store_write",
                    ):
                        try:
                            if self._visual_store is None and not self._visual_store_warned:
                                await self._init_visual_store()
                            if self._visual_store is not None:
                                _now = datetime.now(timezone.utc)
                                _ts = _now.isoformat()
                                from src.memory.cold.visual_schema import VisualMemoryRecord

                                records = [
                                    VisualMemoryRecord(
                                        memory_id="",
                                        timestamp=_ts,
                                        memory_type="scene",
                                        content_text=str(top1.get("primary", "")),
                                        source_window=str(top1.get("title", "")),
                                        tags=[],
                                        embedding=[],
                                        tier="cold",
                                    ),
                                    VisualMemoryRecord(
                                        memory_id="",
                                        timestamp=_ts,
                                        memory_type="window",
                                        content_text=str(top1.get("title", "")),
                                        source_window=str(top1.get("title", "")),
                                        tags=[],
                                        embedding=[],
                                        tier="cold",
                                    ),
                                ]
                                await self._visual_store.insert_batch(records)
                                SharedContext.get_instance().set(
                                    NS_PERCEPTION, "last_visual_write_ts", _ts,
                                )
                                self._log("[视觉记忆] scene+window written", flush=True)
                        except Exception as e:
                            logger.warning(
                                "Visual memory write failed (non-fatal): %s", e,
                            )
                else:
                    # No windows detected — clear snapshot
                    self._latest = WindowAttentionSnapshot(
                        timestamp=time.time(),
                        cycle=_cycle_count,
                    )

                self._visual_last_capture = time.monotonic()
                self._log("[视觉] snapshot built", flush=True)
                self._log(f"[POLL-{_cycle_count}] CYCLE OK", flush=True)
            except asyncio.CancelledError:
                self._log(
                    "Window attention poller cancelled (shutdown).", flush=True
                )
                break
            except Exception as e:
                self._log(
                    f"Window attention capture failed: {e}\n{traceback.format_exc()}",
                    flush=True,
                )
                self._log(f"[POLL-{_cycle_count}] CYCLE FAIL: {e}", flush=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_context_tags(self, top_windows: list[dict]) -> list[str]:
        if not top_windows:
            return ["unknown"]
        tw = top_windows[0]
        tags = [
            str(tw.get("title", "")),
            str(tw.get("app", "")),
            str(tw.get("primary", "other")),
        ]
        return [t for t in tags if t]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        return self._region_proposer is not None and self._region_proposer.available

    @property
    def latest(self) -> Optional[WindowAttentionSnapshot]:
        """Latest typed snapshot from the background poller, or None if no cycle yet."""
        return self._latest
