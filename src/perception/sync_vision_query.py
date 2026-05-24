"""
SyncVisionQuery — async-safe region-of-interest visual query with timeout and caching.

v4.5.0 §1.7: 按需视觉查询
  ThreadPoolExecutor for blocking GPU inference
  asyncio.wait_for with timeout
  TTL-based cache fallback on timeout (replaces single cached snapshot)
  VisionSnapshot metadata: stale, failed flags
  Callers MUST check metadata.stale and metadata.failed before trusting results

v4.5.0 v4.5.0 changes §10: 现在异步安全，含超时回退与缓存
"""

from __future__ import annotations

import asyncio
import copy
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from src.perception.visual.types import VisionSnapshot, VisionMetadata

logger = logging.getLogger(__name__)

DEFAULT_MAX_WAIT_MS = 10
DEFAULT_CACHE_TTL_MS = 500


class SyncVisionQuery:
    """
    Async-safe visual query wrapper for thread-pooled GPU inference.

    v4.5.0 §1.7:
      - Uses ThreadPoolExecutor(max_workers=1)
      - asyncio.wait_for with configurable timeout
      - TTL-based cache with stale fallback on timeout
      - Returns VisionSnapshot.empty(failed=True) on complete failure
      - Callers MUST check metadata.stale and metadata.failed

    v4.5.0 v4.5.0 changes §10: 现在异步安全，含超时回退与缓存
    """

    def __init__(
        self,
        max_wait_ms: int = DEFAULT_MAX_WAIT_MS,
        cache_ttl_ms: int = DEFAULT_CACHE_TTL_MS,
    ):
        """
        Args:
            max_wait_ms: Maximum wait time for inference in milliseconds.
                If exceeded, returns a stale cached snapshot or empty.
                Configure from RuntimeConfig or thresholds.yaml.
            cache_ttl_ms: Time-to-live for cached snapshots in milliseconds.
                After TTL expires, stale cache is treated as expired
                and query falls through to empty/failed.
        """
        self.max_wait_ms = max_wait_ms
        self.cache_ttl_ms = cache_ttl_ms
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._cached_snapshot: Optional[VisionSnapshot] = None
        self._cache_timestamp: float = 0.0
        self._lane = None  # Lazy-init in _sync_infer — OmniParser YOLOv11nLane

    @property
    def has_valid_cache(self) -> bool:
        """Check if the cache is non-empty and within TTL."""
        if self._cached_snapshot is None:
            return False
        if self.cache_ttl_ms <= 0:
            return self._cached_snapshot is not None
        elapsed_ms = (time.monotonic() - self._cache_timestamp) * 1000.0
        return elapsed_ms < self.cache_ttl_ms

    async def query_roi(
        self, x: int, y: int, width: int, height: int
    ) -> VisionSnapshot:
        """
        Query a region of interest with timeout and cache fallback.

        v4.5.0 §1.7: async def query_roi(self, x, y, width, height) -> VisionSnapshot

        Args:
            x: ROI left coordinate.
            y: ROI top coordinate.
            width: ROI width.
            height: ROI height.

        Returns:
            Fresh VisionSnapshot on success, stale cached copy on timeout,
            or empty snapshot on complete failure.
            Callers MUST check metadata.stale and metadata.failed.
        """
        try:
            loop = asyncio.get_event_loop()
            snapshot = await asyncio.wait_for(
                loop.run_in_executor(
                    self._executor, self._sync_infer, x, y, width, height
                ),
                timeout=self.max_wait_ms / 1000.0,
            )
            self._cached_snapshot = snapshot
            self._cache_timestamp = time.monotonic()
            snapshot.metadata.stale = False
            snapshot.metadata.failed = False
            return snapshot

        except asyncio.TimeoutError:
            if self._cached_snapshot is not None and self.has_valid_cache:
                snap = copy.copy(self._cached_snapshot)
                snap.metadata.stale = True
                snap.metadata.failed = False
                logger.debug(
                    "SyncVisionQuery timeout (%.1f ms) for ROI (%d,%d,%d,%d). "
                    "Returning TTL-valid stale cache. "
                    "(cache_age=%.0fms, ttl=%.0fms) (v4.5.0 §1.7)",
                    self.max_wait_ms, x, y, width, height,
                    (time.monotonic() - self._cache_timestamp) * 1000.0,
                    self.cache_ttl_ms,
                )
                return snap
            # v4.5.0 §1.7: return VisionSnapshot.empty(failed=True, reason="timeout and no cache")
            logger.warning(
                "SyncVisionQuery timeout and no valid cache available "
                "(cache_exists=%s, ttl_valid=%s). "
                "Returning empty snapshot. (v4.5.0 §1.7)",
                self._cached_snapshot is not None,
                self.has_valid_cache,
            )
            return VisionSnapshot.empty(
                failed=True, reason="timeout and no cache")

        except Exception:
            # Catch any unexpected inference failure — log at WARNING
            # with trace context per 项目宪法 §4.3.  Fall back to stale
            # cache if available, otherwise return empty.
            logger.warning(
                "SyncVisionQuery query_roi failed for ROI (%d,%d,%d,%d). "
                "(v4.5.0 §1.7)",
                x, y, width, height,
                exc_info=True,
            )
            if self._cached_snapshot is not None and self.has_valid_cache:
                snap = copy.copy(self._cached_snapshot)
                snap.metadata.stale = True
                return snap
            return VisionSnapshot.empty(failed=True, reason="inference error")

    def _sync_infer(
        self, x: int, y: int, width: int, height: int
    ) -> VisionSnapshot:
        """
        Synchronous inference in the thread pool.

        v4.5.0 §1.7: 线程池中执行的同步推理（使用独立推理上下文）
        v4.5.0 §1.7: ROI inference backed by OmniParser L2 icon-detect
        """
        # v4.5.0 §1.7: ROI inference backed by OmniParser L2 icon-detect
        try:
            # Lazy-import to avoid circular deps at module load
            from src.perception.visual.yolov11n import YOLOv11nLane
            from src.perception.visual.screenshot import capture_screenshot

            # Lazy-init OmniParser lane (reuses model across calls)
            if self._lane is None:
                self._lane = YOLOv11nLane()

            if not self._lane.available:
                return VisionSnapshot.empty(
                    failed=True, reason="OmniParser unavailable")

            # Step 1: Capture full screenshot
            frame = capture_screenshot()  # (H, W, 3) uint8 numpy array

            # Step 2: Crop to ROI, clamped to screen boundaries
            h, w = frame.shape[:2]
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(w, x + width)
            y2 = min(h, y + height)

            if x2 <= x1 or y2 <= y1:
                return VisionSnapshot.empty(
                    failed=True, reason="ROI out of screen bounds")

            roi_frame = frame[y1:y2, x1:x2]

            # Step 3: Run OmniParser L2 icon-detect on cropped region
            raw_results = self._lane._infer(roi_frame)

            # Step 4: Parse results and offset bboxes to screen coordinates
            ui_elements = self._lane._parse_results(
                raw_results, offset_x=float(x1), offset_y=float(y1))

            # Step 5: Build and return populated VisionSnapshot
            return VisionSnapshot(
                ui_elements=ui_elements,
                metadata=VisionMetadata(stale=False, failed=False),
            )

        except Exception as exc:
            # v4.5.0 §1.7: on any inference failure, return empty with
            # failed=True so callers can fall back to stale cache or
            # degrade gracefully.
            logger.warning(
                "SyncVisionQuery _sync_infer failed for ROI "
                "(%d,%d,%d,%d): %s (v4.5.0 §1.7)",
                x, y, width, height, exc,
                exc_info=True,
            )
            return VisionSnapshot.empty(failed=True, reason=str(exc))

    def cache(self, snapshot: VisionSnapshot) -> None:
        """Manually update the cache without running inference.

        Resets the TTL timer so the cached entry is fresh from this moment.
        """
        self._cached_snapshot = snapshot
        self._cache_timestamp = time.monotonic()

    def invalidate_cache(self) -> None:
        """Clear the cached snapshot and reset TTL timer."""
        self._cached_snapshot = None
        self._cache_timestamp = 0.0

    def shutdown(self) -> None:
        """Clean up the thread pool executor."""
        try:
            self._executor.shutdown(wait=False)
        except RuntimeError:
            # Executor already shut down — safe to ignore.
            pass
