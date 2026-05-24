"""
Window attention pipeline — Steps 1-4 of the new visual frontend.

v4.5.0 §T2: Window-level attention replaces full-screen multi-lane pipeline.
  Step 1: Enumerate windows via PowerShell
  Step 2: Z-order crop + L2D detection
  Step 3: Spatial attention scoring (mouse proximity, size, Z-order)
  Step 4: Temporal change scoring + top-window selection

All scoring is self-contained within WindowAttentionPipeline — no external
scene classification models. Scene tags default to "other".
"""

from __future__ import annotations

import logging
import time
from typing import Any, cast

import numpy as np

from src.perception.visual.types import BBox
from src.perception.visual.window_enum import get_window_hierarchy

logger = logging.getLogger(__name__)

TOP_WINDOW_COUNT = 5


# ══════════════════════════════════════════════════════════════════════
# ASR dedup helpers — v5.x §T2
# ══════════════════════════════════════════════════════════════════════

def _levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein distance between two strings.

    v5.x §T2: Used to detect ASR text duplication in visual context.
    Safe: no external dependencies, pure Python, O(n*m) with n ≤ 30.
    """
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    if len(s2) == 0:
        return len(s1)
    # Reason: only called for short strings (≤30 chars) in _text_matches_asr
    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            curr_row.append(min(
                curr_row[j] + 1,          # insertion
                prev_row[j + 1] + 1,      # deletion
                prev_row[j] + (c1 != c2), # substitution
            ))
        prev_row = curr_row
    return prev_row[-1]


def _text_matches_asr(text: str, asr_text: str | None) -> bool:
    """Check if *text* duplicates ASR user input.

    Uses substring matching first (O(n)), then edit distance ≤ 3 for
    short strings (O(n*m), n,m ≤ 30).

    v5.x §T2: Dedup check to strip ASR text from OCR/VLM context.
    """
    if not asr_text or not text:
        return False
    text_lower = text.lower().strip()
    asr_lower = asr_text.lower().strip()

    # Fast path: substring containment
    if text_lower in asr_lower or asr_lower in text_lower:
        return True

    # Slower path: Levenshtein ≤ 3 (only for short texts)
    if len(text_lower) <= 30 and len(asr_lower) <= 30:
        if _levenshtein_distance(text_lower, asr_lower) <= 3:
            return True

    return False


# ══════════════════════════════════════════════════════════════════════
# Embedding model (lazy-loaded singleton) — v5.x Step 4b
# ══════════════════════════════════════════════════════════════════════

_embedding_model: Any = None


def _get_embedding_model() -> Any:
    """Lazy-load bge-small-zh-v1.5 SentenceTransformer for semantic matching.

    v5.x §T2 Step 4b: Returns model or None on failure.  Caller falls back
    to substring matching when model is unavailable.
    """
    global _embedding_model
    if _embedding_model is None:
        try:
            # Reason: sentence-transformers is optional; import deferred
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415
            _embedding_model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
        except Exception:
            # Safe: embedding failures degrade to substring — no action blocked
            logger.warning(
                "semantic_match: sentence-transformers / bge-small-zh-v1.5 "
                "unavailable — falling back to substring matching"
            )
            _embedding_model = False  # Sentinel: tried and failed
    return _embedding_model if _embedding_model is not False else None


# v5.x: Filter system utility windows
_MIN_WINDOW_AREA = 200 * 200  # ~1% of 1920x1080
_SKIP_TITLES = ["windows 输入体验", "任务栏",
                "开始", "操作中心", "通知", "msctf", "textinputhost"]

# v5.x FIX: Hidden/offscreen/system windows — negative coords or system titles
_HIDDEN_TITLE_BLACKLIST = ["vcxsrv", "hcontrol", "qt"]

# Task switcher (Alt+Tab) window filter — these should never compete
# for top attention. Matched against title and class_name.
_VBA_FILTER_TITLES = ["microsoft visual basic"]
_TASK_SWITCHER_TITLES = ["任务切换"]
_TASK_SWITCHER_CLASSES = ["xamlexplorer", "hostisland", "hcontrol", "vcxsrv"]

# v5.x: System tray / notification area window patterns
_TRAY_TITLES = ["shell_traywnd", "traynotify", "notification"]

# L2D window detection — title/class_name patterns (not z-order)
_L2D_TITLES = ["openheart-l2d"]
_L2D_CLASSES = ["glfw30", "live2d"]


class WindowAttentionPipeline:
    """Steps 1-4 of the new visual frontend.

    Window enumeration, cropping, spatial attention, and temporal change
    tracking. Does NOT import L2/L3/VLM at init — those are external
    dependencies consumed by downstream modules.

    v4.5.0 §T2: WindowAttentionPipeline interface
    """

    def __init__(self) -> None:
        self._prev_windows: list[dict[str, Any]] = []
        self._vlm_cache: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # 主入口 — v4.5.0 §T2 Step 1-4
    # ------------------------------------------------------------------

    async def process_frame(
        self,
        screenshot: np.ndarray,
        mouse_xy: tuple[int, int],
        asr_text: str | None = None
    ) -> dict[str, Any]:
        """Process a single screenshot frame through Steps 1-4.

        Args:
            screenshot: numpy array (H, W, 3) — full screen capture.
            mouse_xy: (x, y) mouse cursor position in screen coordinates.
            asr_text: Optional ASR transcript (reserved for future cross-modal).

        Returns:
            dict with keys: windows, top_windows, l2d_crop.
            Each window dict has: title, bounds, z, crop, primary, app, tags,
            attention_score, change_score.
        """
        if screenshot is None:
            return {"windows": [], "top_windows": [], "l2d_crop": None}

        raw_windows = get_window_hierarchy()
        if not raw_windows:
            return {"windows": [], "top_windows": [], "l2d_crop": None}

        windows = sorted(
            cast(list[dict[str, Any]], raw_windows),
            key=lambda w: int(w.get("z", 0)),
        )
        windows = [w for w in windows if not self._is_utility_window(w)]

        for w in windows:
            w["crop"] = self._crop_window(screenshot, w)
            w["bounds"] = BBox(
                x=float(w["left"]),
                y=float(w["top"]),
                w=float(w["width"]),
                h=float(w["height"]),
            )
            scene_tag = "other"
            w["primary"] = scene_tag or "other"
            w["app"] = w.get("class_name", "unknown") or "unknown"
            w["tags"] = [scene_tag] if scene_tag else ["other"]

        # v5.x: L2D is topmost by z-order (smaller z = more foreground).
        # Fallback to title/class detection for non-topmost L2D windows.
        l2d_crop: np.ndarray | None = None
        if windows:
            topmost = windows[0]  # smallest z = topmost
            if self._is_l2d_window(topmost) or topmost.get("z", 0) == 0:
                l2d_crop = topmost.get("crop")
                logger.warning("[L2D] topmost: z=%s title=%s class=%s",
                               topmost.get("z", "?"), topmost.get("title", "?")[:30], topmost.get("class_name", "?"))
        # Fallback: scan all windows for L2D by title/class
        if l2d_crop is None:
            for w in windows[1:]:
                if self._is_l2d_window(w):
                    l2d_crop = w.get("crop")
                    logger.warning("[L2D] fallback by class: z=%s title=%s", w.get("z", "?"), w.get("title", "?")[:30])
                    break

        # v5.x: Remove L2D windows from attention — character, not content
        windows = [w for w in windows if not self._is_l2d_window(w)]

        screen_h, screen_w = screenshot.shape[:2]
        mx, my = mouse_xy

        # v5.x FIX: Filter windows with area larger than screen
        # (common for off-screen/minimized windows that report inflated rects)
        screen_area = screen_w * screen_h
        windows = [w for w in windows
                   if int(w.get("width", 0)) * int(w.get("height", 0)) <= screen_area]

        # v5.x: Filter minimized / hidden / inactive windows
        windows = [w for w in windows
                   if "minimized" not in str(w.get("state", "")).lower()
                   and "hidden" not in str(w.get("state", "")).lower()
                   and "inactive" not in str(w.get("state", "")).lower()]

        # v5.x: Filter background windows (Z-order > 5 = deep background stack)
        windows = [w for w in windows
                   if int(w.get("z", 9999)) <= 5]

        # v5.x: Filter system tray / notification area windows
        windows = [w for w in windows
                   if not any(
                       t in str(w.get("class_name", "")).lower()
                       or t in str(w.get("title", "")).lower()
                       for t in _TRAY_TITLES
                   )]

        # v5.x FIX: compute largest window area for dominant-window
        # heuristic in _spatial_weight (multi-monitor fullscreen detection)
        largest_window_area = max(
            (float(w.get("width", 0)) * float(w.get("height", 0))
             for w in windows),
            default=0.0,
        )
        for w in windows:
            w["attention_score"] = self._spatial_weight(w, mx, my,
                                                       screen_w, screen_h,
                                                       largest_window_area)

        # v5.x Step 4b: Semantic matching (ASR text vs window metadata)
        if asr_text:
            self._semantic_match(asr_text, windows)

        for w in windows:
            w["change_score"] = self._compute_change(w)

        self._prev_windows = list(windows)

        # Filter out task switcher (Alt+Tab) overlay windows
        # before scoring — they should never compete for top attention
        filtered: list[dict[str, Any]] = []
        for w in windows:
            title = str(w.get("title", "")).lower()
            class_name = str(w.get("class_name", "")).lower()
            skip = False
            for t in _TASK_SWITCHER_TITLES:
                if t in title:
                    skip = True
                    break
            if not skip:
                for v in _VBA_FILTER_TITLES:
                    if v in title:
                        skip = True
                        break
            if not skip:
                for c in _TASK_SWITCHER_CLASSES:
                    if c in class_name:
                        skip = True
                        break
            if not skip:
                filtered.append(w)
        windows = filtered

        # v5.x: Simple Z-order — no complex scoring
        top_windows = sorted(
            windows, key=lambda w: int(w.get("z", 9999)),  # v5.x: Z=0 is topmost
        )[:TOP_WINDOW_COUNT]

        return {
            "windows": windows,
            "top_windows": top_windows,
            "l2d_crop": l2d_crop,
        }

    # ------------------------------------------------------------------
    # Step 2 helpers
    # ------------------------------------------------------------------

    def _crop_window(
        self, screenshot: np.ndarray, window: dict[str, Any]
    ) -> np.ndarray | None:
        """Crop screenshot to window boundaries. Returns None if invalid."""
        x1 = max(0, int(window.get("left", 0)))
        y1 = max(0, int(window.get("top", 0)))
        w = int(window.get("width", 0))
        h = int(window.get("height", 0))
        if w <= 0 or h <= 0:
            return None
        x2 = min(screenshot.shape[1], x1 + w)
        y2 = min(screenshot.shape[0], y1 + h)
        if x2 <= x1 or y2 <= y1:
            return None
        return screenshot[y1:y2, x1:x2].copy()

    @staticmethod
    def _is_utility_window(w: dict[str, Any]) -> bool:
        """Filter out IME popups, system overlays, tiny tooltips,
        off-screen windows (negative coords), and known system windows
        (VcXsrv, HControl, Qt*).

        v5.x FIX: Added negative-coord and system-title checks.
        """
        title = str(w.get("title", "")).lower()
        class_name = str(w.get("class_name", "")).lower()
        w_px = int(w.get("width", 0))
        h_px = int(w.get("height", 0))
        if w_px * h_px < _MIN_WINDOW_AREA:
            return True
        for skip in _SKIP_TITLES:
            if skip in title or skip in class_name:
                return True
        # v5.x FIX: Off-screen windows (negative position)
        left = int(w.get("left", 0))
        top = int(w.get("top", 0))
        if left < 0 or top < 0:
            logger.debug("Filtered off-screen window: title=%s pos=(%d,%d)",
                         title[:40], left, top)
            return True
        # v5.x FIX: System windows that should never be scored
        for sys_title in _HIDDEN_TITLE_BLACKLIST:
            if sys_title in title:
                logger.debug("Filtered system window: title=%s", title[:40])
                return True
        return False

    def _is_l2d_window(self, w: dict[str, Any]) -> bool:
        """Detect L2D window by title or class_name, not z-order."""
        title = str(w.get("title", "")).lower().strip()
        class_name = str(w.get("class_name", "")).lower()
        for t in _L2D_TITLES:
            if t in title:
                return True
        for c in _L2D_CLASSES:
            if c in class_name:
                return True
        return False

    # ------------------------------------------------------------------
    # Step 3 helpers
    # ------------------------------------------------------------------

    def _spatial_weight(
        self, window: dict[str, Any], mx: int | None = None, my: int | None = None,
        screen_w: int = 1920, screen_h: int = 1080,
        largest_window_area: float = 0.0,
    ) -> float:
        """Compute attention score from spatial cues + foreground flag.

        v5.x FIX (multi-monitor fullscreen): three detection methods replace
        the old single-monitor-only area_ratio > 0.75 / touches_edges checks:
          (a) area > 45% of total virtual desktop (catches borderless games
              spanning most of a single monitor on multi-monitor setups)
          (b) touches screen edges with relaxed tolerance (catches
              exclusive-fullscreen on primary monitor)
          (c) window is the largest by area AND covers >25% of screen
              (heuristic: the dominant window on multi-monitor is likely
              the user's active application)

        v5.x FIX: When fullscreen_bonus triggers, foreground_bonus is
        suppressed (set to 0).  GetForegroundWindow() is unreliable for
        DirectX exclusive-fullscreen games — they often do not register as
        the Win32 foreground window, leaving a stale terminal/IDE with the
        +2.0 bonus.  A fullscreen game IS the real foreground from the
        user's perspective.

        v4.5.0 §T2 Step 3
        """
        bounds = window.get("bounds")
        if bounds is None:
            return 0.0

        screen_area = float(screen_w * screen_h)
        area = float(bounds.w * bounds.h)
        area_ratio = area / screen_area
        area_score = min(area_ratio / 0.5, 1.0)

        # v5.x FIX: multi-monitor fullscreen detection
        touches_edges = (
            abs(bounds.x) < 10
            and abs(bounds.y) < 10
            and abs(bounds.w - screen_w) < 200
            and abs(bounds.h - screen_h) < 200
        )
        # v5.x FIX: largest-window heuristic — on multi-monitor (~3840×2160),
        # a 2560×1440 game covering 35-45% is clearly the dominant window
        is_dominant = (largest_window_area > 0
                       and area >= largest_window_area * 0.95
                       and area_ratio > 0.25)
        is_fullscreen = (area_ratio > 0.45 or touches_edges or is_dominant)
        # v5.x FIX: raised from 2.0 to 3.0 so fullscreen games decisively
        # outrank any non-fullscreen window even without foreground flag
        fullscreen_bonus = 3.0 if is_fullscreen else 0.0

        z = int(window.get("z", 0))
        z_score = 1.0 / (z + 1.5)

        if mx is not None and my is not None:
            cx = float(bounds.x + bounds.w / 2)
            cy = float(bounds.y + bounds.h / 2)
            dist = ((mx - cx) ** 2 + (my - cy) ** 2) ** 0.5
            contained = (bounds.x <= mx <= bounds.x + bounds.w and
                         bounds.y <= my <= bounds.y + bounds.h)
            mouse_score = max(0, 1.0 - dist / 1200) if contained else 0.0
        else:
            mouse_score = 0.0

        # v5.x FIX: GetForegroundWindow() unreliable for DirectX fullscreen
        # games.  When is_fullscreen is true, the fullscreen window IS the
        # real foreground — suppress the foreground bonus so a stale terminal
        # with keyboard focus cannot outrank a fullscreen game.
        foreground_bonus = 2.0 if window.get("foreground", False) else 0.0
        if is_fullscreen:
            foreground_bonus = 0.0

        return (area_score * 0.4 + z_score * 0.2 + mouse_score * 0.2
                + foreground_bonus + fullscreen_bonus)

    # ------------------------------------------------------------------
    # Step 4a helpers
    # ------------------------------------------------------------------

    def _compute_change(self, window: dict[str, Any]) -> float:
        """Temporal change score vs previous frame. 0.0=unchanged, 1.0=new."""
        title = window.get("title", "")
        if not title:
            return 1.0

        prev = None
        for pw in self._prev_windows:
            if pw.get("title") == title:
                prev = pw
                break

        if prev is None:
            return 1.0

        bounds = window.get("bounds")
        prev_bounds = prev.get("bounds")
        if bounds is None or prev_bounds is None:
            return 0.5

        dx = abs(bounds.x - prev_bounds.x)
        dy = abs(bounds.y - prev_bounds.y)
        dw = abs(bounds.w - prev_bounds.w)
        dh = abs(bounds.h - prev_bounds.h)

        norm_dx = min(1.0, dx / max(bounds.w, 1.0))
        norm_dy = min(1.0, dy / max(bounds.h, 1.0))
        norm_dw = min(1.0, dw / max(bounds.w, 1.0))
        norm_dh = min(1.0, dh / max(bounds.h, 1.0))

        return 0.25 * (norm_dx + norm_dy + norm_dw + norm_dh)

    # ------------------------------------------------------------------
    # Step 4b: Semantic matching (ASR text vs window metadata) — v5.x
    # ------------------------------------------------------------------

    def _semantic_match(self, asr_text: str, windows: list[dict[str, Any]]) -> None:
        """Boost attention_score for windows semantically matching ASR text.

        v5.x Step 4b: Compare ASR embedding against window title + primary + app.
        Adds sim * 0.5 to attention_score (additive, consistent with other scoring
        components). Falls back to substring matching if embedding model unavailable.

        Raises nothing — all exceptions caught and degraded to substring fallback.
        """
        if not asr_text or not windows:
            return

        # Build window text representations (title + primary + app)
        window_texts: list[str] = []
        for w in windows:
            parts = [
                str(w.get("title", "") or ""),
                str(w.get("primary", "") or ""),
                str(w.get("app", "") or ""),
            ]
            window_texts.append(" ".join(p for p in parts if p))

        # ── Primary: embedding-based semantic matching ──
        model = _get_embedding_model()
        if model is not None:
            try:
                # Reason: model.encode/similarity may fail on OOM or corrupted model
                asr_vec = model.encode([asr_text])[0]
                win_vecs = model.encode(window_texts)
                sims = model.similarity(asr_vec, win_vecs)[0]
                for i, w in enumerate(windows):
                    sim = max(0.0, sims[i].item())
                    if sim > 0.0:
                        w["attention_score"] += sim * 0.5
                return
            except Exception:
                # Safe: embedding lookup failure degrades to substring — no action blocked
                logger.warning(
                    "semantic_match: embedding failed for '%s' — "
                    "falling back to substring", asr_text[:30],
                )

        # ── Fallback: bidirectional substring matching ──
        asr_lower = asr_text.lower()
        for i, w in enumerate(windows):
            window_lower = window_texts[i].lower()
            if asr_lower in window_lower or window_lower in asr_lower:
                w["attention_score"] += 0.3

    # ------------------------------------------------------------------
    # VLM cache — Heartbeat cycle reuse (v4.5.0 §T4)
    # ------------------------------------------------------------------

    def _check_vlm_cache(self, window_title: str) -> str | None:
        """Return cached VLM description if available and not expired.

        v4.5.0 §T4: Cache descriptions for 3s heartbeat reuse cycle.
        Cache is used only for reusing descriptions across heartbeat ticks
        within the same cycle — it never replaces VLM inference.
        """
        if window_title in self._vlm_cache:
            entry = self._vlm_cache[window_title]
            if time.time() - entry["timestamp"] < 3.0:
                return entry["description"]
        return None

    def _update_vlm_cache(self, window_title: str, description: str) -> None:
        """Store VLM description in cache with current timestamp."""
        self._vlm_cache[window_title] = {
            "description": description,
            "timestamp": time.time(),
        }

    # ------------------------------------------------------------------
    # Step 7: Layered LLM context assembly — v4.5.0 §T2 Step 7
    # ------------------------------------------------------------------

    def assemble_llm_context(
        self,
        top_window: dict[str, Any],
        asr_text: str | None = None,
    ) -> dict[str, str | list[str]]:
        """Build focused LLM context from top-1 window + VLM description + intent match.

        Returns dict with keys: text, scene, position, vlm_description, matched.
        Total context kept < 200 tokens — raw ui/text arrays are discarded.

        v4.5.0 §T2 Step 7
        """
        ui = top_window.get("ui", [])
        text_items = top_window.get("text", [])
        # v5.x: Use icon_labels from orchestrator concept classifier when available
        icon_labels: list[str] = top_window.get("icon_labels", []) or []
        ui_desc = ""
        if icon_labels:
            ui_desc = f"含 {', '.join(icon_labels[:5])}"
        elif ui:
            ui_items = []
            for u in ui:
                if hasattr(u, "type"):
                    ui_items.append(u.type)
                elif isinstance(u, dict):
                    ui_items.append(u.get("label", u.get("type", "")))
            if ui_items:
                ui_desc = f"按钮: {', '.join(ui_items[:8])}"
        text_desc = ""
        if text_items:
            text_parts = []
            for t in text_items:
                if hasattr(t, "content"):
                    content = str(t.content)[:30]
                elif isinstance(t, dict):
                    content = str(t.get("content", t.get("text", "")))[:30]
                else:
                    continue
                # v5.x §T2: Skip text items that duplicate ASR input
                if _text_matches_asr(content, asr_text):
                    continue
                text_parts.append(content)
            if text_parts:
                text_desc = f"文字: {'; '.join(text_parts[:5])}"
        # v5.x §T2: Strip ASR duplications from VLM description
        vlm_desc = top_window.get("vlm_description", "") or ""
        if vlm_desc and _text_matches_asr(vlm_desc[:40], asr_text):
            vlm_desc = ""
        ctx: dict[str, str | list[str]] = {
            "text": f"用户说：{asr_text}" if asr_text else "",
            "ui": ui_desc,
            "ocr_text": text_desc,
            "scene": (
                f"前台应用：{top_window.get('app', '未知')}，"
                f"场景：{top_window.get('primary', '其他')}"
            ),
            "position": self._describe_position(top_window),
            "vlm_description": vlm_desc,
            "matched": [],
        }

        # Conditional: only inject UI/text matched to user intent
        if asr_text:
            ctx["matched"] = self._match_intent_elements(asr_text, top_window)

        return ctx

    def _describe_position(self, window: dict[str, Any]) -> str:
        """Describe window position in natural language.

        v4.5.0 §T2 Step 7 helper
        """
        bounds = window.get("bounds")
        if bounds is None:
            return "位置未知"
        z = window.get("z", 0)
        title = window.get("title", "")[:20]
        pos = (
            f"窗口'{title}' "
            f"位于({int(bounds.x)},{int(bounds.y)}) "
            f"尺寸{int(bounds.w)}x{int(bounds.h)}"
        )
        if z == 0:
            pos += "，前台窗口"
        else:
            pos += f"，Z={z}"
        return pos

    def _match_intent_elements(
        self,
        asr_text: str,
        window: dict[str, Any],
    ) -> list[str]:
        """Match ASR intent to specific UI/text elements.

        Keyword-match between asr_text and element type/content.
        Returns human-readable descriptions, max 3.

        v4.5.0 §T2 Step 7 helper
        """
        matched: list[str] = []
        ui_elements = window.get("ui", [])
        text_elements = window.get("text", [])

        asr_lower = asr_text.lower()

        for ui in ui_elements:
            ui_type = str(ui.get("type", "")).lower()
            if any(kw in asr_lower for kw in ui_type.split()):
                matched.append(f"可交互元素'{ui.get('type')}'在窗口内")

        for txt in text_elements:
            txt_content = str(txt.get("content", "")).lower()
            if any(kw in asr_lower for kw in txt_content.split()):
                snippet = str(txt.get("content", ""))[:30]
                matched.append(f"屏幕上显示'{snippet}'")

        return matched[:3]

    def assemble_heartbeat_context(
        self,
        vlm_description: str,
        top_window: dict[str, Any],
    ) -> str:
        """For ProactiveHeartbeat: simple scene + VLM description. No ASR text.

        Returns a plain string (not dict) for heartbeat prompts.

        v4.5.0 §T2 Step 7
        """
        return f"前台：{top_window.get('app', '未知')}。{vlm_description}"
