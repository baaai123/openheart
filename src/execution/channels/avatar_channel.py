"""
AvatarChannel — unified avatar execution channel.

v4.5.0 §7.3.6: AvatarChannel with Live2DRenderer.
v4.5.0 §7.3.4: Live2D rendering MUST run in a dedicated thread, NOT on the asyncio event loop.
项目宪法 §1.3: Live2D 渲染必须在独立子线程中执行，禁止占用 asyncio 主事件循环。

Degradation logic (v4.5.0 §7.3.7):
  - Live2D init failure → degraded=true, recovery scheduled every 60s
  - 3+ consecutive runtime errors → degraded=true
  - Recovery: attempt Live2D recreation every 60 seconds via _recovery_loop()

v5.x (GLX rewrite): All GLUT code replaced with raw ctypes GLX/X11.
  Rendering window created via XCreateSimpleWindow + glXChooseFBConfig + glXCreateNewContext.
"""

from __future__ import annotations

import asyncio
import ctypes
import logging
import math
import os
import sys
import threading
import time
from typing import Optional

import numpy as np

from src.infra.live2d_loader import Live2DModel  # v4.5.0 §7.3 — Cubism 3.0 ctypes
from src.execution.channels.companion_animation import (
    CompanionAnimationManager,
    RELIABLE_EMOTIONS,
)

logger = logging.getLogger(__name__)

# v4.5.0 §7.3.4: 5-second init timeout for Live2DRenderer sub-thread
_LIVE2D_INIT_TIMEOUT_SEC = 5.0
# v4.5.0 §7.3.7: retry interval for Live2D recovery
_RECOVERY_INTERVAL_SEC = 60.0
# Maximum consecutive failures before degrading to fallback (§7.3.7)
_MAX_FAIL_COUNT = 3

# ---------------------------------------------------------------------------
# GLX / X11 / OpenGL constants (mirrors test_l2d_glx.py)
# ---------------------------------------------------------------------------
_GL_LIB_PATH = "/usr/lib/x86_64-linux-gnu/libGL.so.1"
_X11_LIB_PATH = "libX11.so.6"

# X11
_ExposureMask = 1 << 15

# GL
_GL_COLOR_BUFFER_BIT = 0x4000
_GL_BLEND = 0x0BE2
_GL_SRC_ALPHA = 0x0302
_GL_ONE_MINUS_SRC_ALPHA = 0x0303
_GL_VENDOR = 0x1F00
_GL_RENDERER = 0x1F01

# GLX
_GLX_RGBA_TYPE = 0x8014

# Minimal FBConfig attributes: matches working test_l2d_glx.py (0x0002 ≈ GLX_USE_GL or platform-equiv)
_FBCONFIG_ATTRS = (ctypes.c_int * 5)(0x0002, 1, 0, 0)


def _load_gl_library() -> ctypes.CDLL:
    """Return a ctypes handle to libGL.so.1.  Must be called in the render thread."""
    if not os.path.exists(_GL_LIB_PATH):
        raise RuntimeError(f"OpenGL library not found: {_GL_LIB_PATH}")
    lib = ctypes.CDLL(_GL_LIB_PATH)

    # glX
    lib.glXChooseFBConfig.argtypes = [
        ctypes.c_void_p, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
    ]
    lib.glXChooseFBConfig.restype = ctypes.POINTER(ctypes.c_void_p)

    lib.glXCreateNewContext.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_int,
    ]
    lib.glXCreateNewContext.restype = ctypes.c_void_p

    lib.glXMakeContextCurrent.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ]
    lib.glXMakeContextCurrent.restype = ctypes.c_int

    lib.glXSwapBuffers.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.glXDestroyContext.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

    # gl
    lib.glClear.argtypes = [ctypes.c_uint]
    lib.glClearColor.argtypes = [ctypes.c_float] * 4
    lib.glEnable.argtypes = [ctypes.c_uint]
    lib.glBlendFunc.argtypes = [ctypes.c_uint, ctypes.c_uint]
    lib.glFinish.argtypes = []
    lib.glGetString.argtypes = [ctypes.c_uint]
    lib.glGetString.restype = ctypes.c_char_p

    return lib


def _load_x11_library() -> ctypes.CDLL:
    """Return a ctypes handle to libX11.so.6."""
    lib = ctypes.CDLL(_X11_LIB_PATH)

    lib.XOpenDisplay.argtypes = [ctypes.c_char_p]
    lib.XOpenDisplay.restype = ctypes.c_void_p
    lib.XDefaultScreen.argtypes = [ctypes.c_void_p]
    lib.XDefaultScreen.restype = ctypes.c_int
    lib.XDefaultRootWindow.argtypes = [ctypes.c_void_p]
    lib.XDefaultRootWindow.restype = ctypes.c_int
    lib.XCreateSimpleWindow.argtypes = [ctypes.c_void_p] * 9
    lib.XCreateSimpleWindow.restype = ctypes.c_int
    lib.XSelectInput.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_long]
    lib.XMapRaised.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.XSync.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.XStoreName.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p]
    lib.XPending.argtypes = [ctypes.c_void_p]
    lib.XPending.restype = ctypes.c_int
    lib.XNextEvent.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.XDestroyWindow.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.XCloseDisplay.argtypes = [ctypes.c_void_p]

    return lib


class Live2DRenderer:
    """
    Live2D model renderer running in a dedicated sub-thread with its own OpenGL context
    created via raw GLX/X11 (no GLUT dependency).

    v4.5.0 §7.3.4:
      - Rendering thread runs _render_loop() in a dedicated os thread.
      - Main thread pushes commands via queue.Queue (non-blocking).
      - Status/heartbeat flows back via asyncio.Queue.
      - set_expression, start_motion, set_parameter, set_lip_sync_volume: all non-blocking.
      - Uses Cubism Core ctypes wrapper (live2d_loader.py) + raw ctypes GLX/X11 for windowing.

    v5.x: All GLUT references removed; GLX/X11 via pure ctypes.
    """

    def __init__(self, config_path: str = "config/live2d.yaml") -> None:
        import queue

        self._cmd_queue: queue.Queue[tuple[str, tuple]] = queue.Queue()
        self._status_queue: asyncio.Queue[dict] = asyncio.Queue()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._healthy: bool = False
        self._last_heartbeat: float = time.monotonic()
        self._config_path: str = config_path

        # GLX state (populated in render thread)
        self._gl: Optional[ctypes.CDLL] = None
        self._x11: Optional[ctypes.CDLL] = None
        self._dpy: Optional[int] = None
        self._win: int = 0
        self._ctx: Optional[int] = None

        # Start the render thread with a ready-event for init-timeout detection
        self._ready_event = threading.Event()
        self._thread = threading.Thread(
            target=self._render_loop,
            name="live2d-render-thread",
            daemon=True,
        )
        self._thread.start()

        # v4.5.0 §7.3.4: wait up to 5s for ready signal
        if not self._ready_event.wait(timeout=_LIVE2D_INIT_TIMEOUT_SEC):
            self._thread = None
            raise RuntimeError(
                f"Live2DRenderer init timeout ({_LIVE2D_INIT_TIMEOUT_SEC}s). "
                "Render thread did not signal ready."
            )
        self._healthy = True
        logger.info("Live2DRenderer initialized successfully (thread=%s)", self._thread.name)

    # -------------------------------------------------------------------
    # Public command interface — all non-blocking (§7.3.4)
    # -------------------------------------------------------------------

    def set_expression(self, expression_name: str) -> None:
        self._cmd_queue.put(("expression", (expression_name,)))

    def start_motion(self, motion_name: str) -> None:
        self._cmd_queue.put(("motion", (motion_name,)))

    def set_parameter(self, param_id: str, value: float) -> None:
        self._cmd_queue.put(("parameter", (param_id, value)))

    def set_lip_sync_volume(self, volume: float) -> None:
        self._cmd_queue.put(("lip_sync", (volume,)))

    def close(self) -> None:
        self._stop_event.set()
        self._cmd_queue.put(("__exit__", ()))
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        self._healthy = False

    @property
    def is_healthy(self) -> bool:
        return self._healthy

    # -------------------------------------------------------------------
    # Render thread
    # -------------------------------------------------------------------

    def _render_loop(self) -> None:
        try:
            self._init_gl_context()
            self._ready_event.set()
        except Exception:
            logger.exception("Live2D render thread failed to initialise OpenGL context")
            return

        frame_count = 0
        while not self._stop_event.is_set():
            try:
                self._render_one_frame()
                self._drain_commands()
                # Drain pending X11 events so the window stays responsive
                while self._x11.XPending(self._dpy):
                    evt = (ctypes.c_char * 96)()
                    self._x11.XNextEvent(self._dpy, ctypes.cast(evt, ctypes.c_void_p))
                frame_count += 1
                if frame_count % 30 == 0:
                    self._send_heartbeat()
            except Exception:
                logger.exception("Live2D render loop exception")
                self._healthy = False
                break
        self._cleanup_gl_context()

    # -------------------------------------------------------------------
    # GLX window init (v5.x: raw ctypes, no GLUT)
    # -------------------------------------------------------------------

    def _init_gl_context(self) -> None:
        # v4.5.0 §7.3.4 — raw GLX/X11 window creation (same approach as test_l2d_glx.py)
        self._gl = _load_gl_library()
        self._x11 = _load_x11_library()

        # --- Open display ---
        self._dpy = self._x11.XOpenDisplay(b":0")
        if not self._dpy:
            raise RuntimeError("XOpenDisplay returned NULL — cannot connect to X server")
        scr = self._x11.XDefaultScreen(self._dpy)
        root = self._x11.XDefaultRootWindow(self._dpy)

        # --- Read window size from config ---
        w, h = 400, 600
        try:
            import yaml
            cfg = yaml.safe_load(open(self._config_path))
            if isinstance(cfg, dict):
                w = int(cfg.get("window_width", w))
                h = int(cfg.get("window_height", h))
        except Exception:
            logger.debug("Could not read window size from config; using defaults")

        # --- Create window ---
        self._win = self._x11.XCreateSimpleWindow(
            self._dpy, root, 200, 200, w, h, 0, 0, 0,
        )
        self._x11.XSelectInput(self._dpy, self._win, _ExposureMask)
        self._x11.XStoreName(self._dpy, self._win, b"OpenHeart L2D")
        self._x11.XMapRaised(self._dpy, self._win)
        self._x11.XSync(self._dpy, 0)
        time.sleep(0.3)  # Let WM settle (same as test_l2d_glx.py)

        # --- Choose FBConfig + create GLX context ---
        n_fbc = ctypes.c_int()
        fbc_list = self._gl.glXChooseFBConfig(
            self._dpy, scr, _FBCONFIG_ATTRS, ctypes.byref(n_fbc),
        )
        if not fbc_list or n_fbc.value == 0:
            raise RuntimeError("glXChooseFBConfig returned no configs")

        self._ctx = self._gl.glXCreateNewContext(
            self._dpy, fbc_list[0], _GLX_RGBA_TYPE, None, 1,
        )
        if not self._ctx:
            raise RuntimeError("glXCreateNewContext returned NULL")

        if not self._gl.glXMakeContextCurrent(self._dpy, self._win, self._win, self._ctx):
            raise RuntimeError("glXMakeContextCurrent failed")

        vid = self._gl.glGetString(_GL_VENDOR)
        rnd = self._gl.glGetString(_GL_RENDERER)
        logger.info(
            "GLX window=%d FBConfigs=%d ctx=%s vendor=%s renderer=%s",
            self._win, n_fbc.value,
            hex(self._ctx) if self._ctx else "NULL",
            vid.decode() if vid else "N/A",
            rnd.decode() if rnd else "N/A",
        )

        # --- Set up GL state ---
        self._gl.glClearColor(0.0, 0.0, 0.0, 0.0)
        self._gl.glEnable(_GL_BLEND)
        self._gl.glBlendFunc(_GL_SRC_ALPHA, _GL_ONE_MINUS_SRC_ALPHA)

        # --- Load Live2D model ---
        self._live2d_model = None
        self._expression_map: dict[str, list[tuple[str, float]]] = {}

        try:
            import yaml
            cfg = yaml.safe_load(open(self._config_path))
            cfg_expressions = cfg.get("expressions", {}) if isinstance(cfg, dict) else {}
            for exp_name, param_str in cfg_expressions.items():
                pairs: list[tuple[str, float]] = []
                if param_str and param_str.strip():
                    for token in param_str.strip().split():
                        if ":" in token:
                            pname, _, pval = token.partition(":")
                            try:
                                pairs.append((pname.strip(), float(pval.strip())))
                            except ValueError:
                                logger.warning("Invalid expression param value: %r", token)
                self._expression_map[exp_name] = pairs

            model_path = cfg.get("model_path") if isinstance(cfg, dict) else None
            if model_path and os.path.exists(model_path):
                m = Live2DModel()
                m.load(model_path)
                self._live2d_model = m
                self._healthy = True
                logger.info("Live2D model loaded via Cubism Core: %s (params=%d drawables=%d)",
                            model_path, m.param_count, m.drawable_count)
        except Exception as e:
            logger.warning("Live2D model load skipped: %s", e)

        self._last_heartbeat = time.monotonic()

    # -------------------------------------------------------------------
    # Per-frame rendering (v5.x: raw GLX swap, no GLUT)
    # -------------------------------------------------------------------

    def _render_one_frame(self) -> None:
        # v4.5.0 §7.3.4 — raw OpenGL + GLX swap
        self._gl.glClear(_GL_COLOR_BUFFER_BIT)
        if self._live2d_model:
            self._live2d_model.update()
        self._gl.glFinish()
        self._gl.glXSwapBuffers(self._dpy, self._win)

    # -------------------------------------------------------------------
    # Command processing
    # -------------------------------------------------------------------

    def _drain_commands(self) -> None:
        import queue
        while True:
            try:
                cmd, args = self._cmd_queue.get_nowait()
            except queue.Empty:
                break
            if cmd == "__exit__":
                self._stop_event.set()
                return

            model = self._live2d_model
            if cmd == "expression":
                expression_name = args[0]
                params = self._expression_map.get(expression_name)
                if params is None:
                    logger.warning("Unknown expression %r, falling back to neutral", expression_name)
                    params = self._expression_map.get("neutral", [])
                if model:
                    for pname, pval in params:
                        model.set_parameter(pname, pval)
                    model.update()
                else:
                    logger.debug("expression %r ignored — no Live2D model loaded", expression_name)

            elif cmd == "motion":
                motion_name = args[0]
                if model:
                    model.start_motion(motion_name)
                else:
                    logger.debug("motion %r ignored — no Live2D model loaded", motion_name)

            elif cmd == "parameter":
                param_id, value = args[0], args[1]
                if model:
                    model.set_parameter(param_id, value)

            elif cmd == "lip_sync":
                volume = args[0]
                if model:
                    model.set_parameter("ParamMouthOpen", volume)

            self._cmd_queue.task_done()

    # -------------------------------------------------------------------
    # Heartbeat
    # -------------------------------------------------------------------

    def _send_heartbeat(self) -> None:
        self._last_heartbeat = time.monotonic()
        try:
            if self._live2d_model:
                self._live2d_model.update()
            self._healthy = True
        except Exception:
            logger.warning("Live2D heartbeat failed — marking unhealthy", exc_info=True)
            self._healthy = False
        try:
            self._status_queue.put_nowait({"type": "heartbeat", "ts": self._last_heartbeat})
        except asyncio.QueueFull:
            pass

    # -------------------------------------------------------------------
    # Cleanup (v5.x: raw GLX/X11, no GLUT)
    # -------------------------------------------------------------------

    def _cleanup_gl_context(self) -> None:
        # v4.5.0 §7.3.4 — clean up GLX context, X11 window, model
        if hasattr(self, "_live2d_model") and self._live2d_model:
            try:
                self._live2d_model.close()
            except Exception:
                pass

        # Make context not-current, then destroy it
        if self._dpy and self._ctx:
            try:
                self._gl.glXMakeContextCurrent(self._dpy, 0, 0, None)
            except Exception:
                pass
            try:
                self._gl.glXDestroyContext(self._dpy, self._ctx)
            except Exception:
                pass
            self._ctx = None

        # Destroy window
        if self._dpy and self._win:
            try:
                self._x11.XDestroyWindow(self._dpy, self._win)
            except Exception:
                pass
            self._win = 0

        logger.info("Live2D render thread stopped")


# =================================================================== AvatarChannel

class AvatarChannel:
    def __init__(self, config_path="config/live2d.yaml"):
        self._renderer = None
        self._config_path = config_path
        self._degraded = False
        self._failure_count = 0
        self._companion_anim = None
        self._idle_task = None
        self._loop = None
        self._recovery_task = None
        self._init_renderer()
        # v4.5.0 §7.3.7: start background recovery loop if asyncio loop is available
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None:
            self._recovery_task = loop.create_task(self._recovery_loop())

    def _init_renderer(self):
        try:
            self._renderer = Live2DRenderer(self._config_path)
            self._degraded = False
            self._failure_count = 0
        except Exception as exc:
            # v4.5.0 §7.3.7: Live2D init failure → degraded mode, no fallback
            logger.warning(
                "Live2D init failed (%s) → degraded=true, recovery scheduled in %ss",
                exc,
                _RECOVERY_INTERVAL_SEC,
            )
            self._renderer = None
            self._degraded = True

    def set_expression(self, name):
        if self._renderer:
            self._renderer.set_expression(name)

    def start_motion(self, name, prio=2):
        if self._renderer:
            self._renderer.start_motion(name)

    def set_lip_sync(self, vol):
        if self._renderer:
            self._renderer.set_lip_sync_volume(vol)

    def send_audio(self, audio_bytes: bytes) -> None:
        """v4.5.0 §7.3.3: TTS audio → lip-sync via RMS volume."""
        if self._degraded:
            return  # silent degrade — captions handle text display
        if not audio_bytes:
            return
        try:
            samples = (
                np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
            )
            rms = compute_rms_volume(samples)
            self.set_lip_sync(int(rms * 2.0))  # scale to 0-2 range
        except Exception:
            pass  # v4.5.0 §7.3.4: audio processing failure is non-critical

    def set_parameter(self, pid, val):
        if self._renderer:
            self._renderer.set_parameter(pid, val)

    @property
    def degraded(self) -> bool:
        return self._degraded

    def _try_recover(self):
        self._failure_count += 1
        if self._failure_count >= 3:
            if self._renderer:
                self._renderer.close()
            self._degraded = True
            self._renderer = None

    async def _recovery_loop(self):
        """v4.5.0 §7.3.7: attempt Live2D recovery every 60s."""
        while True:
            await asyncio.sleep(_RECOVERY_INTERVAL_SEC)
            if self._degraded:
                try:
                    self._init_renderer()
                    if not self._degraded:
                        logger.info("Live2D recovered successfully")
                except Exception:
                    pass


def compute_rms_volume(samples) -> float:
    import math
    if not samples: return 0.0
    return math.sqrt(sum(s*s for s in samples) / len(samples))
