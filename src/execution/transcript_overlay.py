"""
TranscriptOverlay — synchronized text display window for TTS.

v4.5.0 §7.5.6: Transcript Overlay 同步显示
- tkinter-based floating window (borderless, semi-transparent, topmost)
- Crash recovery: rebuilds every 60 s if the window dies
- Decoupled from audio playback; voice_channel invokes sync methods
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import tkinter

logger = logging.getLogger(__name__)

_REBUILD_INTERVAL_S = 60.0

_DEFAULT_CONFIG: dict[str, object] = {
    "enabled": True,
    "font_size": 32,
    "color": "#FFFFFF",
    "background": "#000000",
    "opacity": 0.7,
    "position": "auto",
    "word_highlight": True,
    "mouse_pass_through": True,
    "idle_hide_seconds": 300,
    "conversation_enabled": True,
    "max_conversation_lines": 10,
    "width": 900,
    "height": 420,
    "conversation_bg": "#1a1a2e",
    "user_color": "#7ec8e3",
    "assistant_color": "#ffb7c5",
    "separator_color": "#333355",
    "l2d_window_title": "Live2D",
    "conversation_font_size": 18,
}


def _load_config() -> dict[str, object]:
    """Load overlay config from YAML, falling back to defaults."""
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "config", "transcript_overlay.yaml"
    )
    config = dict(_DEFAULT_CONFIG)
    try:
        import yaml

        with open(config_path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
        if loaded:
            config.update(loaded)
    except Exception:
        pass
    return config


_CJK_RANGES = (
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    (0x3400, 0x4DBF),  # CJK Extension A
    (0x3000, 0x303F),  # CJK Symbols & Punctuation
    (0xFF00, 0xFFEF),  # Fullwidth Forms
)


def _is_cjk(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _CJK_RANGES)


def _char_width(ch: str) -> float:
    if ch.isspace():
        return 0.0
    return 1.0 if _is_cjk(ch) else 0.5


def _wrap_text(text: str, max_cjk: int = 7) -> str:
    if not text:
        return text

    paragraphs = text.split("\n")
    result_paragraphs: list[str] = []

    for para in paragraphs:
        if not para or para.isspace():
            result_paragraphs.append("")
            continue

        tokens: list[str] = []
        i = 0
        while i < len(para):
            ch = para[i]
            if _is_cjk(ch):
                tokens.append(ch)
                i += 1
            elif ch.isspace():
                tokens.append(ch)
                i += 1
            else:
                j = i
                while j < len(para) and not _is_cjk(para[j]) and not para[j].isspace():
                    j += 1
                tokens.append(para[i:j])
                i = j

        lines: list[str] = []
        cur_tok: list[str] = []
        cur_width = 0.0

        for token in tokens:
            tw = sum(_char_width(c) for c in token)

            if token.isspace():
                cur_tok.append(token)
                continue

            if tw > max_cjk:
                if cur_tok:
                    while cur_tok and cur_tok[-1].isspace():
                        cur_tok.pop()
                    line = "".join(cur_tok)
                    if line:
                        lines.append(line)
                    cur_tok = []
                    cur_width = 0.0

                remaining = token
                while remaining:
                    acc = 0.0
                    split = 0
                    for ch in remaining:
                        cw = _char_width(ch)
                        if acc + cw > max_cjk:
                            break
                        acc += cw
                        split += 1
                    if split == 0:
                        split = 1
                    lines.append(remaining[:split])
                    remaining = remaining[split:]
                continue

            if cur_width + tw > max_cjk and cur_tok:
                while cur_tok and cur_tok[-1].isspace():
                    cur_tok.pop()
                line = "".join(cur_tok)
                if line:
                    lines.append(line)
                cur_tok = [token]
                cur_width = tw
            else:
                cur_tok.append(token)
                cur_width += tw

        if cur_tok:
            while cur_tok and cur_tok[-1].isspace():
                cur_tok.pop()
            line = "".join(cur_tok)
            if line:
                lines.append(line)

        result_paragraphs.append("\n".join(lines))

    # Strip trailing empty paragraphs (trailing newlines in input)
    while result_paragraphs and not result_paragraphs[-1]:
        result_paragraphs.pop()

    return "\n".join(result_paragraphs)


class _OverlayWindow:
    """Internal helper that owns the actual Tkinter window in a worker thread."""

    def __init__(self, config: dict[str, object]) -> None:
        self.config: dict[str, object] = config
        self._tk: Any = None
        self._root: tkinter.Tk | None = None
        self._label: tkinter.Label | None = None
        self._conv_text: tkinter.Text | None = None
        self._conversation_text: str = ""
        self._thread: threading.Thread | None = None
        self._alive: bool = False
        self._lock: threading.Lock = threading.Lock()
        self._pending_text: str | None = None
        self._pending_highlight: int | None = None
        self._visible: bool = False

    def start(self) -> bool:
        """Spawn the Tk thread. Return True if thread started, False if tk unavailable."""
        try:
            __import__("tkinter")
        except ImportError:
            logger.warning("TranscriptOverlay: tkinter not available")
            return False

        with self._lock:
            if self._thread and self._thread.is_alive():
                return True
            self._alive = True
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
            return True

    def stop(self) -> None:
        with self._lock:
            self._alive = False
            if self._root:
                try:
                    self._root.quit()
                except Exception:
                    pass

    def _run(self) -> None:
        """Tk main loop — runs in dedicated thread."""
        import tkinter

        try:
            self._root = tkinter.Tk()
            self._root.withdraw()
            self._configure_window()
            self._build_ui()
            self._root.deiconify()
            self._apply_pending_state()
            self._apply_auto_position()
            self._root.mainloop()
        except Exception as exc:
            logger.warning(f"TranscriptOverlay window crashed: {exc}")
        finally:
            try:
                if self._root:
                    self._root.destroy()
            except Exception:
                pass
            self._root = None

    def _configure_window(self) -> None:
        """Apply borderless, topmost, transparency, position settings."""
        root = self._root
        if root is None:
            return
        cfg = self.config

        root.overrideredirect(True)
        root.attributes("-topmost", True)
        root.attributes("-alpha", float(cfg.get("opacity", 0.7)))

        if cfg.get("mouse_pass_through", True):
            try:
                root.attributes("-disabled", True)
            except Exception:
                pass

        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        conv_enabled = cfg.get("conversation_enabled", True)
        if conv_enabled:
            width = int(cfg.get("width", 500))  # type: ignore[arg-type]
            height = int(cfg.get("height", 200))  # type: ignore[arg-type]
        else:
            width = int(screen_w * 0.6)
            height = int(cfg.get("font_size", 24) * 3)

        pos = cfg.get("position", "auto")
        x = max(0, (screen_w - width) // 2)
        initial_y = 20 if pos == "top" else screen_h - height - 40
        initial_y = max(0, initial_y)

        root.geometry(f"{width}x{height}+{x}+{initial_y}")

    def _build_ui(self) -> None:
        """Create the label widget."""
        root = self._root
        if root is None:
            return
        cfg = self.config
        bg = cfg.get("background", "#000000")
        conv_bg = cfg.get("conversation_bg", "#1a1a2e")
        fg = cfg.get("color", "#FFFFFF")
        font_size = cfg.get("font_size", 24)
        conv_font_size = int(cfg.get("conversation_font_size", 14))  # type: ignore[arg-type]
        user_color = str(cfg.get("user_color", "#7ec8e3"))
        assistant_color = str(cfg.get("assistant_color", "#ffb7c5"))

        import tkinter
        import tkinter.ttk as ttk

        root.configure(bg=bg if not cfg.get("conversation_enabled", True) else conv_bg)
        self._label = tkinter.Label(
            root,
            text="",
            font=("Helvetica", font_size),
            fg=fg,
            bg=conv_bg,
            justify="center",
        )
        self._label.pack(side="top", fill="x")

        conv_enabled = cfg.get("conversation_enabled", True)
        if conv_enabled:
            sep_color = str(cfg.get("separator_color", "#333355"))
            separator = ttk.Separator(root, orient="horizontal")
            separator.pack(side="top", fill="x", padx=5, pady=2)
            try:
                separator.configure(style="TSeparator")
                style = ttk.Style()
                style.configure("TSeparator", background=sep_color)
            except Exception:
                pass

            self._conv_text = tkinter.Text(
                root,
                font=("Helvetica", conv_font_size),
                fg=fg,
                bg=conv_bg,
                wrap="word",
                height=int(cfg.get("max_conversation_lines", 10)),  # type: ignore[arg-type]
                relief="flat",
                borderwidth=0,
                highlightthickness=0,
                padx=10,
                pady=5,
                state="disabled",
            )
            self._conv_text.tag_config("user", foreground=user_color)
            self._conv_text.tag_config("assistant", foreground=assistant_color)
            self._conv_text.pack(side="bottom", fill="both", expand=True)

    def _apply_pending_state(self) -> None:
        with self._lock:
            if self._pending_text is not None:
                self._set_text_unsafe(self._pending_text)
            if self._pending_highlight is not None:
                self._highlight_unsafe(self._pending_highlight)
            if self._conversation_text:
                self._set_conversation_unsafe(self._conversation_text)
            if self._visible:
                self._show_unsafe()
            else:
                self._hide_unsafe()

    def show_sentence(self, text: str) -> None:
        with self._lock:
            self._pending_text = text
            self._pending_highlight = None
        if self._root:
            self._root.after(0, lambda: self._set_text_unsafe(text))

    def highlight_word(self, word_index: int) -> None:
        with self._lock:
            self._pending_highlight = word_index
        if self._root:
            self._root.after(0, lambda: self._highlight_unsafe(word_index))

    def clear(self) -> None:
        with self._lock:
            self._pending_text = ""
            self._pending_highlight = None
        if self._root:
            self._root.after(0, lambda: self._set_text_unsafe(""))

    def hide(self) -> None:
        with self._lock:
            self._visible = False
        if self._root:
            self._root.after(0, self._hide_unsafe)

    def show(self) -> None:
        with self._lock:
            self._visible = True
        if self._root:
            self._root.after(0, self._show_unsafe)

    def show_conversation(self, text: str) -> None:
        """Update conversation text — thread-safe, schedules tkinter update."""
        with self._lock:
            self._conversation_text = text
        if self._root:
            self._root.after(0, lambda: self._set_conversation_unsafe(text))

    def _set_text_unsafe(self, text: str) -> None:
        if self._label:
            try:
                wrapped = _wrap_text(text)
                self._label.config(text=wrapped)
            except Exception:
                pass

    def _set_conversation_unsafe(self, text: str) -> None:
        if not self._conv_text:
            return
        try:
            self._conv_text.config(state="normal")
            self._conv_text.delete("1.0", "end")
            lines = text.split("\n")
            for i, line in enumerate(lines):
                if i > 0:
                    self._conv_text.insert("end", "\n")
                if line.startswith("👤 用户:"):
                    tag = "user"
                elif line.startswith("🤖 雪奈:"):
                    tag = "assistant"
                else:
                    tag = None
                self._conv_text.insert("end", line, tag)
            self._conv_text.config(state="disabled")
        except Exception as e:
            logger.warning("_set_conversation_unsafe failed: %s", e)

    def _highlight_unsafe(self, word_index: int) -> None:
        logger.debug(f"TranscriptOverlay highlight word {word_index}")

    def _find_l2d_window(self) -> tuple[int, int, int, int] | None:
        import subprocess

        l2d_title = str(self.config.get("l2d_window_title", "Live2D"))
        try:
            result = subprocess.run(
                ["xdotool", "search", "--name", l2d_title],
                capture_output=True,
                text=True,
                timeout=1,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return None
        except Exception:
            return None

        window_ids = result.stdout.strip().split()
        if not window_ids:
            return None

        wid = window_ids[0]
        try:
            geo = subprocess.run(
                ["xdotool", "getwindowgeometry", wid],
                capture_output=True,
                text=True,
                timeout=1,
            )
        except Exception:
            return None

        x = y = w = h = 0
        for line in geo.stdout.splitlines():
            line = line.strip()
            if line.startswith("Position:"):
                coords = line.split(":")[1].strip().split(",")
                if len(coords) == 2:
                    x, y = int(coords[0].strip()), int(coords[1].strip())
            elif line.startswith("Geometry:"):
                dims = line.split(":")[1].strip().split("x")
                if len(dims) == 2:
                    w, h = int(dims[0].strip()), int(dims[1].strip())
        return (x, y, w, h)

    def _apply_auto_position(self) -> None:
        root = self._root
        if root is None or str(self.config.get("position", "auto")) != "auto":
            return
        l2d_geom = self._find_l2d_window()
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        try:
            win_w = root.winfo_width()
            win_h = root.winfo_height()
        except Exception:
            win_w, win_h = 500, 200
        x = (screen_w - win_w) // 2
        if l2d_geom:
            _, l2d_y, _, l2d_h = l2d_geom
            y = min(l2d_y + l2d_h + 20, screen_h - win_h - 20)
            y = max(y, 20)
        else:
            y = screen_h - win_h - 40
        x = max(x, 0)
        y = max(y, 0)
        try:
            root.geometry(f"+{x}+{y}")
        except Exception:
            pass

    def _hide_unsafe(self) -> None:
        if self._root:
            try:
                self._root.withdraw()
            except Exception:
                pass

    def _show_unsafe(self) -> None:
        if self._root:
            try:
                self._root.deiconify()
            except Exception:
                pass

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()


class TranscriptOverlay:
    """
    Public API for TTS transcript overlay + conversation history display.

    v4.5.0 §7.5.6:
    - show_sentence / highlight_word / clear / hide / show
    - add_user_message / add_assistant_message (conversation mode)
    - Crash recovery via 60-second rebuild watchdog
    - Runs in dedicated thread so audio is never blocked
    """

    def __init__(self, config: dict[str, object] | None = None) -> None:
        self._config = config if config is not None else _load_config()
        self._window: _OverlayWindow | None = None
        self._watchdog_thread: threading.Thread | None = None
        self._watchdog_alive = False
        self._lock = threading.Lock()
        self._conversation_history: list[tuple[str, str]] = []

        if self._config.get("enabled", True):
            self._start_window()
            self._start_watchdog()

    def _start_window(self) -> None:
        with self._lock:
            if self._window is not None and self._window.is_alive:
                return
            self._window = _OverlayWindow(self._config)
            ok = self._window.start()
            if ok:
                logger.info("TranscriptOverlay window started")
            else:
                logger.warning("TranscriptOverlay window could not start (tkinter unavailable?)")

    def _start_watchdog(self) -> None:
        self._watchdog_alive = True
        self._watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True)
        self._watchdog_thread.start()

    def _watchdog_loop(self) -> None:
        """Every 60 s, ensure the window is alive; rebuild if necessary."""
        while self._watchdog_alive:
            time.sleep(_REBUILD_INTERVAL_S)
            with self._lock:
                win = self._window
            if win is None or not win.is_alive:
                logger.warning("TranscriptOverlay window not alive — rebuilding")
                try:
                    self._start_window()
                except Exception as exc:
                    logger.warning(f"TranscriptOverlay rebuild failed: {exc}")

    def show_sentence(self, text: str) -> None:
        """Display a full sentence, replacing current content."""
        with self._lock:
            win = self._window
        if win:
            try:
                win.show_sentence(text)
            except Exception as exc:
                logger.warning(f"TranscriptOverlay.show_sentence failed: {exc}")

    def highlight_word(self, word_index: int) -> None:
        """Highlight word at index (requires TTS word timestamps)."""
        with self._lock:
            win = self._window
        if win:
            try:
                win.highlight_word(word_index)
            except Exception as exc:
                logger.warning(f"TranscriptOverlay.highlight_word failed: {exc}")

    def clear(self) -> None:
        """Clear window content."""
        with self._lock:
            win = self._window
        if win:
            try:
                win.clear()
            except Exception as exc:
                logger.warning(f"TranscriptOverlay.clear failed: {exc}")

    def hide(self) -> None:
        """Hide the overlay window (user idle timeout)."""
        with self._lock:
            win = self._window
        if win:
            try:
                win.hide()
            except Exception as exc:
                logger.warning(f"TranscriptOverlay.hide failed: {exc}")

    def show(self) -> None:
        """Show the overlay window."""
        with self._lock:
            win = self._window
        if win:
            try:
                win.show()
            except Exception as exc:
                logger.warning(f"TranscriptOverlay.show failed: {exc}")

    def add_user_message(self, text: str) -> None:
        """Record a user message and refresh the conversation display."""
        if not self._config.get("conversation_enabled", True):
            return
        with self._lock:
            self._conversation_history.append(("user", text))
        self._refresh_conversation()

    def add_assistant_message(self, text: str) -> None:
        """Record an assistant message and refresh the conversation display."""
        if not self._config.get("conversation_enabled", True):
            return
        logger.info("subtitle add_assistant_message: %s", text[:40])
        with self._lock:
            self._conversation_history.append(("assistant", text))
        self._refresh_conversation()

    def update_last_assistant_message(self, text: str) -> None:
        if not self._config.get("conversation_enabled", True):
            return
        logger.info("subtitle update_last_assistant_message: %s", text[:40])
        with self._lock:
            for i in range(len(self._conversation_history) - 1, -1, -1):
                if self._conversation_history[i][0] == "assistant":
                    self._conversation_history[i] = ("assistant", text)
                    break
        self._refresh_conversation()

    def _render_conversation(self, history: list[tuple[str, str]]) -> str:
        """Format conversation history into multi-line text with role icons."""
        max_lines = int(self._config.get("max_conversation_lines", 10))  # type: ignore[arg-type]
        limited = history[-max_lines:] if len(history) > max_lines else history
        lines: list[str] = []
        for role, msg in limited:
            if role == "user":
                lines.append(f"👤 用户: {msg}")
            else:
                lines.append(f"🤖 雪奈: {msg}")
        return "\n".join(lines)

    def _refresh_conversation(self) -> None:
        """Re-render and push conversation text to overlay window."""
        with self._lock:
            rendered = self._render_conversation(self._conversation_history)
            win = self._window
        if win:
            try:
                win.show_conversation(rendered)
            except Exception as exc:
                logger.warning(f"TranscriptOverlay._refresh_conversation failed: {exc}")

    def stop(self) -> None:
        """Gracefully stop overlay and watchdog."""
        self._watchdog_alive = False
        with self._lock:
            win = self._window
        if win:
            win.stop()
