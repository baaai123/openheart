#!/usr/bin/env python3
"""OpenHeart Desktop UI — tkinter control panel. v5.x
Provides API configuration, persona editing, module status/control,
visual preview, chat log, and one-click backend startup.
"""
from __future__ import annotations

import os
import re
import subprocess
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from pathlib import Path

# ── Constants ──────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env"

# Dark palette
BG        = "#1e1e1e"
BG2       = "#2d2d2d"
FG        = "#d4d4d4"
SELECT_BG = "#264f78"
BTN_BG    = "#0e639c"
BTN_FG    = "#ffffff"
ENTRY_BG  = "#3c3c3c"
GREEN     = "#4ec04e"
RED       = "#f14c4c"
CHAT_BG   = "#111111"

FONT      = ("Segoe UI", 10)
FONT_MONO = ("Consolas", 10)


class StatusDot(tk.Canvas):
    """Small green/red/grey circle indicator."""
    def __init__(self, parent: tk.Widget, size: int = 16, **kw):
        super().__init__(parent, width=size, height=size,
                         highlightthickness=0, bg=BG, **kw)
        self._dot = self.create_oval(2, 2, size - 2, size - 2,
                                     fill="#555", outline="")
        self.set(None)

    def set(self, ok: bool | None):
        fill = GREEN if ok else (RED if ok is False else "#555")
        self.itemconfig(self._dot, fill=fill)


class DesktopUI:
    """Main application window."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("OpenHeart Desktop UI")
        self.root.geometry("1000x700")
        self.root.minsize(850, 550)

        self._proc: subprocess.Popen[str] | None = None
        self._running = False

        # API config variables
        self.baseurl_var = tk.StringVar()
        self.model_var   = tk.StringVar()
        self.apikey_var  = tk.StringVar()
        self.show_key    = False

        # Module switch variables
        self.voice_var  = tk.BooleanVar(value=True)
        self.visual_var = tk.BooleanVar(value=True)
        self.l2d_var    = tk.BooleanVar(value=True)

        # Visual preview
        self._preview_var = tk.StringVar(
            value="Concepts: (waiting)\nOCR: (waiting)"
        )

        self._setup_style()
        self._build_ui()
        self._load_env()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── Dark theme ───────────────────────────────────────────────────
    def _setup_style(self):
        style = ttk.Style()
        style.theme_use("clam")

        style.configure(".", background=BG, foreground=FG,
                        fieldbackground=ENTRY_BG, font=FONT)
        style.configure("TFrame", background=BG)
        style.configure("TLabel", background=BG, foreground=FG)
        style.configure("TLabelframe", background=BG, foreground=FG,
                        fieldbackground=BG)
        style.configure("TLabelframe.Label", background=BG, foreground=FG)
        style.configure("TButton", background=BTN_BG, foreground=BTN_FG,
                        bordercolor=BG, lightcolor=BG, darkcolor=BG)
        style.map("TButton",
                  background=[("active", "#1177bb"), ("disabled", "#555")],
                  foreground=[("disabled", "#999")])
        style.configure("Success.TButton", background="#2d8c2d")
        style.map("Success.TButton",
                  background=[("active", "#3aa83a"), ("disabled", "#555")])
        style.configure("TCheckbutton", background=BG, foreground=FG)
        style.map("TCheckbutton",
                  background=[("active", BG2)],
                  indicatorcolor=[("selected", BTN_BG)])
        style.configure("TEntry", fieldbackground=ENTRY_BG, foreground=FG,
                        insertcolor=FG)
        style.configure("TProgressbar", background=BTN_BG,
                        troughcolor=BG2, bordercolor=BG)
        style.configure("Horizontal.TProgressbar", background=BTN_BG,
                        troughcolor=BG2)

    # ── Build UI ─────────────────────────────────────────────────────
    def _build_ui(self):
        # ── Top row: left (config) + right (status/controls) ──
        top = ttk.Frame(self.root)
        top.pack(fill=tk.BOTH, expand=True, padx=6, pady=(6, 2))

        # -- Left panel: configuration --
        left = ttk.LabelFrame(top, text="Configuration", width=320)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)
        left.pack_propagate(False)

        # API config
        api_lf = ttk.LabelFrame(left, text="API Config")
        api_lf.pack(fill=tk.X, padx=4, pady=(4, 2))
        api_lf.columnconfigure(1, weight=1)

        self._api_fields: list[ttk.Entry] = []
        for i, (label, var) in enumerate([
            ("Base URL:", self.baseurl_var),
            ("Model:",    self.model_var),
            ("API Key:",  self.apikey_var),
        ]):
            ttk.Label(api_lf, text=label).grid(
                row=i, column=0, padx=4, pady=2, sticky="w")
            show = "*" if label == "API Key:" else ""
            ent = ttk.Entry(api_lf, textvariable=var, show=show)
            ent.grid(row=i, column=1, padx=4, pady=2, sticky="ew")
            self._api_fields.append(ent)

        # API key visibility toggle
        ttk.Button(api_lf, text="Show/Hide Key",
                   command=self._toggle_key_visible).grid(
            row=1, column=2, padx=(0, 4), pady=2, sticky="w")
        # Load .env button
        ttk.Button(api_lf, text="Load from .env",
                   command=self._load_env).grid(
            row=2, column=0, columnspan=3, pady=(0, 4))

        # Persona / system prompt
        persona_lf = ttk.LabelFrame(left, text="Persona / System Prompt")
        persona_lf.pack(fill=tk.BOTH, expand=True, padx=4, pady=(2, 4))

        self._persona_text = tk.Text(
            persona_lf, wrap=tk.WORD,
            bg=ENTRY_BG, fg=FG, insertbackground=FG,
            font=FONT_MONO, relief=tk.FLAT, borderwidth=0,
            padx=6, pady=6,
        )
        self._persona_text.pack(fill=tk.BOTH, expand=True)
        self._persona_text.insert("1.0",
            "You are OpenHeart, an empathetic AI companion. "
            "Respond naturally and warmly."
        )

        # -- Right panel: status, switches, preview, start --
        right = ttk.Frame(top)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6, 0))

        # Status indicators
        stat_lf = ttk.LabelFrame(right, text="Module Status")
        stat_lf.pack(fill=tk.X, padx=4, pady=(0, 4))

        self._dots: dict[str, StatusDot] = {}
        for col, name in enumerate(["Voice", "Visual", "L2D"]):
            f = ttk.Frame(stat_lf)
            f.grid(row=0, column=col, padx=8, pady=4, sticky="w")
            dot = StatusDot(f)
            dot.pack(side=tk.LEFT, padx=(0, 4))
            ttk.Label(f, text=name).pack(side=tk.LEFT)
            self._dots[name.lower()] = dot

        # Module switches
        sw_lf = ttk.LabelFrame(right, text="Module Switches")
        sw_lf.pack(fill=tk.X, padx=4, pady=(0, 4))

        for var, text in [
            (self.voice_var,  "Voice Input"),
            (self.visual_var, "Visual Module"),
            (self.l2d_var,    "L2D"),
        ]:
            ttk.Checkbutton(sw_lf, text=text, variable=var).pack(
                anchor="w", padx=6, pady=1)

        # Visual preview
        prev_lf = ttk.LabelFrame(right, text="Visual Preview")
        prev_lf.pack(fill=tk.BOTH, expand=True, padx=4, pady=(0, 4))

        self._preview_lbl = ttk.Label(
            prev_lf, textvariable=self._preview_var,
            font=FONT_MONO, anchor="nw", justify=tk.LEFT,
            wraplength=320, padding=6,
        )
        self._preview_lbl.pack(fill=tk.BOTH, expand=True)

        # Start button + progress bar
        start_frame = ttk.Frame(right)
        start_frame.pack(fill=tk.X, padx=4, pady=(0, 4))

        self._start_btn = ttk.Button(
            start_frame, text="START",
            command=self._start_backend,
            style="Success.TButton",
        )
        self._start_btn.pack(side=tk.LEFT, padx=(0, 8))

        self._progress = ttk.Progressbar(
            start_frame, mode="indeterminate", length=160
        )
        self._progress.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Status label
        self._status_var = tk.StringVar(value="Idle")
        ttk.Label(start_frame, textvariable=self._status_var,
                  font=("Segoe UI", 8)).pack(side=tk.LEFT, padx=(8, 0))

        # ── Chat display ──
        chat_lf = ttk.LabelFrame(self.root, text="Chat / LLM Responses")
        chat_lf.pack(fill=tk.BOTH, expand=True, padx=6, pady=(2, 6))

        self._chat = scrolledtext.ScrolledText(
            chat_lf, wrap=tk.WORD,
            bg=CHAT_BG, fg=FG, insertbackground=FG,
            font=FONT_MONO, state=tk.DISABLED,
            padx=8, pady=6,
        )
        self._chat.pack(fill=tk.BOTH, expand=True)
        # Style the scrollbar for dark theme
        try:
            self._chat.vbar.configure(
                troughcolor=BG2, bg=BG2, activebackground="#555",
                highlightbackground=BG2,
            )
        except tk.TclError:
            pass  # some platforms don't support all options

    # ── .env loading ─────────────────────────────────────────────────
    def _load_env(self):
        """Parse .env and populate API config fields."""
        if not ENV_FILE.exists():
            return
        try:
            text = ENV_FILE.read_text(encoding="utf-8")
        except OSError:
            return
        mapping: dict[str, str] = {}
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            mapping[key.strip()] = val.strip().strip("\"'")

        env_key_map = {
            "DEEPSEEK_BASE_URL": self.baseurl_var,
            "DEEPSEEK_MODEL":    self.model_var,
            "DEEPSEEK_API_KEY":  self.apikey_var,
        }
        for env_key, var in env_key_map.items():
            if env_key in mapping:
                var.set(mapping[env_key])

    def _toggle_key_visible(self):
        self.show_key = not self.show_key
        show = "" if self.show_key else "*"
        for ent in self._api_fields:
            # Only toggle the API key entry (3rd field)
            try:
                if ent.cget("show") == "*" or ent.cget("show") == "":
                    ent.configure(show=show)
            except tk.TclError:
                pass

    # ── Backend lifecycle ────────────────────────────────────────────
    def _start_backend(self):
        if self._running:
            # Toggle off: stop the process
            if self._proc:
                self._proc.terminate()
            return

        sh_path = ROOT / "run_backend.sh"
        if not sh_path.exists():
            messagebox.showerror("Start Error",
                f"run_backend.sh not found:\n{sh_path}")
            return

        self._running = True
        self._start_btn.configure(text="STOP", style="TButton")
        self._status_var.set("Starting...")
        self._progress.start(15)
        self._log("Starting backend...\n")

        def _reader():
            try:
                # Export env vars from UI fields so the subprocess inherits them
                env = os.environ.copy()
                env["DEEPSEEK_BASE_URL"] = self.baseurl_var.get()
                env["DEEPSEEK_MODEL"]    = self.model_var.get()
                env["DEEPSEEK_API_KEY"]  = self.apikey_var.get()

                self._proc = subprocess.Popen(
                    ["bash", str(sh_path)],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1, env=env,
                )
                for line in self._proc.stdout or []:
                    if line:
                        self.root.after(0, self._log, line)
                self._proc.wait()
            except Exception as exc:
                self.root.after(0, self._log, f"Error: {exc}\n")
            finally:
                self.root.after(0, self._on_backend_stopped)

        threading.Thread(target=_reader, daemon=True).start()

    def _on_backend_stopped(self):
        self._running = False
        self._progress.stop()
        self._start_btn.configure(text="START", style="Success.TButton")
        self._status_var.set("Stopped")
        self._log("Backend stopped.\n")
        self._proc = None

    def _log(self, msg: str):
        """Append line to the chat log (thread-safe via root.after)."""
        self._chat.configure(state=tk.NORMAL)
        self._chat.insert(tk.END, msg)
        self._chat.see(tk.END)
        self._chat.configure(state=tk.DISABLED)

    # ── Cleanup ──────────────────────────────────────────────────────
    def _on_close(self):
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    DesktopUI().run()
