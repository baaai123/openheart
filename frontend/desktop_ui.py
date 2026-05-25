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
L2D_WIN_PATH = "D:\\electron-l2d"        # Windows path for electron-l2d
L2D_WSL_PATH = "/mnt/d/electron-l2d"     # WSL mount of the same

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
        self._voice_proc: subprocess.Popen[str] | None = None
        self._running = False

        # Step progress tracking for docker → L2D → voice launch flow
        self._step_statuses: dict[str, bool | None] = {
            "docker": None,
            "l2d":    None,
            "voice":  None,
        }

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
        for col, name in enumerate(["Docker", "L2D", "Voice", "Visual"]):
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
            start_frame, mode="determinate", length=160, maximum=100
        )
        self._progress.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Step status label
        self._status_var = tk.StringVar(value="Idle")
        ttk.Label(start_frame, textvariable=self._status_var,
                  font=("Segoe UI", 8)).pack(side=tk.LEFT, padx=(8, 0))

        # Step indicator label (e.g. "① Docker Compose  ✓ ② L2D  ...")
        self._step_var = tk.StringVar(value="")
        ttk.Label(start_frame, textvariable=self._step_var,
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

    # ── Launch flow: Docker Compose → L2D ────────────────────────────
    def _start_backend(self):
        if self._running:
            return

        self._running = True
        self._start_btn.configure(state=tk.DISABLED, text="Starting...")
        self._status_var.set("Launching services...")
        self._progress["value"] = 0

        self._step_statuses = {"docker": None, "l2d": None, "voice": None}
        for key, dot in self._dots.items():
            if key in self._step_statuses:
                dot.set(None)

        self._log("=== Launch Flow: Docker Compose → L2D → Voice Pipeline ===\n")

        def _launch_flow():
            try:
                self._step_docker_compose()
                self._step_start_l2d()
                self._step_start_voice()
            finally:
                self.root.after(0, self._on_flow_finished)

        threading.Thread(target=_launch_flow, daemon=True).start()

    # ── Step 1: Docker Compose ───────────────────────────────────────
    def _step_docker_compose(self):
        def ui(msg, prog, ok):
            self._step_statuses["docker"] = ok
            self.root.after(0, lambda: (
                self._status_var.set(msg),
                self._progress.configure(value=prog),
                self._step_var.set("① Docker " + ("✓" if ok else "✗") + "  ② L2D …  ③ Voice …"),
                self._log(f"[Docker] {msg}\n"),
                self._dots["docker"].set(ok),
            ))

        ui("Docker Compose: starting...", 10, None)
        try:
            result = subprocess.run(
                ["docker", "compose", "--profile", "infra", "up", "-d"],
                cwd=str(ROOT),
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode == 0:
                ui("Docker Compose: online", 50, True)
            else:
                err = result.stderr.strip() or result.stdout.strip() or "exit code %d" % result.returncode
                ui(f"Docker Compose: failed — {err}", 50, False)
        except FileNotFoundError:
            ui("Docker Compose: 'docker' not found in PATH", 50, False)
        except subprocess.TimeoutExpired:
            ui("Docker Compose: timed out (120s)", 50, False)
        except Exception as exc:
            ui(f"Docker Compose: error — {exc}", 50, False)

    # ── Step 2: L2D from D:\electron-l2d ────────────────────────────
    def _step_start_l2d(self):
        def ui(msg, prog, ok):
            self._step_statuses["l2d"] = ok
            d = self._step_statuses.get("docker")
            d_icon = "✓" if d is True else ("✗" if d is False else "…")
            self.root.after(0, lambda: (
                self._status_var.set(msg),
                self._progress.configure(value=prog),
                self._step_var.set(f"① Docker {d_icon}  ② L2D " + ("✓" if ok else "✗") + "  ③ Voice …"),
                self._log(f"[L2D] {msg}\n"),
                self._dots["l2d"].set(ok),
            ))

        ui("L2D: starting...", 60, None)
        l2d_path = Path(L2D_WSL_PATH)
        if not l2d_path.exists():
            ui(f"L2D: path not found — {L2D_WIN_PATH}", 80, False)
            return

        try:
            self._proc = subprocess.Popen(
                ["cmd.exe", "/c", "start", "L2D", "/d", L2D_WIN_PATH,
                 "cmd", "/k", "npm", "start"],
                cwd=str(l2d_path),
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                text=True,
            )
            ui("L2D: launched", 100, True)
        except Exception as exc:
            ui(f"L2D: error — {exc}", 80, False)

    # ── Step 3: Voice pipeline (demo_full.py) ────────────────────────
    def _step_start_voice(self):
        def ui(msg, prog, ok):
            self._step_statuses["voice"] = ok
            d = self._step_statuses.get("docker")
            l = self._step_statuses.get("l2d")
            d_icon = "✓" if d is True else ("✗" if d is False else "…")
            l_icon = "✓" if l is True else ("✗" if l is False else "…")
            self.root.after(0, lambda: (
                self._status_var.set(msg),
                self._progress.configure(value=prog),
                self._step_var.set(f"① Docker {d_icon}  ② L2D {l_icon}  ③ Voice " + ("✓" if ok else "✗")),
                self._log(f"[Voice] {msg}\n"),
                self._dots["voice"].set(ok),
            ))

        demo_path = ROOT / "scripts" / "demo_full.py"
        if not demo_path.exists():
            ui(f"demo_full.py not found — cannot start voice pipeline", 90, False)
            return

        ui("Voice pipeline: starting...", 85, None)

        # Export env from UI fields so the subprocess inherits them
        env = os.environ.copy()
        env["DEEPSEEK_BASE_URL"] = self.baseurl_var.get()
        env["DEEPSEEK_MODEL"]    = self.model_var.get()
        env["DEEPSEEK_API_KEY"]  = self.apikey_var.get()

        try:
            self._voice_proc = subprocess.Popen(
                ["conda", "run", "-n", "cv311", "python",
                 str(demo_path), "--mode", "voice"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, env=env,
            )
            ui("Voice pipeline: running", 100, True)

            # Read and forward output in background
            def _pipe_output():
                vp = self._voice_proc
                if not vp:
                    return
                try:
                    for line in vp.stdout or []:
                        if line:
                            self.root.after(0, self._log, line)
                    vp.wait()
                except Exception:
                    pass
                finally:
                    self.root.after(0, lambda: self._dots["voice"].set(False))

            threading.Thread(target=_pipe_output, daemon=True).start()

        except Exception as exc:
            ui(f"Voice pipeline: error — {exc}", 90, False)

    def _on_flow_finished(self):
        self._running = False
        self._start_btn.configure(state=tk.NORMAL, text="START", style="Success.TButton")
        all_ok = all(v is True for v in self._step_statuses.values())
        self._status_var.set("All services online" if all_ok else "Launch incomplete")
        self._log("=== Launch flow complete ===\n" if all_ok
                   else "=== Launch flow finished with errors ===\n")
        self._proc = None

    def _log(self, msg: str):
        """Append line to the chat log (thread-safe via root.after)."""
        self._chat.configure(state=tk.NORMAL)
        self._chat.insert(tk.END, msg)
        self._chat.see(tk.END)
        self._chat.configure(state=tk.DISABLED)

    # ── Cleanup ──────────────────────────────────────────────────────
    def _on_close(self):
        # L2D was launched via cmd.exe /c start (detached window).
        # self._proc (cmd.exe) should already have exited — safe-grace
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        # Voice pipeline needs active cleanup
        if self._voice_proc and self._voice_proc.poll() is None:
            self._voice_proc.terminate()
            try:
                self._voice_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._voice_proc.kill()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    DesktopUI().run()
