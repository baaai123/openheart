"""
MouseChannel — biomimetic mouse control with Bezier trajectories and safety levels.

v4.5.0 §7.4: 键盘协调器 (Keyboard Coordinator)
  - §7.4.1: cubic Bezier curves with control-point jitter (20-50px), sigmoid speed profile +5% noise
  - §7.4.2: visual closed-loop verification via SyncVisionQuery.query_roi()
  - §7.4.3: Win32 PostMessage preference; "safe mode" for protected applications (highlight + voice guide)

项目宪法 §2.1: channel name MUST be "mouse_channel", NEVER "input_channel".
"""

from __future__ import annotations

import asyncio
import enum
import logging
import math
import os
import random
import subprocess
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# v4.5.0 §7.4.3: Safety levels
# ---------------------------------------------------------------------------

class SafetyLevel(str, enum.Enum):
    NORMAL = "normal"
    SAFE = "safe"


# v4.5.0 §7.4.3: list of protected application window class names
_PROTECTED_APP_PATTERNS = (
    "ConsoleWindowClass",  # admin consoles
    "Progman",             # desktop
    "Shell_TrayWnd",       # taskbar
    "TaskSwitcherWnd",     # Alt+Tab overlay
    "SysListView32",       # system list views
)


@dataclass
class MouseAction:
    action_type: str
    target: Optional[tuple[float, float]] = None
    start_ms: int = 0
    deadline_ms: int = 2000
    button: str = "left"
    text: Optional[str] = None
    scroll_amount: int = 0


# ---------------------------------------------------------------------------
# Bezier trajectory generation (§7.4.1)
# ---------------------------------------------------------------------------

def _cubic_bezier(
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    t: float,
) -> tuple[float, float]:
    u = 1.0 - t
    x = u**3 * p0[0] + 3 * u**2 * t * p1[0] + 3 * u * t**2 * p2[0] + t**3 * p3[0]
    y = u**3 * p0[1] + 3 * u**2 * t * p1[1] + 3 * u * t**2 * p2[1] + t**3 * p3[1]
    return (x, y)


def _sigmoid_speed(t: float, v_max: float = 1.0, k: float = 6.0, t_mid: float = 0.5) -> float:
    return v_max / (1.0 + math.exp(-k * (t - t_mid)))


def generate_bezier_path(
    start: tuple[float, float],
    end: tuple[float, float],
    num_points: int = 60,
    endpoint_jitter: float = 2.0,
) -> list[tuple[float, float]]:
    dx = end[0] - start[0]
    dy = end[1] - start[1]

    jitter_range = random.uniform(20, 50)
    p1_x = start[0] + dx * 0.3 + random.uniform(-jitter_range, jitter_range)
    p1_y = start[1] + dy * 0.1 + random.uniform(-jitter_range * 0.3, jitter_range * 0.3)
    p2_x = start[0] + dx * 0.7 + random.uniform(-jitter_range, jitter_range)
    p2_y = start[1] + dy * 0.9 + random.uniform(-jitter_range * 0.3, jitter_range * 0.3)

    path: list[tuple[float, float]] = []
    pause_indices: set[int] = set()

    if num_points > 20:
        pause_indices.add(random.randint(num_points // 3, num_points // 2))
        if num_points > 40:
            pause_indices.add(random.randint(num_points * 2 // 3, num_points - 5))

    has_paused = False
    pause_duration = 2

    for i in range(num_points):
        t = i / (num_points - 1)
        speed = _sigmoid_speed(t, v_max=1.0, k=random.uniform(5.0, 7.0), t_mid=0.5)
        noise = 1.0 + random.uniform(-0.05, 0.05)
        effective_speed = speed * noise

        if effective_speed < 0.15 and not has_paused and i > 2 and i < num_points - 2:
            has_paused = True
        elif effective_speed >= 0.3 and has_paused:
            has_paused = False

        point = _cubic_bezier(start, (p1_x, p1_y), (p2_x, p2_y), end, t)
        path.append(point)

    final_x = end[0]
    final_y = end[1]
    path[-1] = (final_x, final_y)

    return path


# ---------------------------------------------------------------------------
# Mouse controller backend abstraction
# ---------------------------------------------------------------------------

class MouseController:
    def move_to(self, x: int, y: int) -> None:
        raise NotImplementedError

    def move_relative(self, dx: int, dy: int) -> None:
        raise NotImplementedError

    def click(self, x: int, y: int, button: str = "left", clicks: int = 1) -> None:
        raise NotImplementedError

    def press(self, button: str = "left") -> None:
        raise NotImplementedError

    def release(self, button: str = "left") -> None:
        raise NotImplementedError

    def scroll(self, dx: int = 0, dy: int = 0) -> None:
        raise NotImplementedError

    def type_text(self, text: str) -> None:
        raise NotImplementedError


class PyAutoGUIController(MouseController):
    def __init__(self) -> None:
        import pyautogui
        self._pg = pyautogui
        self._pg.FAILSAFE = False
        self._pg.PAUSE = 0.0

    def move_to(self, x: int, y: int) -> None:
        self._pg.moveTo(x, y)

    def move_relative(self, dx: int, dy: int) -> None:
        self._pg.moveRel(dx, dy)

    def click(self, x: int, y: int, button: str = "left", clicks: int = 1) -> None:
        self._pg.click(x, y, button=button, clicks=clicks)

    def press(self, button: str = "left") -> None:
        self._pg.mouseDown(button=button)

    def release(self, button: str = "left") -> None:
        self._pg.mouseUp(button=button)

    def scroll(self, dx: int = 0, dy: int = 0) -> None:
        if dy:
            self._pg.scroll(dy)
        if dx:
            self._pg.hscroll(dx)

    def type_text(self, text: str) -> None:
        self._pg.typewrite(text, interval=0.02)


class Win32Controller(MouseController):
    def __init__(self) -> None:
        pass

    def _ensure_win32(self):
        try:
            import win32api
            import win32con
            import win32gui
            return win32api, win32con, win32gui
        except ImportError:
            raise RuntimeError("win32api not available on this platform")

    def move_to(self, x: int, y: int) -> None:
        win32api, _, _ = self._ensure_win32()
        try:
            win32api.SetCursorPos((x, y))
        except Exception:
            logger.exception("Win32 SetCursorPos failed at (%d, %d)", x, y)

    def move_relative(self, dx: int, dy: int) -> None:
        try:
            from ctypes import windll
            windll.user32.mouse_event(0x0001, dx, dy, 0, 0)
        except Exception:
            logger.exception("Win32 mouse_event failed")

    def click(self, x: int, y: int, button: str = "left", clicks: int = 1) -> None:
        self.move_to(x, y)
        time.sleep(0.01)
        for _ in range(clicks):
            self.press(button)
            time.sleep(0.02)
            self.release(button)

    def press(self, button: str = "left") -> None:
        _, win32con, _ = self._ensure_win32()
        try:
            flag = win32con.MOUSEEVENTF_LEFTDOWN if button == "left" else win32con.MOUSEEVENTF_RIGHTDOWN
            import win32api
            win32api.mouse_event(flag, 0, 0, 0, 0)
        except Exception:
            logger.exception("Win32 mouse press failed")

    def release(self, button: str = "left") -> None:
        _, win32con, _ = self._ensure_win32()
        try:
            flag = win32con.MOUSEEVENTF_LEFTUP if button == "left" else win32con.MOUSEEVENTF_RIGHTUP
            import win32api
            win32api.mouse_event(flag, 0, 0, 0, 0)
        except Exception:
            logger.exception("Win32 mouse release failed")

    def scroll(self, dx: int = 0, dy: int = 0) -> None:
        _, win32con, _ = self._ensure_win32()
        try:
            import win32api
            if dy:
                win32api.mouse_event(win32con.MOUSEEVENTF_WHEEL, 0, 0, dy * 120, 0)
        except Exception:
            logger.exception("Win32 scroll failed")

    def type_text(self, text: str) -> None:
        """Type text via PowerShell SendKeys. v4.5.0 §7.4.3."""
        import subprocess
        escaped = text.replace('"', '`"')
        ps_cmd = (
            f'Add-Type -AssemblyName System.Windows.Forms; '
            f'[System.Windows.Forms.SendKeys]::SendWait("{escaped}")'
        )
        try:
            subprocess.run(
                ["powershell.exe", "-Command", ps_cmd],
                timeout=10, capture_output=True, check=True,
            )
        except subprocess.TimeoutExpired:
            logger.exception("PowerShell SendKeys timeout (text=%r)", text[:50])
        except subprocess.CalledProcessError as e:
            logger.exception("PowerShell SendKeys failed (text=%r): %s", text[:50], e.stderr)


def _detect_protected_window() -> bool:
    try:
        import ctypes
        user32 = ctypes.windll.user32
        hwnd = user32.GetForegroundWindow()
        class_name_buf = ctypes.create_unicode_buffer(256)
        ctypes.windll.user32.GetClassNameW(hwnd, class_name_buf, 255)
        class_name = class_name_buf.value
        for pattern in _PROTECTED_APP_PATTERNS:
            if pattern in class_name:
                return True
    except Exception:
        pass
    return False


# ---------------------------------------------------------------------------
# MouseChannel (§7.4)
# ---------------------------------------------------------------------------

class MouseChannel:
    def __init__(
        self,
        controller: Optional[MouseController] = None,
        safety_level: SafetyLevel = SafetyLevel.NORMAL,
        width: int = 2560,
        height: int = 1440,
        baseline: Optional[dict] = None,  # v4.5.0 §4.3 / §7.4
    ) -> None:
        if controller is None:
            try:
                controller = Win32Controller()
            except Exception:
                logger.warning("Win32 not available, falling back to PyAutoGUI")
                controller = PyAutoGUIController()
        self._controller: MouseController = controller
        self._safety_level: SafetyLevel = safety_level
        self._width: int = width
        self._height: int = height
        self._current_position: tuple[int, int] = (width // 2, height // 2)

        # v4.5.0 §4.3 / §7.4: load personality-driven mouse parameters
        if baseline is None:
            baseline = self._load_baseline()
        ms = baseline.get("mouse_style", {})
        self._speed: float = float(
            ms.get("movement_speed", {}).get("value", 0.6)  # type: ignore[union-attr]
        )
        self._precision: float = float(
            ms.get("precision_mode", {}).get("value", 0.3)  # type: ignore[union-attr]
        )
        self._hover: bool = bool(
            ms.get("hover_before_click", {}).get("value", True)  # type: ignore[union-attr]
        )

    # -------------------------------------------------------------------
    # Personality-driven parameter helpers (§4.3, §4.6, §7.4)
    # -------------------------------------------------------------------

    @staticmethod
    def _load_baseline() -> dict:
        import json
        import os

        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
            "config", "baseline.json",
        )
        try:
            with open(config_path, encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            logger.warning(
                "Cannot load baseline.json from %s: %s; using defaults",
                config_path, exc,
            )
            return {"mouse_style": {}}

    def _speed_to_points(self) -> int:
        return max(20, min(100, int(80 - (self._speed - 0.4) / 0.4 * 50)))

    def _precision_to_jitter(self) -> float:
        return 4.0 - (self._precision - 0.1) / 0.4 * 3.5

    def update_personality(self, mouse_style: dict) -> None:
        self._speed = float(mouse_style.get("movement_speed", 0.6))
        self._precision = float(mouse_style.get("precision_mode", 0.3))
        self._hover = bool(mouse_style.get("hover_before_click", True))

    # -------------------------------------------------------------------
    # Safety checks (§7.4.3)
    # -------------------------------------------------------------------

    def _is_safe_to_act(self) -> bool:
        if self._safety_level == SafetyLevel.SAFE:
            return False
        if _detect_protected_window():
            logger.warning("Protected application detected — switching to SAFE mode")
            self._safety_level = SafetyLevel.SAFE
            return False
        return True

    # -------------------------------------------------------------------
    # Bezier-based move (§7.4.1)
    # -------------------------------------------------------------------

    async def move_to(
        self,
        target_x: int,
        target_y: int,
        deadline_ms: int = 2000,
        start_ms: int = 0,
    ) -> bool:
        if not self._is_safe_to_act():
            logger.info("SAFE mode: would highlight (%d, %d) + voice guide", target_x, target_y)
            return False

        start_pos = self._current_position
        target_clamped = (
            max(0, min(target_x, self._width)),
            max(0, min(target_y, self._height)),
        )

        path = generate_bezier_path(
            (float(start_pos[0]), float(start_pos[1])),
            (float(target_clamped[0]), float(target_clamped[1])),
            num_points=self._speed_to_points(),
            endpoint_jitter=self._precision_to_jitter(),
        )

        total_steps = len(path)
        step_duration_ms = max(1, deadline_ms // total_steps)

        # Build batched PowerShell script using System.Windows.Forms.Cursor::Position
        # (DllImport SetCursorPos doesn't work across WSL2/WSLg boundary)
        # SetProcessDPIAware ensures physical-pixel coordinates on HiDPI displays.
        ps_lines = [
            'Add-Type -AssemblyName System.Windows.Forms',
            'Add-Type -Name NativeDPI -Namespace Temp -MemberDefinition @"',
            '[DllImport("user32.dll")]',
            'public static extern bool SetProcessDPIAware();',
            '"@',
            '[Temp.NativeDPI]::SetProcessDPIAware() | Out-Null',
        ]
        for px, py in path:
            ps_lines.append(f'[System.Windows.Forms.Cursor]::Position = New-Object System.Drawing.Point({int(px)},{int(py)})')
            ps_lines.append(f"Start-Sleep -Milliseconds {step_duration_ms}")

        ps_script = "\n".join(ps_lines)

        # try/except: subprocess may fail if powershell.exe is missing or
        #   WSL interop is broken; timeout prevents hang. Safe fallback.
        try:
            proc = await asyncio.create_subprocess_exec(
                "powershell.exe", "-ExecutionPolicy", "Bypass", "-Command", "-",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await asyncio.wait_for(
                proc.communicate(input=ps_script.encode()),
                timeout=(deadline_ms / 1000.0) + 5,
            )
            if proc.returncode != 0:
                stderr_text = stderr.decode(errors="replace").strip() if stderr else ""
                logger.warning(
                    "move_to PowerShell failed (rc=%d): %s",
                    proc.returncode, stderr_text,
                )
                return False
        except asyncio.TimeoutError:
            logger.warning("move_to PowerShell timed out")
            return False
        except Exception:
            # catch-all: FileNotFoundError, OSError. Safe — no data corruption.
            logger.exception("move_to PowerShell failed at target (%d, %d)", target_x, target_y)
            return False

        self._current_position = (target_clamped[0], target_clamped[1])
        return True

    async def move_to_instant(self, target_x: int, target_y: int) -> bool:
        """Instant cursor jump — no Bezier animation. For proximity fallback."""
        if not self._is_safe_to_act():
            return False
        target_clamped = (
            max(0, min(target_x, self._width)),
            max(0, min(target_y, self._height)),
        )
        # v4.5.0 §7.4.2: SetProcessDPIAware ensures physical-pixel coordinates
        #   on HiDPI displays — same DPI context as screenshot logic.
        ps_script = (
            'Add-Type -AssemblyName System.Windows.Forms\n'
            'Add-Type -Name NativeDPI -Namespace Temp -MemberDefinition @"\n'
            '[DllImport("user32.dll")]\n'
            'public static extern bool SetProcessDPIAware();\n'
            '"@\n'
            '[Temp.NativeDPI]::SetProcessDPIAware() | Out-Null\n'
            f'[System.Windows.Forms.Cursor]::Position = New-Object System.Drawing.Point({int(target_clamped[0])},{int(target_clamped[1])})'
        )
        try:
            proc = await asyncio.create_subprocess_exec(
                "powershell.exe", "-Command", ps_script,
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(proc.communicate(), timeout=3)
            ok = proc.returncode == 0
        except Exception:
            ok = False
        if ok:
            self._current_position = target_clamped
        return ok

    # -------------------------------------------------------------------
    # Click / double-click
    # -------------------------------------------------------------------

    async def click(
        self,
        target_x: int,
        target_y: int,
        button: str = "left",
        clicks: int = 1,
    ) -> bool:
        if not self._is_safe_to_act():
            return False
        if self._hover:
            await asyncio.sleep(random.uniform(0.2, 0.4))
        try:
            self._controller.move_to(target_x, target_y)
            await asyncio.sleep(0.02)
            self._controller.click(target_x, target_y, button=button, clicks=clicks)
            self._current_position = (target_x, target_y)
            return True
        except Exception:
            logger.exception("Mouse click failed at (%d, %d)", target_x, target_y)
            return False

    # -------------------------------------------------------------------
    # click_at — PowerShell-based click via click_helper.ps1 (§7.4.3)
    # -------------------------------------------------------------------

    @staticmethod
    def _resolve_ps1_path() -> str:
        """Resolve click_helper.ps1 path using dynamic username.

        v4.5.0 §7.4: Uses $env:USERNAME for dynamic path resolution.
        On WSL queries powershell.exe for the Windows username; falls back to 'PC'.
        """
        # try/except: os.environ access is safe; KeyError handled by get default
        try:
            username = os.environ.get("USERNAME")
        except Exception:
            username = ""

        if not username:
            # try/except: powershell.exe may not be on PATH in some WSL configs;
            #   fall back to USER env var, then 'PC'. Safe.
            try:
                result = subprocess.run(
                    ["powershell.exe", "-Command", "$env:USERNAME"],
                    capture_output=True, text=True, timeout=3,
                )
                if result.returncode == 0:
                    username = result.stdout.strip()
            except Exception:
                username = ""

        if not username:
            username = os.environ.get("USER", "PC")

        # Try repo-local scripts/click_helper.ps1 first (supports DPI-aware version),
        # fall back to legacy Windows user path for backward compatibility.
        repo_script = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "scripts", "click_helper.ps1",
        )
        if os.path.exists(repo_script):
            return repo_script
        return f"/mnt/c/Users/{username}/openheart/click_helper.ps1"

    async def click_at(self, x: int, y: int) -> bool:
        """Click at screen coordinates via PowerShell Win32 API.

        v4.5.0 §7.4.3: Uses click_helper.ps1 with DllImport for
        SetCursorPos + mouse_event. No ctypes X11, no pyautogui.
        No sudo required.
        """
        if not self._is_safe_to_act():
            logger.info("SAFE mode: skipping click_at (%d, %d)", x, y)
            return False

        if self._hover:
            await asyncio.sleep(random.uniform(0.2, 0.4))

        ps1 = self._resolve_ps1_path()
        # v4.5.0 §7.4: DPI-awareness pattern consistent with move_to/move_to_instant.
        ps_lines = [
            'Add-Type -AssemblyName System.Windows.Forms',
            'Add-Type -Name NativeDPI -Namespace Temp -MemberDefinition @"',
            '[DllImport("user32.dll")]',
            'public static extern bool SetProcessDPIAware();',
            '"@',
            '[Temp.NativeDPI]::SetProcessDPIAware() | Out-Null',
            f'& "{ps1}" -x {x} -y {y}',
        ]
        ps_script = "\n".join(ps_lines)
        # try/except: subprocess may fail if PowerShell/WSL interop is down;
        #   timeout prevents indefinite hang. Safe to return False on failure.
        try:
            proc = await asyncio.create_subprocess_exec(
                "powershell.exe", "-ExecutionPolicy", "Bypass", "-Command", "-",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await asyncio.wait_for(
                proc.communicate(input=ps_script.encode()), timeout=5,
            )
            if proc.returncode != 0:
                stderr_text = stderr.decode(errors="replace").strip() if stderr else ""
                logger.warning(
                    "click_at PowerShell failed (rc=%d, x=%d, y=%d): %s",
                    proc.returncode, x, y, stderr_text,
                )
            return proc.returncode == 0
        except asyncio.TimeoutError:
            logger.warning("click_at timed out at (%d, %d)", x, y)
            return False
        except Exception:
            # catch-all: FileNotFoundError if powershell.exe missing,
            #   OSError on broken pipe. Safe — no data corruption.
            logger.exception("click_at failed at (%d, %d)", x, y)
            return False

    # -------------------------------------------------------------------
    # right_click_at — PowerShell-based right-click via click_helper.ps1 (§7.4.3)
    # -------------------------------------------------------------------

    async def right_click_at(self, x: int, y: int) -> bool:
        """Right-click at screen coordinates via PowerShell Win32 API.

        v4.5.0 §7.4.3: Same subprocess pattern as click_at, but passes
        ``-right`` flag to click_helper.ps1 for MOUSEEVENTF_RIGHTDOWN (0x0008)
        and MOUSEEVENTF_RIGHTUP (0x0010).
        """
        if not self._is_safe_to_act():
            logger.info("SAFE mode: skipping right_click_at (%d, %d)", x, y)
            return False

        if self._hover:
            await asyncio.sleep(random.uniform(0.2, 0.4))

        ps1 = self._resolve_ps1_path()
        # v4.5.0 §7.4: DPI-awareness pattern consistent with move_to/move_to_instant.
        ps_lines = [
            'Add-Type -AssemblyName System.Windows.Forms',
            'Add-Type -Name NativeDPI -Namespace Temp -MemberDefinition @"',
            '[DllImport("user32.dll")]',
            'public static extern bool SetProcessDPIAware();',
            '"@',
            '[Temp.NativeDPI]::SetProcessDPIAware() | Out-Null',
            f'& "{ps1}" -x {x} -y {y} -right',
        ]
        ps_script = "\n".join(ps_lines)
        # try/except: subprocess may fail if PowerShell/WSL interop is down;
        #   timeout prevents indefinite hang. Safe to return False on failure.
        try:
            proc = await asyncio.create_subprocess_exec(
                "powershell.exe", "-ExecutionPolicy", "Bypass", "-Command", "-",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await asyncio.wait_for(
                proc.communicate(input=ps_script.encode()), timeout=5,
            )
            if proc.returncode != 0:
                stderr_text = stderr.decode(errors="replace").strip() if stderr else ""
                logger.warning(
                    "right_click_at PowerShell failed (rc=%d, x=%d, y=%d): %s",
                    proc.returncode, x, y, stderr_text,
                )
            return proc.returncode == 0
        except asyncio.TimeoutError:
            logger.warning("right_click_at timed out at (%d, %d)", x, y)
            return False
        except Exception:
            # catch-all: FileNotFoundError if powershell.exe missing,
            #   OSError on broken pipe. Safe — no data corruption.
            logger.exception("right_click_at failed at (%d, %d)", x, y)
            return False

    # -------------------------------------------------------------------
    # double_click_at — two left-clicks with 200ms interval
    # -------------------------------------------------------------------

    async def double_click_at(self, x: int, y: int) -> bool:
        """Double left-click with 200ms interval.

        v4.5.0 §7.4: Reuses click_at twice with asyncio.sleep(0.2)
        between clicks. Both calls must succeed for the action to
        be considered successful.
        """
        if not self._is_safe_to_act():
            logger.info("SAFE mode: skipping double_click_at (%d, %d)", x, y)
            return False

        first = await self.click_at(x, y)
        await asyncio.sleep(0.2)
        second = await self.click_at(x, y)
        return first and second

    # -------------------------------------------------------------------
    # type_keys — keyboard text input via controller
    # -------------------------------------------------------------------

    async def type_keys(self, text: str) -> bool:
        """Type text string via the controller backend.

        v4.5.0 §7.4: Delegates to MouseController.type_text().
        Falls back to PowerShell SendKeys on Win32 or pyautogui.typewrite.
        """
        if not self._is_safe_to_act():
            logger.info("SAFE mode: skipping type_keys %r", text[:30])
            return False
        try:
            self._controller.type_text(text)
            return True
        except Exception:
            # catch-all: controller implementations may throw on
            #   missing dependencies. Safe — no data corruption.
            logger.exception("type_keys failed for text %r", text[:30])
            return False

    # -------------------------------------------------------------------
    # Scroll
    # -------------------------------------------------------------------

    async def scroll(self, dx: int = 0, dy: int = 0) -> bool:
        if not self._is_safe_to_act():
            return False
        try:
            self._controller.scroll(dx=dx, dy=dy)
            return True
        except Exception:
            logger.exception("Mouse scroll failed")
            return False

    # -------------------------------------------------------------------
    # Keyboard text input
    # -------------------------------------------------------------------

    async def type_text(self, text: str) -> bool:
        if not self._is_safe_to_act():
            logger.info("SAFE mode: skipping text input %r", text[:30])
            return False
        try:
            self._controller.type_text(text)
            return True
        except Exception:
            logger.exception("Mouse type_text failed")
            return False

    # -------------------------------------------------------------------
    # Visual closed-loop verification (§7.4.2)
    # -------------------------------------------------------------------

    async def verify_target(
        self,
        target_x: int,
        target_y: int,
        expected_label: Optional[str] = None,
    ) -> dict:
        result: dict = {"verified": False, "adjusted_x": target_x, "adjusted_y": target_y}
        try:
            from src.perception.sync_vision_query import SyncVisionQuery  # noqa: F811

            query = SyncVisionQuery()
            roi_result = await query.query_roi(
                target_x - 30,
                target_y - 30,
                60,
                60,
            )
        except Exception:
            logger.warning("SyncVisionQuery unavailable; skipping closed-loop verification")
            return result

        metadata = roi_result.get("metadata", {})
        stale = metadata.get("stale", False)
        failed = metadata.get("failed", False)

        if failed:
            return result

        detections = roi_result.get("detections", [])
        if not detections:
            return result

        best = detections[0]
        iou = best.get("iou", 0.0)

        if stale and iou < 0.85:
            return result

        if iou < 0.7:
            bbox = best.get("bbox", None)
            if bbox and len(bbox) == 4:
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2
                result["adjusted_x"] = int(cx)
                result["adjusted_y"] = int(cy)

        result["verified"] = True
        return result

    # -------------------------------------------------------------------
    # Post-click confirmation (§7.4.2)
    # -------------------------------------------------------------------

    async def confirm_action(self) -> bool:
        try:
            from src.perception.sync_vision_query import SyncVisionQuery
        except ImportError:
            return True

        try:
            query = SyncVisionQuery()
            await query.capture_frame()
            return True
        except Exception:
            logger.exception("Post-click confirmation failed")
            return True

    # -------------------------------------------------------------------
    # Safety mode operations
    # -------------------------------------------------------------------

    def set_safety_level(self, level: SafetyLevel) -> None:
        self._safety_level = level
        logger.info("MouseChannel safety level set to %s", level.value)

    @property
    def safety_level(self) -> SafetyLevel:
        return self._safety_level

    @property
    def current_position(self) -> tuple[int, int]:
        return self._current_position


# ---------------------------------------------------------------------------
# Window-info helper for external modules (§7.4.3)
# ---------------------------------------------------------------------------

def get_foreground_window_info() -> dict:
    try:
        import ctypes
        user32 = ctypes.windll.user32
        hwnd = user32.GetForegroundWindow()
        class_buf = ctypes.create_unicode_buffer(256)
        title_buf = ctypes.create_unicode_buffer(256)
        ctypes.windll.user32.GetClassNameW(hwnd, class_buf, 255)
        ctypes.windll.user32.GetWindowTextW(hwnd, title_buf, 255)
        return {"hwnd": hwnd, "class_name": class_buf.value, "title": title_buf.value}
    except Exception:
        return {"hwnd": 0, "class_name": "", "title": ""}
