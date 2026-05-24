"""Desktop screen capture utility.

Captures the current desktop screen as an RGB numpy array.
Primary method: PowerShell bridge via subprocess (DPI-aware, uses
  SetProcessDPIAware() to get physical pixel resolution).
Fallback: PIL.ImageGrab.grab() (works via WSLg $DISPLAY).

v4.5.0 §4.1.1
"""

import base64
import io
import subprocess

import numpy as np
from PIL import Image, ImageGrab

_CAPTURE_TIMEOUT = 5


def capture_screenshot() -> np.ndarray:
    """Capture current desktop screen.

    Primary method: PowerShell bridge with SetProcessDPIAware() for correct
    physical-pixel resolution on HiDPI displays.
    Falls back to PIL.ImageGrab.grab() if PowerShell is unavailable.

    Returns:
        np.ndarray: RGB image array with shape (H, W, 3), dtype uint8.

    Raises:
        RuntimeError: If both primary and fallback capture methods fail.
    """
    # Primary: PowerShell bridge (DPI-aware, physical pixel coords)
    # v4.5.0 §4.1.1 — avoid PIL on HiDPI where ImageGrab may return
    # logical coords (e.g. 1707×960) while mouse uses physical (2560×1440).
    try:
        return _capture_via_powershell()
    except (RuntimeError, Exception):
        pass  # Fall through to PIL fallback

    # Fallback: PIL ImageGrab (works via WSLg $DISPLAY)
    # Called from visual_pool thread already — no need for nested executor
    try:
        img = ImageGrab.grab()
        return np.array(img.convert("RGB"))
    except (OSError, Exception):
        pass

    raise RuntimeError("All screenshot capture methods failed")


def _capture_via_powershell() -> np.ndarray:
    """Capture screenshot via PowerShell fallback.

    Uses System.Windows.Forms.Screen + System.Drawing.Bitmap to capture
    the screen, serialises as base64 PNG via MemoryStream, and decodes
    in Python.  Wrapped in a 5-second subprocess timeout.

    Returns:
        np.ndarray: RGB image array with shape (H, W, 3), dtype uint8.

    Raises:
        RuntimeError: If PowerShell subprocess times out, returns no data,
            or the decoded image is empty.
    """
    # v4.5.0 §4.1.1 — chr(36) = $ to avoid PowerShell interpolation issues
    # Call SetProcessDPIAware() before reading screen bounds, otherwise
    # HiDPI-scaled displays (e.g. 2560×1440 @150%) report scaled res (1707×960).
    ps_script = (
        "Add-Type -AssemblyName System.Windows.Forms\n"
        + "Add-Type -AssemblyName System.Drawing\n"
        + "Add-Type @'\n"
        + "using System.Runtime.InteropServices;\n"
        + "public class NativeDPI {\n"
        + "    [DllImport(\"user32.dll\")]\n"
        + "    public static extern bool SetProcessDPIAware();\n"
        + "}\n"
        + "'@\n"
        + "[NativeDPI]::SetProcessDPIAware() | Out-Null\n"
        + chr(36) + "screen = [System.Windows.Forms.Screen]::PrimaryScreen\n"
        + chr(36) + "bounds = " + chr(36) + "screen.Bounds\n"
        + chr(36) + "bmp = New-Object System.Drawing.Bitmap " + chr(36) + "bounds.Width, " + chr(36) + "bounds.Height\n"
        + chr(36) + "g = [System.Drawing.Graphics]::FromImage(" + chr(36) + "bmp)\n"
        + chr(36) + "g.CopyFromScreen(" + chr(36) + "bounds.X, " + chr(36) + "bounds.Y, 0, 0, " + chr(36) + "bmp.Size)\n"
        + chr(36) + "ms = New-Object System.IO.MemoryStream\n"
        + chr(36) + "bmp.Save(" + chr(36) + "ms, [System.Drawing.Imaging.ImageFormat]::Png)\n"
        + chr(36) + "bmp.Dispose()\n"
        + chr(36) + "g.Dispose()\n"
        + "Write-Output ([System.Convert]::ToBase64String(" + chr(36) + "ms.ToArray()))"
    )

    try:
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", ps_script],
            capture_output=True,
            timeout=_CAPTURE_TIMEOUT,
            check=True,
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(
            f"PowerShell screenshot timed out after {_CAPTURE_TIMEOUT}s: {e}"
        ) from e
    except subprocess.CalledProcessError as e:
        stderr_bytes = e.stderr  # pyright: ignore[reportAny] — subprocess stderr is loosely typed
        stderr_text = (
            stderr_bytes.decode("utf-8", errors="replace").strip()
            if isinstance(stderr_bytes, bytes)
            else "no stderr"
        )
        raise RuntimeError(
            f"PowerShell screenshot process failed (exit {e.returncode}): {stderr_text}"
        ) from e

    stdout = result.stdout.strip()
    if not stdout:
        raise RuntimeError("PowerShell screenshot returned empty output")

    try:
        png_bytes = base64.b64decode(stdout)
        img = Image.open(io.BytesIO(png_bytes))
        array = np.array(img.convert("RGB"))
        if array.size == 0:
            raise RuntimeError("Decoded screenshot image is empty")
        return array
    except Exception as e:
        raise RuntimeError(
            f"Failed to decode PowerShell screenshot output: {e}"
        ) from e
