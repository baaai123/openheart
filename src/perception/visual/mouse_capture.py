"""Mouse coordinate capture via PowerShell subprocess.

v4.5.0 §1.3.2: 以鼠标坐标为中心动态裁剪. Captures global mouse
position via PowerShell System.Windows.Forms.Cursor (same pattern
as screenshot.py fallback).  Zero additional pip dependencies.

Returns (root_x, root_y) or None on failure.
"""

import logging
import re
import subprocess
from typing import Optional

_logger = logging.getLogger(__name__)


def get_mouse_position() -> Optional[tuple[int, int]]:
    """Query the global mouse pointer position via PowerShell.

    Spawns a PowerShell subprocess to call
    [System.Windows.Forms.Cursor]::Position, which returns the
    absolute screen coordinates in WSLg/Windows.

    Returns:
        Tuple (x, y) of absolute screen coordinates, or None
        if the PowerShell query fails.

    Raises:
        Only caught exceptions are logged; the function always returns
        None on failure.
    """
    # PS script: print X and Y coordinates as two lines of integers.
    # v4.5.0 §4.1.1 — chr(36) = $ to avoid shell interpolation
    ps_script = (
        "Add-Type -AssemblyName System.Windows.Forms; "
        + "Write-Output ([System.Windows.Forms.Cursor]::Position.X); "
        + "Write-Output ([System.Windows.Forms.Cursor]::Position.Y)"
    )

    try:
        result = subprocess.run(
            ["powershell.exe", "-Command", ps_script],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if result.returncode != 0 or not result.stdout.strip():
            _logger.warning(
                "PowerShell mouse query failed (rc=%d); mouse capture degraded.",
                result.returncode,
            )
            return None

        lines = result.stdout.strip().splitlines()
        nums = [int(n) for n in lines if n.strip().isdigit()]
        if len(nums) >= 2:
            return (nums[0], nums[1])
        _logger.warning("PowerShell mouse query returned unexpected output: %r", result.stdout)
        return None

    except (subprocess.TimeoutExpired, OSError, ValueError, Exception):
        # Catch-all: subprocess may fail if PowerShell/WSL interop is down.
        # Safe because we always return None on failure; no data corruption risk.
        _logger.warning(
            "PowerShell mouse capture failed unexpectedly", exc_info=True
        )
        return None


def _resolve_display() -> str:
    """Return the value of $DISPLAY, or '<not set>'."""
    import os
    return os.environ.get("DISPLAY", "<not set>")
