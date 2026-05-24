"""Smoke test for mouse coordinate capture via ctypes+X11.

v4.5.0 §1.3.2: 以鼠标坐标为中心动态裁剪
"""

import importlib.util
import logging
import sys


def _load_mouse_capture():
    """Load mouse_capture module bypassing __init__.py (ultralytics may not be available)."""
    spec = importlib.util.spec_from_file_location(
        "mouse_capture", "src/perception/visual/mouse_capture.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_get_mouse_position():
    """get_mouse_position() returns valid (x, y) tuple or None on headless."""
    logging.basicConfig(level=logging.WARNING)
    mod = _load_mouse_capture()
    pos = mod.get_mouse_position()

    assert pos is not None, "get_mouse_position() returned None"
    assert isinstance(pos, tuple), f"Expected tuple, got {type(pos)}"
    assert len(pos) == 2, f"Expected tuple of 2, got {len(pos)}"
    x, y = pos
    assert isinstance(x, int), f"Expected int x, got {type(x)}"
    assert isinstance(y, int), f"Expected int y, got {type(y)}"
    assert x >= 0, f"x must be >= 0, got {x}"
    assert y >= 0, f"y must be >= 0, got {y}"
    print(f"PASS: mouse position ({x}, {y})")


def test_multiple_calls():
    """Calling get_mouse_position() 3 times rapidly should not crash."""
    logging.basicConfig(level=logging.WARNING)
    mod = _load_mouse_capture()
    for i in range(3):
        pos = mod.get_mouse_position()
        assert pos is not None, f"Call {i+1} returned None"
    print("PASS: 3 rapid calls succeeded")


if __name__ == "__main__":
    test_get_mouse_position()
    test_multiple_calls()
    print("ALL TESTS PASSED")
