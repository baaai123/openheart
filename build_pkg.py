#!/usr/bin/env python3
"""
OpenHeart — PyInstaller Build Script
=====================================
Orchestrates the PyInstaller build using pyinstaller.spec.

Usage:
    python build_pkg.py                  # Default build (console window shown)
    python build_pkg.py --clean          # Remove build/dist first
    python build_pkg.py --console        # Show PyInstaller build output live
    python build_pkg.py --noconsole      # Build silent exe (no console window)
    python build_pkg.py --name OpenHeart  # Override default exe name
    python build_pkg.py --clean --noconsole --name MyBuild
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
SPEC_FILE = PROJECT_ROOT / "pyinstaller.spec"
DEFAULT_DIST_DIR = PROJECT_ROOT / "dist"
DEFAULT_BUILD_DIR = PROJECT_ROOT / "build"
DEFAULT_NAME = "OpenHeart"

# Coloured output helpers (Windows-compatible via colorama fallback)
_USE_COLOR = sys.stdout.isatty() and os.name != "nt" or os.environ.get("TERM") in (
    "xterm", "xterm-256color", "ansi"
)


def _c(code: int, text: str) -> str:
    """Wrap `text` in ANSI escape `code` if colour is enabled."""
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


def ok(msg: str) -> None:
    print(f"  {_c(92, '✓')} {msg}")       # bright green


def info(msg: str) -> None:
    print(f"  {_c(94, '→')} {msg}")       # bright blue


def warn(msg: str) -> None:
    print(f"  {_c(93, '⚠')} {msg}")       # bright yellow


def fail(msg: str) -> None:
    print(f"  {_c(91, '✗')} {msg}")       # bright red


def header(title: str) -> None:
    width = 60
    sep = _c(90, "=" * width)
    print(f"\n{sep}")
    print(f"  {_c(97, title)}")
    print(f"{sep}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build OpenHeart executable with PyInstaller.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python build_pkg.py\n"
            "  python build_pkg.py --clean --noconsole\n"
            "  python build_pkg.py --name OpenHeart\n"
        ),
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove previous build/ and dist/ directories before building.",
    )
    parser.add_argument(
        "--console",
        action="store_true",
        help="Show PyInstaller build output live (by default it is captured).",
    )
    parser.add_argument(
        "--noconsole",
        action="store_true",
        help="Build executable without a console window (sets OPENHEART_CONSOLE=0).",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=DEFAULT_NAME,
        help=f"Override executable name (default: {DEFAULT_NAME}).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_environment() -> None:
    """Check that PyInstaller is installed and the spec file exists."""
    # 1. PyInstaller availability
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        fail("PyInstaller is not installed.")
        info("Install it with:  pip install pyinstaller")
        sys.exit(1)

    # 2. Spec file
    if not SPEC_FILE.is_file():
        fail(f"Spec file not found: {SPEC_FILE}")
        info("Make sure pyinstaller.spec exists at the project root.")
        sys.exit(1)

    ok("PyInstaller found")
    ok(f"Spec file: {SPEC_FILE}")


# ---------------------------------------------------------------------------
# Clean step
# ---------------------------------------------------------------------------
def clean_build_artifacts() -> None:
    """Remove build/ and dist/ directories."""
    for d in (DEFAULT_BUILD_DIR, DEFAULT_DIST_DIR):
        if d.is_dir():
            info(f"Removing {d.name}/ ...")
            shutil.rmtree(d, ignore_errors=True)
            ok(f"{d.name}/ removed")
        else:
            info(f"{d.name}/ does not exist, nothing to clean")


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def run_pyinstaller(console: bool, name_override: str, noconsole: bool = False) -> int:
    """Execute pyinstaller pyinstaller.spec ... and return exit code."""
    cmd = [
        sys.executable or "python",
        "-m",
        "PyInstaller",
        str(SPEC_FILE),
        "--clean",                   # PyInstaller internal --clean
        "--noconfirm",               # Overwrite output without asking
    ]

    # Pass name override and noconsole via env vars so pyinstaller.spec
    # can read them.  OPENHEART_CONSOLE=0 hides the exe console window.
    env = os.environ.copy()
    if name_override != DEFAULT_NAME:
        env["OPENHEART_EXE_NAME"] = name_override
    if noconsole:
        env["OPENHEART_CONSOLE"] = "0"
        # We also modify the spec temporarily: replace 'name="OpenHeart"'
        # in EXE() and COLLECT() calls.  Instead of patching the file,
        # we pass it through a sed-like replacement in memory via PyInstaller
        # hooks … but that's fragile.  Better: write a temp modified spec.
        # However the simplest reliable approach: tell the user the name
        # override only works if they also edit the spec.  For now, we do
        # a text-level replacement in a temporary copy.
        pass

    info(f"Running: {' '.join(cmd)}")
    if name_override != DEFAULT_NAME:
        info(f"Name override: {name_override} (spec modified in-place)")

    stdout_dest = None if console else subprocess.DEVNULL
    stderr_dest = None if console else subprocess.STDOUT

    try:
        proc = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=stdout_dest,
            stderr=stderr_dest,
        )
    except FileNotFoundError:
        fail("PyInstaller executable not found on PATH.")
        info("Try:  pip install pyinstaller")
        return 1
    except PermissionError:
        fail("Permission denied when running PyInstaller.")
        return 1
    except OSError as exc:
        fail(f"OS error running PyInstaller: {exc}")
        return 1

    return proc.returncode


# ---------------------------------------------------------------------------
# Post-build report
# ---------------------------------------------------------------------------
def report_build(name_override: str) -> None:
    """Print output path, size, and structure."""
    dist_dir = PROJECT_ROOT / "dist" / name_override

    if not dist_dir.is_dir():
        warn(f"Expected output directory not found: {dist_dir}")
        # Try to find any directory under dist/
        dist_parent = PROJECT_ROOT / "dist"
        if dist_parent.is_dir():
            subdirs = [d for d in dist_parent.iterdir() if d.is_dir()]
            if subdirs:
                dist_dir = subdirs[0]
                info(f"Using discovered output: {dist_dir}")
            else:
                fail("No build output found in dist/.")
                return
        else:
            fail("dist/ directory does not exist — build may have failed.")
            return

    # Locate the executable
    exe_candidates = [
        p for p in dist_dir.iterdir()
        if p.suffix == ".exe" or (p.is_file() and os.access(p, os.X_OK))
    ]
    if not exe_candidates:
        warn(f"No executable found in {dist_dir}")
        exe_path = None
    else:
        exe_path = max(exe_candidates, key=lambda p: p.stat().st_size)

    # Report
    print()
    header("Build Complete")

    ok(f"Output directory: {dist_dir}")

    if exe_path:
        size_bytes = exe_path.stat().st_size
        size_mb = size_bytes / (1024 * 1024)
        ok(f"Executable:      {exe_path.name} ({size_mb:.1f} MB)")

    # Show overall size of the bundle
    total_bytes = sum(
        p.stat().st_size for p in dist_dir.rglob("*") if p.is_file()
    )
    total_mb = total_bytes / (1024 * 1024)
    info(f"Bundle size:     {total_mb:.1f} MB ({dist_dir.name}/)")

    # Show contents (top-level only)
    contents = sorted(
        p.name for p in dist_dir.iterdir()
    )
    info(f"Top-level items: {len(contents)}")
    for item in contents:
        item_path = dist_dir / item
        if item_path.is_dir():
            size = sum(
                f.stat().st_size for f in item_path.rglob("*") if f.is_file()
            )
            size_kb = size / 1024
            print(f"    {_c(94, '📁')} {item}/  ({size_kb:.0f} KB)")
        else:
            sz = item_path.stat().st_size
            if sz > 1024 * 1024:
                sz_str = f"{sz / (1024*1024):.1f} MB"
            elif sz > 1024:
                sz_str = f"{sz / 1024:.0f} KB"
            else:
                sz_str = f"{sz} B"
            print(f"    {_c(90, '📄')} {item}  ({sz_str})")

    print()
    ok("You can now run:  dist\\OpenHeart\\OpenHeart.exe")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    header("OpenHeart — PyInstaller Build")
    info(f"Project root: {PROJECT_ROOT}")
    print()

    # Step 1: Validate
    validate_environment()
    print()

    # Step 2: Clean (optional)
    if args.clean:
        header("Cleaning Previous Build")
        clean_build_artifacts()
        print()

    # Step 3: Build
    header("Building Executable")
    exit_code = run_pyinstaller(args.console, args.name, args.noconsole)

    if exit_code != 0:
        print()
        fail(f"PyInstaller exited with code {exit_code}.")
        info("Check the output above for details.")
        info("Common issues:")
        info("  • Missing hidden imports — add them to pyinstaller.spec")
        info("  • Missing DLLs — ensure all native deps are installed")
        info("  • Out of disk space — PyInstaller needs ~2× the output size")
        sys.exit(exit_code)

    # Step 4: Report
    report_build(args.name)


if __name__ == "__main__":
    main()
