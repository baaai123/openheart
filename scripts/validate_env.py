#!/usr/bin/env python3
"""
validate_env.py — runtime dependency validator for OpenHeart.

Checks:
  1. Python ≥ 3.11
  2. CUDA availability (torch.cuda.is_available())
   3. GPU VRAM detection and auto-tier selection (high / low)
  4. Model path existence (from config/model_paths.yaml)
  5. Redis connection (ping)
  6. LanceDB accessibility

Exit code 0 when all checks pass, non-zero otherwise.
v4.5.0 §12.1 — three-tier VRAM design.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ── constants ──────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATHS_YAML = PROJECT_ROOT / "config" / "model_paths.yaml"

# Two-tier VRAM thresholds in GB
VRAM_HIGH_THRESHOLD_GB = 12.0
VRAM_LOW_THRESHOLD_GB = 0.0

# ── helpers ────────────────────────────────────────────────────────────────


def ok(msg: str) -> str:
    return f"✅  {msg}"


def fail(msg: str) -> str:
    return f"❌  {msg}"


def warn(msg: str) -> str:
    return f"⚠️  {msg}"


def info(msg: str) -> str:
    return f"ℹ️  {msg}"


# ── check 1: python version ───────────────────────────────────────────────


def check_python() -> tuple[bool, str]:
    """Python ≥ 3.11 required (spec §13.0)."""
    v = sys.version_info
    if v >= (3, 11):
        return True, ok(f"Python {v.major}.{v.minor}.{v.micro}")
    return False, fail(f"Python {v.major}.{v.minor}.{v.micro} — 3.11+ required")


# ── check 2 & 3: cuda + vram ──────────────────────────────────────────────


def check_cuda_and_vram() -> tuple[bool, str]:
    """
    Check torch.cuda.is_available() and detect VRAM total,
    then resolve VRAM tier per spec §12.1 thresholds.
    """
    try:
        import torch  # type: ignore[import-untyped]
    except ImportError:
        return False, fail("PyTorch not installed")

    if not torch.cuda.is_available():
        return False, fail("CUDA not available — GPU required")

    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    except Exception as exc:
        return False, fail(f"Failed to query GPU memory: {exc}")

    total_gb = total_bytes / (1024.0 ** 3)
    free_gb = free_bytes / (1024.0 ** 3)

    # resolve tier (replicating runtime.py _resolve_vram_tier logic)
    if total_gb >= VRAM_HIGH_THRESHOLD_GB:
        tier = "high"
        tier_note = "all models, 4096 context"
    elif total_gb > VRAM_LOW_THRESHOLD_GB:
        tier = "low"
        tier_note = "reduced models, 2048 context, CosyVoice on CPU"
    else:
        return False, fail(
            f"VRAM {total_gb:.1f} GB — below minimum requirements"
        )

    lines = [
        ok(f"CUDA available — total={total_gb:.1f} GB, free={free_gb:.1f} GB"),
        info(f"VRAM tier: {tier.upper()} ({tier_note})"),
    ]
    return True, "\n".join(lines)


# ── check 4: model paths ──────────────────────────────────────────────────


def check_model_paths() -> tuple[bool, str]:
    """Verify every path in config/model_paths.yaml exists on disk."""
    if not MODEL_PATHS_YAML.exists():
        return False, fail(f"Model paths config missing: {MODEL_PATHS_YAML}")

    try:
        import yaml
    except ImportError:
        return False, fail("PyYAML not installed — cannot read model_paths.yaml")

    try:
        with open(MODEL_PATHS_YAML, encoding="utf-8") as fh:
            raw: object = yaml.safe_load(fh)
    except Exception as exc:
        return False, fail(f"Failed to parse {MODEL_PATHS_YAML}: {exc}")

    if not isinstance(raw, dict):
        return False, fail(f"{MODEL_PATHS_YAML} does not contain a mapping")
    model_paths: dict[str, str] = raw  # type: ignore[assignment]  # narrowed by isinstance above

    missing: list[str] = []
    present: list[str] = []
    for name, rel_path in model_paths.items():
        abs_path = PROJECT_ROOT / rel_path
        if abs_path.exists():
            present.append(f"  • {name} → {rel_path}")
        else:
            missing.append(f"  • {name} → {rel_path}")

    lines: list[str] = []
    if present:
        lines.append(ok(f"{len(present)} model paths found:"))
        lines.extend(present)
    if missing:
        lines.append(warn(f"{len(missing)} model paths missing (run scripts/download_models.py):"))
        lines.extend(missing)

    # model paths are not fatal — models may be downloaded later
    success = len(missing) == 0
    indicator = ok if success else warn
    lines.insert(0, indicator(f"Model paths check: {len(present)}/{len(present)+len(missing)} found"))
    return success, "\n".join(lines)


# ── check 5: redis ────────────────────────────────────────────────────────


def check_redis() -> tuple[bool, str]:
    """Ping the local Redis server."""
    try:
        import redis  # type: ignore[import-untyped]
    except ImportError:
        return False, warn("redis-py not installed (optional for hot memory)")

    try:
        r = redis.Redis(host="localhost", port=6379, socket_connect_timeout=3)
        if r.ping():
            return True, ok("Redis ping succeeded (localhost:6379)")
        return False, fail("Redis ping returned falsy")
    except redis.ConnectionError:
        return False, warn("Redis not reachable at localhost:6379 (optional for hot memory)")
    except Exception as exc:
        return False, warn(f"Redis check failed: {exc} (optional for hot memory)")


# ── check 6: lancedb ──────────────────────────────────────────────────────


def check_lancedb() -> tuple[bool, str]:
    """Verify LanceDB is importable and can create an ephemeral table."""
    try:
        import lancedb  # type: ignore[import-untyped]
    except ImportError:
        return False, warn("LanceDB not installed (optional for cold memory)")

    tmp_dir = "/tmp/openheart_lancedb_validate"
    try:
        db = lancedb.connect(tmp_dir)
        # create a trivial table to verify write access
        import pyarrow as pa
        data = pa.table({"id": [1]})
        db.create_table("validate_test", data, mode="overwrite")
        # cleanup
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return True, ok("LanceDB accessible (ephemeral table created/cleaned)")
    except Exception as exc:
        return False, warn(f"LanceDB check failed: {exc} (optional for cold memory)")


# ── check: import core deps ────────────────────────────────────────────────


def check_core_imports() -> tuple[bool, str]:
    """Check that key Python packages are importable."""
    packages = [
        ("torch", "PyTorch"),
        ("asyncio", "asyncio"),
        ("numpy", "numpy"),
        ("cv2", "opencv-python"),
        ("PIL", "pillow"),
        ("networkx", "networkx"),
        ("spacy", "spaCy"),
        ("snownlp", "snownlp"),
    ]
    ok_list: list[str] = []
    fail_list: list[str] = []
    for mod, pkg in packages:
        try:
            __import__(mod)
            ok_list.append(f"  • {mod}")
        except ImportError:
            fail_list.append(f"  • {mod} ({pkg})")

    lines: list[str] = []
    lines.append(ok(f"{len(ok_list)} core packages importable:"))
    lines.extend(ok_list)
    if fail_list:
        lines.append(warn(f"{len(fail_list)} core packages missing:"))
        lines.extend(fail_list)
    return len(fail_list) == 0, "\n".join(lines)


# ── main ───────────────────────────────────────────────────────────────────


def main() -> int:
    print("=" * 60)
    print("  OpenHeart — Environment Validation")
    print("=" * 60)
    print()

    results: list[tuple[str, bool, str]] = []

    # 1. Python version
    ok_flag, msg = check_python()
    results.append(("Python Version", ok_flag, msg))

    # 2 & 3. CUDA + VRAM
    ok_flag, msg = check_cuda_and_vram()
    results.append(("CUDA / VRAM", ok_flag, msg))

    # 4. Model paths
    ok_flag, msg = check_model_paths()
    results.append(("Model Paths", ok_flag, msg))

    # 5. Redis
    ok_flag, msg = check_redis()
    results.append(("Redis", ok_flag, msg))

    # 6. LanceDB
    ok_flag, msg = check_lancedb()
    results.append(("LanceDB", ok_flag, msg))

    # Bonus: core imports
    ok_flag, msg = check_core_imports()
    results.append(("Core Imports", ok_flag, msg))

    # ── print report ──────────────────────────────────────────────────────
    for name, passed, detail in results:
        marker = "PASS" if passed else "WARN/FAIL"
        print(f"[{marker}] {name}")
        print(detail)
        print()

    # ── summary ────────────────────────────────────────────────────────────
    hard_checks = ["Python Version", "CUDA / VRAM"]
    hard_failed = any(not p for n, p, _ in results if n in hard_checks)
    soft_failed = sum(1 for n, p, _ in results if n not in hard_checks and not p)
    total = len(results)

    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)

    if hard_failed:
        print(fail(f"Hard checks FAILED — OpenHeart cannot start."))
        return 1

    if soft_failed:
        print(warn(f"{soft_failed} optional check(s) did not pass — some features may be degraded."))
        print(info("See above for details on what is missing."))
    else:
        print(ok(f"All {total} checks passed. Environment is ready for OpenHeart."))

    return 0


if __name__ == "__main__":
    sys.exit(main())
