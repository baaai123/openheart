# -*- mode: python ; coding: utf-8 -*-
#
# PyInstaller spec for OpenHeart — Full voice dialogue pipeline.
#
# Build:
#   pyinstaller pyinstaller.spec --clean
#
# Output:
#   dist/OpenHeart.exe   (single-directory bundle, console-enabled)
#
# Layout:
#   scripts/demo_full.py       # Entry point
#   src/                       # Application source (bundled as package)
#   config/                    # Runtime YAML/JSON configs
#   models/         [excluded] # Downloaded weight files (user-provided at runtime)
#   deps/           [excluded] # Build-time dependencies
#   learning/       [excluded] # Training data
#   electron-l2d/   [excluded] # Electron frontend

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Collect all .py from src/ so PyInstaller treats it as a top-level package.
# ---------------------------------------------------------------------------
def collect_src_py(relative_dir: str) -> list:
    """Return [(str source_path, str dest)] for every .py under src/."""
    src_dir = PROJECT_ROOT / relative_dir
    entries = []
    for py_file in src_dir.rglob("*.py"):
        # dest = relative/from/src/... preserving subdir structure
        dest = str(py_file.relative_to(PROJECT_ROOT).parent)
        entries.append((str(py_file), dest))
    return entries


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
a = Analysis(
    [str(PROJECT_ROOT / "scripts" / "demo_full.py")],
    pathex=[str(PROJECT_ROOT)],
    binaries=[],
    datas=(
        # --add-data for config/ directory (runtime YAML/JSON)
        [(str(PROJECT_ROOT / "config"), "config")]
        +
        # Bundle all source .py files so imports resolve correctly
        collect_src_py("src")
    ),
    hiddenimports=[
        # Core ML / DL frameworks
        "torch",
        "torchaudio",
        "numpy",
        "transformers",
        "accelerate",
        "bitsandbytes",
        "sentence_transformers",
        # Inference engines
        "vllm",
        "cosyvoice",
        "ultralytics",
        # Audio / VAD
        "pywhispercpp",
        "silero_vad",
        "ten_vad",
        # Vision
        "cv2",
        "PIL",
        # NLP
        "spacy",
        "snownlp",
        "textblob",
        # Network / Async
        "aiohttp",
        "websockets",
        # Storage / Cache
        "lancedb",
        "redis",
        # Graph / ML
        "networkx",
        "scipy",
        "scikit_learn",
        # Config / Serialization
        "yaml",
        "json",
    ],
    hookspath=[],
    hooksconfig={},
    excludes=[
        # Excluded project directories (weights, deps, training, frontend)
        "models",
        "deps",
        "learning",
        "electron_l2d",
        # Unnecessary heavy packages
        "matplotlib",
        "tensorflow",
        "tensorboard",
        "notebook",
        "jupyter",
        "ipython",
        "setuptools",
        "pip",
        "pkg_resources",
        # Dev / test only
        "pytest",
        "black",
        "ruff",
        "mypy",
    ],
    noarchive=False,
    optimize=0,  # 0 = no optimization (fastest build)
)

# ---------------------------------------------------------------------------
# PyInstaller ZIP archive
# ---------------------------------------------------------------------------
pyz = PYZ(a.pure, a.zipped_data, cipher=None)

# ---------------------------------------------------------------------------
# Executable  (console=True → show CLI output)
# ---------------------------------------------------------------------------
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="OpenHeart",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,          # Show console window for CLI output
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

# ---------------------------------------------------------------------------
# Collection  (output → dist/OpenHeart/)
# ---------------------------------------------------------------------------
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="OpenHeart",
)

# ---------------------------------------------------------------------------
# NOTE on runtime weight files
#
# Models (models/, ~10-30 GB) are NOT bundled. The user must place them
# alongside the executable at first launch, or set OPENHEART_MODELS_PATH.
#
# The spec avoids `tree` helper for src/ because PyInstaller's tree()
# only collects files from within the spec's directory and may miss
# nested packages.  collect_src_py() gives explicit control.
# ---------------------------------------------------------------------------
