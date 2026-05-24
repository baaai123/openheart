"""
Hot-to-cold memory sync engine — v4.5.0 §3.2.4.

Exports:
    MemorySyncEngine  — incremental Redis→LanceDB sync
    SyncConfig        — sync configuration
    SyncResult        — per-cycle sync outcome
    filter_scene_sensitive — sensitive information check
    compute_importance_score — v4.5.0 importance formula
"""

from .sync_engine import (
    MemorySyncEngine,
    SyncConfig,
    SyncResult,
    filter_scene_sensitive,
    compute_importance_score,
)

__all__ = [
    "MemorySyncEngine",
    "SyncConfig",
    "SyncResult",
    "filter_scene_sensitive",
    "compute_importance_score",
]
