"""
Memory decay engine with emotional protection — v4.5.0 §3.3.2.

Exports:
    MemoryDecayEngine  — Ebbinghaus-based decay evaluation
    DecayConfig        — decay configuration
    DecayResult        — per-entry decay calculation result
    MemoryType         — EMOTION, FACT, ACTION enum
    compute_decayed_importance — decay formula
    classify_memory_type — classify scene into memory type
    is_emotionally_protected — check emotional protection
"""

from .decay_engine import (
    MemoryDecayEngine,
    DecayConfig,
    DecayResult,
    MemoryType,
    compute_decayed_importance,
    classify_memory_type,
    is_emotionally_protected,
)

__all__ = [
    "MemoryDecayEngine",
    "DecayConfig",
    "DecayResult",
    "MemoryType",
    "compute_decayed_importance",
    "classify_memory_type",
    "is_emotionally_protected",
]
