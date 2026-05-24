"""
OpenHeart runtime configuration package.

Exports:
    RuntimeConfig          — immutable config from env vars + GPU detection
    VRAMTier               — enum: HIGH, MID, LOW
    SystemRequirementError — raised if system below minimum requirements

Usage (call once at startup):
    from config.runtime import RuntimeConfig
    rc = RuntimeConfig.from_environ()

v4.5.0 §0.5: All mode switches resolved once into RuntimeConfig.
"""

from .runtime import RuntimeConfig, VRAMTier, SystemRequirementError

__all__ = [
    "RuntimeConfig",
    "VRAMTier",
    "SystemRequirementError",
]
