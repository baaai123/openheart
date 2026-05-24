"""
Perception layer — four-lane visual + audio processing and perception bus output.

v4.5.0 §1: 感知层
"""


def __getattr__(name: str):
    if name == "PerceptionBus":
        from .perception_bus import PerceptionBus
        return PerceptionBus
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "PerceptionBus",
]
