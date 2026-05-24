"""InfraProvider Protocol — unified infrastructure provider interface.

v4.5.0 §10/§13: The top-level protocol that aggregates the three infra
sub-protocols (personality, memory, safety) into a single surface for DI.

All three properties are read-only. Concrete implementations wire together
the corresponding subsystem's resources via constructor injection.
"""

from __future__ import annotations

from typing import Protocol

from src.decision.safety_infra import SafetyInfra
from src.memory.memory_infra import MemoryInfra
from src.personality.personality_infra import PersonalityInfra


class InfraProvider(Protocol):
    """Aggregate protocol for all infrastructure sub-protocols.

    Any concrete container (e.g. InfraServices) must provide these three
    properties, each exposing the corresponding sub-protocol interface.

    v4.5.0 §10, §13
    """

    @property
    def personality(self) -> PersonalityInfra:
        """Personality subsystem — baseline, offsets, auditing, lifecycle."""
        ...

    @property
    def memory(self) -> MemoryInfra:
        """Memory subsystem — hot/cold storage, privacy filtering."""
        ...

    @property
    def safety(self) -> SafetyInfra:
        """Safety subsystem — classification, reflex rule matching."""
        ...
