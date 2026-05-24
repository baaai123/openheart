"""InfraServices — concrete InfraProvider container.

v4.5.0 §10/§13: Wires together the three infra subsystems via constructor
injection. Owns no business logic — merely a container that satisfies the
InfraProvider Protocol and provides a unified shutdown() entry point.

Usage:
    services = InfraServices(personality=..., memory=..., safety=...)
    # … application runs …
    await services.shutdown()
"""

from __future__ import annotations

from src.decision.safety_infra import SafetyInfra
from src.infra.infra_provider import InfraProvider
from src.memory.memory_infra import MemoryInfra
from src.personality.personality_infra import PersonalityInfra


class InfraServices(InfraProvider):
    """Concrete infrastructure container with constructor injection.

    Each property delegates to the instance passed at construction time.
    ``shutdown()`` calls each subsystem's lifecycle hook in a fixed order.
    """

    def __init__(
        self,
        personality: PersonalityInfra,
        memory: MemoryInfra,
        safety: SafetyInfra,
    ) -> None:
        self._personality: PersonalityInfra = personality
        self._memory: MemoryInfra = memory
        self._safety: SafetyInfra = safety

    # ── InfraProvider properties ─────────────────────────────────────────

    @property
    def personality(self) -> PersonalityInfra:
        return self._personality

    @property
    def memory(self) -> MemoryInfra:
        return self._memory

    @property
    def safety(self) -> SafetyInfra:
        return self._safety

    # ── Lifecycle ────────────────────────────────────────────────────────

    async def shutdown(self) -> None:
        """Gracefully shut down all infra subsystems.

        Calls the ``shutdown()`` coroutine on each sub-provider in the
        order: personality → memory → safety.
        """
        await self._personality.shutdown()
        await self._memory.shutdown()
        await self._safety.shutdown()
