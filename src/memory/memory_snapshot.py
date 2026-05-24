"""MemorySnapshot — pure dataclass for structured memory context injection.

v5.x architecture: thin data container with no I/O, no external dependencies.
recent_dialog removed — conversation history belongs to SessionState, not memory.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MemorySnapshot:
    """Structured memory snapshot for LLM context assembly — v5.x architecture.

    Pure data container with no I/O, network, or storage dependencies.
    Designed for injection into DecisionBridge prompt (v4.5.0 §5.1).

    Attributes:
        historical_summary: Privacy-filtered summary of recent hot memory context.
        memory_drawer: Semantic search results from cold memory for current topic.
    """

    historical_summary: str = ""
    memory_drawer: str = ""

    def to_prompt_text(self) -> str:
        """Render the snapshot as a compact Chinese prompt-text block.

        Returns:
            Formatted string with non-empty fields, or empty string if both
            fields are empty. Each field is rendered as a labelled section:
            - [历史记忆] {historical_summary}
            - [相关记忆] {memory_drawer}

        v4.5.0 §5.1: injected into LLM decision prompt for context awareness.
        """
        parts: list[str] = []
        if self.historical_summary:
            parts.append("[历史记忆] " + self.historical_summary)
        if self.memory_drawer:
            parts.append("[相关记忆] " + self.memory_drawer)
        return "\n".join(parts)
