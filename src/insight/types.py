"""v5.x insight-memory-joint: Prompt learning type definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np


@dataclass
class PromptRef:
    """A stored visual concept reference for SAVPE inference."""
    id: str = ""
    name: str = ""
    crop: Optional[np.ndarray] = field(default=None, repr=False)
    vpe_embedding: Optional[np.ndarray] = field(default=None, repr=False)
    context_tags: list[str] = field(default_factory=list)
    confidence: float = 0.5
    last_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    vpe_expires_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
