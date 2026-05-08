"""Novelty-driven intrinsic reward."""

from __future__ import annotations

import math


class NoveltyReward:
    """Count-based novelty bonus over color-mixing events."""

    def __init__(self, alpha: float = 0.5, persistent_counts: bool = True):
        self.alpha = float(alpha)
        self.persistent_counts = persistent_counts
        self.counts = {}

    def reset(self) -> None:
        if not self.persistent_counts:
            self.counts = {}

    def compute(self, info: dict) -> float:
        """Return a count-based bonus for the current mixing event."""
        event = info.get("mixing_event")
        if event is None:
            return 0.0

        count = self.counts.get(event, 0)
        return 1.0 / ((count + 1) ** self.alpha)

    def update(self, info: dict) -> None:
        """Increment the count for the current mixing event."""
        event = info.get("mixing_event")
        if event is None:
            return

        self.counts[event] = self.counts.get(event, 0) + 1
