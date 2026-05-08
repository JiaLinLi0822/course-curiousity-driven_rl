"""First-pass information-oriented intrinsic reward.

This Phase A version rewards surprising outcomes under a simple learned
conditional outcome table. It is intentionally lightweight and can later be
replaced with a Bayesian/KL-based information-gain calculation.
"""

from __future__ import annotations

import math


class InfoGainReward:
    """Reward outcomes that are currently hard to predict."""

    def __init__(
        self,
        num_possible_outcomes: int = 4,
        smoothing: float = 1.0,
        reward_clip: float | None = None,
    ):
        self.num_possible_outcomes = num_possible_outcomes
        self.smoothing = smoothing
        self.reward_clip = reward_clip
        self.counts = {}

    def reset(self) -> None:
        # Counts intentionally persist across episodes so the learned outcome
        # model reflects cumulative experience across training.
        pass

    def compute(self, info: dict) -> float:
        """Return -log p(outcome | condition) using current counts."""
        event = info.get("mixing_event")
        if event is None:
            return 0.0

        condition, outcome = self._split_event(event)
        probability = self._predicted_probability(condition, outcome)
        reward = -math.log(max(probability, 1e-12))
        if self.reward_clip is not None:
            reward = min(reward, self.reward_clip)
        return reward

    def update(self, info: dict) -> None:
        """Record the observed outcome under its condition."""
        event = info.get("mixing_event")
        if event is None:
            return

        condition, outcome = self._split_event(event)

        if condition not in self.counts:
            self.counts[condition] = {}

        outcome_counts = self.counts[condition]
        outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1

    def _split_event(
        self,
        event: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
    ) -> tuple[tuple[tuple[int, int], tuple[int, int]], tuple[int, int]]:
        agent_color_before, tile_color_before, outcome_color = event
        condition = (agent_color_before, tile_color_before)
        return condition, outcome_color

    def _predicted_probability(
        self,
        condition: tuple[tuple[int, int], tuple[int, int]],
        outcome: tuple[int, int],
    ) -> float:
        outcome_counts = self.counts.get(condition, {})
        outcome_count = outcome_counts.get(outcome, 0)
        total_count = sum(outcome_counts.values())

        numerator = outcome_count + self.smoothing
        denominator = total_count + self.smoothing * self.num_possible_outcomes
        return numerator / denominator
