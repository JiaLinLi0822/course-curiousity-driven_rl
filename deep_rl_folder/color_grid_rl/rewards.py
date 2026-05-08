"""Extrinsic and intrinsic reward helpers.

Step 1 only uses sparse terminal extrinsic reward. Intrinsic reward functions
are defined as small placeholders so later modules can share a common interface.
"""

from __future__ import annotations

import numpy as np

from .config import DEFAULT_REWARD_CONFIG, RewardConfig


def compute_extrinsic_reward(
    solved: bool,
    timeout: bool,
    solved_reward: float,
    timeout_reward: float,
    step_reward: float,
) -> float:
    """Return sparse terminal reward for the current transition."""
    if solved:
        return solved_reward
    if timeout:
        return timeout_reward
    return step_reward


def combine_rewards(
    extrinsic_reward: float,
    intrinsic_reward: float = 0.0,
    config: RewardConfig = DEFAULT_REWARD_CONFIG,
) -> float:
    """Combine extrinsic and intrinsic reward using explicit weights."""
    return (
        config.extrinsic_weight * extrinsic_reward
        + config.intrinsic_coef * intrinsic_reward
    )


class NoIntrinsicReward:
    """Minimal intrinsic reward object used before Step 3 and Step 4 exist."""

    def reset(self) -> None:
        """Reset internal state.

        This placeholder has no state, but the method keeps the future
        interface consistent.
        """

    def compute(self, info: dict) -> float:
        """Return zero intrinsic reward."""
        return 0.0

    def update(self, info: dict) -> None:
        """No-op update for interface compatibility."""
        pass


class RandomIntrinsicReward:
    """Random intrinsic reward baseline over observed color-mixing events."""

    def reset(self) -> None:
        """No episodic state to reset."""

    def compute(self, info: dict) -> float:
        """Return a uniform random bonus for mixing events, otherwise zero."""
        if info.get("mixing_event") is None:
            return 0.0
        return float(np.random.random())

    def update(self, info: dict) -> None:
        """No-op update for interface compatibility."""
        pass
