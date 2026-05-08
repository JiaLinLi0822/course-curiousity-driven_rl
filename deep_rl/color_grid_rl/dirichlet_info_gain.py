from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np
from scipy.special import gammaln, digamma


Color = Tuple[int, int]
Condition = Tuple[Color, Color]
Event = Tuple[Color, Color, Color]


def _dirichlet_kl(alpha_p: np.ndarray, alpha_q: np.ndarray) -> float:
    """
    KL[ Dir(alpha_p) || Dir(alpha_q) ]
    """
    a0_p = np.sum(alpha_p)
    a0_q = np.sum(alpha_q)

    kl = (
        gammaln(a0_p)
        - gammaln(a0_q)
        - np.sum(gammaln(alpha_p) - gammaln(alpha_q))
        + np.sum((alpha_p - alpha_q) * (digamma(alpha_p) - digamma(a0_p)))
    )
    return max(0.0, float(kl))


class DirichletInfoGainReward:
    """
    Dirichlet belief-update KL version of information-oriented intrinsic reward.

    For each condition = (agent_color_before, tile_color_before),
    maintain a Dirichlet posterior over possible outcome colors.

    Intrinsic reward for one observation is:
        KL( posterior || prior )
    where:
        prior = belief before seeing this outcome
        posterior = belief after updating with this outcome
    """

    def __init__(
        self,
        prior_alpha: float = 1.0,
        outcome_space: tuple[Color, ...] = (
            (0, 0),  # black
            (1, 0),  # blue
            (0, 1),  # yellow
            (1, 1),  # white
        ),
        persistent_across_episodes: bool = True,
        reward_clip: float | None = None,
    ):
        self.prior_alpha = float(prior_alpha)
        self.outcome_space = outcome_space
        self.persistent_across_episodes = persistent_across_episodes
        self.reward_clip = reward_clip

        self.outcome_to_idx = {c: i for i, c in enumerate(self.outcome_space)}
        self.num_outcomes = len(self.outcome_space)

        # counts / alphas by condition
        self.alphas: Dict[Condition, np.ndarray] = {}

    def reset(self) -> None:
        """
        Keep beliefs across episodes by default.
        If you want episodic reset instead, set persistent_across_episodes=False.
        """
        if not self.persistent_across_episodes:
            self.alphas = {}

    def _get_alpha(self, condition: Condition, create: bool = True) -> np.ndarray:
        if condition not in self.alphas:
            if not create:
                return np.full(
                    self.num_outcomes,
                    self.prior_alpha,
                    dtype=np.float64,
                )
            self.alphas[condition] = np.full(
                self.num_outcomes,
                self.prior_alpha,
                dtype=np.float64,
            )
        return self.alphas[condition]

    def _split_event(self, event: Event) -> tuple[Condition, Color]:
        agent_before, tile_before, outcome = event
        condition = (agent_before, tile_before)
        return condition, outcome

    def compute(self, info: dict) -> float:
        """
        Compute intrinsic reward from the current mixing_event,
        but do NOT update the internal belief yet.
        """
        event = info.get("mixing_event")
        if event is None:
            return 0.0

        condition, outcome = self._split_event(event)
        if outcome not in self.outcome_to_idx:
            return 0.0

        alpha_old = self._get_alpha(condition, create=False).copy()
        alpha_new = alpha_old.copy()

        out_idx = self.outcome_to_idx[outcome]
        alpha_new[out_idx] += 1.0

        reward = _dirichlet_kl(alpha_new, alpha_old)
        if self.reward_clip is not None:
            reward = min(reward, self.reward_clip)
        return reward

    def update(self, info: dict) -> None:
        """
        Actually apply the Bayesian update after reward has been computed.
        """
        event = info.get("mixing_event")
        if event is None:
            return

        condition, outcome = self._split_event(event)
        if outcome not in self.outcome_to_idx:
            return

        alpha = self._get_alpha(condition)
        out_idx = self.outcome_to_idx[outcome]
        alpha[out_idx] += 1.0

    def get_expected_probs(self) -> Dict[Condition, Dict[Color, float]]:
        """
        Return the expected categorical distribution for each condition.
        """
        result: Dict[Condition, Dict[Color, float]] = {}
        for condition, alpha in self.alphas.items():
            probs = alpha / np.sum(alpha)
            result[condition] = {
                color: float(probs[i]) for i, color in enumerate(self.outcome_space)
            }
        return result

    def get_alpha_table(self) -> Dict[Condition, np.ndarray]:
        return {k: v.copy() for k, v in self.alphas.items()}
