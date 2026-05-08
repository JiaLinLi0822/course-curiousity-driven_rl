"""Intrinsic reward modules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict

import numpy as np

from chromatic_white.world_model import WorldModel
from chromatic_white.rules import COLOR_TO_IDX


class IntrinsicRewardModule(ABC):
    name: str = "base"

    @abstractmethod
    def compute(self, obs, next_obs, transition, was_oob) -> float:
        ...

    def train_step(self, batch: Dict) -> Dict[str, float]:
        return {}


class ZeroIntrinsic(IntrinsicRewardModule):
    name = "zero"

    def compute(self, obs, next_obs, transition, was_oob):
        return 0.0


class InfoGainIntrinsic(IntrinsicRewardModule):
    name = "info_gain"

    def __init__(self, world_model: WorldModel, scale: float = 1.0):
        self.world_model = world_model
        self.scale = scale

    def compute(self, obs, next_obs, transition, was_oob):
        if was_oob:
            return 0.0
        c_A, c_B, outcome = transition
        ig = self.world_model.info_gain(c_A, c_B, outcome)
        return self.scale * ig


class SurprisalIntrinsic(IntrinsicRewardModule):
    name = "surprisal"

    def __init__(self, world_model: WorldModel, scale: float = 1.0):
        self.world_model = world_model
        self.scale = scale

    def compute(self, obs, next_obs, transition, was_oob):
        if was_oob:
            return 0.0
        c_A, c_B, outcome = transition
        pred = self.world_model.posterior_mean(c_A, c_B)
        p = max(float(pred[COLOR_TO_IDX[outcome]]), 1e-12)
        return self.scale * (-np.log(p))


class ExpectedInfoGainIntrinsic(IntrinsicRewardModule):
    name = "expected_info_gain"

    def __init__(self, world_model: WorldModel, scale: float = 1.0):
        self.world_model = world_model
        self.scale = scale

    def compute(self, obs, next_obs, transition, was_oob):
        if was_oob:
            return 0.0
        c_A, c_B, _outcome = transition
        eig = self.world_model.expected_info_gain(c_A, c_B)
        return self.scale * eig


class NoveltyIntrinsic(IntrinsicRewardModule):
    name = "novelty"

    def __init__(self, scale: float = 1.0, n_states_estimate: int = 10000):
        self.scale = scale
        self.n_states_estimate = n_states_estimate
        self.counts: Dict[bytes, int] = {}
        self.total_visits = 0

    def _hash(self, obs: np.ndarray) -> bytes:
        return (obs[:-1] > 0.5).astype(np.int8).tobytes()

    def compute(self, obs, next_obs, transition, was_oob):
        h = self._hash(next_obs)
        self.counts[h] = self.counts.get(h, 0) + 1
        self.total_visits += 1
        c = self.counts[h]
        p = (c + 1.0 / self.n_states_estimate) / (self.total_visits + 1.0)
        p = max(p, 1e-12)
        return self.scale * (-np.log(p))


class HybridIntrinsic(IntrinsicRewardModule):
    name = "hybrid"

    def __init__(self, world_model: WorldModel, alpha: float = 0.5, scale: float = 1.0,
                 n_states_estimate: int = 10000):
        self.world_model = world_model
        self.alpha = alpha
        self.scale = scale
        self.novelty_mod = NoveltyIntrinsic(scale=1.0, n_states_estimate=n_states_estimate)
        self.info_gain_mod = InfoGainIntrinsic(world_model, scale=1.0)

        self._ig_sum = 0.0
        self._ig_sq_sum = 0.0
        self._nov_sum = 0.0
        self._nov_sq_sum = 0.0
        self._n = 1e-4

    def _update_stats(self, ig: float, nov: float):
        self._n += 1
        self._ig_sum += ig
        self._ig_sq_sum += ig * ig
        self._nov_sum += nov
        self._nov_sq_sum += nov * nov

    def _std(self, sum_, sq_sum_):
        mean = sum_ / self._n
        var = max(sq_sum_ / self._n - mean * mean, 1e-8)
        return float(np.sqrt(var))

    def compute(self, obs, next_obs, transition, was_oob):
        ig = self.info_gain_mod.compute(obs, next_obs, transition, was_oob)
        nov = self.novelty_mod.compute(obs, next_obs, transition, was_oob)
        self._update_stats(ig, nov)
        ig_std = self._std(self._ig_sum, self._ig_sq_sum)
        nov_std = self._std(self._nov_sum, self._nov_sq_sum)
        combined = self.alpha * (ig / ig_std) + (1.0 - self.alpha) * (nov / nov_std)
        return self.scale * combined
