"""Dirichlet-Categorical posterior over color-mixing outcomes."""

from __future__ import annotations

from typing import Dict, FrozenSet

import numpy as np
from scipy.special import digamma, gammaln

from chromatic_white.rules import (
    COLORS, COLOR_TO_IDX, NUM_COLORS,
    TRUE_MIXING_RULE, ALL_EDGES, EDGE_LABELS, edge_key,
)


class WorldModel:
    NUM_OUTCOMES = NUM_COLORS

    def __init__(self, prior_alpha: float = 1.0):
        self.prior_alpha = float(prior_alpha)
        self.alpha: Dict[FrozenSet[int], np.ndarray] = {
            e: np.full(self.NUM_OUTCOMES, self.prior_alpha, dtype=np.float64)
            for e in ALL_EDGES
        }
        self.counts: Dict[FrozenSet[int], int] = {e: 0 for e in ALL_EDGES}

    def info_gain(self, c_A, c_B, outcome) -> float:
        key = edge_key(c_A, c_B)
        alpha = self.alpha[key]
        o_idx = COLOR_TO_IDX[outcome]
        alpha_new = alpha.copy()
        alpha_new[o_idx] += 1.0
        return _dirichlet_kl(alpha_new, alpha)

    def update(self, c_A, c_B, outcome) -> None:
        key = edge_key(c_A, c_B)
        o_idx = COLOR_TO_IDX[outcome]
        self.alpha[key][o_idx] += 1.0
        self.counts[key] += 1

    def observe(self, transition) -> float:
        c_A, c_B, outcome = transition
        ig = self.info_gain(c_A, c_B, outcome)
        self.update(c_A, c_B, outcome)
        return ig

    def posterior_mean(self, c_A, c_B) -> np.ndarray:
        key = edge_key(c_A, c_B)
        a = self.alpha[key]
        return a / a.sum()

    def expected_info_gain(self, c_A, c_B) -> float:
        key = edge_key(c_A, c_B)
        alpha = self.alpha[key]
        p_pred = alpha / alpha.sum()
        eig = 0.0
        for o_idx in range(self.NUM_OUTCOMES):
            alpha_new = alpha.copy()
            alpha_new[o_idx] += 1.0
            eig += p_pred[o_idx] * _dirichlet_kl(alpha_new, alpha)
        return float(eig)

    def eig_per_edge(self) -> Dict[FrozenSet[int], float]:
        result = {}
        for edge in ALL_EDGES:
            idxs = sorted(edge)
            if len(idxs) == 1:
                idxs = [idxs[0], idxs[0]]
            c_A, c_B = COLORS[idxs[0]], COLORS[idxs[1]]
            result[edge] = self.expected_info_gain(c_A, c_B)
        return result

    def kl_to_truth(self) -> float:
        total = 0.0
        for edge, true_idx in TRUE_MIXING_RULE.items():
            p = self.alpha[edge] / self.alpha[edge].sum()
            p_true = max(p[true_idx], 1e-12)
            total += -np.log(p_true)
        return total / len(TRUE_MIXING_RULE)

    def top1_accuracy(self) -> float:
        correct = 0
        for edge, true_idx in TRUE_MIXING_RULE.items():
            pred_idx = int(np.argmax(self.alpha[edge]))
            if pred_idx == true_idx:
                correct += 1
        return correct / len(TRUE_MIXING_RULE)

    def edge_coverage(self, min_obs: int = 1) -> float:
        covered = sum(1 for c in self.counts.values() if c >= min_obs)
        return covered / len(self.counts)

    def total_observations(self) -> int:
        return sum(self.counts.values())

    def snapshot(self) -> Dict[str, float]:
        return {
            "wm_kl_to_truth": self.kl_to_truth(),
            "wm_top1_acc": self.top1_accuracy(),
            "wm_coverage_1": self.edge_coverage(1),
            "wm_coverage_5": self.edge_coverage(5),
            "wm_total_obs": self.total_observations(),
        }

    def edge_snapshot(self) -> Dict[str, float]:
        eigs = self.eig_per_edge()
        result = {}
        for edge in ALL_EDGES:
            label = EDGE_LABELS[edge]
            result[f"eig_{label}"] = eigs[edge]
            result[f"count_{label}"] = self.counts[edge]
        return result


def _dirichlet_kl(alpha_p: np.ndarray, alpha_q: np.ndarray) -> float:
    sum_p = alpha_p.sum()
    sum_q = alpha_q.sum()
    kl = (
        gammaln(sum_p) - gammaln(sum_q)
        - np.sum(gammaln(alpha_p) - gammaln(alpha_q))
        + np.sum((alpha_p - alpha_q) * (digamma(alpha_p) - digamma(sum_p)))
    )
    return float(max(kl, 0.0))
