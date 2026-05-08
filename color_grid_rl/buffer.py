"""Rollout buffer for PPO."""

from __future__ import annotations

import numpy as np
import torch


class RolloutBuffer:
    """Store one on-policy rollout before a PPO update."""

    def __init__(self):
        self.clear()

    def clear(self) -> None:
        self.observations = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.log_probs = []
        self.advantages = None
        self.returns = None

    def __len__(self) -> int:
        return len(self.rewards)

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        value: float,
        log_prob: float,
    ) -> None:
        """Store one transition."""
        self.observations.append(obs.copy())
        self.actions.append(action)
        self.rewards.append(float(reward))
        self.dones.append(float(done))
        self.values.append(float(value))
        self.log_probs.append(float(log_prob))

    def compute_returns_and_advantages(
        self,
        last_value: float,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        """Compute GAE advantages and bootstrapped returns."""
        advantages = np.zeros(len(self.rewards), dtype=np.float32)
        last_advantage = 0.0

        for step in reversed(range(len(self.rewards))):
            if step == len(self.rewards) - 1:
                next_non_terminal = 1.0 - self.dones[step]
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[step]
                next_value = self.values[step + 1]

            delta = (
                self.rewards[step]
                + gamma * next_value * next_non_terminal
                - self.values[step]
            )
            last_advantage = (
                delta
                + gamma * gae_lambda * next_non_terminal * last_advantage
            )
            advantages[step] = last_advantage

        self.advantages = advantages
        self.returns = advantages + np.array(self.values, dtype=np.float32)

    def as_tensors(self, device: str) -> dict[str, torch.Tensor]:
        """Convert stored rollout arrays to PyTorch tensors."""
        if self.advantages is None or self.returns is None:
            raise RuntimeError("Call compute_returns_and_advantages before as_tensors.")

        return {
            "observations": torch.as_tensor(
                np.array(self.observations), dtype=torch.float32, device=device
            ),
            "actions": torch.as_tensor(self.actions, dtype=torch.long, device=device),
            "old_log_probs": torch.as_tensor(
                self.log_probs, dtype=torch.float32, device=device
            ),
            "advantages": torch.as_tensor(
                self.advantages, dtype=torch.float32, device=device
            ),
            "returns": torch.as_tensor(self.returns, dtype=torch.float32, device=device),
        }
