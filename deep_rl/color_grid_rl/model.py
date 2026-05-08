"""Simple actor-critic model for PPO."""

from __future__ import annotations

import torch
from torch import nn


class ActorCritic(nn.Module):
    """Small MLP with separate policy and value heads."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int = 64):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden_size, action_dim)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return action logits and scalar state value."""
        features = self.shared(obs)
        action_logits = self.policy_head(features)
        value = self.value_head(features).squeeze(-1)
        return action_logits, value
