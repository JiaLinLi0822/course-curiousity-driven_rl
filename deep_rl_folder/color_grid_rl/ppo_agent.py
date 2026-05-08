"""Minimal PPO agent."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .buffer import RolloutBuffer
from .config import DEFAULT_PPO_CONFIG, PPOConfig
from .model import ActorCritic


class PPOAgent:
    """Wrap an actor-critic network with PPO action selection and updates."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        config: PPOConfig = DEFAULT_PPO_CONFIG,
        device: str = "cpu",
    ):
        self.config = config
        self.device = device
        self.model = ActorCritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_size=config.hidden_size,
        ).to(device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
        )

    def select_action(
        self,
        obs: np.ndarray,
        deterministic: bool = False,
    ) -> tuple[int, float, float]:
        """Select an action and return action, value, and log probability."""
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        obs_tensor = obs_tensor.unsqueeze(0)

        with torch.no_grad():
            logits, value = self.model(obs_tensor)
            dist = Categorical(logits=logits)
            action = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
            log_prob = dist.log_prob(action)

        return action.item(), value.item(), log_prob.item()

    def act(self, obs: np.ndarray, deterministic: bool = False) -> int:
        """Small evaluation-friendly action API."""
        action, _, _ = self.select_action(obs, deterministic=deterministic)
        return action

    def get_value(self, obs: np.ndarray) -> float:
        """Estimate V(s) for bootstrapping at rollout boundaries."""
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        obs_tensor = obs_tensor.unsqueeze(0)

        with torch.no_grad():
            _, value = self.model(obs_tensor)

        return value.item()

    def update(self, buffer: RolloutBuffer, last_value: float = 0.0) -> dict[str, float]:
        """Run PPO updates over the current rollout buffer."""
        if len(buffer) == 0:
            return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        buffer.compute_returns_and_advantages(
            last_value=last_value,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )
        data = buffer.as_tensors(self.device)

        observations = data["observations"]
        actions = data["actions"]
        old_log_probs = data["old_log_probs"]
        advantages = data["advantages"]
        returns = data["returns"]

        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        num_steps = len(buffer)
        minibatch_size = min(self.config.minibatch_size, num_steps)

        last_stats = {}
        for _ in range(self.config.update_epochs):
            indices = torch.randperm(num_steps, device=self.device)

            for start in range(0, num_steps, minibatch_size):
                batch_idx = indices[start : start + minibatch_size]

                logits, values = self.model(observations[batch_idx])
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(actions[batch_idx])
                entropy = dist.entropy().mean()

                log_ratio = new_log_probs - old_log_probs[batch_idx]
                ratio = torch.exp(log_ratio)

                unclipped = ratio * advantages[batch_idx]
                clipped = torch.clamp(
                    ratio,
                    1.0 - self.config.clip_epsilon,
                    1.0 + self.config.clip_epsilon,
                ) * advantages[batch_idx]
                policy_loss = -torch.min(unclipped, clipped).mean()

                value_loss = nn.functional.mse_loss(values, returns[batch_idx])

                loss = (
                    policy_loss
                    + self.config.value_loss_coef * value_loss
                    - self.config.entropy_coef * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm,
                )
                self.optimizer.step()

                last_stats = {
                    "loss": loss.item(),
                    "policy_loss": policy_loss.item(),
                    "value_loss": value_loss.item(),
                    "entropy": entropy.item(),
                }

        return last_stats
