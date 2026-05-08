"""Final evaluation entry point for trained policies.

Run from the repository root with:
    python -m color_grid_rl.test_policy
"""

from __future__ import annotations

from pathlib import Path

import torch

from .config import (
    DEFAULT_ENV_CONFIG,
    DEFAULT_OBS_CONFIG,
    DEFAULT_PPO_CONFIG,
    DEFAULT_REWARD_CONFIG,
    DEFAULT_TRAIN_CONFIG,
)
from .env import ColorGridEnv
from .evaluation import append_eval_row, evaluate_policy
from .main_train import _display_reward_mode, _make_reward_module
from .ppo_agent import PPOAgent


def load_checkpoint_if_available(agent: PPOAgent, checkpoint_path: str | Path | None) -> bool:
    """
    Load a policy checkpoint when one exists.

    TODO: If your training script later saves optimizer state or richer metadata,
    extend this helper to restore that format too.
    """
    if checkpoint_path is None:
        return False

    path = Path(checkpoint_path)
    if not path.exists():
        print(f"Checkpoint not found: {path}. Evaluating the current initialized policy.")
        return False

    checkpoint = torch.load(path, map_location=DEFAULT_TRAIN_CONFIG.device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    agent.model.load_state_dict(state_dict)
    print(f"Loaded checkpoint from {path}")
    return True


def main(
    reward_mode: str = DEFAULT_REWARD_CONFIG.reward_type,
    seed: int = DEFAULT_TRAIN_CONFIG.seed,
    checkpoint_path: str | Path | None = None,
    num_eval_episodes: int = 100,
    output_path: str | Path = "outputs/final_eval_results.csv",
) -> dict:
    env = ColorGridEnv(config=DEFAULT_ENV_CONFIG)
    agent = PPOAgent(
        obs_dim=DEFAULT_OBS_CONFIG.obs_dim,
        action_dim=env.action_space.n,
        config=DEFAULT_PPO_CONFIG,
        device=DEFAULT_TRAIN_CONFIG.device,
    )
    load_checkpoint_if_available(agent, checkpoint_path)

    reward_model = _make_reward_module(reward_mode)
    reward_label = _display_reward_mode(reward_mode)
    metrics = evaluate_policy(
        env=env,
        agent=agent,
        true_rule=env.get_true_rule(),
        outcome_space=env.get_outcome_space(),
        num_eval_episodes=num_eval_episodes,
        deterministic=True,
        reward_model=reward_model,
        device=DEFAULT_TRAIN_CONFIG.device,
    )

    row = {
        "reward_type": reward_label,
        "seed": seed,
        "train_episode": "final",
    }
    row.update(metrics)
    row.update(
        {
            "intrinsic_coef": DEFAULT_REWARD_CONFIG.intrinsic_coef,
            "learning_rate": DEFAULT_PPO_CONFIG.learning_rate,
            "entropy_coef": DEFAULT_PPO_CONFIG.entropy_coef,
            "gamma": DEFAULT_PPO_CONFIG.gamma,
            "gae_lambda": DEFAULT_PPO_CONFIG.gae_lambda,
            "clip_eps": DEFAULT_PPO_CONFIG.clip_epsilon,
            "novelty_alpha": DEFAULT_REWARD_CONFIG.novelty_alpha,
            "info_gain_smoothing": DEFAULT_REWARD_CONFIG.info_gain_smoothing,
            "dirichlet_prior_alpha": DEFAULT_REWARD_CONFIG.dirichlet_prior_alpha,
        }
    )
    saved_path = append_eval_row(output_path, row)
    print(f"Saved final evaluation metrics to {saved_path}")
    return metrics


if __name__ == "__main__":
    main()
