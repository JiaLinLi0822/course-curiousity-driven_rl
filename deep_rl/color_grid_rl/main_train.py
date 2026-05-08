"""Minimal PPO training script with switchable intrinsic reward.

Run from the repository root with:
    python -m color_grid_rl.main_train
"""

from __future__ import annotations

import csv
import numpy as np
from pathlib import Path
import torch

from .buffer import RolloutBuffer
from .config import (
    DEFAULT_ENV_CONFIG,
    DEFAULT_EVAL_CONFIG,
    DEFAULT_OBS_CONFIG,
    DEFAULT_PPO_CONFIG,
    DEFAULT_REWARD_CONFIG,
    DEFAULT_TRAIN_CONFIG,
)
from .dirichlet_info_gain import DirichletInfoGainReward
from .env import ColorGridEnv
from .evaluation import append_eval_row, evaluate_policy
from .info_gain import InfoGainReward
from .novelty import NoveltyReward
from .ppo_agent import PPOAgent
from .rewards import NoIntrinsicReward, RandomIntrinsicReward, combine_rewards


REWARD_MODE_ALIASES = {
    "none": "none",
    "random": "random",
    "novelty": "novelty",
    "surprisal": "info_gain",
    "info_gain": "info_gain",
    "dirichlet IG": "dirichlet_info_gain",
    "dirichlet_ig": "dirichlet_info_gain",
    "dirichlet_info_gain": "dirichlet_info_gain",
}

REWARD_MODE_DISPLAY_NAMES = {
    "none": "none",
    "random": "random",
    "novelty": "novelty",
    "info_gain": "surprisal",
    "dirichlet_info_gain": "dirichlet IG",
}


def _canonical_reward_mode(reward_mode: str) -> str:
    try:
        return REWARD_MODE_ALIASES[reward_mode]
    except KeyError as exc:
        raise ValueError(f"Unknown reward_mode: {reward_mode}") from exc


def _display_reward_mode(reward_mode: str) -> str:
    canonical_mode = _canonical_reward_mode(reward_mode)
    return REWARD_MODE_DISPLAY_NAMES[canonical_mode]


def _print_update_log(update_count: int, buffer_size: int, stats: dict[str, float]) -> None:
    """Print one PPO update log line."""
    print(
        "[update] "
        f"update={update_count} "
        f"buffer_size={buffer_size} "
        f"loss={stats['loss']:.4f} "
        f"policy_loss={stats['policy_loss']:.4f} "
        f"value_loss={stats['value_loss']:.4f} "
        f"entropy={stats['entropy']:.4f}"
    )


def _make_reward_module(reward_mode: str):
    """Create the intrinsic reward module requested by reward_mode."""
    reward_mode = _canonical_reward_mode(reward_mode)
    if reward_mode == "none":
        return NoIntrinsicReward()
    if reward_mode == "random":
        return RandomIntrinsicReward()
    if reward_mode == "novelty":
        return NoveltyReward(
            alpha=DEFAULT_REWARD_CONFIG.novelty_alpha,
            persistent_counts=DEFAULT_REWARD_CONFIG.novelty_persistent_counts,
        )
    if reward_mode == "info_gain":
        return InfoGainReward(
            smoothing=DEFAULT_REWARD_CONFIG.info_gain_smoothing,
            reward_clip=DEFAULT_REWARD_CONFIG.info_gain_clip,
        )
    if reward_mode == "dirichlet_info_gain":
        return DirichletInfoGainReward(
            prior_alpha=DEFAULT_REWARD_CONFIG.dirichlet_prior_alpha,
            persistent_across_episodes=(
                DEFAULT_REWARD_CONFIG.dirichlet_persistent_across_episodes
            ),
            reward_clip=DEFAULT_REWARD_CONFIG.dirichlet_clip,
        )

    raise ValueError(f"Unknown reward_mode: {reward_mode}")


def _save_episode_records(
    episode_records: list[dict],
    reward_mode: str,
    seed: int,
    results_dir: str | Path = "results",
) -> Path:
    """Save episode logs to one CSV file."""
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    csv_path = results_path / f"{reward_mode}_seed{seed}.csv"
    fieldnames = [
        "reward_mode",
        "seed",
        "episode",
        "return",
        "ext_return",
        "int_return",
        "solved",
        "steps",
        "overloads",
    ]

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(episode_records)

    return csv_path


def _make_eval_row(reward_mode: str, seed: int, train_episode: int, metrics: dict) -> dict:
    """Combine evaluation metrics with experiment metadata for fair comparison."""
    row = {
        "reward_type": reward_mode,
        "seed": seed,
        "train_episode": train_episode,
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
    return row


def train(
    reward_mode: str = "dirichlet IG",
    seed: int | None = None,
    total_timesteps: int | None = None,
    results_dir: str | Path = "results",
    save_csv: bool = True,
) -> list[dict]:
    """Run a small end-to-end PPO training loop."""
    if seed is None:
        seed = DEFAULT_TRAIN_CONFIG.seed
    if total_timesteps is None:
        total_timesteps = DEFAULT_TRAIN_CONFIG.total_timesteps

    np.random.seed(seed)
    torch.manual_seed(seed)

    env = ColorGridEnv(config=DEFAULT_ENV_CONFIG)
    agent = PPOAgent(
        obs_dim=DEFAULT_OBS_CONFIG.obs_dim,
        action_dim=env.action_space.n,
        config=DEFAULT_PPO_CONFIG,
        device=DEFAULT_TRAIN_CONFIG.device,
    )
    buffer = RolloutBuffer()
    intrinsic_reward = _make_reward_module(reward_mode)
    reward_label = _display_reward_mode(reward_mode)
    eval_env = ColorGridEnv(config=DEFAULT_ENV_CONFIG)
    episode_records = []

    print(f"Starting training with reward_mode={reward_label} seed={seed}")

    obs, info = env.reset(seed=seed)

    episode_count = 0
    update_count = 0

    episode_return = 0.0
    episode_extrinsic_return = 0.0
    episode_intrinsic_return = 0.0

    for timestep in range(1, total_timesteps + 1):
        action, value, log_prob = agent.select_action(obs)

        next_obs, extrinsic_reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        intrinsic = intrinsic_reward.compute(info)
        total_reward = combine_rewards(
            extrinsic_reward=extrinsic_reward,
            intrinsic_reward=intrinsic,
            config=DEFAULT_REWARD_CONFIG,
        )

        buffer.add(
            obs=obs,
            action=action,
            reward=total_reward,
            done=done,
            value=value,
            log_prob=log_prob,
        )

        episode_return += total_reward
        episode_extrinsic_return += extrinsic_reward
        episode_intrinsic_return += intrinsic

        # Update intrinsic reward model after computing reward for this step.
        intrinsic_reward.update(info)

        obs = next_obs

        if len(buffer) >= DEFAULT_PPO_CONFIG.rollout_steps:
            last_value = 0.0 if done else agent.get_value(obs)
            buffer_size = len(buffer)

            update_stats = agent.update(buffer, last_value=last_value)
            update_count += 1
            _print_update_log(update_count, buffer_size, update_stats)

            buffer.clear()

        if done:
            episode_count += 1
            solved = info["solved"]

            print(
                "episode="
                f"{episode_count:04d} "
                f"return={episode_return:.2f} "
                f"ext_return={episode_extrinsic_return:.2f} "
                f"int_return={episode_intrinsic_return:.2f} "
                f"solved={solved} "
                f"steps={info['step_count']} "
                f"overloads={info['overload_count']}"
            )

            episode_records.append(
                {
                    "reward_mode": reward_label,
                    "seed": seed,
                    "episode": episode_count,
                    "return": episode_return,
                    "ext_return": episode_extrinsic_return,
                    "int_return": episode_intrinsic_return,
                    "solved": int(solved),
                    "steps": info["step_count"],
                    "overloads": info["overload_count"],
                }
            )

            if episode_count % DEFAULT_EVAL_CONFIG.eval_every_episodes == 0:
                eval_metrics = evaluate_policy(
                    env=eval_env,
                    agent=agent,
                    true_rule=eval_env.get_true_rule(),
                    outcome_space=eval_env.get_outcome_space(),
                    num_eval_episodes=DEFAULT_EVAL_CONFIG.num_eval_episodes,
                    deterministic=DEFAULT_EVAL_CONFIG.deterministic,
                    reward_model=intrinsic_reward,
                    device=DEFAULT_TRAIN_CONFIG.device,
                )
                eval_row = _make_eval_row(
                    reward_mode=reward_label,
                    seed=seed,
                    train_episode=episode_count,
                    metrics=eval_metrics,
                )
                if DEFAULT_EVAL_CONFIG.save_eval_csv:
                    eval_path = append_eval_row(DEFAULT_EVAL_CONFIG.output_path, eval_row)
                    print(f"Saved eval metrics to {eval_path}")
                else:
                    print(
                        "[eval] "
                        f"episode={episode_count} "
                        f"success_rate={eval_metrics['success_rate']:.2f} "
                        f"rule_accuracy_all={eval_metrics['rule_accuracy_all']:.2f}"
                    )

            obs, info = env.reset()

            # For novelty / info modules, reset() should usually preserve
            # across-episode knowledge unless the module is configured otherwise.
            intrinsic_reward.reset()

            episode_return = 0.0
            episode_extrinsic_return = 0.0
            episode_intrinsic_return = 0.0

    if len(buffer) > 0:
        buffer_size = len(buffer)
        last_value = agent.get_value(obs)

        update_stats = agent.update(buffer, last_value=last_value)
        update_count += 1
        _print_update_log(update_count, buffer_size, update_stats)

        buffer.clear()

    if save_csv:
        csv_path = _save_episode_records(
            episode_records=episode_records,
            reward_mode=reward_label,
            seed=seed,
            results_dir=results_dir,
        )
        print(f"Saved episode logs to {csv_path}")

    return episode_records


if __name__ == "__main__":
    train()
