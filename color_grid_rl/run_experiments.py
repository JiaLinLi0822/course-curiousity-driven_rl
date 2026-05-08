"""Run PPO experiments across reward modes and random seeds.

Run from the repository root with:
    python -m color_grid_rl.run_experiments
"""

from __future__ import annotations

from .config import DEFAULT_TRAIN_CONFIG
from .main_train import train


REWARD_MODES = ["random", "novelty", "surprisal", "dirichlet IG"]
SEEDS = [0, 1, 2]


def main() -> None:
    for reward_mode in REWARD_MODES:
        for seed in SEEDS:
            print("=" * 60)
            print(f"Running reward_mode={reward_mode}, seed={seed}")
            train(
                reward_mode=reward_mode,
                seed=seed,
                total_timesteps=DEFAULT_TRAIN_CONFIG.total_timesteps,
                results_dir="results",
                save_csv=True,
            )


if __name__ == "__main__":
    main()
