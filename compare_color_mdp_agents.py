import argparse
from pathlib import Path
from statistics import mean
from typing import Dict, List

import numpy as np

from color_mdp_agents import AGENT_FACTORIES, ColorMDPDirichletAgentAdapter
from color_mdp_tables import (
    dirichlet_alpha_rows,
    dirichlet_w_rows,
    extract_dirichlet_alpha_table,
    extract_dirichlet_w_table,
    extract_q_table,
    mean_q_table_rows,
    q_table_rows,
    rollout_greedy,
    write_rows,
)
from env import ColorMixingMDPEnv
from plotting import plot_all_q_tables, plot_dirichlet_diagnostics, plot_training_curves
from train import run_training


def train_one_seed(agent_label: str, factory, seed: int, n_episodes: int, env_kwargs: Dict):
    env = ColorMixingMDPEnv(**env_kwargs)
    agent = factory(seed)

    print(f"Training {agent_label} | episodes={n_episodes} | seed={seed}")
    history = run_training(env, agent, n_episodes=n_episodes)
    returns = np.array([h.episode_return_ext for h in history], dtype=float)
    greedy = rollout_greedy(ColorMixingMDPEnv(**env_kwargs), agent)
    q_table = extract_q_table(agent)

    return agent, returns, greedy, q_table


def save_outputs(
    out_dir: Path,
    smooth: int,
    curves: Dict[str, List[np.ndarray]],
    summary_rows: List[Dict],
    raw_rows: List[Dict],
    q_rows: List[Dict],
    q_tables: Dict[str, List[np.ndarray]],
    dirichlet_w_tables: List[np.ndarray],
    dirichlet_alpha_tables: List[np.ndarray],
) -> None:
    curve_arrays = {label: np.vstack(values) for label, values in curves.items()}
    mean_q_tables = {
        agent_label: np.mean(np.stack(values, axis=0), axis=0)
        for agent_label, values in q_tables.items()
    }

    plot_training_curves(curve_arrays, out_dir / "color_mdp_training_returns.png", smooth)
    plot_all_q_tables(mean_q_tables, out_dir)

    write_rows(
        out_dir / "color_mdp_episode_returns.csv",
        ["agent", "seed_index", "seed", "episode", "episode_return_ext"],
        raw_rows,
    )
    write_rows(out_dir / "color_mdp_summary.csv", summary_rows[0].keys(), summary_rows)
    write_rows(
        out_dir / "color_mdp_q_tables.csv",
        ["agent", "seed_index", "seed", "state_id", "state", "action_id", "action", "q_value"],
        q_rows,
    )
    write_rows(
        out_dir / "color_mdp_mean_q_tables.csv",
        ["agent", "state_id", "state", "action_id", "action", "mean_q_value"],
        mean_q_table_rows(mean_q_tables),
    )

    if dirichlet_w_tables and dirichlet_alpha_tables:
        mean_w_table = np.mean(np.stack(dirichlet_w_tables, axis=0), axis=0)
        mean_alpha_table = np.mean(np.stack(dirichlet_alpha_tables, axis=0), axis=0)
        plot_dirichlet_diagnostics(mean_w_table, mean_alpha_table, out_dir)
        write_rows(
            out_dir / "dirichlet_w_tables.csv",
            ["state_id", "state", "action_id", "action", "mean_w_value"],
            dirichlet_w_rows(mean_w_table),
        )
        write_rows(
            out_dir / "dirichlet_alpha_model.csv",
            ["context", "state", "action", "outcome", "mean_alpha"],
            dirichlet_alpha_rows(mean_alpha_table),
        )


def print_summary(out_dir: Path, summary_rows: List[Dict], has_dirichlet_diagnostics: bool) -> None:
    print("\nColor Mixing MDP comparison")
    print("-" * 104)
    print(
        f"{'Agent':38s}"
        f"{'Train mean':>12s}"
        f"{'Last 100':>12s}"
        f"{'Success':>10s}"
        f"{'Best path':>10s}"
        f"  Example greedy path"
    )
    print("-" * 104)
    for row in summary_rows:
        print(
            f"{row['agent']:38s}"
            f"{row['mean_return_across_training']:12.4f}"
            f"{row['mean_return_last_100_episodes']:12.4f}"
            f"{row['greedy_success_rate']:10.3f}"
            f"{row['greedy_best_path_rate']:10.3f}"
            f"  {row['example_greedy_actions']}"
        )

    print(f"\nSaved plot:        {out_dir / 'color_mdp_training_returns.png'}")
    print(f"Saved summary:     {out_dir / 'color_mdp_summary.csv'}")
    print(f"Saved raw returns: {out_dir / 'color_mdp_episode_returns.csv'}")
    print(f"Saved Q tables:    {out_dir / 'color_mdp_q_tables.png'}")
    print(f"Saved Q CSV:       {out_dir / 'color_mdp_q_tables.csv'}")
    if has_dirichlet_diagnostics:
        print(f"Saved Dirichlet w: {out_dir / 'dirichlet_diagnostics' / 'dirichlet_w_table.png'}")
        print(f"Saved alpha model: {out_dir / 'dirichlet_diagnostics' / 'dirichlet_alpha_model.png'}")


def run_experiment(
    n_episodes: int,
    n_seeds: int,
    master_seed: int,
    out_dir: Path,
    smooth: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    env_kwargs = dict(max_steps=6, step_cost=0.02, overload_penalty=0.0)

    raw_rows: List[Dict] = []
    summary_rows: List[Dict] = []
    curves: Dict[str, List[np.ndarray]] = {}
    q_tables: Dict[str, List[np.ndarray]] = {}
    q_rows: List[Dict] = []
    dirichlet_w_tables: List[np.ndarray] = []
    dirichlet_alpha_tables: List[np.ndarray] = []

    for agent_idx, (agent_label, factory) in enumerate(AGENT_FACTORIES):
        seed_mean_returns = []
        seed_last_returns = []
        greedy_rows = []

        for seed_idx in range(n_seeds):
            seed = master_seed + 1000 * agent_idx + seed_idx
            agent, returns, greedy, q_table = train_one_seed(
                agent_label,
                factory,
                seed,
                n_episodes,
                env_kwargs,
            )

            curves.setdefault(agent_label, []).append(returns)
            q_tables.setdefault(agent_label, []).append(q_table)
            q_rows.extend(q_table_rows(agent_label, seed_idx, seed, q_table))
            greedy_rows.append(greedy)
            seed_mean_returns.append(float(returns.mean()))
            seed_last_returns.append(float(returns[-min(100, len(returns)) :].mean()))

            if isinstance(agent, ColorMDPDirichletAgentAdapter):
                dirichlet_w_tables.append(extract_dirichlet_w_table(agent))
                dirichlet_alpha_tables.append(extract_dirichlet_alpha_table(agent))

            for episode, ret in enumerate(returns, start=1):
                raw_rows.append(
                    {
                        "agent": agent_label,
                        "seed_index": seed_idx,
                        "seed": seed,
                        "episode": episode,
                        "episode_return_ext": ret,
                    }
                )

        summary_rows.append(
            {
                "agent": agent_label,
                "n_seeds": n_seeds,
                "n_episodes": n_episodes,
                "mean_return_across_training": mean(seed_mean_returns),
                "mean_return_last_100_episodes": mean(seed_last_returns),
                "greedy_success_rate": mean(row["greedy_success"] for row in greedy_rows),
                "greedy_best_path_rate": mean(
                    row["greedy_actions"] == "red -> green -> blue" for row in greedy_rows
                ),
                "example_greedy_actions": greedy_rows[0]["greedy_actions"],
                "example_greedy_colors": greedy_rows[0]["greedy_colors"],
                "example_greedy_return": greedy_rows[0]["greedy_return"],
            }
        )

    save_outputs(
        out_dir,
        smooth,
        curves,
        summary_rows,
        raw_rows,
        q_rows,
        q_tables,
        dirichlet_w_tables,
        dirichlet_alpha_tables,
    )
    print_summary(out_dir, summary_rows, bool(dirichlet_w_tables and dirichlet_alpha_tables))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare four Q-learning agents on the color mixing MDP."
    )
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--master-seed", type=int, default=123)
    parser.add_argument("--out-dir", type=Path, default=Path("results/color_mdp"))
    parser.add_argument("--smooth", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiment(
        n_episodes=args.episodes,
        n_seeds=args.seeds,
        master_seed=args.master_seed,
        out_dir=args.out_dir,
        smooth=args.smooth,
    )


if __name__ == "__main__":
    main()
