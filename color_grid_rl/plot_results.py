"""Plot learning curves from saved episode CSV logs.

Run from the repository root with:
    python -m color_grid_rl.plot_results
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")
ROLLING_WINDOW = 20
REWARD_LABELS = {
    "random": "random",
    "novelty": "novelty",
    "info_gain": "surprisal",
    "surprisal": "surprisal",
    "dirichlet_info_gain": "dirichlet IG",
    "dirichlet IG": "dirichlet IG",
}
REWARD_ORDER = ["surprisal", "dirichlet IG", "novelty", "random"]
REWARD_COLORS = {
    "surprisal": "#d5912d",
    "dirichlet IG": "#6d63c2",
    "novelty": "#35a983",
    "random": "#8d8b86",
}


def _canonical_reward_mode(reward_mode: str) -> str:
    return REWARD_LABELS.get(reward_mode, reward_mode)


def _ordered_rewards(rewards) -> list[str]:
    reward_set = set(rewards)
    ordered = [reward for reward in REWARD_ORDER if reward in reward_set]
    ordered.extend(sorted(reward_set - set(ordered)))
    return ordered


def _load_rows(results_dir: Path) -> list[dict]:
    latest_paths = {}
    for csv_path in sorted(results_dir.glob("*.csv")):
        if csv_path.name == "summary.csv":
            continue
        with csv_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            file_rows = list(reader)
        if not file_rows:
            continue

        reward_mode = _canonical_reward_mode(file_rows[0]["reward_mode"])
        seed = int(file_rows[0]["seed"])
        key = (reward_mode, seed)
        previous = latest_paths.get(key)
        if previous is None or csv_path.stat().st_mtime > previous.stat().st_mtime:
            latest_paths[key] = csv_path

    rows = []
    for key, csv_path in sorted(latest_paths.items()):
        with csv_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            rows.extend(reader)
    print(
        "Using latest training logs: "
        + ", ".join(
            f"{reward}/seed{seed}={path.name}"
            for (reward, seed), path in sorted(latest_paths.items())
        )
    )
    return rows


def rolling_average(values: list[float], window: int = ROLLING_WINDOW) -> list[float]:
    """Return a simple rolling average with a shorter window at the start."""
    averages = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        window_values = values[start : i + 1]
        averages.append(sum(window_values) / len(window_values))
    return averages


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _average_metric_by_mode_and_episode(
    rows: list[dict],
    metric_name: str,
) -> dict[str, tuple[list[int], list[float]]]:
    grouped = {}
    for row in rows:
        reward_mode = _canonical_reward_mode(row["reward_mode"])
        episode = int(row["episode"])
        value = float(row[metric_name])
        grouped.setdefault(reward_mode, {}).setdefault(episode, []).append(value)

    curves = {}
    for reward_mode, episode_values in grouped.items():
        episodes = sorted(episode_values.keys())
        values = [_mean(episode_values[episode]) for episode in episodes]
        curves[reward_mode] = (episodes, rolling_average(values))
    return curves


def _plot_metric(rows: list[dict], metric_name: str, title: str, ylabel: str, filename: str) -> None:
    curves = _average_metric_by_mode_and_episode(rows, metric_name)

    plt.figure()
    for reward_mode in _ordered_rewards(curves):
        episodes, values = curves[reward_mode]
        plt.plot(
            episodes,
            values,
            color=REWARD_COLORS.get(reward_mode),
            label=reward_mode,
        )

    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURES_DIR / filename
    plt.savefig(output_path)
    plt.close()
    print(f"Saved {output_path}")


def main() -> None:
    rows = _load_rows(RESULTS_DIR)
    if not rows:
        print(f"No result CSV files found in {RESULTS_DIR}")
        return

    _plot_metric(
        rows,
        metric_name="solved",
        title=f"Rolling Success Rate (window={ROLLING_WINDOW})",
        ylabel="Success Rate",
        filename="success_rate.png",
    )
    _plot_metric(
        rows,
        metric_name="steps",
        title=f"Rolling Average Steps (window={ROLLING_WINDOW})",
        ylabel="Steps",
        filename="average_steps.png",
    )
    _plot_metric(
        rows,
        metric_name="overloads",
        title=f"Rolling Average Overloads (window={ROLLING_WINDOW})",
        ylabel="Overloads",
        filename="overloads.png",
    )
    _plot_metric(
        rows,
        metric_name="int_return",
        title=f"Rolling Intrinsic Return (window={ROLLING_WINDOW})",
        ylabel="Intrinsic Return",
        filename="intrinsic_return.png",
    )
    _plot_metric(
        rows,
        metric_name="ext_return",
        title=f"Rolling Extrinsic Return (window={ROLLING_WINDOW})",
        ylabel="Extrinsic Return",
        filename="extrinsic_return.png",
    )


if __name__ == "__main__":
    main()
