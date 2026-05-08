"""Plot evaluation metrics from outputs/eval_results.csv.

Run from the repository root with:
    python -m color_grid_rl.plot_eval_results
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


EVAL_CSV = Path("outputs/eval_results.csv")
PLOTS_DIR = Path("outputs/plots")
DEFAULT_NUM_EVAL_EPISODES = 50
REWARD_LABELS = {
    "random": "random",
    "novelty": "novelty",
    "info_gain": "surprisal",
    "surprisal": "surprisal",
    "dirichlet_info_gain": "dirichlet IG",
    "dirichlet IG": "dirichlet IG",
}
PRESENTATION_LABELS = {
    "surprisal": "Predictive\nsurprisal",
    "dirichlet IG": "Dirichlet IG",
    "novelty": "Novelty",
    "random": "Random",
}
REWARD_ORDER = ["surprisal", "dirichlet IG", "novelty", "random"]
REWARD_COLORS = {
    "surprisal": "#d5912d",
    "dirichlet IG": "#6d63c2",
    "novelty": "#35a983",
    "random": "#8d8b86",
}
BUDGET_HATCHES = ("////", "", "....")
MATPLOTLIB_STYLE = {
    "font.family": "Arial",
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.75,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "legend.loc": "upper right",
}
plt.rcParams.update(MATPLOTLIB_STYLE)


def _load_rows(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        print(f"No evaluation CSV found at {csv_path}")
        return []

    with csv_path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def _to_float(value):
    if value in ("", "None", None):
        return math.nan
    return float(value)


def _reward_label(row: dict) -> str:
    return REWARD_LABELS.get(row["reward_type"], row["reward_type"])


def _comparison_rows(rows: list[dict]) -> list[dict]:
    """
    Build the comparison set from the longest available segment per reward/seed.

    outputs/eval_results.csv is append-only, so after re-running experiments it
    can contain both original long runs and later shorter reruns.
    """
    segments_by_key = {}
    active_key = None
    active_segment = []
    active_last_episode = None

    def flush_active_segment() -> None:
        if active_key is not None and active_segment:
            segments_by_key.setdefault(active_key, []).append(list(active_segment))

    for row in rows:
        key = (row["reward_type"], _reward_label(row), int(row["seed"]))
        train_episode = int(row["train_episode"])

        starts_new_segment = (
            key != active_key
            or active_last_episode is None
            or train_episode <= active_last_episode
        )
        if starts_new_segment:
            flush_active_segment()
            active_segment = []

        active_key = key
        active_segment.append(row)
        active_last_episode = train_episode

    flush_active_segment()

    selected_segments = {}
    for (_raw_reward, reward, seed), segments in segments_by_key.items():
        selected = max(segments, key=len)
        canonical_key = (reward, seed)
        current = selected_segments.get(canonical_key)
        if current is None:
            selected_segments[canonical_key] = selected
        elif len(selected) > len(current):
            selected_segments[canonical_key] = selected

    filtered_rows = [row for key in sorted(selected_segments) for row in selected_segments[key]]
    print(
        "Using comparison eval segments: "
        + ", ".join(
            f"{reward}/seed{seed}={len(segment)} rows"
            for (reward, seed), segment in sorted(selected_segments.items())
        )
    )
    return filtered_rows


def _ordered_rewards(rewards) -> list[str]:
    reward_set = set(rewards)
    ordered = [reward for reward in REWARD_ORDER if reward in reward_set]
    ordered.extend(sorted(reward_set - set(ordered)))
    return ordered


def _mean(values: list[float]) -> float:
    clean_values = [value for value in values if not math.isnan(value)]
    return float(np.mean(clean_values)) if clean_values else math.nan


def _sem(values: list[float]) -> float:
    clean_values = [value for value in values if not math.isnan(value)]
    if len(clean_values) <= 1:
        return 0.0
    return float(np.std(clean_values, ddof=1) / math.sqrt(len(clean_values)))


def _select_three_budgets(rows: list[dict]) -> list[int]:
    episodes_by_reward = {}
    for row in rows:
        episodes_by_reward.setdefault(_reward_label(row), set()).add(
            int(row["train_episode"])
        )

    common_episodes = set.intersection(*episodes_by_reward.values()) if episodes_by_reward else set()
    episodes = sorted(common_episodes or {int(row["train_episode"]) for row in rows})
    if len(episodes) <= 3:
        return episodes
    return [episodes[0], episodes[len(episodes) // 2], episodes[-1]]


def _save_current_figure(output_path: Path, dpi: int | None = None) -> Path:
    """Save a figure, falling back to a suffixed filename if the target is locked."""
    try:
        plt.savefig(output_path, dpi=dpi) if dpi is not None else plt.savefig(output_path)
        return output_path
    except PermissionError:
        for idx in range(1, 100):
            fallback_path = output_path.with_name(
                f"{output_path.stem}_new{idx}{output_path.suffix}"
            )
            if fallback_path.exists():
                continue
            plt.savefig(fallback_path, dpi=dpi) if dpi is not None else plt.savefig(
                fallback_path
            )
            print(f"Could not overwrite {output_path}; saved {fallback_path} instead")
            return fallback_path
        raise


def _curves_by_reward(rows: list[dict], metric_name: str) -> dict[str, tuple[list[int], list[float]]]:
    grouped = {}
    for row in rows:
        reward_type = _reward_label(row)
        train_episode = int(row["train_episode"])
        grouped.setdefault(reward_type, {}).setdefault(train_episode, []).append(
            _to_float(row[metric_name])
        )

    curves = {}
    for reward_type, episode_values in grouped.items():
        episodes = sorted(episode_values)
        values = []
        for episode in episodes:
            clean_values = [v for v in episode_values[episode] if not math.isnan(v)]
            values.append(sum(clean_values) / len(clean_values) if clean_values else math.nan)
        curves[reward_type] = (episodes, values)
    return curves


def _curve_stats_by_reward(
    rows: list[dict],
    metric_name: str,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    grouped = _metric_series_by_reward_seed(rows, metric_name)
    curves = {}
    for reward_type, seed_series in grouped.items():
        episode_values = {}
        for series in seed_series.values():
            for episode, value in series:
                episode_values.setdefault(episode, []).append(value)

        episodes = np.array(sorted(episode_values), dtype=np.float64)
        means = np.array([_mean(episode_values[int(episode)]) for episode in episodes])
        sems = np.array([_sem(episode_values[int(episode)]) for episode in episodes])
        curves[reward_type] = (episodes, means, sems)
    return curves


def _plot_metric(rows: list[dict], metric_name: str, ylabel: str, filename: str) -> None:
    curves = _curve_stats_by_reward(rows, metric_name)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for reward_type in _ordered_rewards(curves):
        episodes, means, sems = curves[reward_type]
        color = REWARD_COLORS.get(reward_type)
        ax.plot(
            episodes,
            means,
            linewidth=1.6,
            color=color,
            label=reward_type,
        )
        ax.fill_between(
            episodes,
            means - sems,
            means + sems,
            color=color,
            alpha=0.16,
            linewidth=0,
        )

    ax.set_xlabel("Train Episode")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=True)
    fig.tight_layout()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PLOTS_DIR / filename
    output_path = _save_current_figure(output_path)
    plt.close(fig)
    print(f"Saved {output_path}")


def _plot_checkpoint_bar(
    rows: list[dict],
    metric_name: str = "success_rate",
    filename: str = "success_rate_checkpoint_bars.png",
) -> None:
    budgets = _select_three_budgets(rows)
    if not budgets:
        return

    grouped = {}
    for row in rows:
        reward_type = _reward_label(row)
        train_episode = int(row["train_episode"])
        if train_episode not in budgets:
            continue
        grouped.setdefault(reward_type, {}).setdefault(train_episode, []).append(
            _to_float(row[metric_name])
        )

    rewards = _ordered_rewards(grouped)
    if not rewards:
        return

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    x = np.array(budgets, dtype=np.float64)
    for reward in rewards:
        means = np.array([_mean(grouped.get(reward, {}).get(budget, [])) for budget in budgets])
        sems = np.array([_sem(grouped.get(reward, {}).get(budget, [])) for budget in budgets])
        color = REWARD_COLORS.get(reward, "#777777")
        ax.plot(x, means, linewidth=1.6, color=color, label=reward)
        ax.fill_between(x, means - sems, means + sems, color=color, alpha=0.16, linewidth=0)

    ax.set_title("Eval success rate at three training checkpoints")
    ax.set_xlabel("Train Episode")
    ax.set_ylabel(f"Eval success rate (seeds x {DEFAULT_NUM_EVAL_EPISODES} eval episodes)")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=True)
    fig.tight_layout()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PLOTS_DIR / filename
    output_path = _save_current_figure(output_path, dpi=180)
    plt.close(fig)
    print(f"Saved {output_path}")


def _metric_series_by_reward_seed(
    rows: list[dict],
    metric_name: str,
) -> dict[str, dict[int, list[tuple[int, float]]]]:
    grouped = {}
    for row in rows:
        value = _to_float(row.get(metric_name))
        if math.isnan(value):
            continue
        reward_type = _reward_label(row)
        seed = int(row["seed"])
        episode = int(row["train_episode"])
        grouped.setdefault(reward_type, {}).setdefault(seed, []).append((episode, value))

    for seed_series in grouped.values():
        for seed, series in seed_series.items():
            seed_series[seed] = sorted(series)
    return grouped


def _plot_mean_with_sem(ax, grouped, reward_type: str, label: str) -> None:
    seed_series = grouped.get(reward_type, {})
    if not seed_series:
        return

    episode_values = {}
    for series in seed_series.values():
        for episode, value in series:
            episode_values.setdefault(episode, []).append(value)

    episodes = sorted(episode_values)
    means = np.array([_mean(episode_values[episode]) for episode in episodes])
    sems = np.array([_sem(episode_values[episode]) for episode in episodes])
    color = REWARD_COLORS.get(reward_type)

    ax.plot(episodes, means, color=color, linewidth=1.8, label=label)
    ax.fill_between(episodes, means - sems, means + sems, color=color, alpha=0.16, linewidth=0)


def _plot_rule_and_speed_panels(
    rows: list[dict],
    filename: str = "rule_learning_and_speed_panels.png",
) -> None:
    kl_grouped = _metric_series_by_reward_seed(rows, "belief_kl_true_to_learned")
    steps_grouped = _metric_series_by_reward_seed(rows, "mean_steps_to_solve")
    rewards = _ordered_rewards(set(kl_grouped) | set(steps_grouped))
    if not rewards:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for reward_type in rewards:
        label = PRESENTATION_LABELS.get(reward_type, reward_type).replace("\n", " ")
        _plot_mean_with_sem(axes[0], kl_grouped, reward_type, label)
        _plot_mean_with_sem(axes[1], steps_grouped, reward_type, label)

    axes[0].set_title("Rule learning over training\nlower = closer to truth")
    axes[0].set_xlabel("Train Episode")
    axes[0].set_ylabel("KL(true rule || learned posterior) (log)")
    axes[0].set_yscale("log")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].set_title("Eval steps to solve over training\nrule learned -> faster solving")
    axes[1].set_xlabel("Train Episode")
    axes[1].set_ylabel("Eval steps to solve")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PLOTS_DIR / filename
    try:
        fig.savefig(output_path, dpi=180)
    except PermissionError:
        for idx in range(1, 100):
            fallback_path = output_path.with_name(
                f"{output_path.stem}_new{idx}{output_path.suffix}"
            )
            if fallback_path.exists():
                continue
            fig.savefig(fallback_path, dpi=180)
            print(f"Could not overwrite {output_path}; saved {fallback_path} instead")
            output_path = fallback_path
            break
    plt.close(fig)
    print(f"Saved {output_path}")


def _linear_slope(series: list[tuple[int, float]]) -> float | None:
    clean_series = [(x, y) for x, y in series if not math.isnan(y)]
    if len(clean_series) < 2:
        return None
    x = np.array([item[0] for item in clean_series], dtype=np.float64)
    y = np.array([item[1] for item in clean_series], dtype=np.float64)
    slope, _ = np.polyfit(x, y, deg=1)
    return float(slope * 100.0)


def _plot_speedup_slopes(
    rows: list[dict],
    filename: str = "speedup_slopes.png",
) -> None:
    grouped = _metric_series_by_reward_seed(rows, "mean_steps_to_solve")
    rewards = _ordered_rewards(grouped)
    if not rewards:
        return

    slope_values = {
        reward: [
            slope
            for series in grouped[reward].values()
            if (slope := _linear_slope(series)) is not None
        ]
        for reward in rewards
    }
    slope_values = {reward: values for reward, values in slope_values.items() if values}
    rewards = _ordered_rewards(slope_values)
    if not rewards:
        return

    x = np.arange(len(rewards))
    means = [_mean(slope_values[reward]) for reward in rewards]
    errors = [_sem(slope_values[reward]) for reward in rewards]
    colors = [REWARD_COLORS.get(reward, "#777777") for reward in rewards]

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.axhline(0.0, color="#222222", linewidth=0.75)
    ax.bar(
        x,
        means,
        color=colors,
        edgecolor="#333333",
        linewidth=0.75,
        yerr=errors,
        capsize=3,
        error_kw={"linewidth": 0.75, "capthick": 0.75},
    )

    ax.set_title("Speedup slopes\nnegative = solves get faster across training; positive = slower or flat")
    ax.set_ylabel("Eval steps-to-solve change\nper 100 train episodes")
    ax.set_xticks(x)
    ax.set_xticklabels([PRESENTATION_LABELS.get(reward, reward) for reward in rewards])
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PLOTS_DIR / filename
    output_path = _save_current_figure(output_path, dpi=180)
    plt.close(fig)
    print(f"Saved {output_path}")


def main() -> None:
    rows = _load_rows(EVAL_CSV)
    if not rows:
        return
    rows = _comparison_rows(rows)

    plots = [
        ("blue_yellow_rule_accuracy", "Blue+Yellow -> White Accuracy", "blue_yellow_rule_accuracy.png"),
        ("blue_yellow_condition_coverage", "Blue+Yellow Condition Coverage", "blue_yellow_condition_coverage.png"),
        ("blue_yellow_white_events", "Blue+Yellow -> White Events", "blue_yellow_white_events.png"),
        ("rule_accuracy_all", "Rule Accuracy (All Conditions)", "rule_accuracy_all.png"),
        ("condition_coverage", "Condition Coverage", "condition_coverage.png"),
        ("belief_kl_true_to_learned", "KL(true || learned)", "belief_kl_true_to_learned.png"),
        ("success_rate", "Success Rate", "success_rate.png"),
        ("mean_steps_to_solve", "Mean Steps to Solve", "mean_steps_to_solve.png"),
        ("mean_intrinsic_return", "Mean Intrinsic Return", "mean_intrinsic_return.png"),
        ("overload_rate", "Overload Rate", "overload_rate.png"),
    ]

    for metric_name, ylabel, filename in plots:
        if metric_name in rows[0]:
            _plot_metric(rows, metric_name, ylabel, filename)

    if "success_rate" in rows[0]:
        _plot_checkpoint_bar(rows)
    if {"belief_kl_true_to_learned", "mean_steps_to_solve"}.issubset(rows[0]):
        _plot_rule_and_speed_panels(rows)
        _plot_speedup_slopes(rows)


if __name__ == "__main__":
    main()
