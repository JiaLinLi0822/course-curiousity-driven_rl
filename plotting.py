import os
from pathlib import Path
from typing import Dict

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parent / ".mplconfig"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from color_mdp_constants import ACTION_LABELS, COLOR_STATE_LABELS
from color_mdp_tables import alpha_context_labels


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


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    kernel = np.ones(window, dtype=float) / float(window)
    padded = np.pad(values, (window - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def plot_training_curves(curves: Dict[str, np.ndarray], out_path: Path, smooth: int) -> None:
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    episode_axis = np.arange(1, next(iter(curves.values())).shape[1] + 1)

    for agent_name, returns_by_seed in curves.items():
        mean_returns = returns_by_seed.mean(axis=0)
        if returns_by_seed.shape[0] > 1:
            sem = returns_by_seed.std(axis=0, ddof=1) / np.sqrt(returns_by_seed.shape[0])
        else:
            sem = np.zeros_like(mean_returns)

        y = moving_average(mean_returns, smooth)
        lo = moving_average(mean_returns - sem, smooth)
        hi = moving_average(mean_returns + sem, smooth)
        ax.plot(episode_axis, y, label=agent_name, linewidth=2.2)
        ax.fill_between(episode_axis, lo, hi, alpha=0.16, linewidth=0)

    ax.set_xlabel("Training episode")
    ax.set_ylabel("Average Return")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _annotate_heatmap(ax, table: np.ndarray) -> None:
    for row in range(table.shape[0]):
        for col in range(table.shape[1]):
            ax.text(
                col,
                row,
                f"{table[row, col]:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="black",
            )


def plot_q_table_heatmap(
    q_table: np.ndarray,
    title: str,
    out_path: Path,
    vmin: float,
    vmax: float,
) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 2.4))
    im = ax.imshow(q_table, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("State color")
    ax.set_ylabel("Action")
    ax.set_xticks(range(len(COLOR_STATE_LABELS)))
    ax.set_xticklabels(COLOR_STATE_LABELS, rotation=35, ha="right")
    ax.set_yticks(range(len(ACTION_LABELS)))
    ax.set_yticklabels(ACTION_LABELS)
    _annotate_heatmap(ax, q_table)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03, label="Q value")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_dirichlet_diagnostics(
    mean_w_table: np.ndarray,
    mean_alpha_table: np.ndarray,
    out_dir: Path,
) -> None:
    diag_dir = out_dir / "dirichlet_diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    max_w = float(mean_w_table.max())
    min_w = float(mean_w_table.min())
    plot_q_table_heatmap(
        mean_w_table,
        "Curiosity-based Q-learning: w(s,a)",
        diag_dir / "dirichlet_w_table.png",
        vmin=min_w,
        vmax=max_w if max_w != min_w else min_w + 1.0,
    )

    fig, ax = plt.subplots(figsize=(7.2, 7.8))
    im = ax.imshow(mean_alpha_table, cmap="magma", aspect="auto")
    ax.set_title("Curiosity-based Q-learning: Dirichlet alpha model")
    ax.set_xlabel("Outcome color")
    ax.set_ylabel("Transition context")
    ax.set_xticks(range(len(COLOR_STATE_LABELS)))
    ax.set_xticklabels(COLOR_STATE_LABELS, rotation=35, ha="right")
    ax.set_yticks(range(len(alpha_context_labels())))
    ax.set_yticklabels(alpha_context_labels(), fontsize=6)

    for row in range(mean_alpha_table.shape[0]):
        for col in range(mean_alpha_table.shape[1]):
            value = mean_alpha_table[row, col]
            if value >= mean_alpha_table.max() * 0.25:
                ax.text(
                    col,
                    row,
                    f"{value:.0f}",
                    ha="center",
                    va="center",
                    fontsize=5.5,
                    color="white",
                )

    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03, label="alpha")
    fig.tight_layout()
    fig.savefig(diag_dir / "dirichlet_alpha_model.png", dpi=300)
    plt.close(fig)


def plot_all_q_tables(q_tables: Dict[str, np.ndarray], out_dir: Path) -> None:
    q_dir = out_dir / "q_tables"
    q_dir.mkdir(parents=True, exist_ok=True)

    all_values = np.concatenate([table.ravel() for table in q_tables.values()])
    vmin = float(all_values.min())
    vmax = float(all_values.max())
    if vmin == vmax:
        vmin -= 1.0
        vmax += 1.0

    n_agents = len(q_tables)
    n_cols = 2
    n_rows = int(np.ceil(n_agents / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(10.0, 2.2 * n_rows),
        squeeze=False,
        constrained_layout=True,
    )

    flat_axes = list(axes.flat)
    for ax, (agent_name, table) in zip(flat_axes, q_tables.items()):
        im = ax.imshow(table, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(agent_name)
        ax.set_xlabel("State color")
        ax.set_ylabel("Action")
        ax.set_xticks(range(len(COLOR_STATE_LABELS)))
        ax.set_xticklabels(COLOR_STATE_LABELS, rotation=35, ha="right")
        ax.set_yticks(range(len(ACTION_LABELS)))
        ax.set_yticklabels(ACTION_LABELS)
        _annotate_heatmap(ax, table)

        safe_name = agent_name.lower().replace(" ", "_").replace("-", "_").replace("/", "_")
        plot_q_table_heatmap(
            table,
            agent_name,
            q_dir / f"{safe_name}_q_table.png",
            vmin=vmin,
            vmax=vmax,
        )

    for ax in flat_axes[n_agents:]:
        ax.axis("off")

    fig.colorbar(im, ax=flat_axes[:n_agents], fraction=0.025, pad=0.02, label="Q value")
    fig.savefig(out_dir / "color_mdp_q_tables.png", dpi=300)
    plt.close(fig)
