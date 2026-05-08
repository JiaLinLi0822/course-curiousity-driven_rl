"""Create a summary table from episode CSV logs.

Run from the repository root with:
    python -m color_grid_rl.summarize_results
"""

from __future__ import annotations

import csv
from pathlib import Path


RESULTS_DIR = Path("results")
SUMMARY_PATH = RESULTS_DIR / "summary.csv"


def _load_rows(results_dir: Path) -> list[dict]:
    rows = []
    for csv_path in sorted(results_dir.glob("*.csv")):
        if csv_path.name == SUMMARY_PATH.name:
            continue
        with csv_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            rows.extend(reader)
    return rows


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def main() -> None:
    rows = _load_rows(RESULTS_DIR)
    if not rows:
        print(f"No result CSV files found in {RESULTS_DIR}")
        return

    grouped = {}
    for row in rows:
        grouped.setdefault(row["reward_mode"], []).append(row)

    summary_rows = []
    for reward_mode, mode_rows in sorted(grouped.items()):
        solved_rows = [row for row in mode_rows if int(row["solved"]) == 1]

        summary_rows.append(
            {
                "reward_mode": reward_mode,
                "num_episodes": len(mode_rows),
                "success_rate": _mean([float(row["solved"]) for row in mode_rows]),
                "mean_return": _mean([float(row["return"]) for row in mode_rows]),
                "mean_ext_return": _mean([float(row["ext_return"]) for row in mode_rows]),
                "mean_int_return": _mean([float(row["int_return"]) for row in mode_rows]),
                "mean_steps": _mean([float(row["steps"]) for row in mode_rows]),
                "mean_overloads": _mean([float(row["overloads"]) for row in mode_rows]),
                "mean_steps_solved_only": _mean(
                    [float(row["steps"]) for row in solved_rows]
                ),
            }
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = list(summary_rows[0].keys())
    with SUMMARY_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Saved summary table to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
