"""Aggregate per-run results into summary.csv (consumed by the slide-figures notebook).

Usage:
    python -m experiments.analyze_white --runs_dir ./runs_white_600 --output_dir ./figures_white_600
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


CONDITION_ORDER = ["info_gain", "expected_info_gain", "surprisal", "novelty", "random", "hybrid"]


def load_runs(runs_dir: Path) -> Dict[str, List[Dict]]:
    grouped: Dict[str, List[Dict]] = {c: [] for c in CONDITION_ORDER}
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        name = run_dir.name
        if "_seed" not in name:
            continue
        cond_part, seed_part = name.rsplit("_seed", 1)
        if "_beta" in cond_part:
            cond_part = cond_part.split("_beta")[0]
        try:
            seed = int(seed_part)
        except ValueError:
            continue
        if cond_part not in grouped:
            continue
        try:
            history = json.load(open(run_dir / "history.json"))
            final_eval = json.load(open(run_dir / "final_eval.json"))
        except Exception as e:
            print(f"[warn] Skipping {run_dir.name}: {e}")
            continue
        grouped[cond_part].append({
            "seed": seed, "history": history, "final_eval": final_eval,
        })

    for c, runs in grouped.items():
        print(f"  {c}: {len(runs)} run(s)")
    return grouped


def write_summary_csv(grouped, out_dir: Path) -> pd.DataFrame:
    rows = []
    for cond, runs in grouped.items():
        for run in runs:
            row = {"condition": cond, "seed": run["seed"]}
            row.update(run["final_eval"])
            if run["history"]:
                last = run["history"][-1]
                for k in ["wm_kl_to_truth", "wm_top1_acc", "wm_coverage_5",
                          "success_rate", "mean_overloads_per_ep",
                          "entropy", "mean_int_reward_per_step"]:
                    row[f"final_{k}"] = last.get(k, np.nan)
            rows.append(row)
    df = pd.DataFrame(rows).sort_values(["condition", "seed"]).reset_index(drop=True)
    df.to_csv(out_dir / "summary.csv", index=False)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading runs from {runs_dir}")
    grouped = load_runs(runs_dir)
    if not any(grouped.values()):
        print("No runs found.")
        return

    df = write_summary_csv(grouped, out_dir)
    print(df.to_string())
    print(f"\nWrote {out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
