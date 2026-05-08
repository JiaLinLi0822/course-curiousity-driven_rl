"""Evaluation helpers for task behavior and latent rule learning."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import numpy as np

BLUE = (1, 0)
YELLOW = (0, 1)
WHITE = (1, 1)
BLUE_YELLOW_WHITE_CONDITIONS = ((BLUE, YELLOW), (YELLOW, BLUE))


class RuleTracker:
    """
    Evaluation-only tracker for estimating the color-mixing rule from observed
    mixing events. This should not affect the reward model or policy training.
    """

    def __init__(self, outcome_space, smoothing: float = 1.0):
        self.outcome_space = tuple(outcome_space)
        self.smoothing = float(smoothing)
        self.reset()

    def reset(self) -> None:
        self.counts = {}
        self.total_events = 0

    def update(self, info: dict) -> None:
        event = info.get("mixing_event")
        if event is None:
            return

        agent_color_before, tile_color_before, outcome_color = event
        condition = (agent_color_before, tile_color_before)
        outcome_counts = self.counts.setdefault(condition, {})
        outcome_counts[outcome_color] = outcome_counts.get(outcome_color, 0) + 1
        self.total_events += 1

    def get_probs(self, condition) -> dict:
        outcome_counts = self.counts.get(condition, {})
        total_count = sum(outcome_counts.values())
        denominator = total_count + self.smoothing * len(self.outcome_space)

        return {
            outcome: (outcome_counts.get(outcome, 0) + self.smoothing) / denominator
            for outcome in self.outcome_space
        }

    def predict(self, condition):
        if condition not in self.counts:
            return None

        probs = self.get_probs(condition)
        return max(probs, key=probs.get)

    def num_seen_conditions(self) -> int:
        return len(self.counts)

    def num_total_events(self) -> int:
        return self.total_events


def _is_probabilistic_rule_entry(value: Any) -> bool:
    return isinstance(value, dict)


def _true_distribution(true_rule: dict, condition, outcome_space) -> dict:
    true_value = true_rule[condition]
    if _is_probabilistic_rule_entry(true_value):
        return {outcome: float(true_value.get(outcome, 0.0)) for outcome in outcome_space}
    return {outcome: 1.0 if outcome == true_value else 0.0 for outcome in outcome_space}


def _safe_kl(true_probs: dict, learned_probs: dict, outcome_space, epsilon: float) -> float:
    true_values = np.array([true_probs.get(outcome, 0.0) for outcome in outcome_space])
    learned_values = np.array([learned_probs.get(outcome, 0.0) for outcome in outcome_space])

    true_values = true_values + epsilon
    learned_values = learned_values + epsilon
    true_values = true_values / true_values.sum()
    learned_values = learned_values / learned_values.sum()

    return float(np.sum(true_values * np.log(true_values / learned_values)))


def compute_rule_metrics(
    rule_tracker: RuleTracker,
    true_rule,
    outcome_space,
    epsilon: float = 1e-8,
) -> dict:
    """Compare the evaluation-only learned rule estimate to the true rule."""
    if true_rule is None:
        return {
            "rule_accuracy_seen": np.nan,
            "rule_accuracy_all": np.nan,
            "condition_coverage": np.nan,
            "belief_kl_true_to_learned": np.nan,
            "num_seen_conditions": rule_tracker.num_seen_conditions(),
            "num_total_conditions": 0,
        }

    total_conditions = len(true_rule)
    if total_conditions == 0:
        return {
            "rule_accuracy_seen": 0.0,
            "rule_accuracy_all": 0.0,
            "condition_coverage": 0.0,
            "belief_kl_true_to_learned": np.nan,
            "num_seen_conditions": rule_tracker.num_seen_conditions(),
            "num_total_conditions": 0,
        }

    seen_conditions = [
        condition for condition in rule_tracker.counts.keys() if condition in true_rule
    ]

    correct_seen = 0
    for condition in seen_conditions:
        prediction = rule_tracker.predict(condition)
        true_value = true_rule[condition]
        if _is_probabilistic_rule_entry(true_value):
            best_true_outcome = max(true_value, key=true_value.get)
            correct_seen += int(prediction == best_true_outcome)
        else:
            correct_seen += int(prediction == true_value)

    correct_all = correct_seen
    kls = []
    for condition in true_rule:
        true_probs = _true_distribution(true_rule, condition, outcome_space)
        learned_probs = rule_tracker.get_probs(condition)
        kls.append(_safe_kl(true_probs, learned_probs, outcome_space, epsilon))

    return {
        "rule_accuracy_seen": (
            correct_seen / len(seen_conditions) if seen_conditions else 0.0
        ),
        "rule_accuracy_all": correct_all / total_conditions,
        "condition_coverage": len(seen_conditions) / total_conditions,
        "belief_kl_true_to_learned": float(np.mean(kls)) if kls else np.nan,
        "num_seen_conditions": len(seen_conditions),
        "num_total_conditions": total_conditions,
    }


def compute_blue_yellow_white_metrics(rule_tracker: RuleTracker) -> dict:
    """Focused metrics for the key latent rule: blue + yellow -> white."""
    seen_conditions = [
        condition
        for condition in BLUE_YELLOW_WHITE_CONDITIONS
        if condition in rule_tracker.counts
    ]
    correct_predictions = sum(
        int(rule_tracker.predict(condition) == WHITE) for condition in seen_conditions
    )

    blue_yellow_events = 0
    blue_yellow_white_events = 0
    for condition in BLUE_YELLOW_WHITE_CONDITIONS:
        outcome_counts = rule_tracker.counts.get(condition, {})
        blue_yellow_events += sum(outcome_counts.values())
        blue_yellow_white_events += outcome_counts.get(WHITE, 0)

    return {
        "blue_yellow_rule_seen": int(bool(seen_conditions)),
        "blue_yellow_rule_accuracy": (
            correct_predictions / len(seen_conditions) if seen_conditions else 0.0
        ),
        "blue_yellow_condition_coverage": (
            len(seen_conditions) / len(BLUE_YELLOW_WHITE_CONDITIONS)
        ),
        "blue_yellow_events": blue_yellow_events,
        "blue_yellow_white_events": blue_yellow_white_events,
    }


def _select_eval_action(agent, obs, deterministic: bool) -> int:
    if hasattr(agent, "act"):
        try:
            return int(agent.act(obs, deterministic=deterministic))
        except TypeError:
            return int(agent.act(obs))

    try:
        result = agent.select_action(obs, deterministic=deterministic)
    except TypeError:
        result = agent.select_action(obs)

    if isinstance(result, tuple):
        return int(result[0])
    return int(result)


def evaluate_policy(
    env,
    agent,
    true_rule=None,
    outcome_space=None,
    num_eval_episodes: int = 20,
    deterministic: bool = True,
    reward_model=None,
    device=None,
) -> dict:
    """Run no-gradient policy evaluation with task, behavior, and rule metrics."""
    del device  # Kept for a stable call signature if GPU evaluation is added later.

    if outcome_space is None and hasattr(env, "get_outcome_space"):
        outcome_space = env.get_outcome_space()
    if outcome_space is None:
        outcome_space = []

    if true_rule is None and hasattr(env, "get_true_rule"):
        true_rule = env.get_true_rule()

    rule_tracker = RuleTracker(outcome_space=outcome_space)
    model = getattr(agent, "model", None)
    was_training = bool(model.training) if model is not None else False
    if model is not None:
        model.eval()

    solved_episodes = 0
    total_steps = 0
    total_extrinsic_return = 0.0
    total_intrinsic_return = 0.0
    steps_to_solve = []
    total_mixing_events = 0
    unique_mixing_events = set()
    total_overloads = 0

    with np.errstate(invalid="ignore"), np.errstate(divide="ignore"):
        for episode_idx in range(num_eval_episodes):
            reset_result = env.reset()
            obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result

            done = False
            episode_steps = 0
            while not done:
                action = _select_eval_action(agent, obs, deterministic)
                obs, extrinsic_reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                episode_steps += 1
                total_steps += 1
                total_extrinsic_return += float(extrinsic_reward)

                if reward_model is not None:
                    # Evaluation may log intrinsic reward, but must not call update().
                    total_intrinsic_return += float(reward_model.compute(info))

                event = info.get("mixing_event")
                if event is not None:
                    total_mixing_events += 1
                    unique_mixing_events.add(event)
                if info.get("overload", False):
                    total_overloads += 1

                rule_tracker.update(info)

                if info.get("solved", False):
                    solved_episodes += 1
                    steps_to_solve.append(info.get("step_count", episode_steps))

    if model is not None and was_training:
        model.train()

    unique_count = len(unique_mixing_events)
    unique_ratio = unique_count / total_mixing_events if total_mixing_events else 0.0
    rule_metrics = compute_rule_metrics(rule_tracker, true_rule, outcome_space)
    focused_rule_metrics = compute_blue_yellow_white_metrics(rule_tracker)

    metrics = {
        "success_rate": solved_episodes / num_eval_episodes,
        "mean_episode_steps": total_steps / num_eval_episodes,
        "mean_steps_to_solve": float(np.mean(steps_to_solve)) if steps_to_solve else np.nan,
        "mean_extrinsic_return": total_extrinsic_return / num_eval_episodes,
        "mean_intrinsic_return": total_intrinsic_return / num_eval_episodes,
        "total_mixing_events": total_mixing_events,
        "unique_mixing_events": unique_count,
        "unique_mixing_ratio": unique_ratio,
        "repeated_mixing_ratio": 1.0 - unique_ratio if total_mixing_events else 0.0,
        "overload_rate": total_overloads / total_steps if total_steps else 0.0,
        "mean_overloads_per_episode": total_overloads / num_eval_episodes,
    }
    metrics.update(rule_metrics)
    metrics.update(focused_rule_metrics)
    return metrics


def append_eval_row(csv_path: str | Path, row: dict) -> Path:
    """Append one evaluation row, writing the header when the file is new."""
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = path.exists()
    fieldnames = list(row.keys())
    if file_exists:
        with path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader)
            old_fieldnames = reader.fieldnames or []

        for name in old_fieldnames:
            if name not in fieldnames:
                fieldnames.append(name)

        if old_fieldnames != fieldnames:
            with path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(existing_rows)

    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    return path
