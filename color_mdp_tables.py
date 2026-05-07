import csv
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

from color_mdp_agents import ColorMDPDirichletAgentAdapter
from color_mdp_constants import (
    ACTION_IDS,
    ACTION_LABELS,
    COLOR_STATE_IDS,
    COLOR_STATE_LABELS,
    COLOR_TUPLES,
)
from env import COLOR_ACTION_NAMES, COLOR_ACTIONS, ColorMixingMDPEnv, RGB_COLOR_NAMES


def write_rows(path: Path, fieldnames: Iterable[str], rows: List[Dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def rollout_greedy(env: ColorMixingMDPEnv, agent) -> Dict:
    state = env.reset()
    done = False
    actions = []
    colors = [RGB_COLOR_NAMES[env.id_color(state)]]
    total_return = 0.0
    overloads = 0

    while not done:
        legal_actions = env.legal_actions(state)
        action = agent.select_action(state, legal_actions, greedy=True)
        result = env.step(action)
        actions.append(COLOR_ACTION_NAMES[action])
        colors.append(result.info["color_name"])
        total_return += result.reward_ext
        overloads += int(result.info.get("overload", False))
        state = result.next_state
        done = result.done

    return {
        "greedy_actions": " -> ".join(actions),
        "greedy_colors": " -> ".join(colors),
        "greedy_return": total_return,
        "greedy_success": int(env.goals_reached > 0),
        "greedy_steps": env.steps,
        "greedy_overloads": overloads,
    }


def extract_q_table(agent) -> np.ndarray:
    q_agent = agent.agent if hasattr(agent, "agent") else agent
    table = np.zeros((len(ACTION_IDS), len(COLOR_STATE_IDS)), dtype=float)

    for col, state_id in enumerate(COLOR_STATE_IDS):
        key = (state_id,) if isinstance(agent, ColorMDPDirichletAgentAdapter) else state_id
        q_values = q_agent.q[key]
        for row, action_id in enumerate(ACTION_IDS):
            table[row, col] = float(q_values[action_id])

    return table


def extract_dirichlet_w_table(agent: ColorMDPDirichletAgentAdapter) -> np.ndarray:
    inner = agent.agent
    table = np.zeros((len(ACTION_IDS), len(COLOR_STATE_IDS)), dtype=float)

    for col, state_id in enumerate(COLOR_STATE_IDS):
        state_key = agent._state_key(state_id)
        for row, action_id in enumerate(ACTION_IDS):
            table[row, col] = inner._criterion(state_key, action_id, ACTION_IDS)

    return table


def extract_dirichlet_alpha_table(agent: ColorMDPDirichletAgentAdapter) -> np.ndarray:
    inner = agent.agent
    rows = []
    for state_color in COLOR_TUPLES:
        for action_id in ACTION_IDS:
            action_color = COLOR_ACTIONS[action_id]
            rows.append(inner.alpha_model[(state_color, action_color)].copy())
    return np.vstack(rows)


def alpha_context_labels() -> List[str]:
    labels = []
    for state_color in COLOR_TUPLES:
        state_name = RGB_COLOR_NAMES[state_color]
        for action_id in ACTION_IDS:
            labels.append(f"{state_name}+{COLOR_ACTION_NAMES[action_id]}")
    return labels


def q_table_rows(agent_label: str, seed_idx: int, seed: int, q_table: np.ndarray) -> List[Dict]:
    rows = []
    for state_col, state_id in enumerate(COLOR_STATE_IDS):
        for action_row, action_id in enumerate(ACTION_IDS):
            rows.append(
                {
                    "agent": agent_label,
                    "seed_index": seed_idx,
                    "seed": seed,
                    "state_id": state_id,
                    "state": COLOR_STATE_LABELS[state_col],
                    "action_id": action_id,
                    "action": ACTION_LABELS[action_row],
                    "q_value": q_table[action_row, state_col],
                }
            )
    return rows


def mean_q_table_rows(mean_q_tables: Dict[str, np.ndarray]) -> List[Dict]:
    rows = []
    for agent_label, table in mean_q_tables.items():
        for state_col, state_id in enumerate(COLOR_STATE_IDS):
            for action_row, action_id in enumerate(ACTION_IDS):
                rows.append(
                    {
                        "agent": agent_label,
                        "state_id": state_id,
                        "state": COLOR_STATE_LABELS[state_col],
                        "action_id": action_id,
                        "action": ACTION_LABELS[action_row],
                        "mean_q_value": table[action_row, state_col],
                    }
                )
    return rows


def dirichlet_w_rows(mean_w_table: np.ndarray) -> List[Dict]:
    rows = []
    for state_col, state_id in enumerate(COLOR_STATE_IDS):
        for action_row, action_id in enumerate(ACTION_IDS):
            rows.append(
                {
                    "state_id": state_id,
                    "state": COLOR_STATE_LABELS[state_col],
                    "action_id": action_id,
                    "action": ACTION_LABELS[action_row],
                    "mean_w_value": mean_w_table[action_row, state_col],
                }
            )
    return rows


def dirichlet_alpha_rows(mean_alpha_table: np.ndarray) -> List[Dict]:
    rows = []
    contexts = alpha_context_labels()
    row_idx = 0
    for state_color in COLOR_TUPLES:
        for action_id in ACTION_IDS:
            for outcome_col, outcome_color in enumerate(COLOR_TUPLES):
                rows.append(
                    {
                        "context": contexts[row_idx],
                        "state": RGB_COLOR_NAMES[state_color],
                        "action": COLOR_ACTION_NAMES[action_id],
                        "outcome": RGB_COLOR_NAMES[outcome_color],
                        "mean_alpha": mean_alpha_table[row_idx, outcome_col],
                    }
                )
            row_idx += 1
    return rows
