from typing import Dict

import numpy as np

from color_mdp_agents import ColorMDPDirichletAgentAdapter
from color_mdp_constants import (
    ACTION_IDS,
    COLOR_STATE_IDS,
)
from env import COLOR_ACTION_NAMES, ColorMixingMDPEnv, RGB_COLOR_NAMES


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

