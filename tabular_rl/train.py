from dataclasses import dataclass
from typing import List


@dataclass
class EpisodeStats:
    episode_return_ext: float
    goals_reached: int
    steps: int
    overloads: int


def run_training(env, agent, n_episodes: int = 5000) -> List[EpisodeStats]:
    history: List[EpisodeStats] = []

    for _ in range(n_episodes):
        state = env.reset()
        done = False
        ep_return_ext = 0.0
        ep_overloads = 0
        ep_steps = 0
        ep_goals = 0

        while not done:
            legal_actions = env.legal_actions(state)
            action = agent.select_action(state, legal_actions, greedy=False)
            result = env.step(action)

            next_legal = env.legal_actions(result.next_state)
            agent.update(
                state,
                action,
                result.reward_ext,
                result.next_state,
                result.done,
                next_legal,
            )
            agent.post_transition_update(state, action, result.next_state)

            ep_return_ext += result.reward_ext
            ep_steps += 1
            ep_overloads += int(result.info.get("overload", False))
            ep_goals += int(result.info.get("trial_success", False))

            state = result.next_state
            done = result.done

        history.append(
            EpisodeStats(
                episode_return_ext=ep_return_ext,
                goals_reached=ep_goals,
                steps=ep_steps,
                overloads=ep_overloads,
            )
        )

        agent.decay_exploration()

    return history
