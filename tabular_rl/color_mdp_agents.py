from agent import (
    DirichletInformationGainQLearningAgent,
    InformationGainQLearningAgent,
    TabularQLearningAgent,
    UCBQLearningAgent,
)
from env import COLOR_ACTIONS, ID_TO_RGB_COLOR


class ColorMDPDirichletAgentAdapter:
    """
    Adapts DirichletInformationGainQLearningAgent to ColorMixingMDPEnv.

    The wrapped agent expects tuple-like states and transition metadata.
    ColorMixingMDPEnv uses compact integer color states, so this adapter maps
    state id -> (state id,) and reconstructs the color transition.
    """

    def __init__(self, agent: DirichletInformationGainQLearningAgent):
        self.agent = agent

    def _state_key(self, state: int):
        return (int(state),)

    def select_action(self, state, legal_actions, greedy: bool = False) -> int:
        return self.agent.select_action(self._state_key(state), legal_actions, greedy=greedy)

    def post_transition_update(self, state, action: int, next_state) -> None:
        pass

    def decay_exploration(self) -> None:
        self.agent.decay_exploration()

    def update(
        self,
        state,
        action: int,
        reward_total: float,
        next_state,
        done: bool,
        next_legal_actions,
    ) -> None:
        info = {
            "transition": (
                ID_TO_RGB_COLOR[int(state)],
                COLOR_ACTIONS[action],
                ID_TO_RGB_COLOR[int(next_state)],
            )
        }
        self.agent.update(
            self._state_key(state),
            action,
            reward_total,
            self._state_key(next_state),
            done,
            next_legal_actions,
            info,
        )


def make_epsilon_greedy_agent(seed: int):
    return TabularQLearningAgent(
        n_actions=3,
        alpha=0.3,
        gamma=0.95,
        epsilon=0.3,
        epsilon_min=0.02,
        epsilon_decay=0.99,
        seed=seed,
    )


def make_ucb_agent(seed: int):
    return UCBQLearningAgent(
        n_actions=3,
        alpha=0.3,
        gamma=0.95,
        ucb_c=0.8,
        ucb_c_min=0.05,
        ucb_c_decay=0.995,
        seed=seed,
    )


def make_novelty_agent(seed: int):
    return InformationGainQLearningAgent(
        n_actions=3,
        alpha=0.3,
        beta=0.3,
        gamma=0.95,
        gamma_info=0.95,
        temperature=1.0,
        temperature_min=0.05,
        temperature_decay=0.99,
        info_init=5.0,
        seed=seed,
    )


def make_curiosity_agent(seed: int):
    agent = DirichletInformationGainQLearningAgent(
        n_actions=3,
        alpha_q=0.3,
        alpha_info=0.3,
        gamma=0.95,
        gamma_info=0.95,
        softmax_temperature=1.0,
        temperature_min=0.05,
        temperature_decay=0.99,
        prior_concentration=1.0,
        seed=seed,
    )
    return ColorMDPDirichletAgentAdapter(agent)


AGENT_FACTORIES = [
    ("Epsilon-greedy Q-learning", make_epsilon_greedy_agent),
    ("UCB Q-learning", make_ucb_agent),
    ("Novelty-based Q-learning", make_novelty_agent),
    ("Curiosity-based Q-learning", make_curiosity_agent),
]
