import math
import random
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.special import digamma, gammaln


State = int
Color = Tuple[int, int, int]
StateKey = Tuple

COLORS: List[Color] = [
    (0, 0, 0),  # black
    (1, 0, 0),  # red
    (0, 1, 0),  # green
    (0, 0, 1),  # blue
    (1, 1, 0),  # yellow
    (1, 0, 1),  # magenta
    (0, 1, 1),  # cyan
    (1, 1, 1),  # white
]
COLOR_TO_IDX = {color: idx for idx, color in enumerate(COLORS)}
NUM_COLORS = len(COLORS)


def argmax_random_tie(values: Dict[int, float], rng: random.Random) -> int:
    max_v = max(values.values())
    best = [key for key, value in values.items() if value == max_v]
    return rng.choice(best)


def epsilon_greedy_action(
    q_table: Dict[State, np.ndarray],
    state: State,
    legal_actions: List[int],
    epsilon: float,
    rng: random.Random,
) -> int:
    if rng.random() < epsilon:
        return rng.choice(legal_actions)

    qvals = q_table[state]
    legal_q = {action: float(qvals[action]) for action in legal_actions}
    return argmax_random_tie(legal_q, rng)


class TabularQLearningAgent:
    def __init__(
        self,
        n_actions: int = 3,
        alpha: float = 0.2,
        gamma: float = 0.99,
        epsilon: float = 0.2,
        epsilon_min: float = 0.02,
        epsilon_decay: float = 0.995,
        seed: Optional[int] = None,
    ):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = random.Random(seed)
        self.q: Dict[State, np.ndarray] = defaultdict(
            lambda: np.zeros(self.n_actions, dtype=float)
        )

    def select_action(self, state: State, legal_actions: List[int], greedy: bool = False) -> int:
        eps = 0.0 if greedy else self.epsilon
        return epsilon_greedy_action(self.q, state, legal_actions, eps, self.rng)

    def post_transition_update(self, state: State, action: int, next_state: State) -> None:
        pass

    def update(
        self,
        state: State,
        action: int,
        reward_total: float,
        next_state: State,
        done: bool,
        next_legal_actions: List[int],
    ) -> None:
        q_sa = self.q[state][action]
        if done:
            target = reward_total
        else:
            target = reward_total + self.gamma * max(
                self.q[next_state][a] for a in next_legal_actions
            )
        self.q[state][action] += self.alpha * (target - q_sa)

    def decay_exploration(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


class UCBQLearningAgent(TabularQLearningAgent):
    def __init__(
        self,
        ucb_c: float = 1.0,
        ucb_c_min: float = 0.0,
        ucb_c_decay: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ucb_c = ucb_c
        self.ucb_c_min = ucb_c_min
        self.ucb_c_decay = ucb_c_decay
        self.sa_counts: Counter = Counter()

    def select_action(self, state: State, legal_actions: List[int], greedy: bool = False) -> int:
        if greedy:
            qvals = self.q[state]
            legal_q = {action: float(qvals[action]) for action in legal_actions}
            return argmax_random_tie(legal_q, self.rng)

        untried = [action for action in legal_actions if self.sa_counts[(state, action)] == 0]
        if untried:
            return self.rng.choice(untried)

        total_visits = sum(self.sa_counts[(state, action)] for action in legal_actions)
        log_total = math.log(total_visits + 1.0)
        qvals = self.q[state]
        ucb_values = {}
        for action in legal_actions:
            n_sa = self.sa_counts[(state, action)]
            bonus = self.ucb_c * math.sqrt(log_total / n_sa)
            ucb_values[action] = float(qvals[action]) + bonus

        return argmax_random_tie(ucb_values, self.rng)

    def post_transition_update(self, state: State, action: int, next_state: State) -> None:
        self.sa_counts[(state, action)] += 1

    def decay_exploration(self) -> None:
        self.ucb_c = max(self.ucb_c_min, self.ucb_c * self.ucb_c_decay)


class InformationGainQLearningAgent:
    """
    Novelty-based tabular Q-learning.

    The agent learns Q(s,a) and an information-value table ell(s,a). Actions are
    selected by minimizing w(s,a) = return_loss(s,a) / expected_information_gain(s,a).
    """

    def __init__(
        self,
        n_actions: int = 3,
        alpha: float = 0.2,
        beta: float = 0.2,
        gamma: float = 0.99,
        gamma_info: float = 0.99,
        temperature: float = 1.0,
        temperature_min: float = 0.05,
        temperature_decay: float = 0.995,
        info_init: float = 10.0,
        info_floor: float = 1e-8,
        seed: Optional[int] = None,
    ):
        self.n_actions = n_actions
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.gamma_info = gamma_info
        self.temperature = temperature
        self.temperature_min = temperature_min
        self.temperature_decay = temperature_decay
        self.info_init = info_init
        self.info_floor = info_floor
        self.rng = random.Random(seed)
        self.q: Dict[State, np.ndarray] = defaultdict(
            lambda: np.zeros(self.n_actions, dtype=float)
        )
        self.ell: Dict[State, np.ndarray] = defaultdict(
            lambda: np.ones(self.n_actions, dtype=float) * self.info_init
        )
        self.counts: Dict[State, np.ndarray] = defaultdict(
            lambda: np.zeros(self.n_actions, dtype=float)
        )

    def _return_loss(self, state: State, action: int, legal_actions: List[int]) -> float:
        q_values = self.q[state]
        best_q = max(q_values[a] for a in legal_actions)
        return float(best_q - q_values[action])

    def _criterion(self, state: State, action: int, legal_actions: List[int]) -> float:
        loss = self._return_loss(state, action, legal_actions)
        info = max(float(self.ell[state][action]), self.info_floor)
        return loss / info

    def select_action(self, state: State, legal_actions: List[int], greedy: bool = False) -> int:
        w_values = np.array(
            [self._criterion(state, action, legal_actions) for action in legal_actions],
            dtype=float,
        )
        if greedy:
            return legal_actions[int(np.argmin(w_values))]

        logits = -w_values / max(self.temperature, 1e-8)
        logits = logits - np.max(logits)
        probs = np.exp(logits)
        probs = probs / np.sum(probs)
        return self.rng.choices(legal_actions, weights=probs, k=1)[0]

    def _instant_information_gain(self, state: State, action: int) -> float:
        n = self.counts[state][action]
        return float(1.0 / np.sqrt(n + 1.0))

    def post_transition_update(self, state: State, action: int, next_state: State) -> None:
        self.counts[state][action] += 1.0

    def update(
        self,
        state: State,
        action: int,
        reward_total: float,
        next_state: State,
        done: bool,
        next_legal_actions: List[int],
    ) -> None:
        q_sa = self.q[state][action]
        if done:
            q_target = reward_total
        else:
            q_target = reward_total + self.gamma * max(
                self.q[next_state][a] for a in next_legal_actions
            )
        self.q[state][action] += self.alpha * (q_target - q_sa)

        immediate_info = self._instant_information_gain(state, action)
        ell_sa = self.ell[state][action]
        if done:
            ell_target = immediate_info
        else:
            ell_target = immediate_info + self.gamma_info * max(
                self.ell[next_state][a] for a in next_legal_actions
            )
        self.ell[state][action] += self.beta * (ell_target - ell_sa)

    def decay_exploration(self) -> None:
        self.temperature = max(self.temperature_min, self.temperature * self.temperature_decay)


def dirichlet_kl(alpha_new: np.ndarray, alpha_old: np.ndarray) -> float:
    a = np.asarray(alpha_new, dtype=float)
    b = np.asarray(alpha_old, dtype=float)
    a0 = np.sum(a)
    b0 = np.sum(b)

    log_b_a = np.sum(gammaln(a)) - gammaln(a0)
    log_b_b = np.sum(gammaln(b)) - gammaln(b0)
    kl = log_b_b - log_b_a + np.sum((a - b) * (digamma(a) - digamma(a0)))
    return float(max(0.0, kl))


class DirichletInformationGainQLearningAgent:
    """
    Curiosity-driven Q-learning with a Dirichlet transition model.

    The learned information value comes from the KL divergence between the
    posterior and prior Dirichlet distributions after each observed transition.
    """

    def __init__(
        self,
        n_actions: int = 3,
        alpha_q: float = 0.2,
        alpha_info: float = 0.2,
        gamma: float = 0.99,
        gamma_info: float = 0.99,
        softmax_temperature: float = 1.0,
        temperature_min: float = 0.05,
        temperature_decay: float = 0.995,
        prior_concentration: float = 1.0,
        info_floor: float = 1e-8,
        seed: Optional[int] = None,
    ):
        self.n_actions = n_actions
        self.alpha_q = alpha_q
        self.alpha_info = alpha_info
        self.gamma = gamma
        self.gamma_info = gamma_info
        self.temperature = softmax_temperature
        self.temperature_min = temperature_min
        self.temperature_decay = temperature_decay
        self.prior_concentration = prior_concentration
        self.info_floor = info_floor
        self.rng = random.Random(seed)
        self.q: Dict[StateKey, np.ndarray] = defaultdict(
            lambda: np.zeros(self.n_actions, dtype=float)
        )
        self.ell: Dict[StateKey, np.ndarray] = defaultdict(
            lambda: np.ones(self.n_actions, dtype=float)
        )
        self.alpha_model: Dict[Tuple[Color, Color], np.ndarray] = defaultdict(
            lambda: np.ones(NUM_COLORS, dtype=float) * self.prior_concentration
        )

    def state_to_key(self, obs_or_state) -> StateKey:
        if isinstance(obs_or_state, tuple):
            return obs_or_state
        arr = np.asarray(obs_or_state)
        return tuple(arr.astype(np.float32).round(4).tolist())

    def _model_key(self, c_a: Color, c_b: Color) -> Tuple[Color, Color]:
        return tuple(c_a), tuple(c_b)

    def observed_information_gain(self, c_a: Color, c_b: Color, outcome: Color) -> float:
        key = self._model_key(c_a, c_b)
        alpha_before = self.alpha_model[key].copy()
        alpha_after = alpha_before.copy()
        alpha_after[COLOR_TO_IDX[tuple(outcome)]] += 1.0
        return dirichlet_kl(alpha_after, alpha_before)

    def update_world_model(self, c_a: Color, c_b: Color, outcome: Color) -> None:
        key = self._model_key(c_a, c_b)
        self.alpha_model[key][COLOR_TO_IDX[tuple(outcome)]] += 1.0

    def _return_loss(self, state_key: StateKey, action: int, legal_actions: List[int]) -> float:
        qvals = self.q[state_key]
        best_q = max(qvals[action_id] for action_id in legal_actions)
        return float(best_q - qvals[action])

    def _criterion(self, state_key: StateKey, action: int, legal_actions: List[int]) -> float:
        loss = self._return_loss(state_key, action, legal_actions)
        info = max(float(self.ell[state_key][action]), self.info_floor)
        return loss / info

    def select_action(self, state, legal_actions: List[int], greedy: bool = False) -> int:
        state_key = self.state_to_key(state)
        w_values = np.array(
            [self._criterion(state_key, action, legal_actions) for action in legal_actions],
            dtype=float,
        )
        if greedy:
            return legal_actions[int(np.argmin(w_values))]

        logits = -w_values / max(self.temperature, 1e-8)
        logits = logits - logits.max()
        probs = np.exp(logits)
        probs = probs / probs.sum()
        return self.rng.choices(legal_actions, weights=probs, k=1)[0]

    def update(
        self,
        state,
        action: int,
        reward_ext: float,
        next_state,
        done: bool,
        next_legal_actions: List[int],
        info: dict,
    ) -> None:
        state_key = self.state_to_key(state)
        next_key = self.state_to_key(next_state)
        c_a, c_b, outcome = info["transition"]
        c_a = tuple(c_a)
        c_b = tuple(c_b)
        outcome = tuple(outcome)

        ig = self.observed_information_gain(c_a, c_b, outcome)
        q_sa = self.q[state_key][action]
        if done:
            q_target = reward_ext
        else:
            q_target = reward_ext + self.gamma * max(
                self.q[next_key][a] for a in next_legal_actions
            )
        self.q[state_key][action] += self.alpha_q * (q_target - q_sa)

        ell_sa = self.ell[state_key][action]
        if done:
            ell_target = ig
        else:
            ell_target = ig + self.gamma_info * max(
                self.ell[next_key][a] for a in next_legal_actions
            )
        self.ell[state_key][action] += self.alpha_info * (ell_target - ell_sa)
        self.update_world_model(c_a, c_b, outcome)

    def decay_exploration(self) -> None:
        self.temperature = max(self.temperature_min, self.temperature * self.temperature_decay)
