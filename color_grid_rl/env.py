"""3x3 color-grid environment.

The task is solved when the bottom-right goal tile becomes white. The ordinary
step reward is zero; only solving or timing out produces extrinsic reward.
"""

from __future__ import annotations

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - useful message for local setup
    gym = None
    spaces = None

from .config import BLACK, BLUE, DEFAULT_ENV_CONFIG, EnvConfig, WHITE, YELLOW
from .obs_encoder import ObservationEncoder
from .rewards import compute_extrinsic_reward


class _FallbackDiscrete:
    """Small replacement for gymnasium.spaces.Discrete."""

    def __init__(self, n: int):
        self.n = n

    def contains(self, value: int) -> bool:
        return isinstance(value, int) and 0 <= value < self.n

    def sample(self) -> int:
        return int(np.random.randint(self.n))


class _FallbackBox:
    """Small replacement for gymnasium.spaces.Box used for shape metadata."""

    def __init__(self, low: float, high: float, shape: tuple[int, ...], dtype):
        self.low = low
        self.high = high
        self.shape = shape
        self.dtype = dtype


class ColorGridEnv(gym.Env if gym is not None else object):
    """Small grid world with two-channel compositional colors."""

    metadata = {"render_modes": ["ansi"]}

    # Action ids are kept explicit so training logs are easy to read.
    ACTIONS = {
        0: (0, -1),  # up
        1: (0, 1),   # down
        2: (-1, 0),  # left
        3: (1, 0),   # right
    }
    ACTION_NAMES = {
        0: "up",
        1: "down",
        2: "left",
        3: "right",
    }

    def __init__(
        self,
        config: EnvConfig = DEFAULT_ENV_CONFIG,
        obs_encoder: ObservationEncoder | None = None,
    ):
        super().__init__()
        self.config = config
        self.obs_encoder = obs_encoder or ObservationEncoder()

        space_api = spaces
        discrete_cls = space_api.Discrete if space_api is not None else _FallbackDiscrete
        box_cls = space_api.Box if space_api is not None else _FallbackBox

        self.action_space = discrete_cls(len(self.ACTIONS))
        self.observation_space = box_cls(
            low=0.0,
            high=1.0,
            shape=(self.obs_encoder.obs_dim,),
            dtype=np.float32,
        )

        self.agent_pos = self.config.start_pos
        self.agent_color = np.array(BLACK, dtype=np.int64)
        self.grid_colors = np.zeros(
            (self.config.grid_size, self.config.grid_size, 2),
            dtype=np.int64,
        )
        self.step_count = 0
        self.overload_count = 0

    def reset(self, seed: int | None = None, options: dict | None = None):
        """Start a new episode."""
        if gym is not None:
            super().reset(seed=seed)
        elif seed is not None:
            np.random.seed(seed)

        self.step_count = 0
        self.overload_count = 0
        self.agent_pos = self.config.start_pos
        self.agent_color = np.array(BLACK, dtype=np.int64)
        self.grid_colors.fill(0)

        for position, color in self._initial_tile_layout():
            x, y = position
            self.grid_colors[y, x] = np.array(color, dtype=np.int64)

        return self._get_obs(), self._make_info(mixing_event=None, overload=False)

    def step(self, action: int):
        """Apply one movement action and return the Gymnasium step tuple."""
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")

        self.step_count += 1
        moved = False
        mixing_event = None
        overload = False

        old_x, old_y = self.agent_pos
        dx, dy = self.ACTIONS[action]
        new_pos = (old_x + dx, old_y + dy)

        if self._inside_grid(new_pos):
            moved = True
            self.agent_pos = new_pos
            mixing_event, overload = self._mix_with_current_tile()
            if overload:
                self.overload_count += 1

        solved = self._is_solved()
        timeout = (not solved) and self.step_count >= self.config.max_steps

        reward = compute_extrinsic_reward(
            solved=solved,
            timeout=timeout,
            solved_reward=self.config.solved_reward,
            timeout_reward=self.config.timeout_reward,
            step_reward=self.config.step_reward,
        )

        terminated = solved
        truncated = timeout
        info = self._make_info(
            mixing_event=mixing_event,
            overload=overload,
            moved=moved,
            solved=solved,
        )

        return self._get_obs(), reward, terminated, truncated, info

    def render(self) -> str:
        """Return a compact text view for debugging."""
        symbols = {
            (0, 0): ".",
            (1, 0): "B",
            (0, 1): "Y",
            (1, 1): "W",
        }

        lines = []
        for y in range(self.config.grid_size):
            cells = []
            for x in range(self.config.grid_size):
                color = tuple(self.grid_colors[y, x].tolist())
                label = symbols[color]

                if (x, y) == self.agent_pos:
                    label = f"[{label}]"
                elif (x, y) == self.config.goal_pos:
                    label = f"<{label}>"
                else:
                    label = f" {label} "

                cells.append(label)
            lines.append("".join(cells))

        agent_color_name = symbols[tuple(self.agent_color.tolist())]
        lines.append(f"agent_color={agent_color_name} steps={self.step_count}")
        return "\n".join(lines)

    def get_outcome_space(self) -> tuple[tuple[int, int], ...]:
        """Return all possible outcome colors for mixing-rule evaluation."""
        return (BLACK, BLUE, YELLOW, WHITE)

    def get_true_rule(self) -> dict:
        """Return the deterministic color-mixing rule used by the environment."""
        true_rule = {}
        for agent_color in self.get_outcome_space():
            for tile_color in self.get_outcome_space():
                combined = np.array(agent_color) + np.array(tile_color)
                if np.any(combined > 1):
                    outcome = BLACK
                else:
                    outcome = tuple(combined.astype(np.int64).tolist())
                true_rule[(agent_color, tile_color)] = outcome
        return true_rule

    def _initial_tile_layout(self) -> tuple[tuple[tuple[int, int], tuple[int, int]], ...]:
        """Return the initial colored-tile layout for this episode."""
        if not self.config.randomize_initial_tile_positions:
            return self.config.initial_colored_tiles

        available_positions = [
            (x, y)
            for y in range(self.config.grid_size)
            for x in range(self.config.grid_size)
            if (x, y) not in {self.config.start_pos, self.config.goal_pos}
        ]
        colors = [color for _, color in self.config.initial_colored_tiles]

        if len(colors) > len(available_positions):
            raise ValueError(
                "Not enough non-start/non-goal positions for initial colored tiles"
            )

        if gym is not None:
            indices = self.np_random.choice(
                len(available_positions),
                size=len(colors),
                replace=False,
            )
        else:
            indices = np.random.choice(
                len(available_positions),
                size=len(colors),
                replace=False,
            )

        return tuple(
            (available_positions[int(index)], color)
            for index, color in zip(indices, colors)
        )

    def _mix_with_current_tile(self) -> tuple[tuple[tuple[int, int], tuple[int, int], tuple[int, int]], bool]:
        """Combine agent color with current tile color."""
        x, y = self.agent_pos

        agent_before = tuple(self.agent_color.tolist())
        tile_before = tuple(self.grid_colors[y, x].tolist())

        combined = self.agent_color + self.grid_colors[y, x]
        overload = bool(np.any(combined > 1))

        if overload:
            self.agent_color = np.array(BLACK, dtype=np.int64)
            self.grid_colors[y, x] = np.array(BLACK, dtype=np.int64)
            outcome = BLACK
        else:
            self.agent_color = combined.astype(np.int64)
            self.grid_colors[y, x] = combined.astype(np.int64)
            outcome = tuple(combined.tolist())

        mixing_event = (agent_before, tile_before, outcome)
        return mixing_event, overload

    def _inside_grid(self, position: tuple[int, int]) -> bool:
        x, y = position
        return 0 <= x < self.config.grid_size and 0 <= y < self.config.grid_size

    def _is_solved(self) -> bool:
        goal_x, goal_y = self.config.goal_pos
        return tuple(self.grid_colors[goal_y, goal_x].tolist()) == WHITE

    def _get_obs(self) -> np.ndarray:
        return self.obs_encoder.encode(
            agent_pos=self.agent_pos,
            grid_colors=self.grid_colors,
            agent_color=self.agent_color,
        )

    def _make_info(
        self,
        mixing_event,
        overload: bool,
        moved: bool = False,
        solved: bool = False,
    ) -> dict:
        return {
            "agent_pos": self.agent_pos,
            "agent_color": tuple(self.agent_color.tolist()),
            "mixing_event": mixing_event,
            "moved": moved,
            "overload": overload,
            "overload_count": self.overload_count,
            "solved": solved,
            "step_count": self.step_count,
        }


if __name__ == "__main__":
    env = ColorGridEnv()
    obs, info = env.reset()
    print(env.render())
    print("obs shape:", obs.shape)
