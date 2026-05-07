from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


RGBColor = Tuple[int, int, int]
State = int

RGB_COLOR_TO_ID: Dict[RGBColor, int] = {
    (0, 0, 0): 0,
    (1, 0, 0): 1,
    (0, 1, 0): 2,
    (0, 0, 1): 3,
    (1, 1, 0): 4,
    (1, 0, 1): 5,
    (0, 1, 1): 6,
    (1, 1, 1): 7,
}
ID_TO_RGB_COLOR: Dict[int, RGBColor] = {idx: color for color, idx in RGB_COLOR_TO_ID.items()}

COLOR_ACTIONS = {
    0: (1, 0, 0),  # add red
    1: (0, 1, 0),  # add green
    2: (0, 0, 1),  # add blue
}
COLOR_ACTION_NAMES = {
    0: "red",
    1: "green",
    2: "blue",
}
RGB_COLOR_NAMES = {
    (0, 0, 0): "black",
    (1, 0, 0): "red",
    (0, 1, 0): "green",
    (0, 0, 1): "blue",
    (1, 1, 0): "yellow",
    (1, 0, 1): "magenta",
    (0, 1, 1): "cyan",
    (1, 1, 1): "white",
}


@dataclass
class StepResult:
    next_state: State
    reward_ext: float
    done: bool
    info: Dict


class ColorMixingMDPEnv:
    """
    Eight-state color-mixing MDP.

    States are RGB colors: black, red, green, blue, yellow, magenta, cyan, white.
    Actions add one primary color: red, green, or blue. Adding a duplicate
    channel overloads and resets the state to black. Episodes start at black and
    end when the agent reaches white or max_steps is reached.
    """

    BLACK: RGBColor = (0, 0, 0)
    WHITE: RGBColor = (1, 1, 1)
    DEFAULT_PATH_REWARDS: Dict[Tuple[int, int, int], float] = {
        (0, 1, 2): 1.00,  # red, green, blue
        (0, 2, 1): 0.80,  # red, blue, green
        (1, 0, 2): 0.60,  # green, red, blue
        (1, 2, 0): 0.40,  # green, blue, red
        (2, 0, 1): 0.20,  # blue, red, green
        (2, 1, 0): 0.05,  # blue, green, red
    }

    def __init__(
        self,
        max_steps: int = 6,
        step_cost: float = 0.02,
        overload_penalty: float = 0.0,
        timeout_penalty: float = 0.0,
        path_rewards: Optional[Dict[Tuple[int, int, int], float]] = None,
    ):
        self.max_steps = max_steps
        self.step_cost = step_cost
        self.overload_penalty = overload_penalty
        self.timeout_penalty = timeout_penalty
        self.path_rewards = dict(path_rewards or self.DEFAULT_PATH_REWARDS)
        self.current_color: RGBColor = self.BLACK
        self.steps = 0
        self.goals_reached = 0
        self.total_overloads = 0
        self.action_path: List[int] = []

    def color_id(self, color: RGBColor) -> int:
        return RGB_COLOR_TO_ID[color]

    def id_color(self, idx: int) -> RGBColor:
        return ID_TO_RGB_COLOR[idx]

    def _get_state(self) -> State:
        return self.color_id(self.current_color)

    def reset(self) -> State:
        self.current_color = self.BLACK
        self.steps = 0
        self.goals_reached = 0
        self.total_overloads = 0
        self.action_path = []
        return self._get_state()

    def legal_actions(self, state: Optional[State] = None) -> List[int]:
        return list(COLOR_ACTIONS.keys())

    def _mix(self, color: RGBColor, action_color: RGBColor) -> Tuple[RGBColor, bool]:
        mixed = (
            color[0] + action_color[0],
            color[1] + action_color[1],
            color[2] + action_color[2],
        )
        overload = mixed[0] > 1 or mixed[1] > 1 or mixed[2] > 1
        return (self.BLACK if overload else mixed), overload

    def step(self, action: int) -> StepResult:
        if action not in COLOR_ACTIONS:
            raise ValueError(f"Invalid action: {action}")

        self.steps += 1
        action_color = COLOR_ACTIONS[action]
        old_color = self.current_color
        next_color, overload = self._mix(old_color, action_color)

        reward_ext = -self.step_cost
        trial_success = False
        timed_out = False

        if overload:
            self.total_overloads += 1
            self.action_path = []
            reward_ext += self.overload_penalty
        else:
            self.action_path.append(action)

        self.current_color = next_color

        if self.current_color == self.WHITE:
            trial_success = True
            self.goals_reached = 1
            reward_ext += self.path_rewards.get(tuple(self.action_path), 0.0)

        done = trial_success or self.steps >= self.max_steps
        if done and not trial_success and self.steps >= self.max_steps:
            timed_out = True
            reward_ext += self.timeout_penalty

        info = {
            "overload": overload,
            "trial_success": trial_success,
            "timeout": timed_out,
            "color_name": RGB_COLOR_NAMES[self.current_color],
            "transition": (old_color, action_color, self.current_color),
        }

        return StepResult(
            next_state=self._get_state(),
            reward_ext=reward_ext,
            done=done,
            info=info,
        )
