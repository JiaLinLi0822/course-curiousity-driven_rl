"""6x6 grid environment with 8-color (3-bit RGB) mixing rule."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from chromatic_white.rules import (
    BLACK, RED, GREEN, BLUE, WHITE,
    COLORS, COLOR_TO_IDX, NUM_COLORS,
)


GRID_SIZE = 6
START_XY = (0, 0)
GOAL_XY = (GRID_SIZE - 1, GRID_SIZE - 1)

ACTION_DELTAS = (
    (0, -1),
    (0, 1),
    (-1, 0),
    (1, 0),
)
NUM_ACTIONS = 4


@dataclass
class StepInfo:
    transition: Tuple[Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int]]
    was_overload: bool
    was_oob: bool
    solved: bool
    timed_out: bool


class ChromaticWhiteEnv:
    OBS_DIM = (GRID_SIZE * GRID_SIZE) + NUM_COLORS + (GRID_SIZE * GRID_SIZE) * NUM_COLORS + 1

    def __init__(
        self,
        max_steps: int = 1500,
        seed: Optional[int] = None,
        n_base_color_tiles_frac: float = 0.25,
    ):
        self.max_steps = max_steps
        self.n_base_color_tiles_frac = n_base_color_tiles_frac
        self.rng = np.random.default_rng(seed)

        self.grid: np.ndarray = np.zeros((GRID_SIZE, GRID_SIZE, 3), dtype=np.int8)
        self.px: int = START_XY[0]
        self.py: int = START_XY[1]
        self.pc: Tuple[int, int, int] = BLACK
        self.steps: int = 0
        self.done: bool = True

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.px, self.py = START_XY
        self.pc = BLACK
        self.steps = 0
        self.done = False
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE, 3), dtype=np.int8)

        avail = [
            (x, y)
            for y in range(GRID_SIZE)
            for x in range(GRID_SIZE)
            if (x, y) != START_XY and (x, y) != GOAL_XY
        ]
        self.rng.shuffle(avail)

        mandatory = [RED, GREEN, BLUE]
        i = 0
        for color in mandatory:
            x, y = avail[i]
            self.grid[y, x] = color
            i += 1

        for j in range(i, len(avail)):
            if self.rng.random() < self.n_base_color_tiles_frac:
                x, y = avail[j]
                self.grid[y, x] = self._random_base_color()

        return self._obs()

    def _random_base_color(self) -> Tuple[int, int, int]:
        r = self.rng.random()
        if r < 1 / 3:
            return RED
        if r < 2 / 3:
            return GREEN
        return BLUE

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        if self.done:
            raise RuntimeError("step() on a done env. Call reset().")
        if not (0 <= action < NUM_ACTIONS):
            raise ValueError(f"Invalid action: {action}")

        dx, dy = ACTION_DELTAS[action]
        nx, ny = self.px + dx, self.py + dy
        self.steps += 1

        was_oob = not (0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE)

        if was_oob:
            reward = 0.0
            timed_out = self.steps >= self.max_steps
            if timed_out:
                reward = -0.5
                self.done = True
            info = StepInfo(
                transition=(self.pc, self.pc, self.pc),
                was_overload=False, was_oob=True,
                solved=False, timed_out=timed_out,
            )
            return self._obs(), reward, self.done, info.__dict__

        self.px, self.py = nx, ny
        c_A = self.pc
        c_B = tuple(self.grid[ny, nx].tolist())

        mixed = (c_A[0] + c_B[0], c_A[1] + c_B[1], c_A[2] + c_B[2])
        if mixed[0] > 1 or mixed[1] > 1 or mixed[2] > 1:
            self.pc = BLACK
            self.grid[ny, nx] = BLACK
            outcome = BLACK
            was_overload = True
        else:
            self.pc = mixed
            self.grid[ny, nx] = mixed
            outcome = mixed
            was_overload = False

        solved = tuple(self.grid[GOAL_XY[1], GOAL_XY[0]].tolist()) == WHITE

        reward = 0.0
        timed_out = False
        if solved:
            reward = 1.0
            self.done = True
        elif self.steps >= self.max_steps:
            reward = -0.5
            timed_out = True
            self.done = True

        info = StepInfo(
            transition=(c_A, c_B, outcome),
            was_overload=was_overload, was_oob=False,
            solved=solved, timed_out=timed_out,
        )
        return self._obs(), reward, self.done, info.__dict__

    def _obs(self) -> np.ndarray:
        obs = np.zeros(self.OBS_DIM, dtype=np.float32)
        obs[self.py * GRID_SIZE + self.px] = 1.0

        pos_dim = GRID_SIZE * GRID_SIZE
        obs[pos_dim + COLOR_TO_IDX[self.pc]] = 1.0

        tile_offset = pos_dim + NUM_COLORS
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                tc = tuple(self.grid[y, x].tolist())
                tile_idx = y * GRID_SIZE + x
                obs[tile_offset + tile_idx * NUM_COLORS + COLOR_TO_IDX[tc]] = 1.0

        obs[-1] = (self.max_steps - self.steps) / self.max_steps
        return obs

    def render(self) -> str:
        name_of = {
            BLACK: ".", RED: "R", GREEN: "G", BLUE: "B",
            (1, 1, 0): "Y", (1, 0, 1): "M", (0, 1, 1): "C", WHITE: "W",
        }
        lines = []
        for y in range(GRID_SIZE):
            row = []
            for x in range(GRID_SIZE):
                tc = tuple(self.grid[y, x].tolist())
                g = name_of.get(tc, "?")
                if (x, y) == (self.px, self.py):
                    row.append(f"[{name_of.get(self.pc, '?')}]")
                elif (x, y) == GOAL_XY:
                    row.append(f" {g}*")
                else:
                    row.append(f" {g} ")
            lines.append("".join(row))
        lines.append(f"steps={self.steps}/{self.max_steps} pc={name_of.get(self.pc, '?')}")
        return "\n".join(lines)
