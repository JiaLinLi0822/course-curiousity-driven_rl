"""Observation encoding for the color-grid environment.

The environment owns the game state. This module only decides how that state
is converted into a fixed-size vector for an RL policy.
"""

from __future__ import annotations

import numpy as np

from .config import DEFAULT_OBS_CONFIG, ObservationConfig


class ObservationEncoder:
    """Encode environment state into a flat numeric vector.

    Encoding order:
    1. normalized x position
    2. normalized y position
    3. flattened 3x3 grid colors
    4. current agent color
    """

    def __init__(self, config: ObservationConfig = DEFAULT_OBS_CONFIG):
        self.config = config

    @property
    def obs_dim(self) -> int:
        return self.config.obs_dim

    def encode(
        self,
        agent_pos: tuple[int, int],
        grid_colors: np.ndarray,
        agent_color: np.ndarray,
    ) -> np.ndarray:
        """Return a fixed-size float32 observation vector."""
        x, y = agent_pos
        max_coord = self.config.grid_size - 1

        if max_coord <= 0:
            raise ValueError("grid_size must be at least 2")

        pos_features = np.array([x / max_coord, y / max_coord], dtype=np.float32)
        grid_features = grid_colors.reshape(-1).astype(np.float32)
        agent_features = agent_color.astype(np.float32)

        obs = np.concatenate([pos_features, grid_features, agent_features])

        if obs.shape != (self.obs_dim,):
            raise ValueError(f"Expected observation shape {(self.obs_dim,)}, got {obs.shape}")

        return obs
