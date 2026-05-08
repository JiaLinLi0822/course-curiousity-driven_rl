"""Configuration for the 3x3 color-grid exploration project.

This file intentionally keeps configuration simple and explicit. Later
training code can import these values instead of scattering constants across
the project.
"""

from dataclasses import dataclass


# Color names are represented as two binary channels:
# blue channel, yellow channel.
BLACK = (0, 0)
BLUE = (1, 0)
YELLOW = (0, 1)
WHITE = (1, 1)

COLOR_NAME_TO_VECTOR = {
    "black": BLACK,
    "blue": BLUE,
    "yellow": YELLOW,
    "white": WHITE,
}

COLOR_VECTOR_TO_NAME = {value: key for key, value in COLOR_NAME_TO_VECTOR.items()}


@dataclass(frozen=True)
class EnvConfig:
    """Core environment settings."""

    grid_size: int = 3
    max_steps: int = 50
    start_pos: tuple[int, int] = (0, 0)
    goal_pos: tuple[int, int] = (2, 2)

    # These colors are placed at reset. By default their positions are
    # randomized so the agent cannot rely on fixed color-tile locations.
    initial_colored_tiles: tuple[tuple[tuple[int, int], tuple[int, int]], ...] = (
        ((1, 0), BLUE),
        ((0, 1), YELLOW),
    )
    randomize_initial_tile_positions: bool = True

    solved_reward: float = 1.0
    timeout_reward: float = -0.5
    step_reward: float = 0.0


@dataclass(frozen=True)
class ObservationConfig:
    """Observation shape settings."""

    position_dim: int = 2
    color_dim: int = 2
    grid_size: int = 3

    @property
    def grid_color_dim(self) -> int:
        return self.grid_size * self.grid_size * self.color_dim

    @property
    def obs_dim(self) -> int:
        # normalized (x, y) + flattened grid colors + current agent color
        return self.position_dim + self.grid_color_dim + self.color_dim


@dataclass(frozen=True)
class RewardConfig:
    """Reward weights used by the environment and later training code."""

    reward_type: str = "novelty"
    intrinsic_weight: float = 1.0
    extrinsic_weight: float = 1.0
    intrinsic_coef: float = 0.1

    # Novelty-specific knobs.
    novelty_alpha: float = 0.5
    novelty_persistent_counts: bool = True

    # Information-gain-specific knobs.
    info_gain_smoothing: float = 1.0
    info_gain_clip: float | None = 5.0

    # Dirichlet information-gain-specific knobs.
    dirichlet_prior_alpha: float = 1.0
    dirichlet_persistent_across_episodes: bool = True
    dirichlet_clip: float | None = 1.0


@dataclass(frozen=True)
class PPOConfig:
    """Small PPO training settings."""

    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    minibatch_size: int = 64
    rollout_steps: int = 256
    hidden_size: int = 64


@dataclass(frozen=True)
class TrainConfig:
    """Minimal end-to-end training loop settings."""

    total_timesteps: int = 100000
    num_episodes: int = 300
    log_every_episodes: int = 10
    seed: int = 0
    device: str = "cpu"


@dataclass(frozen=True)
class EvalConfig:
    """Evaluation loop settings."""

    eval_every_episodes: int = 20
    num_eval_episodes: int = 50
    deterministic: bool = False
    save_eval_csv: bool = True
    output_path: str = "outputs/eval_results.csv"


DEFAULT_ENV_CONFIG = EnvConfig()
DEFAULT_OBS_CONFIG = ObservationConfig(grid_size=DEFAULT_ENV_CONFIG.grid_size)
DEFAULT_REWARD_CONFIG = RewardConfig()
DEFAULT_PPO_CONFIG = PPOConfig()
DEFAULT_TRAIN_CONFIG = TrainConfig()
DEFAULT_EVAL_CONFIG = EvalConfig()


# Dictionary-style config placeholders make quick experiment sweeps easy while
# the dataclasses above keep the existing code readable and typed.
TRAIN_CONFIG = {
    "total_timesteps": DEFAULT_TRAIN_CONFIG.total_timesteps,
    "num_episodes": DEFAULT_TRAIN_CONFIG.num_episodes,
    "log_every_episodes": DEFAULT_TRAIN_CONFIG.log_every_episodes,
    "seed": DEFAULT_TRAIN_CONFIG.seed,
    "device": DEFAULT_TRAIN_CONFIG.device,
}

PPO_CONFIG = {
    "learning_rate": DEFAULT_PPO_CONFIG.learning_rate,
    "entropy_coef": DEFAULT_PPO_CONFIG.entropy_coef,
    "gamma": DEFAULT_PPO_CONFIG.gamma,
    "gae_lambda": DEFAULT_PPO_CONFIG.gae_lambda,
    "clip_eps": DEFAULT_PPO_CONFIG.clip_epsilon,
    "value_loss_coef": DEFAULT_PPO_CONFIG.value_loss_coef,
    "max_grad_norm": DEFAULT_PPO_CONFIG.max_grad_norm,
    "update_epochs": DEFAULT_PPO_CONFIG.update_epochs,
    "minibatch_size": DEFAULT_PPO_CONFIG.minibatch_size,
    "rollout_steps": DEFAULT_PPO_CONFIG.rollout_steps,
    "hidden_size": DEFAULT_PPO_CONFIG.hidden_size,
}

REWARD_CONFIG = {
    "reward_type": DEFAULT_REWARD_CONFIG.reward_type,
    "intrinsic_coef": DEFAULT_REWARD_CONFIG.intrinsic_coef,
    "novelty_alpha": DEFAULT_REWARD_CONFIG.novelty_alpha,
    "novelty_persistent_counts": DEFAULT_REWARD_CONFIG.novelty_persistent_counts,
    "info_gain_smoothing": DEFAULT_REWARD_CONFIG.info_gain_smoothing,
    "info_gain_clip": DEFAULT_REWARD_CONFIG.info_gain_clip,
    "dirichlet_prior_alpha": DEFAULT_REWARD_CONFIG.dirichlet_prior_alpha,
    "dirichlet_persistent_across_episodes": (
        DEFAULT_REWARD_CONFIG.dirichlet_persistent_across_episodes
    ),
    "dirichlet_clip": DEFAULT_REWARD_CONFIG.dirichlet_clip,
}

EVAL_CONFIG = {
    "eval_every_episodes": DEFAULT_EVAL_CONFIG.eval_every_episodes,
    "num_eval_episodes": DEFAULT_EVAL_CONFIG.num_eval_episodes,
    "deterministic": DEFAULT_EVAL_CONFIG.deterministic,
    "save_eval_csv": DEFAULT_EVAL_CONFIG.save_eval_csv,
    "output_path": DEFAULT_EVAL_CONFIG.output_path,
}
