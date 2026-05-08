# Deep Color-Grid RL

This folder contains a PPO-based reinforcement-learning project for a small
3x3 color-grid task. The experiments compare different intrinsic-reward
signals for learning a hidden color-mixing rule and using it to solve the
environment.

## Task

The agent moves on a 3x3 grid. Its color and the tile color are represented by
two binary channels:

- `black`: no active channels
- `blue`: blue channel active
- `yellow`: yellow channel active
- `white`: both channels active

The default start position is the top-left cell, and the goal is the
bottom-right cell. At reset, blue and yellow tiles are placed on random
non-start, non-goal cells. When the agent enters a tile, its current color is
mixed with the tile color. Combining blue and yellow makes white; reusing an
already active channel overloads the mixture and resets that color to black.

An episode is solved when the goal tile becomes white.

## Agents and Rewards

The policy is trained with PPO. The experiment compares several intrinsic
reward modes:

- `random`: random intrinsic reward baseline
- `novelty`: count-based novelty reward
- `surprisal`: predictive information-gain style reward
- `dirichlet IG`: Dirichlet transition-model information gain

The environment also provides sparse extrinsic reward for solving the task or
timing out.

## Files

- `color_grid_rl/env.py`: 3x3 color-grid environment
- `color_grid_rl/config.py`: environment, PPO, reward, training, and evaluation settings
- `color_grid_rl/ppo_agent.py`: PPO agent and update logic
- `color_grid_rl/model.py`: policy/value neural network
- `color_grid_rl/buffer.py`: rollout buffer
- `color_grid_rl/rewards.py`: baseline reward helpers
- `color_grid_rl/novelty.py`: count-based novelty reward
- `color_grid_rl/info_gain.py`: predictive surprisal / information-gain reward
- `color_grid_rl/dirichlet_info_gain.py`: Dirichlet information-gain reward
- `color_grid_rl/main_train.py`: single training run
- `color_grid_rl/run_experiments.py`: reward-mode and seed sweep
- `color_grid_rl/test_policy.py`: final evaluation entry point
- `color_grid_rl/plot_results.py`: training-curve plots from `results/`
- `color_grid_rl/plot_eval_results.py`: evaluation plots from `outputs/eval_results.csv`
- `color_grid_rl/summarize_results.py`: summary table from episode logs

## Run

From the repository root, enter this folder first:

```bash
cd deep_rl_folder
```

Install the Python dependencies if needed:

```bash
python -m pip install numpy torch matplotlib gymnasium
```

Run one default PPO training job:

```bash
python -m color_grid_rl.main_train
```

Run the full comparison across reward modes and seeds:

```bash
python -m color_grid_rl.run_experiments
```

Create training summaries and plots:

```bash
python -m color_grid_rl.summarize_results
python -m color_grid_rl.plot_results
python -m color_grid_rl.plot_eval_results
```

Run the final evaluation entry point:

```bash
python -m color_grid_rl.test_policy
```

## Outputs

Training logs are written to `results/`:

- one CSV per reward mode and seed, such as `novelty_seed0.csv`
- `summary.csv`

Training plots are written to `figures/`:

- `success_rate.png`
- `average_steps.png`
- `overloads.png`
- `intrinsic_return.png`
- `extrinsic_return.png`

Evaluation logs and figures are written to `outputs/`:

- `eval_results.csv`
- `final_eval_results.csv`
- `plots/`

## Main Settings

The main defaults live in `color_grid_rl/config.py`:

- grid size: `3`
- max episode steps: `50`
- total training timesteps: `100000`
- PPO rollout steps: `256`
- seeds used by `run_experiments.py`: `0`, `1`, `2`
- reward modes used by `run_experiments.py`: `random`, `novelty`, `surprisal`, `dirichlet IG`
