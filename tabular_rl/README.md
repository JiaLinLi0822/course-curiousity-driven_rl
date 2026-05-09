# Color Mixing MDP Agents

This repository compares four tabular reinforcement-learning agents on a small
eight-state color-mixing MDP.

## Task

The environment starts at black. Each action adds one primary color:

- `red`
- `green`
- `blue`

The goal is to reach white. Reusing a color channel overloads the mixture and
resets the state to black. Valid paths to white receive different terminal
rewards, so agents must learn not only how to reach white but which order is
best.

## Agents

The experiment compares:

- Epsilon-greedy Q-learning
- UCB Q-learning
- Novelty-based Q-learning
- Curiosity-driven Q-learning with a Dirichlet transition model

## Files

- `compare_color_mdp_agents.py`: command-line experiment runner
- `env.py`: color-mixing MDP environment
- `agent.py`: the four learning algorithms
- `color_mdp_agents.py`: agent factory functions and Dirichlet adapter
- `train.py`: training loop
- `color_mdp_tables.py`: rollout, Q-table, diagnostics, and CSV helpers
- `plotting.py`: all plotting code
- `color_mdp_constants.py`: shared labels for states and actions

## Run

Install the Python dependencies if needed:

```bash
python -m pip install numpy scipy matplotlib
```

Run the default comparison:

```bash
python compare_color_mdp_agents.py
```

For a quick smoke test:

```bash
python compare_color_mdp_agents.py --episodes 5 --seeds 1 --out-dir /private/tmp/color_mdp_check --smooth 1
```

## Outputs

By default, results are written to `results/color_mdp/`:

- `color_mdp_training_returns.png`
- `color_mdp_q_tables.png`
- `q_tables/*.png`

## Main Options

```bash
python compare_color_mdp_agents.py \
  --episodes 2000 \
  --seeds 30 \
  --master-seed 123 \
  --out-dir results/color_mdp \
  --smooth 25
```
