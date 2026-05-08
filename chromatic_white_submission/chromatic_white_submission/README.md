# Chromatic White: information-theoretic exploration with deep RL + world model

6x6 grid environment, 8 colors (3-bit RGB), sparse goal: paint the bottom-right tile white.
PPO agent (GRU + MLP), Dirichlet-Categorical world model over the 36 unordered color pairs.
Six exploration conditions: `surprisal`, `expected_info_gain` (BALD), `info_gain` (Dirichlet IG),
`novelty`, `random`, `hybrid`. Three episode budgets: 300, 600, 1500. Three seeds per cell.

## Layout

```
chromatic_white/      # env, agent, world model, intrinsic rewards, PPO trainer, evaluator
experiments/          # train_white.py, analyze_white.py, sweep shell scripts
runs_white_{300,600,1500}/   # per-run history.json, eval_history.json, episode_history.json, agent.pt
figures_white_{300,600,1500}/ # summary.csv (one row per condition x seed)
slide_deck_figures/   # make_slide_figures.ipynb + the seven output figures (PNG/PDF)
```

## Reproducing the figures

The notebook reads from the `runs_white_*/` and `figures_white_*/summary.csv` already in this
folder. To regenerate all seven figures from the included data:

```bash
cd slide_deck_figures
jupyter nbconvert --to notebook --execute --inplace make_slide_figures.ipynb
```

Or open `slide_deck_figures/make_slide_figures.ipynb` in Jupyter and run all cells.
Plot 7 (heatmaps) loads each `agent.pt` and rolls out 30 evaluation episodes per condition
per budget, so it takes a few minutes on CPU.

The seven output figures are:

1. `1_eval_success_rate_3budget_bar` - eval success rate at three episode budgets
2. `2_intrinsic_reward_log` - intrinsic-reward magnitude over training (600-step budget)
3. `3_training_episode_length` - training episode length over time (600-step budget)
4. `4_solve_time_vs_solve_number` - steps-to-solve indexed by solve number (600-step budget)
5. `5_rule_learning_kl_600` - KL(true rule || learned posterior) over training (600-step budget)
6. `6_speedup_slopes_1500` - per-seed eval steps-to-solve regression slope (1500-step budget)
7. `7_trajectory_heatmaps` - tile-visit heatmaps, share of visits per tile (600 + 1500 budgets)

## Reproducing the runs from scratch

Requirements: `numpy`, `scipy`, `pandas`, `matplotlib`, `torch`.

Single condition x seed:

```bash
python -m experiments.train_white \
    --condition surprisal --seed 0 --steps 300000 --max_ep_steps 600 \
    --output_dir ./runs_white_600
```

Full sweep (all conditions x 3 seeds x 3 budgets, ~9 parallel jobs):

```bash
bash experiments/overnight_white.sh
```

After training, regenerate `summary.csv` for each budget:

```bash
python -m experiments.analyze_white \
    --runs_dir ./runs_white_600 --output_dir ./figures_white_600
```

Then re-run the notebook.
