#!/usr/bin/env bash
# Run the full sweep (3 budgets x 6 conditions x 3 seeds) and write summary CSVs.

set -e

STEPS=${STEPS:-300000}
SEEDS=${SEEDS:-"0 1 2"}
N_PARALLEL=${N_PARALLEL:-9}
MAX_EP_STEPS_SET=${MAX_EP_STEPS_SET:-"1500 600 300"}
TILE_FRAC=${TILE_FRAC:-0.25}
ENTROPY_COEF=${ENTROPY_COEF:-0.05}
CONDITIONS=${CONDITIONS:-"info_gain expected_info_gain surprisal novelty random hybrid"}

for MAX_EP_STEPS in $MAX_EP_STEPS_SET; do
    OUTPUT_DIR="./runs_white_${MAX_EP_STEPS}"
    FIGURES_DIR="./figures_white_${MAX_EP_STEPS}"
    STEPS=$STEPS SEEDS="$SEEDS" N_PARALLEL=$N_PARALLEL MAX_EP_STEPS=$MAX_EP_STEPS \
        TILE_FRAC=$TILE_FRAC ENTROPY_COEF=$ENTROPY_COEF CONDITIONS="$CONDITIONS" OUTPUT_DIR="$OUTPUT_DIR" \
        bash experiments/run_sweep_white.sh
    python -m experiments.analyze_white --runs_dir "$OUTPUT_DIR" --output_dir "$FIGURES_DIR"
done
