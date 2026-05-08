#!/usr/bin/env bash
# Run training across {conditions} x {seeds} in parallel.

set -e

STEPS=${STEPS:-300000}
EVAL_EVERY=${EVAL_EVERY:-10000}
OUTPUT_DIR=${OUTPUT_DIR:-./runs_white}
N_PARALLEL=${N_PARALLEL:-9}
CONDITIONS=${CONDITIONS:-"info_gain expected_info_gain surprisal novelty random hybrid"}
SEEDS=${SEEDS:-"0 1 2"}
BETA=${BETA:-1.0}
ENTROPY_COEF=${ENTROPY_COEF:-0.05}
MAX_EP_STEPS=${MAX_EP_STEPS:-1500}
TILE_FRAC=${TILE_FRAC:-0.25}

mkdir -p "$OUTPUT_DIR"

JOB_FILE=$(mktemp)
for cond in $CONDITIONS; do
    for seed in $SEEDS; do
        echo "python -m experiments.train_white --condition $cond --seed $seed --steps $STEPS --eval_every $EVAL_EVERY --output_dir $OUTPUT_DIR --beta $BETA --entropy_coef $ENTROPY_COEF --max_ep_steps $MAX_EP_STEPS --n_base_color_tiles_frac $TILE_FRAC" >> "$JOB_FILE"
    done
done

ACTIVE=0
while IFS= read -r job; do
    bash -c "$job" &
    ACTIVE=$((ACTIVE + 1))
    if [ "$ACTIVE" -ge "$N_PARALLEL" ]; then
        wait
        ACTIVE=0
    fi
done < "$JOB_FILE"
wait
rm "$JOB_FILE"
