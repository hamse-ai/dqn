#!/bin/bash
# Runs Prince's 10 experiments (gamma and batch size sweep).
# Anything not listed uses the baseline: lr 1e-4, gamma 0.99, batch 32,
# eps_start 1.0, eps_end 0.05, eps_decay 100000, 300k steps.
# Finished runs (those with a summary.json) are skipped, so it is safe to stop and re-run.
set -e

run() {
    folder="Prince/exp$(printf %02d "$1")"
    if [ -f "$folder/summary.json" ]; then
        echo "$folder already done, skipping"
    else
        uv run python src/train.py --member Prince --exp "$@"
    fi
}

run 1  --gamma 0.90
run 2  --gamma 0.95
run 3  --gamma 0.97
run 4  --gamma 0.995
run 5  --gamma 0.999
run 6  --batch-size 16
run 7  --batch-size 64
run 8  --batch-size 128
run 9  --gamma 0.95  --batch-size 64
run 10 --gamma 0.995 --batch-size 128

