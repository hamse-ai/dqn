#!/bin/bash
# Runs Hamse's 10 experiments (epsilon schedule sweep).
# Anything not listed uses the baseline: lr 1e-4, gamma 0.99, batch 32,
# eps_start 1.0, eps_end 0.05, eps_decay 100000, 300k steps.
# Finished runs (those with a summary.json) are skipped, so it is safe to stop and re-run.
set -e

run() {
    folder="Hamse/exp$(printf %02d "$1")"
    if [ -f "$folder/summary.json" ]; then
        echo "$folder already done, skipping"
    else
        uv run python train.py --member Hamse --exp "$@"
    fi
}

run 1  --eps-decay 20000
run 2  --eps-decay 50000
run 3  --eps-decay 200000
run 4  --eps-decay 300000
run 5  --eps-end 0.01
run 6  --eps-end 0.1
run 7  --eps-end 0.2
run 8  --eps-start 0.5
run 9  --eps-start 0.5 --eps-end 0.01 --eps-decay 20000
run 10 --eps-start 1.0 --eps-end 0.02 --eps-decay 50000