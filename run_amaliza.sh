#!/bin/bash
# Runs Amaliza's 10 experiments. Finished runs are skipped, so you can stop and re-run safely.
set -e
run() { [ -f "Amaliza/exp$(printf %02d $1)/summary.json" ] || uv run python src/train.py --member Amaliza --exp "$@"; }

run 1  --lr 0.0001
run 2  --lr 0.001
run 3  --lr 0.0005
run 4  --lr 0.00025
run 5  --lr 0.00005
run 6  --lr 0.00001
run 7  --lr 0.00025 --batch-size 64
run 8  --lr 0.00005 --eps-decay 50000
run 9  --lr 0.0005  --gamma 0.95
run 10 --lr 0.00025 --eps-end 0.1
