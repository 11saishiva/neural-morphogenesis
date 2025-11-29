#!/usr/bin/env bash
set -euo pipefail

# edit these commands to point to your python training entry
TRAIN_CMD="python src/experiments/train_local_sorting.py"

mkdir -p exps
# baseline - 200 updates (or smaller, keep short)
python -u src/experiments/train_local_sorting.py --total_updates 200 --log_suffix baseline > exps/log_baseline.txt 2>&1 & pid1=$!

# experiment 1: normalized advantages + higher entropy
python -u src/experiments/train_local_sorting.py --total_updates 200 --ent_coef 0.03 --log_suffix exp1_normadv_entropy > exps/log_exp1.txt 2>&1 & pid2=$!

# experiment 2: normalized advantages + std_init + higher entropy
# you must have changed neuro_fuzzy.logstd to 0.6 in code for this to take effect
python -u src/experiments/train_local_sorting.py --total_updates 200 --ent_coef 0.05 --log_suffix exp2_stdinit_entropy > exps/log_exp2.txt 2>&1 & pid3=$!

echo "Launched pids: $pid1, $pid2, $pid3"
wait
