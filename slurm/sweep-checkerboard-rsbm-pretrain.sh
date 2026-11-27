#!/bin/bash
# RSBM pretraining: one job per learning rate, pretraining until the time limit.
# Write the chosen rate and update count into conf/training/toy2dsb.yaml.
# From the repository root: bash slurm/sweep-checkerboard-rsbm-pretrain.sh
set -euo pipefail
mkdir -p logs
export SEEDS="100 101 102"
for lr in 0.0003 0.0006 0.0012; do
  sbatch -J "sweep-rsbm-pretrain-lr$lr" slurm/checkerboard-rsbm.sh \
    training.pretrain_lr=$lr \
    training.n_pretrain_steps=1000000000 training.n_finetune_steps=0 \
    training.max_training_seconds=900 \
    eval.freq=1000 eval.freq_unit=iterations eval.toy2d.sweep_metrics_only=true \
    "run_name=sweep/uniform-to-checkerboard/rsbm-pretrain/lr$lr/seed\${seed}"
done
