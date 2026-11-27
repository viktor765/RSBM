#!/bin/bash
# RSBM recipes: one job per finetuning learning rate, training for one hour
# including pretraining.
# From the repository root: bash slurm/sweep-checkerboard-rsbm.sh
set -euo pipefail
mkdir -p logs
export SEEDS="100 101 102"
for lr in 0.0003 0.0006 0.0012; do
  sbatch -J "sweep-rsbm-lr$lr" slurm/checkerboard-rsbm.sh \
    training.finetune_lr=$lr \
    training.n_finetune_steps=null training.max_training_seconds=3600 \
    eval.freq=300 eval.toy2d.sweep_metrics_only=true \
    "run_name=sweep/uniform-to-checkerboard/alpha_imf/lr$lr/seed\${seed}"
done
