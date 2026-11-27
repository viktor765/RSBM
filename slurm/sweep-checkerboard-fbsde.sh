#!/bin/bash
# RSB-FBSDE recipes: one job per learning rate, decay factor and cache size,
# training for two hours. The middle decay factor is sqrt(0.8).
# From the repository root: bash slurm/sweep-checkerboard-fbsde.sh
set -euo pipefail
mkdir -p logs
export SEEDS="100 101 102"
for lr in 0.0003 0.0006 0.0012; do
  for gamma in 0.8 0.8944271909999159 1.0; do
    for cache in 32768 262144; do
      sbatch -J "sweep-fbsde-lr$lr-gamma$gamma-cache$cache" slurm/checkerboard-fbsde.sh \
        training.lr=$lr training.lr_gamma=$gamma training.cache_size=$cache \
        training.num_stages=null training.max_training_seconds=7200 \
        eval.freq=300 eval.toy2d.sweep_metrics_only=true \
        "run_name=sweep/uniform-to-checkerboard/rsb_fbsde/lr${lr}_gamma${gamma}_cache$cache/seed\${seed}"
    done
  done
done
