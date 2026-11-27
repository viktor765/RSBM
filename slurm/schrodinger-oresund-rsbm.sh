#!/bin/bash
# RSBM on Schrodinger -> Oresund: three seeds at once on one GPU. Arguments go
# to train.py. From the repository root, once logs/ exists:
#   sbatch slurm/schrodinger-oresund-rsbm.sh
#   SEEDS="100 101 102" sbatch slurm/schrodinger-oresund-rsbm.sh training.finetune_lr=0.0006
#SBATCH -J schrodinger-oresund-rsbm
#SBATCH --gpus=1
#SBATCH -C thin
#SBATCH -t 02:00:00
#SBATCH --output=logs/slurm-%x_%j.out

set -euo pipefail
bash scripts/run_concurrent.sh --seeds ${SEEDS:-0 1 2} --mps -- \
    apptainer exec --nv my-pytorch.sif \
    python train.py experiment=schrodinger_to_oresund "$@"
