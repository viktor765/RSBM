#!/bin/bash
# RSB-FBSDE on uniform -> checkerboard: three seeds at once on one GPU. Arguments
# go to train.py. From the repository root, once logs/ exists:
#   sbatch slurm/checkerboard-fbsde.sh
#   SEEDS="100 101 102" sbatch slurm/checkerboard-fbsde.sh training.lr=0.0006
#SBATCH -J checkerboard-fbsde
#SBATCH --gpus=1
#SBATCH -C thin
#SBATCH -t 03:00:00
#SBATCH --output=logs/slurm-%x_%j.out

set -euo pipefail
bash scripts/run_concurrent.sh --seeds ${SEEDS:-0 1 2} --mps -- \
    apptainer exec --nv my-pytorch.sif \
    python train.py experiment=rsb_fbsde_uniform_to_checkerboard "$@"
