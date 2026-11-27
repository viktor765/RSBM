#!/bin/bash

#SBATCH -p gpu
#SBATCH -t 30:00:00
#SBATCH --gpus 1
#SBATCH --array=0-2
#SBATCH --job-name=mnist_to_emnist_reflected
#SBATCH --output=logs/slurm-%x_%A_%a.out

apptainer exec \
    --nv \
    --env TRITON_LIBCUDA_PATH=/usr/local/cuda/compat/lib.real \
    my-pytorch.sif \
    python train.py experiment=mnist_to_emnist sde=reflected seed=$SLURM_ARRAY_TASK_ID
