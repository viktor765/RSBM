#!/bin/bash

#SBATCH -p gpu
#SBATCH -t 70:00:00
#SBATCH --gpus 1
#SBATCH --array=0-2
#SBATCH --job-name=afhq64_reflected
#SBATCH --output=logs/slurm-%x_%A_%a.out

apptainer exec \
    --nv \
    --env TRITON_LIBCUDA_PATH=/usr/local/cuda/compat/lib.real \
    my-pytorch.sif \
    python train.py experiment=afhq64 sde=reflected seed=$SLURM_ARRAY_TASK_ID