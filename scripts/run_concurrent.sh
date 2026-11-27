#!/bin/bash
# Run one command per seed, all at once, appending seed=<seed> to each.
#
#   bash scripts/run_concurrent.sh --seeds 0 1 2 [--mps] -- python train.py experiment=abc
#
# The runs share the machine's CPU cores evenly through OMP_NUM_THREADS. --mps
# lets their kernels overlap on one GPU instead of taking turns. With a container,
# put it in the command, so the MPS daemon stays outside it:
#
#   bash scripts/run_concurrent.sh --seeds 0 1 2 --mps -- \
#       apptainer exec --nv image.sif python train.py experiment=abc
#
# Exits non-zero if any run fails; the other runs still finish.

set -euo pipefail

usage() {
    echo "Usage: $0 --seeds SEED... [--mps] -- COMMAND..." >&2
    exit 2
}

seeds=()
mps=0
while (( $# )); do
    case "$1" in
        --seeds)
            shift
            while (( $# )) && [[ "$1" != --* ]]; do
                seeds+=("$1")
                shift
            done
            ;;
        --mps)
            mps=1
            shift
            ;;
        --)
            shift
            break
            ;;
        *)
            usage
            ;;
    esac
done
(( ${#seeds[@]} > 0 && $# > 0 )) || usage

cores=${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-$(getconf _NPROCESSORS_ONLN)}}
share=$(( cores / ${#seeds[@]} ))
export OMP_NUM_THREADS=$(( share > 0 ? share : 1 ))

if (( mps )); then
    source "$(dirname "${BASH_SOURCE[0]}")/start_mps.sh"
fi

pids=()
for seed in "${seeds[@]}"; do
    echo "$* seed=$seed"
    "$@" "seed=$seed" &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
exit "$status"
