#!/usr/bin/env bash
# Evaluate every downloaded weight file afresh, then draw from those evaluations
# alone the paper's artifacts that final weights reproduce:
#   - the final-evaluation rows of the two-dimensional comparison table,
#   - both image tables and the image comparison figures,
#   - the Schrödinger ↔ Öresund generation figure (seed 0).
# The training curves, the 6h row and the ablations need the training records
# (scripts/paper_from_records.sh).
#
# Usage, from the repository root, after scripts/download_zenodo.sh:
#   bash scripts/paper_from_weights.sh [EVAL_ARG ...]
# EVAL_ARGs go to every eval.py call, e.g. --device cpu without CUDA (not MPS, untested).
#
# Each run's evaluation goes to outputs/zenodo/evals/<run>/eval/, which reads like
# a training run's eval/. A finished one, with its curve.jsonl, is kept; delete it
# to evaluate again. The artifacts go to outputs/zenodo/paper/from_weights/.
set -euo pipefail
shopt -s nullglob

RUNS=outputs/zenodo/training_records
EVALS=outputs/zenodo/evals
PAPER=outputs/zenodo/paper/from_weights

for model in "$RUNS"/*/*/*/*/*/checkpoints/final_model_ema.safetensors; do
    run=${model#"$RUNS"/}
    run=${run%/checkpoints/*}
    if [[ -f $EVALS/$run/eval/curve.jsonl ]]; then
        echo "Have $EVALS/$run/eval"
        continue
    fi
    rm -rf "$EVALS/$run/eval"  # An unfinished evaluation.
    python eval.py "$model" --out "$EVALS/$run/eval" "$@"
done

# Artifacts whose weights weren't downloaded are skipped.
if [[ -d $EVALS/uniform-to-checkerboard && -d $EVALS/schrodinger-to-oresund ]]; then
    python -m scripts.paper_artifacts_toy2d rsbm_vs_rsb_fbsde_final_table \
        --runs-dir "$EVALS" --out-dir "$PAPER/toy2d"
else
    echo "Skipped the two-dimensional table: it needs the checkerboard_* and oresund_* weights."
fi
if [[ -d $EVALS/afhq64 && -d $EVALS/mnist-to-emnist ]]; then
    python -m scripts.paper_artifacts_imgs --runs-dir "$EVALS" --out-dir "$PAPER/images"
else
    echo "Skipped the image artifacts: they need the afhq64_* and mnist_to_emnist_* weights."
fi
generation=("$EVALS"/schrodinger-to-oresund/alpha_imf/refl/sigma0.5/seed0/eval/*/ema/generation_procedure.pdf)
if [[ ${#generation[@]} -eq 1 ]]; then
    mkdir -p "$PAPER/toy2d"
    cp "${generation[0]}" "$PAPER/toy2d/"
    echo "Wrote $PAPER/toy2d/generation_procedure.pdf"
else
    echo "Skipped the generation figure: no evaluation of the Öresund α-RSBM seed 0 weights."
fi
