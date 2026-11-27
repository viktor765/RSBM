#!/usr/bin/env bash
# Draw all the paper's figures and tables from the published training records,
# without the weights.
#
# Usage, from the repository root, after scripts/download_zenodo.sh:
#   bash scripts/paper_from_records.sh
#
# Reads outputs/zenodo/training_records/ and writes outputs/zenodo/paper/from_records/.
set -euo pipefail

RUNS=outputs/zenodo/training_records
PAPER=outputs/zenodo/paper/from_records

python -m scripts.paper_artifacts_toy2d --runs-dir "$RUNS" --out-dir "$PAPER/toy2d"
python -m scripts.paper_artifacts_imgs --runs-dir "$RUNS" --out-dir "$PAPER/images"
