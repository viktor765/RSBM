"""Images of a non-reflected and a reflected run side by side.

python -m scripts.compare_imgs NONREF_RUN REFLECTED_RUN [--random N ...] [--top-k K]
    [--images-per-row R] [--split SPLIT] [--weights WEIGHTS] [--out-dir DIR]
    [--format {pdf,png}]

Both runs translate the same source images with the same noise. Their samples come
from the cache of each run's final evaluation, eval/<step>/<weights>/cache; a missing
cache is filled there from checkpoints/final_model_<weights>.safetensors, with the
run's own settings, as the final evaluation would have. Each direction gets one
figure of N randomly chosen samples per --random count, the first N of one fixed
order, and one of the K samples whose non-reflected versions leave [0, 1] most.
Columns are the source, the non-reflected sample with out-of-range pixels in magenta,
that sample clipped to [0, 1], and the reflected sample; each cell holds R x R
samples. Figures go to DIR, by default <reflected cache>/../comparison, as PDF, or
PNG with --format png, in paper_style.py's style.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch as th
from omegaconf import OmegaConf

from scripts.paper_style import TEXT_WIDTH, save
from src import data_utils, models
from src.evaluation import posthoc
from src.evaluation.context import EvaluationContext
from src.evaluation.factory import run_evaluator_arguments
from src.evaluation.image.cached_samples import write_cached_samples
from src.evaluation.image.comparison import (
    COMPARISON_LABELS,
    comparison_rows,
    comparison_sublabels,
    load_cache,
    select_random_generated_indices,
    select_top_oob_indices,
    validate_comparison_caches,
)
from src.rsbm import create_sde_and_bridge

DIRECTIONS = ("0_to_1", "1_to_0")
# Printed width of one figure: two fit side by side, one per direction.
WIDTH = TEXT_WIDTH / 2
LABEL_SIZE = 8  # pt
LABEL_BAND = 0.17  # inches above the images
LABEL_PAD = 0.02  # inches above the labels, so that LaTeX's tall glyphs aren't cut
SUBLABEL_BAND = 0.12
# These figures are placed by hand. Matplotlib restores this rc setting on figures it
# has saved as PDF, which would then lay them out again.
plt.rcParams["figure.constrained_layout.use"] = False


def fill_cache(run, weights):
    """Write the run's caches as its final evaluation does, from its final model file."""
    path = run / "checkpoints" / models.final_model_file_name(weights)
    metadata = models.read_final_model_metadata(str(path))
    # The file leaves out output locations; the caches don't use this one.
    config = OmegaConf.merge({"output_root": str(run)}, metadata["config"])
    device = posthoc.default_device(config)
    model, _ = models.load_final_model(str(path), device)
    sde, _ = create_sde_and_bridge(config.sde, data_utils.get_data_normalization(config.data))
    out_dir = run / "eval" / f"{metadata['step']:06d}" / weights
    context = EvaluationContext(**run_evaluator_arguments(config, sde),
                                out_dir=str(out_dir.parent.parent), device=device)
    with th.inference_mode():
        write_cached_samples(context, posthoc.evaluated_network(model, config, weights),
                             out_dir, weights)
    return out_dir / "cache"


def run_cache(run, weights, split, direction):
    """The run's cached samples of its final evaluation, filled first if missing."""
    run = Path(run)
    filename = f"{split}_{direction}.pt"
    caches = sorted(run.glob(f"eval/[0-9]*/{weights}/cache"))
    if caches and (caches[-1] / filename).is_file():
        return load_cache(caches[-1] / filename), caches[-1]
    cache = fill_cache(run, weights)
    return load_cache(cache / filename), cache


def grid_figure(rows, labels, sublabels, images_per_row):
    """The rows as one image, in cells of images_per_row x images_per_row samples,
    labels above the columns, drawn at WIDTH with no margin around it."""
    per_cell = images_per_row**2
    if len(rows) % per_cell:
        raise ValueError(f"{len(rows)} samples don't fill cells of {per_cell}.")
    image_h, image_w = rows[0][0].shape[:2]
    gap = max(1, round(min(image_h, image_w) * 0.04))
    column_gap = max(gap + 1, round(min(image_h, image_w) * 0.10))
    cell_h = images_per_row * image_h + (images_per_row - 1) * gap
    cell_w = images_per_row * image_w + (images_per_row - 1) * gap
    n_cells = len(rows) // per_cell
    n_columns = len(rows[0])
    canvas_h = n_cells * cell_h + (n_cells - 1) * gap
    canvas_w = n_columns * cell_w + (n_columns - 1) * column_gap
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.float32)
    for i, row in enumerate(rows):
        cell, place = divmod(i, per_cell)
        subrow, subcol = divmod(place, images_per_row)
        y = cell * (cell_h + gap) + subrow * (image_h + gap)
        for j, image in enumerate(row):
            x = j * (cell_w + column_gap) + subcol * (image_w + gap)
            canvas[y:y + image_h, x:x + image_w] = image

    band = LABEL_PAD + LABEL_BAND + (SUBLABEL_BAND if sublabels else 0)
    images_height = WIDTH * canvas_h / canvas_w
    height = images_height + band
    fig = plt.figure(figsize=(WIDTH, height))
    ax = fig.add_axes([0, 0, 1, images_height / height])
    # "none" embeds the canvas in the PDF at its own resolution, unresampled.
    ax.imshow(canvas, vmin=0, vmax=1, interpolation="none")
    ax.set_axis_off()
    for j, label in enumerate(labels):
        x = (j * (cell_w + column_gap) + cell_w / 2) / canvas_w
        fig.text(x, 1 - LABEL_PAD / height, label, ha="center", va="top", fontsize=LABEL_SIZE)
        if sublabels and sublabels[j]:
            fig.text(x, 1 - (LABEL_PAD + LABEL_BAND) / height, sublabels[j], ha="center", va="top",
                     fontsize=LABEL_SIZE * 0.8)
    return fig


def compare(nonref_run, reflected_run, out_dir=None, *, random=(16,), top_k=4,
            images_per_row=1, split="test", weights="ema", prefix="", format="pdf"):
    """Draw each direction's figures into out_dir as <prefix><split>_<direction>_<kind>;
    return out_dir."""
    for direction in DIRECTIONS:
        nonref, _ = run_cache(nonref_run, weights, split, direction)
        reflected, reflected_cache = run_cache(reflected_run, weights, split, direction)
        validate_comparison_caches(nonref, reflected)
        meta = nonref["metadata"]
        generated = nonref["generated_pixels"]
        selections = {
            f"random{count}": select_random_generated_indices(
                generated.shape[0], count, meta["seed"], meta["split"], meta["direction"])
            for count in random
        }
        selections[f"top{top_k}_oob"] = select_top_oob_indices(generated, top_k)
        sublabels = comparison_sublabels(meta, reflected["metadata"])
        for kind, indices in selections.items():
            rows = comparison_rows(nonref_cache=nonref, reflected_cache=reflected,
                                   generated_indices=indices)
            fig = grid_figure(rows, COMPARISON_LABELS, sublabels, images_per_row)
            out_dir = Path(out_dir or reflected_cache.parent / "comparison")
            save(fig, out_dir / f"{prefix}{split}_{direction}_{kind}", format)
    return out_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("nonref_run", type=Path)
    parser.add_argument("reflected_run", type=Path)
    parser.add_argument("--random", type=int, nargs="+", default=[16],
                        help="Random sample counts, one figure each (default: 16)")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--images-per-row", type=int, default=1,
                        help="Samples per side of a cell (default: 1)")
    parser.add_argument("--split", default="test", choices=("test", "train"))
    parser.add_argument("--weights", default="ema")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--format", choices=("pdf", "png"), default="pdf",
                        help="File format of the figures (default: pdf)")
    args = parser.parse_args()
    out_dir = compare(args.nonref_run, args.reflected_run, args.out_dir, random=args.random,
                      top_k=args.top_k, images_per_row=args.images_per_row,
                      split=args.split, weights=args.weights, format=args.format)
    print(f"Wrote the figures to {out_dir}")


if __name__ == "__main__":
    main()
