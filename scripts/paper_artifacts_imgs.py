"""The paper's figures and tables comparing reflected and non-reflected image runs.

python -m scripts.paper_artifacts_imgs [--runs-dir DIR] [--out-dir OUT] [--format {pdf,png}]

Writes the paper's comparison figures from the final EMA test caches for seed 0
and both tables from the final EMA evaluations over all seeds. Reads the runs
from DIR, by default outputs/, and writes the artifacts to OUT, by default
outputs/paper/images/.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.compare_imgs import compare
from scripts.paper_style import save
from src.evaluation.curve import load_curve

OUTPUT = Path("outputs/paper/images")
COMPARISON_SEED = "seed0"
# Each dataset's directory of seed*/ paper runs, as REPRODUCE.md's commands name it,
# with refl or non-refl filled in, and compare's settings for it.
DATASETS = {
    "afhq64": ("afhq64/alpha_imf/{sde}/sigma0.75",
               {"random": (2, 8), "top_k": 4, "images_per_row": 1}),
    "mnist_to_emnist": ("mnist-to-emnist/alpha_imf/{sde}/sigma1.0",
                        {"random": (8, 32), "top_k": 16, "images_per_row": 2}),
}
METHODS = {"non-refl": r"\(\alpha\)-DSBM", "refl": r"\(\alpha\)-RSBM"}
DIRECTIONS = {"0_to_1": r"\(\Pi_0 \to \Pi_1\)", "1_to_0": r"\(\Pi_0 \gets \Pi_1\)"}
# Each dataset's columns: metric, decimals, integer digits the mean is padded to.
COLUMNS = {
    "mnist_to_emnist": [("msd", 3, 1), ("fid", 2, 2)],
    "afhq64": [("msd", 3, 1), ("lpips", 3, 1), ("fid", 1, 2)],
}
NUMBERS = {2: "two", 3: "three", 4: "four", 5: "five"}
# Evaluation records are keyed by target dataset.
OOB_TARGETS = {
    "mnist_to_emnist": {"ds0": "MNIST", "ds1": "EMNIST"},
    "afhq64": {"ds0": r"AFHQ \texttt{cat}", "ds1": r"AFHQ \texttt{wild}"},
}


def image_runs(runs_dir, dataset, sde):
    directory = runs_dir / DATASETS[dataset][0].format(sde=sde)
    runs = sorted(directory.glob("seed*"), key=lambda run: int(run.name.removeprefix("seed")))
    if not runs:
        raise ValueError(f"No seed*/ runs in {directory}.")
    return runs


def final_rows(runs_dir, dataset, sde):
    """Each seed's final EMA evaluation."""
    return [load_curve(run / "eval", "ema")[-1] for run in image_runs(runs_dir, dataset, sde)]


def mean_sd(values, decimals, digits):
    """Mean and sample SD, the mean padded to digits integer digits so that the
    plus-minus signs of a centred column line up."""
    mean = f"{np.mean(values):.{decimals}f}"
    padding = r"\phantom{0}" * (digits - len(mean.split(".")[0]))
    return rf"\({padding}{mean} \pm {np.std(values, ddof=1):.{decimals}f}\)"


def image_metrics_table(runs_dir):
    rows = {(dataset, sde): final_rows(runs_dir, dataset, sde)
            for dataset in COLUMNS for sde in METHODS}
    seeds = {len(seed_rows) for seed_rows in rows.values()}
    if len(seeds) != 1:
        raise ValueError(f"The image runs have different numbers of seeds: {sorted(seeds)}.")
    body = []
    for direction, title in DIRECTIONS.items():
        for i, (sde, label) in enumerate(METHODS.items()):
            cells = [mean_sd([row[metric][direction] for row in rows[dataset, sde]], *format)
                     for dataset, columns in COLUMNS.items()
                     for metric, *format in columns]
            first = rf"\multirow{{2}}{{*}}{{{title}}}" if i == 0 else ""
            body.append(f"        {first} & {label} & " + " & ".join(cells) + r" \\")
        body.append(r"        \cmidrule(l){1-2}" if direction == "0_to_1" else r"        \bottomrule")
    return "\n".join([
        r"\begin{table}[t]",
        r"    \caption{Evaluation metrics for the image-to-image syntheses. For \(\alpha\)-DSBM,"
        r" the MSD is computed for the unclipped samples, whereas both FID and LPIPS require"
        r" valid pixel values so the clipped samples are used. The columns show mean \(\pm\)"
        rf" one standard deviation computed over {NUMBERS[seeds.pop()]} seeds.}}",
        r"    \label{tab:image-eval-scores}",
        r"    \centering",
        # Fits the text width with every cell a mean and SD.
        r"    \small",
        r"    \setlength{\tabcolsep}{4pt}",
        r"    \begin{tabular}{llccccc}",
        r"        \toprule",
        r"        & & \multicolumn{2}{c}{MNIST \(\leftrightarrow\) EMNIST} & \multicolumn{3}{c}{AFHQ (\(64 \times 64\))} \\",
        r"        \cmidrule(lr){3-4} \cmidrule(lr){5-7}",
        r"        & & MSD \textdownarrow & FID \textdownarrow & MSD \textdownarrow & LPIPS \textdownarrow & FID \textdownarrow \\",
        r"        \midrule",
        *body,
        r"    \end{tabular}",
        r"\end{table}",
        "",
    ])


def image_oob_table(runs_dir):
    """Pool channel-level statistics from the unreflected final EMA test evaluations."""
    body = []
    for dataset, targets in OOB_TARGETS.items():
        evaluations = []
        for run in image_runs(runs_dir, dataset, "non-refl"):
            eval_dir = run / "eval"
            step = load_curve(eval_dir, "ema")[-1]["step"]
            path = eval_dir / f"{step:06d}" / "ema" / "eval_metrics.json"
            evaluations.append(json.loads(path.read_text(encoding="utf-8")))
        for key, target in targets.items():
            records = [evaluation[key]["oob"] for evaluation in evaluations]
            n_images = sum(record["n_images"] for record in records)
            n_channels = sum(record["total_pixel_values"] for record in records)
            n_oob = sum(record["total_oob_pixels"] for record in records)
            # The saved image fraction comes from an integer count within each seed.
            n_affected = sum(round(record["n_images"] * record["fraction_images_with_any_oob"])
                             for record in records)
            total_overshoot = sum(record["total_oob_pixels"] * record["oob_magnitude_mean"]
                                  for record in records if record["total_oob_pixels"])
            mean = f"{total_overshoot / n_oob:.5f}" if n_oob else "--"
            body.append(f"        {target} & {n_images:,} & {100 * n_oob / n_channels:.2f} & "
                        f"{100 * n_affected / n_images:.2f} & {mean}" + r" \\")
    return "\n".join([
        r"\begin{table}[t]",
        r"    \caption{Out-of-bounds (OOB) statistics for \(\alpha\)-DSBM."
        r" For each sample in the test datasets, we generate one sample from the"
        r" opposite distribution. This is repeated for three seeds, retraining"
        r" from scratch for each. For AFHQ, a channel value is an R, G or B value"
        r" for a pixel; for (E)MNIST it equates to pixel value. Mean overshoot is the"
        r" average distance to \([0, 1]\) among OOB channel values.}",
        r"    \label{tab:image-oob}",
        r"    \centering",
        r"    \small",
        r"    \setlength{\tabcolsep}{4pt}",
        r"    \begin{tabular}{@{}lcccc@{}}",
        r"        \toprule",
        r"        Target & Generated images & OOB channel values (\%) & Images with OOB (\%) & Mean overshoot \\",
        r"        \midrule",
        *body,
        r"        \bottomrule",
        r"    \end{tabular}",
        r"\end{table}",
        "",
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=Path("outputs"),
                        help="Directory holding the paper runs (default: outputs/)")
    parser.add_argument("--out-dir", type=Path, default=OUTPUT,
                        help=f"Directory the artifacts go to (default: {OUTPUT}/)")
    parser.add_argument("--format", choices=("pdf", "png"), default="pdf",
                        help="File format of the figures (default: pdf)")
    args = parser.parse_args()
    for dataset, (run, settings) in DATASETS.items():
        compare(args.runs_dir / run.format(sde="non-refl") / COMPARISON_SEED,
                args.runs_dir / run.format(sde="refl") / COMPARISON_SEED,
                args.out_dir, prefix=f"compare_{dataset}_", format=args.format, **settings)
    print(f"Wrote the figures to {args.out_dir}/compare_*.{args.format}")
    for table in (image_metrics_table, image_oob_table):
        print(f"Wrote {save(table(args.runs_dir), args.out_dir / table.__name__)}")


if __name__ == "__main__":
    main()
