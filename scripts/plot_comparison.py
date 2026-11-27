"""Compare experiments: W2 and control cost per direction against training time.

python -m scripts.plot_comparison EXPERIMENT [EXPERIMENT ...] --out-dir DIR

An experiment is a directory of seed*/ runs, such as
outputs/uniform-to-checkerboard/alpha_imf/refl/sigma0.5. Each evaluation pairs
the seeds' k-th evaluations of the EMA weights and is placed at the latest of
their times. Points are seed means with min-max whiskers, one colour per
experiment. Each W2 panel draws the exact-sample ceiling of its direction's
target. Writes comparison.pdf, comparison.png, comparison.csv and summary.txt,
and prints the summary.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from omegaconf import OmegaConf

from src.evaluation.common import training_summary_path
from src.evaluation.curve import load_curve
from src.evaluation.toy2d.utils import exact_sample_w2_percentile

DIRECTIONS = {"0_to_1": "0 → 1", "1_to_0": "1 → 0"}
W2 = "known_target_grid_w2"
COST = "control_cost"
# Categorical colours, in a fixed order.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK = "#0b0b0b"
EXACT_STYLE = {"color": INK, "ls": "--", "lw": 1}


def seed_number(run):
    return int(run.name.removeprefix("seed"))


def load_experiment(path):
    """An experiment's seeds, paired by their k-th evaluations up to the shortest seed.

    w2 and cost map each direction to a (seeds, evaluations) array.
    """
    path = Path(path)
    runs = sorted(path.glob("seed*"), key=seed_number)
    if not runs:
        raise ValueError(f"No seed*/ runs in {path}.")
    curves = [load_curve(run / "eval", "ema") for run in runs]
    rounds = min(map(len, curves))
    field = lambda name, direction: np.array(
        [[row[name][direction] for row in curve[:rounds]] for curve in curves])
    configs = [OmegaConf.load(run / ".hydra" / "config.yaml") for run in runs]
    return {
        "path": path,
        "configs": configs,
        "method": configs[0].method,
        "seeds": [run.name for run in runs],
        "evaluations": [len(curve) for curve in curves],
        "summaries": [json.loads(Path(training_summary_path(run / "eval")).read_text()) for run in runs],
        "seconds": np.array([[row["training_seconds"] for row in curve[:rounds]]
                             for curve in curves]).max(axis=0),
        "pretrain_seconds": max(curve[-1]["pretrain_seconds"] for curve in curves),
        "w2": {direction: field(W2, direction) for direction in DIRECTIONS},
        "cost": {direction: field(COST, direction) for direction in DIRECTIONS},
    }


def exact_ceiling(config, n_seeds):
    """Per direction, the 95th percentile of the mean W2 of n_seeds batches of exact
    samples of that direction's target, with the run's evaluation settings."""
    targets = {"0_to_1": config.data.data1, "1_to_0": config.data.data0}
    return {direction: exact_sample_w2_percentile(
                target.dataset, phase=target.get("phase", 1), n_samples=config.eval.n_samples,
                n_in_mean=n_seeds,
                subdivisions_per_checker_cell=config.eval.toy2d.known_target_grid_subdivisions)
            for direction, target in targets.items()}


def after_pretraining(experiment):
    """Evaluations after training started and, for a run that finetunes, after
    pretraining ended. Before then, cost and W2 are far above the rest and would
    stretch the axes."""
    seconds = experiment["seconds"]
    started = np.arange(len(seconds)) > 0
    if OmegaConf.select(experiment["configs"][0], "training.n_finetune_steps") == 0:
        return started
    return started & (seconds > experiment["pretrain_seconds"])


def draw_points(ax, seconds, values, color, **kwargs):
    """Seed means with min-max whiskers."""
    mean = values.mean(axis=0)
    ax.errorbar(seconds, mean, yerr=(mean - values.min(axis=0), values.max(axis=0) - mean),
                fmt="o", ms=3, color=color, elinewidth=0.8, capsize=1.5, capthick=0.8, **kwargs)


def limits(arrays, bottom_fraction=0.0):
    """Value range with a margin; bottom_fraction of the axis is left free below it."""
    low = min(a.min() for a in arrays if a.size)
    high = max(a.max() for a in arrays if a.size)
    margin = 0.05 * (high - low) or 1e-3
    low, high = low - margin, high + margin
    return low - bottom_fraction / (1 - bottom_fraction) * (high - low), high


def style_axes(ax):
    ax.grid(True, color="#e5e5e5", lw=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def labels(experiments):
    """Each experiment's path after the leading components they all share."""
    parts = [experiment["path"].parts for experiment in experiments]
    shared = 0
    while (len(experiments) > 1 and all(len(p) > shared + 1 for p in parts)
           and len({p[shared] for p in parts}) == 1):
        shared += 1
    return [str(Path(*p[shared:])) for p in parts], str(Path(*parts[0][:shared])) if shared else ""


def plot(experiments, ceilings, path):
    names, shared = labels(experiments)
    colors = PALETTE[:len(experiments)]
    fig, axes = plt.subplots(2, 2, sharex=True, sharey="row", figsize=(11, 7))
    for column, direction in enumerate(DIRECTIONS):
        w2_ax, cost_ax = axes[:, column]
        for experiment, color in zip(experiments, colors):
            draw_points(w2_ax, experiment["seconds"], experiment["w2"][direction], color)
            draw_points(cost_ax, experiment["seconds"], experiment["cost"][direction], color)
        for value in {ceiling[direction] for ceiling in ceilings}:
            w2_ax.axhline(value, **EXACT_STYLE)
        w2_ax.set_title(f"Direction {DIRECTIONS[direction]}", fontsize=10)
        cost_ax.set_xlabel("Training time (s)")
    for row, (key, label) in enumerate((("w2", "W2 to target (grid)"), ("cost", "Control cost"))):
        kept = [experiment[key][direction][:, after_pretraining(experiment)]
                for experiment in experiments for direction in DIRECTIONS]
        if key == "w2":
            kept += [np.array([ceiling[d] for ceiling in ceilings]) for d in DIRECTIONS]
        axes[row, 0].set_ylim(limits(kept))
        axes[row, 0].set_ylabel(label, fontsize=9)
    for ax in axes.flat:
        style_axes(ax)
    handles = [Line2D([], [], ls="", marker="o", ms=4, color=color, label=name)
               for name, color in zip(names, colors)]
    handles += [Line2D([], [], **EXACT_STYLE, label="Exact-sample ceiling (95th percentile)")]
    fig.legend(handles=handles, loc="upper center", ncol=min(len(handles), 3), frameon=False,
               fontsize=8, title=shared or None, title_fontsize=8)
    rows = -(-len(handles) // 3) + bool(shared)
    fig.text(0.5, 0.005, "Seed mean of each k-th evaluation, whiskers seed min and max, EMA weights. "
             "Axes leave out evaluations before training and before pretraining ends.",
             ha="center", va="bottom", fontsize=8, color="#444444")
    fig.tight_layout(rect=(0, 0.03, 1, 1 - 0.035 * rows))
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=200)
    plt.close(fig)


def evaluation_text(experiment, index):
    parts = [f"t={experiment['seconds'][index]:.0f} s"]
    for key, name, digits in (("w2", "W2", 5), ("cost", "cost", 3)):
        for direction in DIRECTIONS:
            values = experiment[key][direction][:, index]
            parts.append(f"{name} {direction} {values.mean():.{digits}f} "
                         f"(SD {values.std(ddof=1):.{digits}f})")
    return ", ".join(parts)


def summary_lines(experiment, label, ceiling):
    summaries = experiment["summaries"]
    lines = [f"{label}: {experiment['path']} ({experiment['method']})",
             f"  seeds {', '.join(experiment['seeds'])}",
             f"  completed {[summary['completed'] for summary in summaries]}",
             f"  evaluations {experiment['evaluations']}"]
    for key, name, text in (("completed_updates", "updates", "{:d}"),
                            ("training_seconds", "training seconds", "{:.0f}"),
                            ("evaluation_seconds", "evaluation seconds", "{:.0f}"),
                            ("peak_cuda_reserved_fraction", "peak reserved GPU fraction", "{:.2f}")):
        values = [summary.get(key) for summary in summaries]
        lines.append(f"  {name} " + ", ".join("n/a" if v is None else text.format(v) for v in values))
    lines.append(f"  final: {evaluation_text(experiment, len(experiment['seconds']) - 1)}")
    met = np.logical_and.reduce([experiment["w2"][d].mean(axis=0) <= ceiling[d] for d in DIRECTIONS])
    lines.append(f"  exact-sample ceilings {ceiling['0_to_1']:.5f}, {ceiling['1_to_0']:.5f}: "
                 f"both met at {met.sum()} of {len(met)} evaluations")
    settled = met & after_pretraining(experiment)
    if settled.any():
        indices = np.flatnonzero(settled)
        cost = sum(experiment["cost"][d] for d in DIRECTIONS).mean(axis=0)
        lines.append(f"    first after pretraining: {evaluation_text(experiment, indices[0])}")
        lines.append(f"    lowest mean cost: {evaluation_text(experiment, indices[np.argmin(cost[indices])])}")
    return lines


def write_csv(experiments, names, path):
    with open(path, "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["experiment", "evaluation", "seconds",
                         *(f"{stat}_{key}_{d}" for key in ("w2", "cost") for d in DIRECTIONS
                           for stat in ("mean", "min", "max"))])
        for experiment, name in zip(experiments, names):
            for index, seconds in enumerate(experiment["seconds"]):
                values = [experiment[key][d][:, index] for key in ("w2", "cost") for d in DIRECTIONS]
                writer.writerow([name, index, seconds, *(stat for v in values
                                                         for stat in (v.mean(), v.min(), v.max()))])


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiments", nargs="+", help="Directories of seed*/ runs")
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    if len(args.experiments) > len(PALETTE):
        parser.error(f"At most {len(PALETTE)} experiments, one per colour.")
    experiments = [load_experiment(path) for path in args.experiments]
    ceilings = [exact_ceiling(e["configs"][0], len(e["seeds"])) for e in experiments]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot(experiments, ceilings, args.out_dir / "comparison.pdf")
    names, _ = labels(experiments)
    write_csv(experiments, names, args.out_dir / "comparison.csv")
    summary = "\n".join(line for experiment, name, ceiling in zip(experiments, names, ceilings)
                        for line in summary_lines(experiment, name, ceiling))
    (args.out_dir / "summary.txt").write_text(summary + "\n")
    print(summary)


if __name__ == "__main__":
    main()
