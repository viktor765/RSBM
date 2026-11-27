"""The paper's figures and tables for the two-dimensional experiments.

python -m scripts.paper_artifacts_toy2d [FIGURE ...] [--runs-dir DIR] [--out-dir OUT] [--format {pdf,png}]

Draws the named figures and tables, by default all of them, from the paper runs
(see REPRODUCE.md) under DIR, by default outputs/. They go to OUT, by default
outputs/paper/toy2d/: each figure as PDF, or PNG with --format png, and each table as LaTeX, in
paper_style.py's style. Table captions are placeholders; the paper has the real ones.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
import numpy as np
from omegaconf import OmegaConf

from scripts.paper_style import TEXT_WIDTH, save
from src.evaluation.curve import load_curve
from src.evaluation.toy2d.utils import exact_sample_w2_percentile, image_density_w2_percentile

OUTPUT = Path("outputs/paper/toy2d")
# Set from --runs-dir.
RUNS_DIR = Path("outputs")

# Directories of seed*/ runs under RUNS_DIR, as REPRODUCE.md's commands name them.
RUNS = {
    "checkerboard_rsbm": "uniform-to-checkerboard/alpha_imf/refl/sigma0.5",
    "checkerboard_rsb_fbsde": "uniform-to-checkerboard/rsb_fbsde/refl/sigma0.5",
    "checkerboard_rsb_fbsde_long": "uniform-to-checkerboard/rsb_fbsde-long/refl/sigma0.5",
    "checkerboard_rsb_fbsde_exact_divergence": "uniform-to-checkerboard/rsb_fbsde-exact-divergence/refl/sigma0.5",
    "checkerboard_rsb_fbsde_boundary_crossing": "uniform-to-checkerboard/rsb_fbsde-boundary-crossing/refl/sigma0.5",
    "checkerboard_rsb_fbsde_left_quadrature": "uniform-to-checkerboard/rsb_fbsde-left-quadrature/refl/sigma0.5",
    "checkerboard_rsb_fbsde_right_quadrature": "uniform-to-checkerboard/rsb_fbsde-right-quadrature/refl/sigma0.5",
    "oresund_rsbm": "schrodinger-to-oresund/alpha_imf/refl/sigma0.5",
    "oresund_rsb_fbsde": "schrodinger-to-oresund/rsb_fbsde/refl/sigma0.5",
    "oresund_rsb_fbsde_long": "schrodinger-to-oresund/rsb_fbsde-long/refl/sigma0.5",
}

# Line style as well as colour, so that methods stay apart in black and white.
METHODS = {
    "rsbm": {"label": r"$\alpha$-RSBM", "color": "#1c5fb4", "ls": "-"},
    "rsb_fbsde": {"label": "RSB-FBSDE", "color": "#d9531e", "ls": "--"},
}
# The direction whose target is data1.
UNIFORM_TO_CHECKERBOARD = "0_to_1"
# Each direction's target, for exact_sample_w2_percentile. Its defaults are the selection
# sweep's ceilings, computed with the paper runs' evaluation settings.
TARGETS = {"0_to_1": "checkerboard", "1_to_0": "uniform_box"}

FIGURES = {}


def figure(function):
    """Register a function returning a figure or a LaTeX table."""
    FIGURES[function.__name__] = function
    return function


def run_dir(name):
    return RUNS_DIR / RUNS[name]


def load_runs(name):
    """Each seed's EMA evaluation rows, by seed directory name."""
    runs = sorted(run_dir(name).glob("seed*"), key=lambda run: int(run.name.removeprefix("seed")))
    if not runs:
        raise ValueError(f"No seed*/ runs in {run_dir(name)}.")
    return {run.name: load_curve(run / "eval", "ema") for run in runs}


def over_seeds(runs, metric, direction):
    """Hours, and each seed's last evaluation at or before those times (seeds, times).

    The times are all seeds' evaluation times, within the range every seed covers.
    """
    curves = [(np.array([row["training_seconds"] for row in rows]),
               np.array([row[metric][direction] for row in rows])) for rows in runs.values()]
    start, stop = max(x[0] for x, _ in curves), min(x[-1] for x, _ in curves)
    grid = np.unique(np.concatenate([x for x, _ in curves]))
    grid = grid[(grid >= start) & (grid <= stop)]
    values = np.stack([y[np.searchsorted(x, grid, side="right") - 1] for x, y in curves])
    return grid / 3600, values


def draw_method(ax, method, metric, direction, runs):
    """Draw the curve of runs, an entry in RUNS, in the style of method, a key of METHODS."""
    hours, values = over_seeds(load_runs(runs), metric, direction)
    style = METHODS[method]
    ax.fill_between(hours, values.min(0), values.max(0), step="post", color=style["color"],
                    alpha=0.2, lw=0)
    ax.plot(hours, values.mean(0), drawstyle="steps-post", **style)


@figure
def checkerboard_rsbm_vs_rsb_fbsde():
    fig, (w2, cost) = plt.subplots(2, 1, sharex=True, figsize=(TEXT_WIDTH / 2, 2.55))
    # As little white margin as possible; the hair of padding keeps edges from clipping.
    fig.get_layout_engine().set(w_pad=0.005, h_pad=0.005, hspace=0.04)
    for method in METHODS:
        draw_method(w2, method, "known_target_grid_w2", UNIFORM_TO_CHECKERBOARD, f"checkerboard_{method}")
        draw_method(cost, method, "control_cost", UNIFORM_TO_CHECKERBOARD, f"checkerboard_{method}")
    w2.axhline(exact_sample_w2_percentile(TARGETS[UNIFORM_TO_CHECKERBOARD]),
               color="black", ls=":", lw=1)
    w2.set_ylabel("2-Wasserstein")
    w2.set_ylim(0, 0.08)
    # Plain text ticks, so that LaTeX also sets the digits in the text font.
    w2.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    w2.legend(loc="upper right")
    cost.set_ylabel("Control cost")
    cost.set_ylim(0, 2.5)
    cost.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    cost.set_xlabel("Training time")
    cost.set_xlim(0, 1)
    cost.set_xticks([0, 1], ["0", "1h"])
    cost.xaxis.set_minor_locator(MultipleLocator(1 / 6))
    fig.align_ylabels()
    return fig


def both_directions(runs, w2_metric, w2_ylim, cost_ylim, cost_format, reference, xlim):
    """W2 above control cost, one column per direction. runs maps methods to RUNS entries;
    reference maps directions to a dotted W2 level; xlim is in hours."""
    fig, axes = plt.subplots(2, 2, sharex=True, sharey="row", figsize=(TEXT_WIDTH, 2.55))
    fig.get_layout_engine().set(w_pad=0.005, h_pad=0.005, hspace=0.04, wspace=0.02)
    # \leftarrow rather than its LaTeX alias \gets, which Matplotlib's own math lacks.
    columns = {"0_to_1": r"$\Pi_0 \to \Pi_1$", "1_to_0": r"$\Pi_0 \leftarrow \Pi_1$"}
    for (w2, cost), (direction, title) in zip(axes.T, columns.items()):
        for method, entry in runs.items():
            draw_method(w2, method, w2_metric, direction, entry)
            draw_method(cost, method, "control_cost", direction, entry)
        w2.axhline(reference[direction], color="black", ls=":", lw=1)
        w2.set_title(title)
        cost.set_xlabel("Training time")
        cost.set_xlim(0, xlim)
        hours = range(1, int(xlim) + 1)
        cost.set_xticks([0, *hours], ["0", *(f"{hour}h" for hour in hours)])
        cost.xaxis.set_minor_locator(MultipleLocator(1 / 6))
    axes[0, 0].set_ylabel("2-Wasserstein")
    axes[0, 0].set_ylim(*w2_ylim)
    axes[0, 0].yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    axes[0, 0].legend(loc="upper right")
    axes[1, 0].set_ylabel("Control cost")
    axes[1, 0].set_ylim(*cost_ylim)
    axes[1, 0].yaxis.set_major_formatter(FormatStrFormatter(cost_format))
    fig.align_ylabels()
    return fig


@figure
def checkerboard_rsbm_vs_rsb_fbsde_long():
    runs = load_runs("checkerboard_rsb_fbsde_long")
    # All three seeds' curves stop where the first seed stopped.
    end = min(rows[-1]["training_seconds"] for rows in runs.values()) / 3600
    return both_directions(
        {"rsbm": "checkerboard_rsbm", "rsb_fbsde": "checkerboard_rsb_fbsde_long"},
        "known_target_grid_w2", (0, 0.08), (0, 20), "%d",
        reference={direction: exact_sample_w2_percentile(dataset)
                   for direction, dataset in TARGETS.items()},
        xlim=end)


@figure
def oresund_rsbm_vs_rsb_fbsde_long():
    runs = load_runs("oresund_rsb_fbsde_long")
    end = min(rows[-1]["training_seconds"] for rows in runs.values()) / 3600
    data = OmegaConf.load(run_dir("oresund_rsbm") / "seed0" / ".hydra" / "config.yaml").data
    endpoints = [data.data0, data.data1]
    return both_directions(
        {"rsbm": "oresund_rsbm", "rsb_fbsde": "oresund_rsb_fbsde_long"}, "w2", (0, 0.25), (0, 120), "%d",
        reference={"0_to_1": image_density_w2_percentile(endpoints, 1),
                   "1_to_0": image_density_w2_percentile(endpoints, 0)},
        xlim=end)


def mean_sd(entry, metric, direction, decimals, hours=None, mean_digits=1):
    """Mean and sample SD over seeds of each seed's final evaluation, or of its evaluation
    nearest to hours. Means are padded to mean_digits integer digits, so that the plus-minus
    signs of a centred column line up."""
    values = []
    for rows in load_runs(entry).values():
        row = rows[-1] if hours is None else min(
            rows, key=lambda row: abs(row["training_seconds"] - hours * 3600))
        values.append(row[metric][direction])
    mean = f"{np.mean(values):.{decimals}f}"
    padding = r"\phantom{0}" * (mean_digits - len(mean.split(".")[0]))
    return rf"{padding}{mean} $\pm$ {np.std(values, ddof=1):.{decimals}f}"


# The rows of rsbm_vs_rsb_fbsde_table: label, checkerboard run, Oresund run, hours
# (None for the final evaluation).
COMPARISON_ROWS = [
    (r"\(\alpha\)-RSBM (1h)", "checkerboard_rsbm", "oresund_rsbm", None),
    (r"RSB-FBSDE (2h)", "checkerboard_rsb_fbsde", "oresund_rsb_fbsde", None),
    (r"RSB-FBSDE (6h)", "checkerboard_rsb_fbsde_long", "oresund_rsb_fbsde_long", 6),
]


@figure
def rsbm_vs_rsb_fbsde_table():
    return comparison_table(COMPARISON_ROWS)


@figure
def rsbm_vs_rsb_fbsde_final_table():
    """rsbm_vs_rsb_fbsde_table's rows of final evaluations, which the final weights reproduce."""
    return comparison_table([row for row in COMPARISON_ROWS if row[3] is None])


def comparison_table(methods):
    w2_decimals = 3
    body = []
    for direction, title in {"0_to_1": r"\(\Pi_0 \to \Pi_1\)", "1_to_0": r"\(\Pi_0 \gets \Pi_1\)"}.items():
        for i, (label, checkerboard, oresund, hours) in enumerate(methods):
            cells = [mean_sd(checkerboard, "known_target_grid_w2", direction, w2_decimals, hours),
                     mean_sd(checkerboard, "control_cost", direction, 2, hours, mean_digits=2),
                     mean_sd(oresund, "w2", direction, w2_decimals, hours),
                     mean_sd(oresund, "control_cost", direction, 2, hours, mean_digits=2)]
            first = rf"\multirow{{{len(methods)}}}{{*}}{{{title}}}" if i == 0 else ""
            body.append(f"        {first} & {label} & " + " & ".join(cells) + r" \\")
        body.append(r"        \cmidrule(l){1-2}" if direction == "0_to_1" else r"        \bottomrule")
    return "\n".join([
        r"\begin{table}[t]",
        r"    \caption{Caption}",
        r"    \centering",
        r"    \small",
        r"    \begin{tabular}{@{}llcccc@{}}",
        r"        \toprule",
        r"        & & \multicolumn{2}{c}{Uniform--Checkerboard} & \multicolumn{2}{c}{Schr\"odinger--\"Oresund} \\",
        r"        \cmidrule(lr){3-4} \cmidrule(lr){5-6}",
        r"        & & \(W_2\) \textdownarrow & Cost \textdownarrow & \(W_2\) \textdownarrow & Cost \textdownarrow \\",
        r"        \midrule",
        *body,
        r"    \end{tabular}",
        r"\end{table}",
        "",
    ])


@figure
def checkerboard_rsb_fbsde_ablation_table():
    w2_decimals = 3
    variants = {"checkerboard_rsb_fbsde": "Default",
                "checkerboard_rsb_fbsde_exact_divergence": "Exact divergence",
                "checkerboard_rsb_fbsde_boundary_crossing": "Boundary crossing",
                "checkerboard_rsb_fbsde_left_quadrature": "Left quadrature",
                "checkerboard_rsb_fbsde_right_quadrature": "Right quadrature"}
    body = []
    for entry, label in variants.items():
        cells = [mean_sd(entry, metric, direction, decimals)
                 for direction in ("0_to_1", "1_to_0")
                 for metric, decimals in (("known_target_grid_w2", w2_decimals), ("control_cost", 2))]
        hours = [rows[-1]["training_seconds"] / 3600 for rows in load_runs(entry).values()]
        cells.append(rf"{np.mean(hours):.2f} $\pm$ {np.std(hours, ddof=1):.2f}")
        body.append(f"        {label} & " + " & ".join(cells) + r" \\")
        if entry == "checkerboard_rsb_fbsde":
            body.append(r"        \midrule")
    return "\n".join([
        r"\begin{table}[t]",
        r"    \caption{Caption}",
        r"    \centering",
        r"    \small",
        r"    \begin{tabular}{@{}lccccc@{}}",
        r"        \toprule",
        r"        & \multicolumn{2}{c}{\(\Pi_0 \to \Pi_1\)} & \multicolumn{2}{c}{\(\Pi_0 \gets \Pi_1\)} & \\",
        r"        \cmidrule(lr){2-3} \cmidrule(lr){4-5}",
        r"        & \(W_2\) \textdownarrow & Cost \textdownarrow & \(W_2\) \textdownarrow & Cost \textdownarrow & Time (h) \textdownarrow \\",
        r"        \midrule",
        *body,
        r"        \bottomrule",
        r"    \end{tabular}",
        r"\end{table}",
        "",
    ])


def main():
    global RUNS_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("figures", nargs="*", metavar="FIGURE", help="Default: all")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR,
                        help="Directory holding the paper runs (default: outputs/)")
    parser.add_argument("--out-dir", type=Path, default=OUTPUT,
                        help=f"Directory the artifacts go to (default: {OUTPUT}/)")
    parser.add_argument("--format", choices=("pdf", "png"), default="pdf",
                        help="File format of the figures (default: pdf)")
    args = parser.parse_args()
    RUNS_DIR = args.runs_dir
    for name in args.figures or FIGURES:
        print(f"Wrote {save(FIGURES[name](), args.out_dir / name, args.format)}")


if __name__ == "__main__":
    main()
