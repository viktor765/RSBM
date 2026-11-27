"""Sweep plots: one figure per method, W2 per direction and control cost.

python -m scripts.plot_sweep PATH [PATH ...] --out-dir DIR [--ceiling {exact,best}] [--overlay PARAM]

A PATH is an experiment, a directory of seed*/ runs, or a directory searched for
them, such as outputs/sweep/uniform-to-checkerboard. Each experiment is one
recipe. The swept hyperparameters are the settings of the runs' resolved
.hydra/config.yaml that differ between a method's recipes; seed, run_name and
out_dir are ignored.

Rows: W2 0 → 1, W2 1 → 0 and control cost averaged over directions within each
seed. Points are seed means of the seeds' k-th evaluations of the EMA weights,
placed at the latest of their times, with min-max whiskers. The swept
hyperparameter with the fewest values is overlaid in each panel, one colour per
value (--overlay picks another); the other hyperparameters' combinations are the
columns. A single swept hyperparameter is the columns.

Every W2 panel draws the exact-sample ceiling. --ceiling best adds the method's
best observed ceiling: per direction, the lowest seed-mean W2 over all recipes and
evaluations plus 2 standard errors (2 / sqrt(seeds) sample SDs) at that point.
Strips under each panel mark the evaluations whose seed-mean W2 is at or below
the chosen ceiling; in the cost row, where both directions are. Writes
sweep-<method>.pdf, .png and .txt, and prints the text.
"""

import argparse
from itertools import product
from pathlib import Path
from textwrap import wrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from omegaconf import OmegaConf

from scripts.plot_comparison import (
    DIRECTIONS,
    EXACT_STYLE,
    INK,
    PALETTE,
    after_pretraining,
    draw_points,
    exact_ceiling,
    limits,
    load_experiment,
    style_axes,
)

METHOD_NAMES = {"alpha_imf": "RSBM", "rsb_fbsde": "RSB-FBSDE"}
RUN_SPECIFIC = {"seed", "run_name", "out_dir"}
BEST_STYLE = {"color": INK, "ls": ":", "lw": 1.2}
LANE = 0.03  # Strip height per overlaid value, as a fraction of the axes.


def find_experiments(paths):
    found = []
    for path in map(Path, paths):
        configs = list(path.glob("seed*/.hydra/config.yaml"))
        if not configs:
            configs = path.rglob("seed*/.hydra/config.yaml")
        found += [config.parents[2] for config in configs]
    if not found:
        raise ValueError(f"No seed*/ runs under {', '.join(map(str, paths))}.")
    return sorted(set(found))


def flatten(node, prefix=""):
    """{dotted key: value}, with lists as tuples."""
    if isinstance(node, dict):
        return {key: value for name, child in node.items()
                for key, value in flatten(child, f"{prefix}{name}.").items()}
    return {prefix[:-1]: tuple(node) if isinstance(node, list) else node}


def recipe_settings(experiment):
    """The settings an experiment's seeds share; run-specific keys are left out."""
    settings = [{key: value for key, value in flatten(OmegaConf.to_container(config, resolve=True)).items()
                 if key not in RUN_SPECIFIC} for config in experiment["configs"]]
    differing = sorted({key for s in settings for key in s if any(t.get(key) != s.get(key) for t in settings)})
    if differing:
        raise ValueError(f"{experiment['path']}: seeds differ in {', '.join(differing)}.")
    return settings[0]


def order(values):
    return sorted(values, key=lambda value: (value is None, value))


def swept(recipes):
    """{key: its values in order} for the settings that differ between recipes."""
    keys = sorted({key for recipe in recipes for key in recipe["settings"]})
    values = {key: {recipe["settings"].get(key) for recipe in recipes} for key in keys}
    return {key: order(found) for key, found in values.items() if len(found) > 1}


def leaf(key):
    return key.rsplit(".", 1)[-1]


def choose_overlay(hyperparameters, requested=None):
    """The hyperparameter to overlay: the one with the fewest values, None if only one is swept."""
    if requested is not None:
        matches = [key for key in hyperparameters if requested in (key, leaf(key))]
        if len(matches) != 1:
            raise ValueError(f"--overlay {requested} names none or several of: "
                             f"{', '.join(hyperparameters)}.")
        return matches[0]
    if len(hyperparameters) < 2:
        return None
    return min(hyperparameters, key=lambda key: (len(hyperparameters[key]), key))


def value_text(key, value):
    return f"{leaf(key)} {value:g}" if isinstance(value, float) else f"{leaf(key)} {value}"


def mean_cost(recipe):
    """Control cost averaged over directions within each seed: (seeds, evaluations)."""
    return sum(recipe["cost"][d] for d in DIRECTIONS) / len(DIRECTIONS)


def best_ceiling(recipes):
    """Per direction: the lowest seed-mean W2 over all recipes and evaluations, plus 2
    standard errors at that point, and where it was found."""
    ceilings = {}
    for direction in DIRECTIONS:
        recipe, index = min(((recipe, index) for recipe in recipes
                             for index in range(len(recipe["seconds"]))),
                            key=lambda pair: pair[0]["w2"][direction][:, pair[1]].mean())
        values = recipe["w2"][direction][:, index]
        ceilings[direction] = {
            "value": values.mean() + 2 * values.std(ddof=1) / np.sqrt(len(values)),
            "mean": values.mean(), "recipe": recipe["label"], "seconds": recipe["seconds"][index]}
    return ceilings


def within(recipe, ceiling):
    """Per row, whether each evaluation's seed-mean W2 is at or below the ceiling;
    the cost row needs both directions."""
    met = {d: recipe["w2"][d].mean(axis=0) <= ceiling[d] for d in DIRECTIONS}
    return [*met.values(), np.logical_and.reduce(list(met.values()))]


def strip_segments(seconds, met, spacing):
    """(start, width) of a short segment centred under each evaluation that meets the condition."""
    return [(x - spacing / 4, spacing / 2) for x in seconds[met]]


def layout(recipes, hyperparameters, overlay):
    """Columns (tuples of the other hyperparameters' values) and overlaid values."""
    others = [key for key in hyperparameters if key != overlay]
    combinations = [values for values in product(*(hyperparameters[key] for key in others))
                    if any(all(r["settings"].get(k) == v for k, v in zip(others, values))
                           for r in recipes)]
    overlaid = hyperparameters[overlay] if overlay else [None]
    return others, combinations, overlaid


def plot(method, recipes, hyperparameters, overlay, exact, best, path):
    others, columns, overlaid = layout(recipes, hyperparameters, overlay)
    active = {d: best[d]["value"] for d in DIRECTIONS} if best else exact
    colors = dict(zip(overlaid, PALETTE))
    spacing = np.median(np.concatenate([np.diff(r["seconds"]) for r in recipes]))
    shift = 0.2 * spacing
    offsets = {value: (i - (len(overlaid) - 1) / 2) * shift for i, value in enumerate(overlaid)}
    strip_bottom = LANE * len(overlaid) + 0.01

    fig, axes = plt.subplots(3, len(columns), sharex=True, sharey="row", squeeze=False,
                             figsize=(max(2.7 * len(columns), 9), 8.2))
    for column, values in enumerate(columns):
        for recipe in recipes:
            if any(recipe["settings"].get(k) != v for k, v in zip(others, values)):
                continue
            value = recipe["settings"].get(overlay) if overlay else None
            seconds = recipe["seconds"] + offsets[value]
            rows = [recipe["w2"]["0_to_1"], recipe["w2"]["1_to_0"], mean_cost(recipe)]
            lane = overlaid.index(value)
            for ax, data, met in zip(axes[:, column], rows, within(recipe, active)):
                draw_points(ax, seconds, data, colors[value])
                ax.broken_barh(strip_segments(seconds, met, spacing),
                               (lane * LANE + 0.005, 0.8 * LANE), facecolor=colors[value], lw=0,
                               transform=ax.get_xaxis_transform())
        for ax, direction in zip(axes[:2, column], DIRECTIONS):
            ax.axhline(exact[direction], **EXACT_STYLE)
            if best:
                ax.axhline(best[direction]["value"], **BEST_STYLE)
        axes[0, column].set_title(", ".join(value_text(k, v) for k, v in zip(others, values)),
                                  fontsize=9)
        axes[2, column].set_xlabel("Training time (s)")

    for row, label in enumerate(["W2 0 → 1", "W2 1 → 0", "Control cost,\nmean of directions"]):
        if row < 2:
            direction = list(DIRECTIONS)[row]
            kept = [r["w2"][direction][:, after_pretraining(r)] for r in recipes]
            kept.append(np.array([exact[direction]] + ([best[direction]["value"]] if best else [])))
        else:
            kept = [mean_cost(r)[:, after_pretraining(r)] for r in recipes]
        axes[row, 0].set_ylim(limits(kept, strip_bottom))
        axes[row, 0].set_ylabel(label, fontsize=9)
    for ax in axes.flat:
        style_axes(ax)

    handles = [Line2D([], [], ls="", marker="o", ms=4, color=colors[value],
                      label=value_text(overlay, value)) for value in overlaid if overlay]
    handles.append(Line2D([], [], **EXACT_STYLE, label="Exact-sample ceiling"))
    if best:
        handles.append(Line2D([], [], **BEST_STYLE, label="Best observed ceiling"))
    handles.append(Patch(color="#888888", lw=0, label=f"Strip: W2 within the "
                         f"{'best observed' if best else 'exact-sample'} ceiling "
                         "(cost row: both directions)"))
    width = fig.get_figwidth()
    columns_in_legend = len(handles) if width >= 16 else 2
    fig.legend(handles=handles, loc="upper center", ncol=columns_in_legend, frameon=False, fontsize=8)
    seeds = len(recipes[0]["seeds"])
    caption = (f"{METHOD_NAMES.get(method, method)} sweep, {len(recipes)} recipes, {seeds} seeds each: "
               "seed mean of each k-th evaluation with seed min and max, EMA weights. "
               "Axes leave out evaluations before training and before pretraining ends.")
    if best:
        caption += "\nBest observed ceiling: lowest seed-mean W2 + 2 SD/√seeds, " + "; ".join(
            f"{DIRECTIONS[d]} {best[d]['mean']:.5f} → {best[d]['value']:.5f} "
            f"({best[d]['recipe']} at {best[d]['seconds']:.0f} s)" for d in DIRECTIONS)
    caption = "\n".join(line for part in caption.split("\n") for line in wrap(part, int(16 * width)))
    fig.text(0.5, 0.005, caption, ha="center", va="bottom", fontsize=8, color="#444444")
    height = fig.get_figheight()
    legend_rows = -(-len(handles) // columns_in_legend)
    fig.tight_layout(rect=(0, (0.16 * (caption.count("\n") + 1) + 0.1) / height,
                           1, 1 - (0.2 * legend_rows + 0.1) / height))
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=200)
    plt.close(fig)


def summary_lines(method, recipes, hyperparameters, exact, best):
    active = {d: best[d]["value"] for d in DIRECTIONS} if best else exact
    lines = [f"{METHOD_NAMES.get(method, method)}: {len(recipes)} recipes, swept "
             + ", ".join(f"{key} {list(values)}" for key, values in hyperparameters.items()),
             "  exact-sample ceilings " + ", ".join(f"{d} {exact[d]:.5f}" for d in DIRECTIONS)]
    if best:
        lines.append("  best observed ceilings " + ", ".join(
            f"{d} {best[d]['value']:.5f} (mean {best[d]['mean']:.5f}, {best[d]['recipe']} "
            f"at {best[d]['seconds']:.0f} s)" for d in DIRECTIONS))
    lines.append(f"  within the {'best observed' if best else 'exact-sample'} ceilings in both "
                 "directions, after pretraining where the run finetunes: first time, and lowest "
                 "seed-mean cost")
    rows = []
    for recipe in recipes:
        met = within(recipe, active)[-1] & after_pretraining(recipe)
        cost = mean_cost(recipe).mean(axis=0)
        if met.any():
            first, cheapest = np.flatnonzero(met)[0], np.flatnonzero(met)[np.argmin(cost[met])]
            text = (f"    {recipe['label']}: from {recipe['seconds'][first]:.0f} s, "
                    f"cost {cost[cheapest]:.4f} at {recipe['seconds'][cheapest]:.0f} s")
            rows.append((cost[cheapest], text))
        else:
            rows.append((np.inf, f"    {recipe['label']}: never"))
    return lines + [text for _, text in sorted(rows, key=lambda row: row[0])]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", help="Experiments or directories holding them")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--ceiling", choices=("exact", "best"), default="exact")
    parser.add_argument("--overlay", help="Hyperparameter to overlay, by its dotted or last name")
    args = parser.parse_args()
    by_method = {}
    for path in find_experiments(args.paths):
        experiment = load_experiment(path)
        experiment["settings"] = recipe_settings(experiment)
        by_method.setdefault(experiment["method"], []).append(experiment)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for method, recipes in by_method.items():
        if len({len(recipe["seeds"]) for recipe in recipes}) > 1:
            raise ValueError(f"{method}: recipes have different numbers of seeds.")
        hyperparameters = swept(recipes)
        overlay = choose_overlay(hyperparameters, args.overlay)
        for recipe in recipes:
            recipe["label"] = ", ".join(value_text(key, recipe["settings"].get(key))
                                        for key in hyperparameters) or str(recipe["path"])
        exact = exact_ceiling(recipes[0]["configs"][0], len(recipes[0]["seeds"]))
        best = best_ceiling(recipes) if args.ceiling == "best" else None
        stem = args.out_dir / f"sweep-{method}"
        plot(method, recipes, hyperparameters, overlay, exact, best, stem.with_suffix(".pdf"))
        summary = "\n".join(summary_lines(method, recipes, hyperparameters, exact, best))
        stem.with_suffix(".txt").write_text(summary + "\n")
        print(summary)


if __name__ == "__main__":
    main()
