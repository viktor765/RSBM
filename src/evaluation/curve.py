"""Training curves: quality against cumulative training work, one row per evaluation."""

import json
import os

from .common import COLORS, plt, savefig

# The first two colours of scripts/plot_comparison.py's palette.
DIRECTION_COLORS = {"0_to_1": "#2a78d6", "1_to_0": "#eb6834"}
CURVE_METRICS = {
    "known_target_grid_w2": "W2 to target",
    "w2": "W2 to test samples",
    "control_cost": "Control cost",
    "region_total_variation": "TV of 8x8 cell masses",
    "outside_support_mass": "Mass outside target support",
    "fid": "FID",
    "kid": "KID",
    "lpips": "LPIPS",
    "msd": "MSD",
    "total_oob_pixel_fraction": "Out-of-bounds pixel fraction",
}


def append_curve_row(
    eval_dir,
    *,
    step,
    weights,
    training_seconds,
    sampling_seconds,
    pretrain_seconds,
    directional_updates,
    fresh_paths,
    fields,
):
    """Append the evaluation of one weight set to ``eval_dir/curve.jsonl``.

    An evaluation of several weight sets appends one row per set, with the
    same step and training work. ``training_seconds`` excludes evaluation and checkpointing on both methods,
    so rows from different methods share one time axis. ``sampling_seconds``
    (simulating training paths) and ``pretrain_seconds`` are parts of it.
    ``fields`` are the evaluator's ``curve_fields`` of its record, mostly
    ``{metric: {direction name: value}}``.
    """
    row = {
        "step": step,
        "weights": weights,
        "training_seconds": training_seconds,
        "sampling_seconds": sampling_seconds,
        "pretrain_seconds": pretrain_seconds,
        "directional_updates": {str(k): v for k, v in directional_updates.items()},
        "fresh_paths": {str(k): v for k, v in fresh_paths.items()},
        **fields,
    }
    with open(os.path.join(eval_dir, "curve.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(row, allow_nan=False) + "\n")


def curve_weights(eval_dir):
    """The weight sets in ``eval_dir/curve.jsonl``, in the order first evaluated."""
    with open(os.path.join(eval_dir, "curve.jsonl"), encoding="utf-8") as stream:
        return list(dict.fromkeys(json.loads(line)["weights"] for line in stream))


def load_curve(eval_dir, weights):
    """Rows of one weight set in ``eval_dir/curve.jsonl`` by step, the last per step."""
    rows = {}
    with open(os.path.join(eval_dir, "curve.jsonl"), encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["weights"] == weights:
                rows[row["step"]] = row  # A resumed run may re-evaluate a step.
    if not rows:
        raise ValueError(f"No {weights} rows in {eval_dir}/curve.jsonl.")
    return [rows[step] for step in sorted(rows)]


def plot_curve(eval_dir, out_dir):
    """Plot this run's curve metrics against training step, one panel per metric.

    Each weight set is one line style per direction: solid for the first set
    evaluated, dashed for the second.
    """
    if not os.path.isfile(os.path.join(eval_dir, "curve.jsonl")):
        return
    curves = {weights: load_curve(eval_dir, weights) for weights in curve_weights(eval_dir)}
    metrics = [
        m for m in CURVE_METRICS if any(row.get(m) for rows in curves.values() for row in rows)
    ]
    if not metrics:
        return
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 3.5), squeeze=False)
    for ax, metric in zip(axes[0], metrics):
        for (weights, rows), style in zip(curves.items(), ("-", "--")):
            for direction, color in DIRECTION_COLORS.items():
                points = [(row["step"], row[metric][direction]) for row in rows
                          if row.get(metric, {}).get(direction) is not None]
                if points:
                    ax.plot(*zip(*points), color=color, ls=style, lw=2, marker="o", ms=3,
                            label=f"{direction}, {weights}")
        ax.set_title(CURVE_METRICS[metric], fontsize=10)
        ax.set_xlabel("Training step")
        ax.grid(True, color="#e5e5e5", lw=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=COLORS["grey"])
    axes[0][0].legend(frameon=False, fontsize=8)
    savefig(fig, os.path.join(out_dir, "training_curve.png"))
    plt.close(fig)
