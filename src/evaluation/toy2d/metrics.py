import numpy as np
import ot

from ...data_utils import make_known_target_grid
from ..common import TOY2D_DEFAULTS


def empirical_w2(
    x: np.ndarray,
    y: np.ndarray,
) -> dict[str, float | int]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    out = {
        "w2_n_source": int(x.shape[0]),
        "w2_n_target": int(y.shape[0]),
    }
    if np.any(np.isnan(x)) or np.any(np.isnan(y)):
        out["w2"] = float(np.inf)
    else:
        out["w2"] = float(np.sqrt(max(ot.solve_sample(x, y).value, 0.0)))
    return out


def known_target_grid_w2(
    generated_samples: np.ndarray,
    dataset: str,
    *,
    subdivisions_per_checker_cell: int,
    phase: int = 0,
) -> dict[str, float | int | str | list[float]] | None:
    """Estimate W2 to a known uniform target without target sampling noise.

    Samples must already use unit-square coordinates. The W2 interval bounds
    target discretization error for the full empirical source;
    it is not a sampling-confidence interval for the generated distribution.
    """

    grid = make_known_target_grid(dataset, subdivisions_per_checker_cell, phase)
    if grid is None:
        return None
    target = grid.points

    generated = np.asarray(generated_samples, dtype=np.float64)
    if generated.ndim != 2 or generated.shape[1] != 2:
        raise ValueError("generated_samples must have shape [num_samples, 2].")
    if generated.shape[0] == 0:
        raise ValueError("generated_samples cannot be empty.")
    if not np.isfinite(generated).all():
        w2_squared = float("inf")
    else:
        source_weights = np.full(
            generated.shape[0],
            1.0 / generated.shape[0],
            dtype=np.float64,
        )
        target_weights = np.full(
            target.shape[0],
            1.0 / target.shape[0],
            dtype=np.float64,
        )
        cost = ot.dist(generated, target, metric="sqeuclidean")
        w2_squared = float(
            ot.emd2(
                source_weights,
                target_weights,
                cost,
                numItermax=1_000_000,
            )
        )

    w2 = float(np.sqrt(max(w2_squared, 0.0)))
    target_error = grid.target_discretization_error_upper_bound
    return {
        "dataset": dataset,
        "w2_n_source": int(generated.shape[0]),
        "w2_n_target": int(target.shape[0]),
        "w2": w2,
        "subdivisions_per_checker_cell": int(subdivisions_per_checker_cell),
        "target_discretization_error_upper_bound": target_error,
        "w2_interval": [
            max(0.0, w2 - target_error),
            w2 + target_error,
        ],
    }


def mean_squared_endpoint_displacement(
    start_points: np.ndarray,
    end_points: np.ndarray,
) -> float:
    """Return paired MSD averaged over samples and dimensions.

    Both arrays must already use the same unit-cube coordinate system.
    Endpoints outside the cube are retained for raw displacement reporting.
    """

    starts = np.asarray(start_points, dtype=np.float64)
    ends = np.asarray(end_points, dtype=np.float64)
    if starts.ndim != 2:
        raise ValueError("start_points must have shape [num_samples, dimension].")
    if ends.shape != starts.shape:
        raise ValueError(
            f"end_points must match start_points shape {starts.shape}."
        )
    if starts.shape[0] == 0 or starts.shape[1] == 0:
        raise ValueError("Endpoint arrays cannot be empty.")
    return float(np.mean((ends - starts) ** 2))


def region_masses(samples: np.ndarray, target_masses: np.ndarray) -> dict:
    """Compare sample masses on a grid of unit-square cells with the target's.

    target_masses[ix, iy] are the exact target masses of the cells (summing to
    one). Samples outside the unit square, where the target has no mass, are
    counted as excess mass. The total variation is half the L1 distance over
    the cells and that excess; the outside-support mass is the sample mass in
    cells of zero target mass or outside the square.
    """
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 2 or x.shape[0] == 0:
        raise ValueError("samples must have shape [num_samples, 2].")
    cells = target_masses.shape[0]
    inside = np.all((x >= 0.0) & (x <= 1.0), axis=1)
    index = np.minimum(np.floor(x[inside] * cells).astype(np.int64), cells - 1)
    masses = np.zeros_like(target_masses, dtype=np.float64)
    np.add.at(masses, (index[:, 0], index[:, 1]), 1.0 / x.shape[0])
    outside_box = 1.0 - inside.mean()
    return {
        "cells_per_axis": int(cells),
        "total_variation": float(
            0.5 * (np.abs(masses - target_masses).sum() + outside_box)
        ),
        "outside_support_mass": float(masses[target_masses == 0].sum() + outside_box),
    }


def support_metrics(
    x: np.ndarray, boundary_eps: float = TOY2D_DEFAULTS["boundary_eps"]
) -> dict[str, float]:
    x = np.asarray(x)
    flat = x.reshape(-1, x.shape[-1])
    violation = np.maximum(np.maximum(-flat, flat - 1.0), 0.0).max(axis=1)
    return {
        "out_of_bounds_fraction": float(np.mean(violation > boundary_eps)),
        "max_violation": float(np.max(violation)) if violation.size else 0.0,
    }


def control_cost(sde, model, trajectory, direction, device, *, per_path=False):
    """Girsanov control cost E sum_i |b|^2 / (2 sigma^2) dt along Euler paths.

    ``trajectory`` holds the states before each step plus the endpoint, as
    returned by ``forward_euler``; b is the learned drift at the same times the
    sampler used, and the reference is driftless with diffusion sigma. Paths
    and reference share the reflection, so this estimates KL(P || R) minus
    the KL between the initial laws: the quantity the Schrodinger bridge
    minimizes for fixed marginals.

    By default return the scalar mean. With ``per_path=True``, return the
    tensor of path costs on ``device`` for retention and summaries. Accumulate
    in float32 on MPS, which does not support float64; otherwise use float64.
    """
    import torch as th

    n_steps = trajectory.shape[1] - 1
    cost_dtype = th.float32 if th.device(device).type == "mps" else th.float64
    cost = th.zeros(trajectory.shape[0], dtype=cost_dtype, device=device)
    with th.inference_mode():
        for i in range(n_steps):
            x = trajectory[:, i].to(device)
            t = th.full((x.shape[0],), i / n_steps, dtype=x.dtype, device=device)
            drift = model(x, t, direction=direction).to(x.dtype)
            step_cost = drift.square().sum(-1) / (2 * sde.sigma**2)
            cost += step_cost.to(cost_dtype) / n_steps
    return cost if per_path else float(cost.mean())
