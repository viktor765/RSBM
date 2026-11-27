"""Reference levels for the toy metrics, computed once and kept in a JSON file in git."""

import itertools
import json
from pathlib import Path

import numpy as np

from omegaconf import OmegaConf

from src.data_utils import make_image_density, make_known_target_grid
from src.evaluation.toy2d.metrics import empirical_w2, known_target_grid_w2

CACHE_PATH = Path(__file__).with_name("exact_sample_w2_percentiles.json")
DATASETS = ("checkerboard", "uniform_box")


def _percentile_over_subsets(w2, n_in_mean, percentile):
    """Percentile of the mean of w2 over every subset of n_in_mean entries."""
    subsets = np.fromiter(
        itertools.chain.from_iterable(itertools.combinations(range(len(w2)), n_in_mean)),
        dtype=np.int64,
    ).reshape(-1, n_in_mean)
    return float(np.percentile(np.asarray(w2)[subsets].mean(axis=1), percentile))


def image_density_w2_percentile(
    endpoints,
    target,
    *,
    seed=20261105,
    n_samples=4096,
    n_batches=300,
    n_in_mean=3,
    percentile=95,
    cache_path=CACHE_PATH,
):
    """As exact_sample_w2_percentile, for the empirical W2 of image densities.

    These targets have no known grid: the metric compares generated samples with
    an evaluation set drawn from the target, and each run seed has its own. So a
    perfect model's W2 is that between two independent batches of exact samples,
    which is what each of the n_batches computes.

    endpoints holds the two endpoints' data settings and target is the index of
    the one to return. Both take turns on one random stream, so a computation
    fills the cache for both.
    """
    endpoints = [OmegaConf.to_container(OmegaConf.create(conf), resolve=True) for conf in endpoints]
    settings = {
        "endpoints": endpoints,
        "seed": seed,
        "n_samples": n_samples,
        "n_batches": n_batches,
        "n_in_mean": n_in_mean,
        "percentile": percentile,
    }
    cache_path = Path(cache_path)
    entries = json.loads(cache_path.read_text()) if cache_path.exists() else []
    for entry in entries:
        if entry["params"] == {"target": target, **settings}:
            return entry["value"]

    rng = np.random.default_rng(seed)
    confs = [OmegaConf.create(conf) for conf in endpoints]
    w2 = [[], []]
    for _ in range(n_batches):
        for index, conf in enumerate(confs):
            generated = make_image_density(conf, n_samples, rng)
            evaluation_set = make_image_density(conf, n_samples, rng)
            w2[index].append(empirical_w2(generated, evaluation_set)["w2"])

    values = [_percentile_over_subsets(batch, n_in_mean, percentile) for batch in w2]
    entries += [{"params": {"target": index, **settings}, "value": values[index]} for index in (0, 1)]
    cache_path.write_text(json.dumps(entries, indent=2) + "\n")
    return values[target]


def _exact_checkerboard_samples(rng, n_samples, occupied_parity):
    kept = []
    while sum(map(len, kept)) < n_samples:
        x = rng.random((3 * n_samples, 2))
        cells = np.floor(x * 4).astype(int)
        kept.append(x[(cells[:, 0] + cells[:, 1]) % 2 == occupied_parity])
    return np.concatenate(kept)[:n_samples]


def exact_sample_w2_percentile(
    dataset,
    *,
    phase=1,
    seed=20261105,
    n_samples=4096,
    n_batches=300,
    n_in_mean=3,
    percentile=95,
    subdivisions_per_checker_cell=16,
    cache_path=CACHE_PATH,
):
    """Percentile of the W2 that a perfect model's seed mean would show.

    Draws n_batches of n_samples exact target samples and computes each batch's
    W2 to the known target. Each subset of n_in_mean batches stands in for the
    seeds of a run. The result is the percentile of those subsets' mean W2, over
    all subsets.

    The two datasets take turns on one random stream, one batch each, so a
    computation produces both values. The first call for a set of parameters
    adds them to the cache file; later calls read them from there. The defaults
    are the checkerboard selection sweep's: its W2 ceilings are these values.
    """
    settings = {
        "phase": phase,
        "seed": seed,
        "n_samples": n_samples,
        "n_batches": n_batches,
        "n_in_mean": n_in_mean,
        "percentile": percentile,
        "subdivisions_per_checker_cell": subdivisions_per_checker_cell,
    }
    params = {"dataset": dataset, **settings}
    cache_path = Path(cache_path)
    entries = json.loads(cache_path.read_text()) if cache_path.exists() else []
    for entry in entries:
        if entry["params"] == params:
            return entry["value"]

    rng = np.random.default_rng(seed)
    # The parity of the occupied checkerboard cells, as the metric's grid defines them.
    grid = make_known_target_grid("checkerboard", subdivisions_per_checker_cell, phase).points
    occupied_parity = np.floor(grid[0] * 4).astype(int).sum() % 2
    w2 = {name: [] for name in DATASETS}
    for _ in range(n_batches):
        w2["checkerboard"].append(known_target_grid_w2(
            _exact_checkerboard_samples(rng, n_samples, occupied_parity), "checkerboard",
            subdivisions_per_checker_cell=subdivisions_per_checker_cell, phase=phase)["w2"])
        w2["uniform_box"].append(known_target_grid_w2(
            rng.random((n_samples, 2)), "uniform_box",
            subdivisions_per_checker_cell=subdivisions_per_checker_cell)["w2"])

    values = {name: _percentile_over_subsets(w2[name], n_in_mean, percentile) for name in DATASETS}
    entries += [{"params": {"dataset": name, **settings}, "value": values[name]} for name in DATASETS]
    cache_path.write_text(json.dumps(entries, indent=2) + "\n")
    return values[dataset]
