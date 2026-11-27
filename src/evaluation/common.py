# ruff: noqa: E402
import glob
import hashlib
import importlib.metadata
import json
import logging
import os
import shutil
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..models.export import WEIGHTS
from ..rsbm.sdes import ReflectedSDE

COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "vermillion": "#D55E00",
    "black": "#111111",
    "grey": "#5F6368",
}

DPI = 300

TOY2D_DEFAULTS = {
    "final_artifacts": True,
    "n_snapshot_times": 6,
    "n_snapshot_particles": 1000,
    "n_gif_paths": 128,
    "n_gif_frames": 31,
    "n_generation_points": 5000,
    "n_generation_paths": 7,
    "vector_grid_size": 21,
    "boundary_eps": 1e-3,
    # Girsanov control cost of the learned drift along the evaluation paths.
    "control_cost": True,
}

logger = logging.getLogger(__name__)


def file_enum_format(path: str, index: int) -> str:
    base, ext = os.path.splitext(path)
    return f"{base}_({index:03d}){ext}"


def next_filename(path: str) -> str:
    if not os.path.exists(path):
        return path

    i = 0
    new_path = file_enum_format(path, i)
    while os.path.exists(new_path):
        i += 1
        new_path = file_enum_format(path, i)
    return new_path


def savefig(fig, path: str, **kwargs):
    """Saves figure, saves copy if `path` already is a file."""
    fig.tight_layout()
    if not os.path.isfile(path):
        fig.savefig(path, **kwargs)
    else:
        zeroth_path = file_enum_format(path, 0)
        if not os.path.isfile(zeroth_path):
            os.rename(path, zeroth_path)
        fig.savefig(path, **kwargs)
        fig.savefig(next_filename(path), **kwargs)


def imsave(path: str, img: np.ndarray, **kwargs) -> None:
    """Saves image, saves copy if `path` already is a file."""
    if not os.path.isfile(path):
        plt.imsave(path, img, **kwargs)
    else:
        zeroth_path = file_enum_format(path, 0)
        if not os.path.isfile(zeroth_path):
            os.rename(path, zeroth_path)
        plt.imsave(path, img, **kwargs)
        plt.imsave(next_filename(path), img, **kwargs)


def step_dir(eval_dir, step: int) -> str:
    """The evaluations at step, and the records of a run that ended there."""
    return os.path.join(eval_dir, str(step).zfill(6))


def evaluation_dir(eval_dir, step: int, weights: str) -> str:
    """The outputs of the evaluation of one weight set at step."""
    return os.path.join(step_dir(eval_dir, step), weights)


def evaluation_weights(eval_conf) -> list[str]:
    """eval.weights: the weight sets each evaluation scores, in order."""
    weights = list(eval_conf.weights)
    if not weights or len(set(weights)) < len(weights) or not set(weights) <= set(WEIGHTS):
        raise ValueError(
            f"eval.weights must list distinct entries of {list(WEIGHTS)}, got {weights}."
        )
    return weights


def training_summary_path(eval_dir) -> str:
    """The summary of the run's latest end, written in its last step's directory."""
    paths = sorted(glob.glob(os.path.join(eval_dir, "[0-9]" * 6, "training_summary.json")))
    if not paths:
        raise FileNotFoundError(f"No training summary under {eval_dir}.")
    return paths[-1]


def shared_dir(root: str, name: str, identity: dict, write) -> str:
    """root/name/<hash of identity>, filled once by write(directory).

    Concurrent writers each fill a temporary directory; the first rename wins.
    """
    serialized = json.dumps(identity, sort_keys=True)
    key = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]
    path = os.path.join(root, name, key)
    if os.path.isdir(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = tempfile.mkdtemp(prefix=f".{key}-", dir=os.path.dirname(path))
    try:
        with open(os.path.join(temporary, "identity.json"), "w", encoding="utf-8") as f:
            json.dump(identity, f, indent=2, sort_keys=True)
        write(temporary)
        os.rename(temporary, path)
    except OSError:
        if not os.path.isdir(path):
            raise
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    return path


def protocol_with_id(definition: dict, **unhashed) -> dict:
    """definition with an id that changes with any of it, plus unhashed fields.

    Unhashed fields (such as the device) describe an evaluation without
    changing what its metrics measure.
    """
    serialized = json.dumps(definition, sort_keys=True, allow_nan=False)
    key = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]
    return {"id": key, **definition, **unhashed}


def solver_settings(sde) -> dict:
    """The Euler sampler's settings, which generated samples depend on."""
    settings = {
        "sde": type(sde).__name__,
        "sigma": float(sde.sigma),
        "euler_steps": sde.euler_steps,
        "noise_on_last_step": sde.noise_on_last_step,
    }
    if isinstance(sde, ReflectedSDE):
        settings["n_reflections"] = sde.reflector.n_reflections
        settings["project_last_step"] = sde.project_last_step
    return settings


def metric_library_versions() -> dict:
    return {
        name: importlib.metadata.version(name)
        for name in ("torch", "torchvision", "torchmetrics", "torch-fidelity")
    }


def write_json(path: str, record: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def get_toy2d_eval_conf(eval_conf) -> dict:
    conf = dict(TOY2D_DEFAULTS)
    conf.update(dict(eval_conf.get("toy2d", {})))
    return conf


def select_evenly_spaced(x: np.ndarray, n: int) -> np.ndarray:
    if x.shape[0] <= n:
        return x
    indices = np.linspace(0, x.shape[0] - 1, n).round().astype(int)
    return x[indices]


def plot_loss(losses, dir: str) -> None:
    fig, ax = plt.subplots(1)
    ax.plot(np.arange(1, len(losses) + 1), losses)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss")
    savefig(fig, os.path.join(dir, "training_loss.png"))
    plt.close(fig)


def plot_directional_losses(losses_by_direction, dir: str) -> None:
    direction_names = {+1: "0_to_1", -1: "1_to_0"}
    fig, ax = plt.subplots(1)
    plotted = False
    for direction in (+1, -1):
        losses = losses_by_direction.get(direction, [])
        if not losses:
            continue
        ax.plot(
            np.arange(1, len(losses) + 1),
            losses,
            label=direction_names[direction],
        )
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel("Directional optimizer update")
    ax.set_ylabel("Loss")
    ax.legend()
    savefig(fig, os.path.join(dir, "directional_training_losses.png"))
    plt.close(fig)
