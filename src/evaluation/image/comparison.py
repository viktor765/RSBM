from pathlib import Path
from typing import Any

import numpy as np
import torch as th

from ...utils import derive_seed
from .cached_samples import CACHE_SCHEMA_VERSION

COMPARISON_LABELS = [
    "Source",
    r"$\alpha$-DSBM",
    r"Clipped",
    r"$\alpha$-RSBM",
]
# Sampler settings the two runs normally share. Where they differ, the figure
# names each method's values under its label.
SOLVER_SUBLABELS = {
    "sigma": lambda value: rf"$\sigma={value:g}$",
    "euler_steps": lambda value: f"{value} steps",
    "noise_on_last_step": lambda value: "last-step noise" if value else "no last-step noise",
}


def load_cache(path: Path) -> dict[str, Any]:
    return th.load(path, map_location="cpu", weights_only=True)


def tensor_to_image_array(x: th.Tensor) -> np.ndarray:
    x = x.detach().cpu().to(dtype=th.float32)
    if x.ndim != 3:
        raise ValueError(f"Expected image tensor with shape (C,H,W), got {x.shape}.")
    if x.shape[0] == 1:
        x = x.repeat(3, 1, 1)
    elif x.shape[0] != 3:
        raise ValueError(f"Expected image tensor with C==1 or C==3, got {x.shape}.")
    return x.clamp(0, 1).permute(1, 2, 0).numpy()


def projected_image_array(x: th.Tensor) -> np.ndarray:
    return tensor_to_image_array(x.clamp(0, 1))


def oob_overlay_array(x: th.Tensor) -> np.ndarray:
    image = tensor_to_image_array(x)
    oob_mask = ((x < 0) | (x > 1)).detach().cpu()
    if oob_mask.shape[0] > 1:
        oob_mask_2d = oob_mask.any(dim=0).numpy()
    else:
        oob_mask_2d = oob_mask[0].numpy()
    image[oob_mask_2d] = np.array([1.0, 0.0, 1.0])
    return image


def select_random_generated_indices(
    n_generated: int,
    plot_count: int,
    seed: int,
    split: str,
    direction_name: str,
) -> th.Tensor:
    n_plot = min(int(plot_count), int(n_generated))
    rng = np.random.default_rng(
        derive_seed(seed, "random_pairs_plot", split, direction_name, bits=32)
    )
    chosen = rng.permutation(n_generated)[:n_plot]
    return th.as_tensor(chosen, dtype=th.long)


def select_top_oob_indices(generated: th.Tensor, top_k: int) -> th.Tensor:
    oob_counts = ((generated < 0) | (generated > 1)).flatten(1).sum(dim=1)
    n_plot = min(int(top_k), generated.shape[0])
    return th.argsort(oob_counts, descending=True)[:n_plot]


# Generation settings both caches must share: the same starts, noise and data.
SHARED_IDENTITY = (
    "split",
    "direction",
    "source_index",
    "target_index",
    "n_requested",
    "seed",
    "dataset_size",
    "sampling_batch_size",
    "data",
)


def validate_comparison_caches(
    nonref_cache: dict[str, Any], reflected_cache: dict[str, Any]
) -> None:
    nonref_meta = nonref_cache["metadata"]
    reflected_meta = reflected_cache["metadata"]
    for name, meta in (("First", nonref_meta), ("Second", reflected_meta)):
        if meta.get("schema_version") != CACHE_SCHEMA_VERSION:
            raise ValueError(
                f"{name} cache has schema {meta.get('schema_version')}, "
                f"not {CACHE_SCHEMA_VERSION}; regenerate it."
            )
    if nonref_meta["sde_type"] == "reflected":
        raise ValueError("First cache must be from a non-reflected run.")
    if reflected_meta["sde_type"] != "reflected":
        raise ValueError("Second cache must be from a reflected run.")
    for key in SHARED_IDENTITY:
        if nonref_meta[key] != reflected_meta[key]:
            raise ValueError(f"Cache metadata field {key!r} differs.")
    if not th.equal(nonref_cache["selected_indices"], reflected_cache["selected_indices"]):
        raise ValueError("Cache selected_indices differ.")
    for key in ("source_pixels", "generated_pixels"):
        if nonref_cache[key].shape != reflected_cache[key].shape:
            raise ValueError(f"Cache tensor {key!r} shapes differ.")
    if not th.equal(nonref_cache["source_pixels"], reflected_cache["source_pixels"]):
        raise ValueError("Cache source_pixels differ.")


def comparison_sublabels(
    nonref_meta: dict[str, Any], reflected_meta: dict[str, Any]
) -> list[str] | None:
    """Each method's sampler settings that differ between the runs, if any."""
    differing = [
        key
        for key in SOLVER_SUBLABELS
        if nonref_meta["solver"][key] != reflected_meta["solver"][key]
    ]
    if not differing:
        return None

    def settings(meta):
        return ", ".join(SOLVER_SUBLABELS[key](meta["solver"][key]) for key in differing)

    return ["", settings(nonref_meta), "", settings(reflected_meta)]


def comparison_rows(
    *,
    nonref_cache: dict[str, Any],
    reflected_cache: dict[str, Any],
    generated_indices: th.Tensor,
) -> list[list[np.ndarray]]:
    sources = nonref_cache["source_pixels"].detach().cpu()
    nonref_generated = nonref_cache["generated_pixels"].detach().cpu()
    reflected_generated = reflected_cache["generated_pixels"].detach().cpu()

    rows = []
    for generated_index in generated_indices.tolist():
        rows.append(
            [
                tensor_to_image_array(sources[generated_index]),
                oob_overlay_array(nonref_generated[generated_index]),
                projected_image_array(nonref_generated[generated_index]),
                tensor_to_image_array(reflected_generated[generated_index]),
            ]
        )
    return rows
