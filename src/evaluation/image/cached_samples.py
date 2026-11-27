import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch as th

from ... import data_utils, utils
from ...rsbm import sdes
from ...utils import derive_seed
from ..common import protocol_with_id, solver_settings
from ..directions import DIRECTIONS, DirectionSpec
from ..retention import model_state_sha256
from ..sampling import sample_direction

logger = logging.getLogger(__name__)

CACHE_SCHEMA_VERSION = 2
DEFAULT_CACHED_SAMPLES_CONF = {
    "enabled": True,
    "n_samples": 1024,
    "seed": 0,
}
CACHED_SAMPLE_BATCH_SIZE = 32
SPLITS = ("test", "train")


def cached_samples_conf(eval_conf) -> dict[str, Any]:
    conf = dict(DEFAULT_CACHED_SAMPLES_CONF)
    conf.update(dict(eval_conf.get("cached_samples", {}) or {}))
    return conf


def cached_samples_enabled(eval_conf) -> bool:
    return bool(cached_samples_conf(eval_conf)["enabled"])


def select_start_indices(
    dataset_size: int,
    n_samples: int,
    seed: int,
    split: str,
    direction_name: str,
) -> np.ndarray:
    if dataset_size <= 0:
        raise ValueError("Cannot create cached samples from an empty dataset.")
    if int(n_samples) <= 0:
        raise ValueError("eval.cached_samples.n_samples must be positive.")
    n_selected = min(int(n_samples), int(dataset_size))
    rng = np.random.default_rng(
        derive_seed(seed, "start_indices", split, direction_name, bits=32)
    )
    return rng.choice(dataset_size, size=n_selected, replace=False).astype(np.int64)


def _sde_type_name(sde) -> str:
    if isinstance(sde, sdes.ReflectedSDE):
        return "reflected"
    if isinstance(sde, sdes.SDE):
        return "brownian"
    return sde.__class__.__name__


def _source_dataset(context, split: str, direction: DirectionSpec):
    train_data = split == "train"
    return data_utils.get_dataset(
        context.data_conf, direction.source_index, train_data, False, context.seed
    )


def _subset_loader(dataset, selected_indices: np.ndarray):
    subset = th.utils.data.Subset(dataset, selected_indices.tolist())
    return th.utils.data.DataLoader(
        subset,
        batch_size=CACHED_SAMPLE_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )


def generation_identity(
    *,
    context,
    model_sha256: str,
    weights: str,
    split: str,
    direction: DirectionSpec,
    n_samples: int,
    seed: int,
    dataset_size: int,
) -> dict[str, Any]:
    """Everything a cache's samples depend on, with an id that changes with any of it.

    The selection and noise seed is eval.cached_samples.seed, not the run seed:
    runs of every seed translate the same starts with the same noise.
    """
    return protocol_with_id(
        {
            "schema_version": CACHE_SCHEMA_VERSION,
            "split": split,
            "direction": direction.name,
            "source_index": direction.source_index,
            "target_index": direction.target_index,
            "n_requested": int(n_samples),
            "seed": int(seed),
            "dataset_size": int(dataset_size),
            "sampling_batch_size": CACHED_SAMPLE_BATCH_SIZE,
            "device": context.device.type,
            "model_state_sha256": model_sha256,
            "weights": weights,
            "solver": solver_settings(context.sde),
            "data": {
                f"ds{i}": data_utils.dataset_identity(
                    context.data_conf, i, split == "train", context.seed
                )
                for i in (0, 1)
            },
            "revision": utils.git_revision(),
        }
    )


def evaluated_model_sha256(model: th.nn.Module) -> str:
    """model_state_sha256 of the network itself, also when it is compiled."""
    return model_state_sha256(getattr(model, "_orig_mod", model))


def _cache_payload_matches(
    *,
    cache: dict[str, Any],
    selected_indices: np.ndarray,
) -> bool:
    expected_indices = th.from_numpy(selected_indices.copy())
    selected = cache.get("selected_indices")
    source = cache.get("source_pixels")
    generated = cache.get("generated_pixels")
    return (
        isinstance(selected, th.Tensor)
        and isinstance(source, th.Tensor)
        and isinstance(generated, th.Tensor)
        and th.equal(selected.cpu(), expected_indices)
        and source.shape[0] == len(selected_indices)
        and generated.shape[0] == len(selected_indices)
    )


def _load_matching_cache(
    *,
    cache_path: Path,
    identity: dict[str, Any],
    selected_indices: np.ndarray,
) -> dict[str, Any] | None:
    """The cache at cache_path if it was generated with identity, else None.

    Without a committed, clean revision the code that generated an existing
    cache is unknown, so it is never reused.
    """
    if not cache_path.is_file():
        return None
    if identity["revision"]["dirty"] is not False:
        logger.warning(
            "Regenerating %s: the work tree has no clean revision to match.", cache_path
        )
        return None
    try:
        cache = th.load(cache_path, map_location="cpu", weights_only=True)
    except Exception:
        logger.warning("Regenerating unreadable cached samples file: %s", cache_path)
        return None
    metadata = cache.get("metadata") if isinstance(cache, dict) else None
    if not isinstance(metadata, dict):
        logger.warning("Regenerating invalid cached samples file: %s", cache_path)
        return None
    if metadata.get("id") != identity["id"]:
        changed = sorted(
            key
            for key in identity
            if key != "id" and metadata.get(key) != identity[key]
        )
        logger.warning(
            "Regenerating %s: generated with different %s.",
            cache_path,
            ", ".join(changed) or "settings",
        )
        return None
    if not _cache_payload_matches(cache=cache, selected_indices=selected_indices):
        logger.warning("Regenerating inconsistent cached samples file: %s", cache_path)
        return None
    logger.info("Using cached samples from %s", cache_path)
    return cache


def write_cached_samples_for_direction(
    *,
    context,
    model: th.nn.Module,
    split: str,
    direction: DirectionSpec,
    output_dir: Path,
    n_samples: int,
    seed: int,
    model_sha256: str,
    weights: str,
) -> dict[str, Any]:
    """Reuse output_dir's cache of this identity, or generate and replace it.

    model_sha256 is evaluated_model_sha256(model), computed once per model.
    """
    dataset = _source_dataset(context, split, direction)
    selected_indices = select_start_indices(
        len(dataset), n_samples, seed, split, direction.name
    )
    identity = generation_identity(
        context=context,
        model_sha256=model_sha256,
        weights=weights,
        split=split,
        direction=direction,
        n_samples=n_samples,
        seed=seed,
        dataset_size=len(dataset),
    )
    cache_path = output_dir / "cache" / f"{split}_{direction.name}.pt"
    cached = _load_matching_cache(
        cache_path=cache_path,
        identity=identity,
        selected_indices=selected_indices,
    )
    if cached is not None:
        return {"cache": cached, "cache_path": cache_path}

    loader = _subset_loader(dataset, selected_indices)
    generator = th.Generator(device=context.device)
    generator.manual_seed(derive_seed(seed, "sde_noise", split, direction.name, bits=63))

    source_batches = []
    generated_batches = []
    model.eval()
    with th.inference_mode():
        for batch in loader:
            source_model_space = batch[0] if isinstance(batch, (tuple, list)) else batch
            source_batches.append(
                context.postprocessing[direction.source_index](
                    source_model_space.detach().cpu().to(dtype=th.float32)
                )
            )
            sampled = sample_direction(
                sde=context.sde,
                model=model,
                source=source_model_space,
                direction=direction,
                device=context.device,
                generator=generator,
            )
            generated_batches.append(
                context.postprocessing[direction.target_index](
                    sampled.detach().cpu().to(dtype=th.float32)
                )
            )

    source_pixels = th.cat(source_batches, dim=0)
    generated_pixels = th.cat(generated_batches, dim=0)
    cache = {
        "metadata": {
            **identity,
            "n_selected": int(len(selected_indices)),
            "sde_type": _sde_type_name(context.sde),
        },
        "selected_indices": th.from_numpy(selected_indices.copy()),
        "source_pixels": source_pixels,
        "generated_pixels": generated_pixels,
    }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_name(cache_path.name + ".tmp")
    th.save(cache, temporary)
    temporary.replace(cache_path)
    return {"cache": cache, "cache_path": cache_path}


def write_cached_samples(
    context, model: th.nn.Module, output_dir: str | Path, weights: str
) -> dict[str, dict[str, Any]]:
    conf = cached_samples_conf(context.eval_conf)
    if not conf["enabled"]:
        return {}

    output_dir = Path(output_dir)
    model_sha256 = evaluated_model_sha256(model)
    cached_by_item = {}
    for split in SPLITS:
        for direction in DIRECTIONS:
            item_key = f"{split}_{direction.name}"
            logger.info("Writing cached samples for %s", item_key)
            cached_by_item[item_key] = write_cached_samples_for_direction(
                context=context,
                model=model,
                split=split,
                direction=direction,
                output_dir=output_dir,
                n_samples=int(conf["n_samples"]),
                seed=int(conf["seed"]),
                model_sha256=model_sha256,
                weights=weights,
            )
    return cached_by_item
