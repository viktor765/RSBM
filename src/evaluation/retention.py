"""Small, portable evaluation records; training checkpoints remain separate."""

import hashlib
import shutil
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from .. import data_utils, models, utils
from .common import protocol_with_id, solver_settings, write_json


def retention_conf(eval_conf):
    conf = dict(eval_conf.get("retention") or {})
    if set(conf) != {"toy_records", "model_snapshots"} or any(
        type(value) is not bool for value in conf.values()
    ):
        raise ValueError(
            "eval.retention accepts boolean toy_records and model_snapshots."
        )
    return conf


def write_toy_records(
    context,
    directory,
    direction,
    source,
    target,
    generated,
    costs,
    *,
    target_cell_masses,
):
    """Retain the exact metric inputs in unit coordinates, without clipping."""
    if not retention_conf(context.eval_conf)["toy_records"]:
        return
    directory = Path(directory)
    arrays = {"source": source, "target": target, "generated": generated}
    if costs is not None:
        arrays["control_cost"] = costs
    # Keep the exact reference, including image-density pixel integration, so
    # endpoint recovery needs neither source data nor the original image files.
    arrays["target_cell_masses"] = target_cell_masses
    # Float32 endpoints, float64 accumulated costs. Preserve endpoint precision
    # if a caller deliberately evaluates in float64.
    np.savez_compressed(directory / "endpoints.npz", **arrays)
    protocol = protocol_with_id(
        {
            "version": 1,
            "coordinates": "unit cube, unclipped; row order preserves source/generated pairs",
            "direction": direction.name,
            "datasets": {
                f"ds{i}": data_utils.dataset_identity(
                    context.data_conf, i, False, context.seed
                )
                for i in (0, 1)
            },
            "seed": context.seed,
            "noise_stream": "evaluation",
            "noise_draw": context.noise_draw,
            "solver": solver_settings(context.sde),
            "toy2d": dict(context.toy2d_conf),
            "control_cost": "native drift squared / (2 sigma^2), left Euler sum per path",
        },
        device=str(context.device),
    )
    write_json(str(directory / "endpoints.json"), protocol)


def write_manifest(directory, *, policy, step, **identity):
    """Inventory actual step outputs with hashes; paths are relative and portable."""
    directory = Path(directory)
    files = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.name == "manifest.json":
            continue
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        files.append(
            {
                "path": path.relative_to(directory).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": digest,
            }
        )
    write_json(
        str(directory / "manifest.json"),
        {
            "version": 1,
            "step": step,
            "policy": policy,
            "revision": utils.git_revision(),
            **identity,
            "files": files,
            "scope": "evaluation outputs at completion; later training summaries/plots may be added",
        },
    )


def model_state_sha256(model):
    """Identity of the evaluated state even when its weights are not retained."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        tensor = tensor.detach().cpu().contiguous()
        digest.update(f"{name}:{tensor.dtype}:{tuple(tensor.shape)}\n".encode())
        digest.update(memoryview(tensor.reshape(-1).view(torch.uint8).numpy()))
    return digest.hexdigest()


def validate_checkpoint_retention(keep_last):
    if keep_last is not None and (type(keep_last) is not int or keep_last < 1):
        raise ValueError(
            "training.checkpoint.keep_last must be null or a positive integer."
        )


def prune_training_checkpoints(directory, keep_last, step):
    """After saving at `step`, keep the newest K complete numbered RSBM checkpoints.

    Only checkpoints up to `step` count, so the one just saved is always kept.
    Resuming an older checkpoint archives later ones; any that remain are not
    this run's history and are left alone, as are incomplete writes, final/
    latest checkpoints and non-numbered directories. The baseline already
    replaces c_latest and does not call this helper.
    """
    validate_checkpoint_retention(keep_last)
    if keep_last is None:
        return
    checkpoints = [
        path
        for path in Path(directory).glob("c_*")
        if path.name.removeprefix("c_").isdigit()
        and int(path.name.removeprefix("c_")) <= step
        and path.is_dir()
        and not path.is_symlink()
        and (path / "complete.json").is_file()
    ]
    checkpoints.sort(key=lambda path: int(path.name.removeprefix("c_")))
    for path in checkpoints[:-keep_last]:
        shutil.rmtree(path)


def retain_evaluated_model(directory, model, config, *, step, weights):
    """Called by either trainer with the unwrapped, uncompiled evaluated network.

    directory is the evaluation's eval/<step>/<weights>/.
    """
    policy = retention_conf(config.eval)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if policy["model_snapshots"]:
        temporary = directory / "model.safetensors.tmp"
        models.save_final_model(
            str(temporary),
            model,
            config,
            step=step,
            weights=weights,
        )
        temporary.replace(directory / "model.safetensors")
    write_manifest(
        directory,
        policy=policy,
        step=step,
        weights=weights,
        model_state_sha256=model_state_sha256(model),
        method=config.method,
        config=OmegaConf.to_container(
            OmegaConf.masked_copy(config, [k for k in config if k != "out_dir"]),
            resolve=True,
        ),
    )
