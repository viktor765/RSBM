"""Evaluate a final model file without its run's training state.

A final model file (final_model_<weights>.safetensors, or an evaluation's
model.safetensors) holds one evaluated weight set and the resolved run config,
which is all evaluation needs: no optimizer, training checkpoint or Hydra run
directory. Each evaluation writes to a new directory, never to the run's eval/,
so the training-time records stay as training wrote them.
"""

import hashlib
import json
import time
from pathlib import Path

import torch as th
from omegaconf import DictConfig, OmegaConf

from .. import data_utils, models, utils
from ..rsbm import create_sde_and_bridge
from .common import evaluation_weights
from .factory import create_evaluator, run_evaluator_arguments

# Settings an evaluation may change: anything under eval, the Euler solver, EMA
# compilation and the root of the shared dataset plots. Changing the model,
# data, seed or reference process would evaluate another experiment.
OVERRIDABLE = (
    "eval.",
    "sde.euler_steps",
    "sde.n_reflections",
    "sde.noise_on_last_step",
    "sde.project_last_step",
    "training.compile_ema",
    "output_root",
)


def apply_overrides(config: DictConfig, overrides) -> DictConfig:
    """config with key=value overrides, each of an OVERRIDABLE setting."""
    for override in overrides:
        key = override.split("=", 1)[0]
        if not any(
            key.startswith(allowed) if allowed.endswith(".") else key == allowed
            for allowed in OVERRIDABLE
        ):
            raise ValueError(
                f"{key} cannot be overridden for evaluation; allowed: {', '.join(OVERRIDABLE)}."
            )
    return OmegaConf.merge(config, OmegaConf.from_dotlist(list(overrides)))


def default_out_dir(model_path) -> Path:
    """<run>/posthoc/<launch time> for <run>/checkpoints/<file>, else beside the file."""
    parent = Path(model_path).resolve().parent
    run = parent.parent if parent.name == "checkpoints" else parent
    return run / "posthoc" / time.strftime("%Y-%m-%d_%H-%M-%S")


def default_device(config) -> th.device:
    """The device training would use: Accelerate's for RSBM, training.device for RSB-FBSDE."""
    if config.method == "rsb_fbsde":
        device = config.training.device
        if device == "auto":
            device = "cuda" if th.cuda.is_available() else "cpu"
        return th.device(device)
    from accelerate import PartialState
    from accelerate.utils import parse_flag_from_env

    return PartialState(cpu=parse_flag_from_env("ACCELERATE_USE_CPU")).device


def evaluated_network(model, config, weights):
    """The drift model the evaluator samples with, for model's weight set."""
    if (
        config.method == "alpha_imf"
        and config.training.compile_ema
        and weights == "ema"
    ):
        # As RSBM training evaluates its EMA weights.
        return th.compile(model)
    return model


def evaluate_final_model(model_path, out_dir, overrides=(), device=None) -> dict:
    """Evaluate model_path into the new directory out_dir, with final artifacts.

    out_dir receives config.yaml (the evaluated settings), posthoc.json (the
    model file, its step and weights, overrides, device and code revision),
    the evaluator's outputs, <step:06d>/<weights>/ and true_data.json, and
    curve.jsonl, the evaluation's curve row. out_dir thus reads like a run's
    eval/ with this one evaluation. Returns the evaluator's metrics.
    """
    metadata = models.read_final_model_metadata(str(model_path))
    # Output locations belong to this evaluation, not to the published model.
    config = apply_overrides(
        OmegaConf.merge({"output_root": str(out_dir)}, metadata["config"]), overrides
    )
    weights = metadata["weights"]
    if weights not in evaluation_weights(config.eval):
        raise ValueError(
            f"{model_path} holds {weights} weights; eval.weights="
            f"{list(config.eval.weights)} cannot select others."
        )
    # The file's weight set alone, as training evaluated it.
    config.eval.weights = [weights]
    device = default_device(config) if device is None else th.device(device)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True)
    OmegaConf.save(config, out_dir / "config.yaml")
    record = {
        "model": str(Path(model_path).resolve()),
        "sha256": hashlib.sha256(Path(model_path).read_bytes()).hexdigest(),
        "step": metadata["step"],
        "weights": weights,
        "overrides": list(overrides),
        "device": str(device),
        "revision": utils.git_revision(),
    }
    (out_dir / "posthoc.json").write_text(json.dumps(record, indent=2) + "\n")

    utils.set_seed(config.seed)
    model, _ = models.load_final_model(str(model_path), device)
    sde, _ = create_sde_and_bridge(
        config.sde, data_utils.get_data_normalization(config.data)
    )
    evaluator = create_evaluator(
        **run_evaluator_arguments(config, sde),
        out_dir=str(out_dir),
        device=device,
    )
    evaluator.prepare()
    with th.inference_mode():
        metrics = evaluator.evaluate(
            evaluated_network(model, config, weights),
            step=metadata["step"],
            weights=weights,
            final_artifacts=True,
        )
    # The row training appends to its curve, without the training work, which
    # the model file doesn't record. An image evaluation without metrics has none.
    if isinstance(metrics, dict):
        row = {"step": metadata["step"], "weights": weights, **evaluator.curve_fields(metrics)}
        with open(out_dir / "curve.jsonl", "w", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    return metrics
