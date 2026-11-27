"""Weights-only export of a trained network, self-contained in one file.

The file holds the state dict of one evaluated weight set in safetensors format
and, in its metadata, the resolved experiment config needed to rebuild the network
(and the SDE and data normalization it was trained with). Unlike a training
checkpoint, it cannot resume training.
"""

from omegaconf import DictConfig, OmegaConf
from safetensors import safe_open
from safetensors.torch import load_model, save_model
from torch import nn

FORMAT = "1"
# The weight sets a run can evaluate: its EMA and its raw (trained) weights.
WEIGHTS = ("ema", "raw")


def final_model_file_name(weights: str) -> str:
    """The run's export of one evaluated weight set, under checkpoints/."""
    return f"final_model_{weights}.safetensors"


def save_final_model(
    path: str,
    model: nn.Module,
    config: DictConfig,
    *,
    step: int,
    weights: str,
) -> None:
    """Save evaluated weights with public experiment settings.

    `weights` is "ema" or "raw". Detailed launch evidence stays in the run's
    private .hydra/ records, separate from the publishable model file.
    """
    if weights not in WEIGHTS:
        raise ValueError(f"weights must be 'ema' or 'raw', got {weights!r}.")
    # out_dir cannot resolve outside Hydra. Resolve the other settings before
    # selecting the public fields, so references to run_name/output_root work.
    resolved = OmegaConf.create(
        OmegaConf.to_container(
            OmegaConf.masked_copy(config, [key for key in config if key != "out_dir"]),
            resolve=True,
        )
    )
    config = OmegaConf.masked_copy(
        resolved,
        [
            "model", "sde", "data", "training", "eval", "method", "seed",
            "experiment_short_name",
        ],
    )
    config.training.pop("resume", None)
    save_model(
        model,
        path,
        metadata={
            "format": FORMAT,
            "step": str(step),
            "weights": weights,
            "config": OmegaConf.to_yaml(config, resolve=True),
        },
    )


def read_final_model_metadata(path: str) -> dict:
    """Return format, step, weights and config, without loading the weights."""
    with safe_open(path, framework="pt") as f:
        metadata = f.metadata()
    if metadata is None or metadata.get("format") != FORMAT:
        raise ValueError(
            f"{path} is not a final model file of format {FORMAT}."
        )
    return {
        "format": metadata["format"],
        "step": int(metadata["step"]),
        "weights": metadata["weights"],
        "config": OmegaConf.create(metadata["config"]),
    }


def load_final_model(path: str, device="cpu") -> tuple[nn.Module, DictConfig]:
    """Rebuild the model from the stored config and load its weights.

    The model outputs the drift, sde.sigma * Z for model.output = control, and
    is returned in eval mode, as both methods evaluate it.
    """
    from . import create_model

    config = read_final_model_metadata(path)["config"]
    model = create_model(config.model, config.sde.sigma)
    load_model(model, path, strict=True)
    return model.to(device).eval(), config
