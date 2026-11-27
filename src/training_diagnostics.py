"""Pre-step numerical guards and versioned training diagnostic windows."""

import json
import math
from pathlib import Path

import torch

DIRECTION_GROUPS = {+1: "forward", -1: "backward"}


def gradient_norm_limit(max_grad_norm):
    if max_grad_norm is None:
        return float("inf")
    if not math.isfinite(max_grad_norm) or max_grad_norm < 0:
        raise ValueError(
            "training.max_grad_norm must be finite and nonnegative, or null."
        )
    return max_grad_norm


def require_finite(values, context, accelerator=None):
    """All ranks reject together, including before a DDP backward collective."""
    invalid = (~torch.isfinite(values.detach())).any().to(torch.int32)
    if accelerator is not None:
        invalid = accelerator.reduce(invalid, reduction="sum")
    if invalid.item():
        raise FloatingPointError(f"Nonfinite {context}; optimizer update aborted.")


def gradient_diagnostics(norm, max_norm):
    return {
        "grad_norm": norm.detach(),
        "clipped_fraction": (
            norm > max_norm if max_norm is not None else torch.zeros_like(norm)
        ).float(),
    }


def accumulate_diagnostics(totals, diagnostics):
    for group, values in diagnostics.items():
        if group not in totals:
            totals[group] = {
                name: torch.zeros_like(value.detach()).float()
                for name, value in values.items()
            }
            totals[group]["steps"] = torch.zeros_like(
                next(iter(values.values()))
            ).float()
        for name, value in values.items():
            totals[group][name].add_(value.detach().float())
        totals[group]["steps"].add_(1)


def summarize_diagnostics(totals, accelerator=None):
    summaries = {}
    for group, values in totals.items():
        names = [name for name in values if name != "steps"]
        means = torch.stack([values[name] / values["steps"] for name in names])
        if accelerator is not None:
            means = accelerator.reduce(means, reduction="mean")
        summaries[group] = dict(zip(names, means.tolist(), strict=True))
        summaries[group]["steps"] = values["steps"].item()
    totals.clear()
    return summaries


def append_diagnostics(out_dir, *, method, step, phase, max_grad_norm, summaries):
    """Write window means using the same group names as the step diagnostics."""
    record = {
        "schema_version": 1,
        "method": method,
        "step": step,
        "start_step": step
        - int(
            sum(values["steps"] for values in summaries.values())
            if method == "rsb_fbsde"
            else max(values["steps"] for values in summaries.values())
        )
        + 1,
        "phase": phase,
        "max_grad_norm": max_grad_norm,
        "groups": summaries,
    }
    path = Path(out_dir) / "training_diagnostics.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, allow_nan=False) + "\n")
