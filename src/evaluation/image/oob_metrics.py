"""Out-of-bounds statistics of generated images in unit-cube coordinates.

A pixel is one pixel channel value. It is out of bounds below 0 or above 1,
and its violation magnitude is its distance to [0, 1]. Images are the sampling
units: per-image counts and maxima are summarized over images.
"""

from typing import Any

import numpy as np
import torch as th

DEFAULT_OOB_THRESHOLDS = (0.0, 0.001, 0.01, 0.05, 0.1)
OOB_DEFINITION = {
    "input": "generated images in the unit cube, unclipped",
    "pixel": "one pixel channel value",
    "out_of_bounds": "below 0 or above 1",
    "magnitude": "distance to [0, 1]",
    "oob_magnitude_mean": "over out-of-bounds pixels; null without any",
    "fraction_images_oob_fraction_ge_threshold": (
        "images whose out-of-bounds pixel fraction is at least the threshold, "
        "or above 0 for threshold 0"
    ),
    "percentiles": "over images, numpy linear interpolation",
}


def get_oob_thresholds(eval_conf) -> tuple[float, ...]:
    thresholds = tuple(
        float(threshold)
        for threshold in eval_conf.get("oob_thresholds", DEFAULT_OOB_THRESHOLDS)
    )
    if any(threshold < 0 for threshold in thresholds):
        raise ValueError(f"eval.oob_thresholds must be non-negative: {thresholds}")
    return thresholds


class OOBStatistics:
    """Per-image out-of-bounds counts and magnitudes, accumulated over batches."""

    def __init__(self):
        self.pixels_per_image = None
        self.below = []
        self.above = []
        self.magnitude_sum = []
        self.magnitude_max = []

    def update(self, generated_pixels: th.Tensor) -> None:
        """Add generated images (N, C, H, W) in unit-cube coordinates."""
        if generated_pixels.ndim != 4:
            raise ValueError(
                f"Expected generated pixels with shape (N,C,H,W), got {generated_pixels.shape}"
            )
        pixels_per_image = int(np.prod(generated_pixels.shape[1:]))
        if self.pixels_per_image not in (None, pixels_per_image):
            raise ValueError("Generated images of one evaluation differ in size.")
        self.pixels_per_image = pixels_per_image
        x = generated_pixels.detach().flatten(1).to(dtype=th.float32)
        magnitude = (-x).clamp(min=0) + (x - 1).clamp(min=0)
        self.below.append((x < 0).sum(dim=1).cpu())
        self.above.append((x > 1).sum(dim=1).cpu())
        self.magnitude_sum.append(magnitude.sum(dim=1).cpu())
        self.magnitude_max.append(magnitude.amax(dim=1).cpu())

    def summary(
        self, thresholds: list[float] | tuple[float, ...] = DEFAULT_OOB_THRESHOLDS
    ) -> dict[str, Any]:
        if not self.below:
            raise ValueError("No generated images to summarize.")
        below = th.cat(self.below).numpy().astype(np.int64)
        above = th.cat(self.above).numpy().astype(np.int64)
        magnitude_sum = th.cat(self.magnitude_sum).numpy().astype(np.float64)
        per_image_max = th.cat(self.magnitude_max).numpy().astype(np.float64)
        oob_per_image = below + above
        oob_fraction_per_image = oob_per_image / self.pixels_per_image
        n_images = int(oob_per_image.size)
        total_values = n_images * self.pixels_per_image
        total_oob = int(oob_per_image.sum())

        threshold_metrics = {}
        for threshold in thresholds:
            threshold = float(threshold)
            if threshold <= 0:
                hits = oob_fraction_per_image > 0
            else:
                hits = oob_fraction_per_image >= threshold
            threshold_metrics[f"{threshold:g}"] = float(hits.mean())

        return {
            "n_images": n_images,
            "pixels_per_image": self.pixels_per_image,
            "total_pixel_values": total_values,
            "total_oob_pixels": total_oob,
            "total_oob_pixel_fraction": total_oob / total_values,
            "below_zero_pixels": int(below.sum()),
            "below_zero_pixel_fraction": int(below.sum()) / total_values,
            "above_one_pixels": int(above.sum()),
            "above_one_pixel_fraction": int(above.sum()) / total_values,
            "oob_pixels_per_image_mean": float(oob_per_image.mean()),
            "oob_pixels_per_image_median": float(np.median(oob_per_image)),
            "oob_pixels_per_image_max": int(oob_per_image.max()),
            "oob_magnitude_mean": (
                float(magnitude_sum.sum() / total_oob) if total_oob else None
            ),
            "per_image_max_oob_magnitude_mean": float(per_image_max.mean()),
            "per_image_max_oob_magnitude_median": float(np.median(per_image_max)),
            "per_image_max_oob_magnitude_p95": float(np.percentile(per_image_max, 95)),
            "per_image_max_oob_magnitude_max": float(per_image_max.max()),
            "fraction_images_with_any_oob": float((oob_per_image > 0).mean()),
            "fraction_images_oob_fraction_ge_threshold": threshold_metrics,
        }
