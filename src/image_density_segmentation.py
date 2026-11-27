"""The Oresund Bridge photo's density: pixel classes from colour thresholds, one
weight per class and a vertical gradient on the sky."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

CLASS_SKY = 0
CLASS_WATER = 1
CLASS_LIGHT_BRIDGE = 2
CLASS_DARK_BRIDGE = 3


@dataclass(frozen=True)
class BridgeThresholdMaskConfig:
    # Rows and columns are given for an image of the reference size and scaled.
    reference_height: int = 288
    reference_width: int = 384
    light_bridge_red_threshold: float = 0.70
    light_bridge_col_start: int = 175
    sky_row_stop: int = 238
    water_row_start: int = 235
    dark_bridge_exclude_lower_left_row_start: int = 200
    dark_bridge_exclude_lower_left_col_stop: int = 175
    water_blue_threshold: float = 0.50
    dark_bridge_blue_threshold: float = 0.52


@dataclass(frozen=True)
class BridgeGradientConfig:
    sky_top_multiplier: float = 1.8
    sky_bottom_multiplier: float = 0.0
    sky_curve: float = 0.5
    preserve_class_mean: bool = False


@dataclass(frozen=True)
class BridgeClassWeights:
    sky: float = 0.05
    water: float = 0.75
    light_bridge: float = 1.50
    dark_bridge: float = 5.00

    def as_array(self) -> np.ndarray:
        return np.asarray(
            [self.sky, self.water, self.light_bridge, self.dark_bridge],
            dtype=np.float32,
        )


def load_aspect_image(path: str | Path, image_height: int = 288) -> Image.Image:
    image = Image.open(path).convert("RGB")
    aspect = image.width / image.height
    width = int(round(image_height * aspect))
    return image.resize((width, image_height), Image.Resampling.LANCZOS)


def _scaled_row(row: int, height: int, reference_height: int) -> int:
    return int(np.clip(round(row * height / reference_height), 0, height))


def _scaled_col(col: int, width: int, reference_width: int) -> int:
    return int(np.clip(round(col * width / reference_width), 0, width))


def nearest_valid_classes(
    mask_stack: np.ndarray,
    class_ids: tuple[int, ...],
    *,
    fallback_class: int = CLASS_SKY,
) -> np.ndarray:
    """Pixels in exactly one mask take its class; the rest take the nearest such pixel's."""
    masks = np.asarray(mask_stack, dtype=bool)
    if masks.ndim != 3:
        raise ValueError(
            f"Expected mask stack with shape (n_classes, H, W), got {masks.shape}"
        )
    if masks.shape[0] != len(class_ids):
        raise ValueError(f"Expected {masks.shape[0]} class ids, got {len(class_ids)}")

    coverage = masks.sum(axis=0)
    classes = np.full(masks.shape[1:], -1, dtype=np.int16)
    valid = coverage == 1
    for mask, class_id in zip(masks, class_ids):
        classes[valid & mask] = class_id

    if valid.any():
        _, indices = distance_transform_edt(~valid, return_indices=True)
        invalid = ~valid
        classes[invalid] = classes[tuple(indices[:, invalid])]
    else:
        classes[:, :] = fallback_class

    return classes


def bridge_classes(
    image: Image.Image,
    config: BridgeThresholdMaskConfig | None = None,
) -> np.ndarray:
    """Each pixel's class: sky, water, light bridge or dark bridge."""
    if config is None:
        config = BridgeThresholdMaskConfig()
    if image.mode != "RGB":
        image = image.convert("RGB")

    rgb = np.asarray(image, dtype=np.float32) / 255.0
    height, width, _ = rgb.shape

    sky_row_stop = _scaled_row(config.sky_row_stop, height, config.reference_height)
    water_row_start = _scaled_row(
        config.water_row_start, height, config.reference_height
    )
    light_bridge_col_start = _scaled_col(
        config.light_bridge_col_start, width, config.reference_width
    )
    dark_bridge_exclude_row_start = _scaled_row(
        config.dark_bridge_exclude_lower_left_row_start,
        height,
        config.reference_height,
    )
    dark_bridge_exclude_col_stop = _scaled_col(
        config.dark_bridge_exclude_lower_left_col_stop,
        width,
        config.reference_width,
    )

    light_bridge_mask0 = rgb[..., 0] > config.light_bridge_red_threshold
    light_bridge_mask1 = np.zeros((height, width), dtype=bool)
    light_bridge_mask1[:, light_bridge_col_start:] = True
    light_bridge_mask = light_bridge_mask0 & light_bridge_mask1

    sky_mask0 = np.zeros((height, width), dtype=bool)
    sky_mask0[:sky_row_stop, :] = True

    water_mask0 = np.zeros((height, width), dtype=bool)
    water_mask0[water_row_start:, :] = True
    water_mask1 = rgb[..., 2] < config.water_blue_threshold
    water_mask = water_mask0 & water_mask1

    dark_bridge_mask1 = rgb[..., 2] < config.dark_bridge_blue_threshold
    dark_bridge_mask = sky_mask0 & dark_bridge_mask1
    dark_bridge_exclude = np.zeros((height, width), dtype=bool)
    dark_bridge_exclude[
        dark_bridge_exclude_row_start:, :dark_bridge_exclude_col_stop
    ] = True
    dark_bridge_mask = dark_bridge_mask & (~dark_bridge_exclude)

    sky_mask = sky_mask0 & ~(dark_bridge_mask | light_bridge_mask)

    mask_stack = np.stack(
        [sky_mask, water_mask, light_bridge_mask, dark_bridge_mask],
        axis=0,
    )
    return nearest_valid_classes(
        mask_stack,
        (CLASS_SKY, CLASS_WATER, CLASS_LIGHT_BRIDGE, CLASS_DARK_BRIDGE),
    )


def class_weight_map(classes: np.ndarray, weights: BridgeClassWeights) -> np.ndarray:
    result = weights.as_array()[np.asarray(classes, dtype=np.int16)]
    return np.asarray(result, dtype=np.float32)


def apply_class_gradients(
    weights: np.ndarray,
    classes: np.ndarray,
    config: BridgeGradientConfig | None = None,
) -> np.ndarray:
    """Scale sky weights from sky_top_multiplier at the top row to
    sky_bottom_multiplier at the bottom, along row_fraction**sky_curve."""
    if config is None:
        config = BridgeGradientConfig()

    weights = np.asarray(weights, dtype=np.float32)
    classes = np.asarray(classes, dtype=np.int16)
    if weights.shape != classes.shape:
        raise ValueError(
            f"Weight map and class map must have same shape, got {weights.shape} and {classes.shape}"
        )

    height = weights.shape[0]
    row_fraction = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    sky_gradient = config.sky_top_multiplier + (
        config.sky_bottom_multiplier - config.sky_top_multiplier
    ) * np.power(row_fraction, config.sky_curve)
    sky_gradient = np.broadcast_to(sky_gradient, weights.shape).astype(np.float32)

    multipliers = np.ones_like(weights, dtype=np.float32)
    sky_mask = classes == CLASS_SKY
    if sky_mask.any():
        if config.preserve_class_mean:
            sky_mean = float(sky_gradient[sky_mask].mean())
            if sky_mean > 0:
                sky_gradient = sky_gradient / sky_mean
        multipliers[sky_mask] = sky_gradient[sky_mask]

    return np.asarray(weights * multipliers, dtype=np.float32)


def probability_from_weights(weights: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(weights, dtype=np.float64)
    probabilities = np.clip(probabilities, 0.0, None)
    total = probabilities.sum(dtype=np.float64)
    if not np.isfinite(total) or total <= 0:
        raise ValueError(f"Weight map must have positive finite sum, got {total}")
    return probabilities / total
