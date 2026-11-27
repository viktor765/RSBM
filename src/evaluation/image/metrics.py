import contextlib
import hashlib
import inspect
import logging
import math
import os
import re
from urllib.parse import urlparse

import numpy as np
import torch as th
import torchmetrics.functional.image.lpips as lpips_module
import torchvision
from torch_fidelity.feature_extractor_inceptionv3 import URL_INCEPTION_V3
from torchmetrics.image.fid import FrechetInceptionDistance, NoTrainInceptionV3
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from ... import data_utils
from ..common import metric_library_versions, protocol_with_id, solver_settings
from .oob_metrics import OOB_DEFINITION, OOBStatistics, get_oob_thresholds

logger = logging.getLogger(__name__)

# The torch hub directory of the metric networks: their pretrained weights are
# read from data/checkpoints, where offline nodes need them pre-placed, and
# downloaded there otherwise.
METRIC_HUB_DIR = "data"
# Increment when a definition in this module changes what a metric measures.
IMAGE_METRIC_PROTOCOL_VERSION = 3
# torchmetrics' FID network: Inception v3 pool features, resized with antialiasing.
INCEPTION_SETTINGS = {
    "name": "inception-v3-compat",
    "features_list": ["2048"],
    "antialias": True,
}
INCEPTION_INPUT = (
    "unit cube, clamped to [0, 1] and rounded to 8-bit levels; "
    "one channel repeated to three"
)
KID_DEFINITION = {
    "kernel": "(x.y / d + 1)^3, d = 2048",
    "estimator": "unbiased MMD^2 over all pairs, without subsets",
}
LPIPS_SETTINGS = {"net_type": "alex", "reduction": "mean", "normalize": False}
# Rows of the kernel blocks KID sums, which bound its memory.
KERNEL_BLOCK = 4096


@contextlib.contextmanager
def metric_hub_dir():
    previous = th.hub.get_dir()
    th.hub.set_dir(METRIC_HUB_DIR)
    try:
        yield
    finally:
        th.hub.set_dir(previous)


def file_sha256(path) -> str:
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def hub_weights_identity(url) -> dict:
    """The file a torch hub download of url uses, checked against the hash in its name."""
    name = os.path.basename(urlparse(url).path)
    path = os.path.join(METRIC_HUB_DIR, "checkpoints", name)
    digest = file_sha256(path)
    prefix = re.search(r"-([0-9a-f]{8,})\.\w+$", name).group(1)
    if not digest.startswith(prefix):
        raise RuntimeError(f"{path} is not the published file {url}; replace it.")
    return {"file": name, "sha256": digest}


def inception_input(x: th.Tensor) -> th.Tensor:
    """Unit-cube images as the 8-bit RGB images the Inception network reads.

    Rounding, unlike torchmetrics' truncating conversion, returns every 8-bit
    dataset image to its own levels. The clamp is the clipping of unreflected
    samples; reflected ones lie in the cube up to float rounding.
    """
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    return th.round(x.clamp(0, 1) * 255).to(th.uint8)


class PrecomputedFeatures(th.nn.Module):
    """Lets FrechetInceptionDistance accumulate features computed elsewhere."""

    num_features = 2048

    def forward(self, features: th.Tensor) -> th.Tensor:
        return features


def polynomial_kernel_sum(x: th.Tensor, y: th.Tensor) -> th.Tensor:
    """The float64 sum of (x_i . y_j / d + 1)^3 over all rows i of x and j of y."""
    total = th.zeros((), dtype=th.float64, device=x.device)
    d = x.shape[1]
    for i in range(0, len(x), KERNEL_BLOCK):
        xi = x[i : i + KERNEL_BLOCK].double()
        for j in range(0, len(y), KERNEL_BLOCK):
            total += ((xi @ y[j : j + KERNEL_BLOCK].double().T / d + 1) ** 3).sum()
    return total


def within_set_kernel_mean(x: th.Tensor) -> th.Tensor:
    """The mean kernel value over pairs of distinct rows of x."""
    n = len(x)
    if n < 2:
        raise ValueError("KID needs at least two images in each set.")
    diagonal = ((x.double().square().sum(1) / x.shape[1] + 1) ** 3).sum()
    return (polynomial_kernel_sum(x, x) - diagonal) / (n * (n - 1))


class UnbiasedKID:
    """KID of image features against a fixed reference set of features.

    The unbiased estimate of the squared MMD (Binkowski et al., 2018) with
    the cubic polynomial kernel, over all pairs: within-reference mean plus
    within-generated mean minus twice the cross mean. The reference term is
    computed once.
    """

    def __init__(self, reference: th.Tensor):
        self.reference = reference
        self.reference_term = within_set_kernel_mean(reference)

    @property
    def n_reference(self) -> int:
        return len(self.reference)

    def __call__(self, generated: th.Tensor) -> float:
        n = self.n_reference * len(generated)
        cross = polynomial_kernel_sum(self.reference, generated) / n
        within = within_set_kernel_mean(generated)
        return float(self.reference_term + within - 2 * cross)


def finite_score(description: str, value) -> float | None:
    """value as a float, or None (logged) when it is not finite."""
    value = float(value)
    if not math.isfinite(value):
        logger.error("%s is %s; recorded as null.", description, value)
        return None
    return value


class ImageMetricEvaluator:
    """FID, KID, LPIPS and MSD of each direction's translations of the evaluation set.

    eval.compute_fid, eval.compute_kid, eval.compute_lpips and eval.compute_msd
    select them independently; FID and KID share one Inception feature pass.
    A selected metric is required: prepare raises if it cannot be set up. The
    sampler rejects non-finite generated values; a non-finite score is
    recorded as null. Out-of-bounds statistics of the same generated images
    accompany any selected metric.
    """

    def __init__(self, context):
        self.context = context
        self.compute_fid = bool(context.eval_conf.get("compute_fid", False))
        self.compute_kid = bool(context.eval_conf.get("compute_kid", False))
        self.compute_lpips = bool(context.eval_conf.get("compute_lpips", False))
        self.compute_msd = bool(context.eval_conf.get("compute_msd", True))
        self.oob_thresholds = get_oob_thresholds(context.eval_conf)
        self.inception = None
        self.fids = None
        self.kids = None
        self.lpips = None
        self.protocol = None

    def prepare(self) -> None:
        weights = {}
        with metric_hub_dir():
            if self.uses_inception:
                self.inception = NoTrainInceptionV3(**INCEPTION_SETTINGS).to(
                    self.context.device
                )
                weights["inception"] = hub_weights_identity(URL_INCEPTION_V3)
                self._init_references()
            if self.compute_lpips:
                self.lpips = self._init_lpips()
                weights["alexnet"] = hub_weights_identity(
                    torchvision.models.AlexNet_Weights.DEFAULT.url
                )
                linear = os.path.join(
                    os.path.dirname(inspect.getfile(lpips_module)),
                    "lpips_models",
                    f"{LPIPS_SETTINGS['net_type']}.pth",
                )
                weights["lpips_linear"] = {
                    "file": os.path.basename(linear),
                    "sha256": file_sha256(linear),
                }
        if self.enabled:
            self.protocol = self._protocol(weights)

    @property
    def enabled(self) -> bool:
        return self.uses_inception or self.compute_lpips or self.compute_msd

    @property
    def uses_inception(self) -> bool:
        return self.compute_fid or self.compute_kid

    def features(self, x_unit: th.Tensor) -> th.Tensor:
        """Inception features of unit-cube images."""
        with th.no_grad():
            return self.inception(inception_input(x_unit))

    def _init_references(self) -> None:
        """FID's and KID's reference statistics: each target training set."""
        if self.compute_fid:
            self.fids = {
                i: FrechetInceptionDistance(
                    feature=PrecomputedFeatures(),
                    sync_on_compute=False,
                    reset_real_features=False,
                ).to(self.context.device)
                for i in (0, 1)
            }
        reference = {}
        for i in (0, 1):
            batches = []
            for x, _ in self.context.train_data[i]:
                x = self.context.postprocessing[i](x.to(self.context.device))
                features = self.features(x)
                if self.fids is not None:
                    self.fids[i].update(features, real=True)
                if self.compute_kid:
                    batches.append(features)
            if self.compute_kid:
                reference[i] = UnbiasedKID(th.cat(batches))
        if self.compute_kid:
            self.kids = reference

    def _init_lpips(self):
        if not self.context.lpips_supported():
            raise ValueError("eval.compute_lpips requires RGB image datasets.")
        # Only the main process evaluates; other ranks cannot join metric sync.
        return LearnedPerceptualImagePatchSimilarity(
            **LPIPS_SETTINGS, sync_on_compute=False
        ).to(self.context.device)

    def _protocol(self, weights) -> dict:
        """What the scores of compute measure; its id changes with any of it."""
        context = self.context

        def evaluation_set(i, train_data):
            return data_utils.dataset_identity(
                context.data_conf, i, train_data, context.seed
            )

        metrics = {}
        if self.uses_inception:
            inception = {
                "reference": {f"ds{i}": evaluation_set(i, True) for i in (0, 1)},
                "network": INCEPTION_SETTINGS,
                "weights": {"inception": weights.get("inception")},
                "input": INCEPTION_INPUT,
                "generated_clip": [0.0, 1.0] if context.clip_generated else None,
            }
            if self.compute_fid:
                metrics["fid"] = inception
            if self.compute_kid:
                metrics["kid"] = {**inception, **KID_DEFINITION}
        if self.compute_lpips:
            metrics["lpips"] = {
                "settings": LPIPS_SETTINGS,
                "weights": {k: weights[k] for k in ("alexnet", "lpips_linear")},
                "pairs": "each generated image with its source",
                "input": "model coordinates clamped to [-1, 1]",
            }
        if self.compute_msd:
            metrics["msd"] = {
                "pairs": "each generated image with its source",
                "input": "unit cube, unclipped",
                "mean": "over pixel channels, then images",
            }
            if context.clip_generated:
                metrics["msd_clipped"] = {
                    **metrics["msd"],
                    "input": "unit cube, generated images clipped to [0, 1]",
                }
        metrics["oob"] = {
            **OOB_DEFINITION,
            "thresholds": list(self.oob_thresholds),
        }
        return protocol_with_id(
            {
                "kind": "image",
                "version": IMAGE_METRIC_PROTOCOL_VERSION,
                # ds<i>: one path from each test image of the other endpoint.
                "generated": {
                    f"ds{i}": {
                        "source": evaluation_set(1 - i, False),
                        "sampling_direction": sampling_direction,
                    }
                    for i, sampling_direction in ((0, -1), (1, 1))
                },
                "seed": context.seed,
                "noise_stream": "evaluation",
                "noise_draw": context.noise_draw,
                "sampling_batch_size": context.eval_batch_size,
                "solver": solver_settings(context.sde),
                "metrics": metrics,
                "libraries": metric_library_versions(),
            },
            device=str(context.device),
        )

    def compute(self, model: th.nn.Module, step: int) -> dict[str, dict]:
        """Scores of each direction ds<i>, with the numbers of images they used.

        n_generated counts the generated images (and LPIPS/MSD pairs), and
        n_reference the real images of FID and KID. oob holds the
        out-of-bounds statistics of the same generated images.
        """
        generator = self.context.new_generator()
        eval_metrics = {}
        for i, sampling_direction in zip((0, 1), (-1, 1), strict=True):
            fid = self.fids[i] if self.compute_fid else None
            if fid is not None:
                fid.reset()
            if self.lpips is not None:
                self.lpips.reset()

            msd = msd_clipped = 0.0
            oob = OOBStatistics()
            generated_features = []
            n_generated = 0
            for x, _ in self.context.test_data[1 - i]:
                x = x.to(self.context.device)
                # The sampler raises on a non-finite state.
                x_sampled = self.context.sde.forward_euler(
                    model,
                    x,
                    direction=sampling_direction,
                    generator=generator,
                )
                n_generated += x.shape[0]
                sampled_unit = self.context.postprocessing[i](x_sampled)
                oob.update(sampled_unit)
                if self.compute_msd:
                    # Unit-cube coordinates, averaged over pixels/channels.
                    x_unit = self.context.postprocessing[1 - i](x)
                    pixels = int(np.prod(x.shape[1:]))
                    msd += th.sum((sampled_unit - x_unit) ** 2) / pixels
                    if self.context.clip_generated:
                        msd_clipped += (
                            th.sum((sampled_unit.clamp(0, 1) - x_unit) ** 2) / pixels
                        )
                if self.uses_inception:
                    features = self.features(sampled_unit)
                    if fid is not None:
                        fid.update(features, real=False)
                    if self.compute_kid:
                        generated_features.append(features)
                if self.lpips is not None:
                    self.lpips.update(x_sampled.clamp(-1, 1), x.clamp(-1, 1))

            label = f"Step {step}: dataset_{i}"
            record = {}
            if fid is not None:
                record["fid"] = finite_score(f"{label} FID", fid.compute().item())
            if self.compute_kid:
                record["kid"] = finite_score(
                    f"{label} KID", self.kids[i](th.cat(generated_features))
                )
            if self.lpips is not None:
                record["lpips"] = finite_score(
                    f"{label} LPIPS", self.lpips.compute().item()
                )
            if self.compute_msd:
                record["msd"] = finite_score(f"{label} MSD", msd / n_generated)
                if self.context.clip_generated:
                    record["msd_clipped"] = finite_score(
                        f"{label} clipped MSD", msd_clipped / n_generated
                    )
            record["n_generated"] = n_generated
            if fid is not None:
                record["n_reference"] = int(fid.real_features_num_samples)
            elif self.compute_kid:
                record["n_reference"] = self.kids[i].n_reference
            record["oob"] = oob.summary(self.oob_thresholds)
            logger.info(
                "%s %s",
                label,
                ", ".join(
                    f"{name}: {record[key]:.6g}"
                    for key, name in (
                        ("fid", "FID"),
                        ("kid", "KID"),
                        ("msd", "MSD"),
                        ("msd_clipped", "clipped MSD"),
                        ("lpips", "LPIPS"),
                    )
                    if record.get(key) is not None
                ),
            )
            logger.info(
                "%s out of bounds: %.6g of pixels, %.6g of images",
                label,
                record["oob"]["total_oob_pixel_fraction"],
                record["oob"]["fraction_images_with_any_oob"],
            )
            eval_metrics[f"ds{i}"] = record
        return eval_metrics

