import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch as th
from omegaconf import OmegaConf
from PIL import Image, ImageFilter, ImageOps
from torchvision import datasets, transforms

from . import utils
from .rsbm import normalizations


def resize_aspect(image: Image.Image, image_height: int) -> Image.Image:
    aspect = image.width / image.height
    width = int(round(image_height * aspect))
    return image.resize((width, image_height), Image.Resampling.LANCZOS)


def _transpose_emnist(image: th.Tensor) -> th.Tensor:
    """Correct EMNIST orientation with a callable picklable by spawned workers."""
    return image.transpose(1, 2)


def postprocessing(data_conf) -> normalizations.AffineBox:
    """Map the configured sampling domain to the unit cube without clipping.

    Domain bounds describe data after preprocessing. A single bound applies
    to every coordinate, including pixels/channels in image tensors.
    """
    domain = data_conf.get("domain")
    if domain is None:
        raise ValueError(f"{data_conf.dataset} requires domain.lower and domain.upper.")
    return normalizations.AffineBox(domain.lower, domain.upper)


def create_dataloaders(
    data_conf,
    batch_size: int,
    train_data: bool,
    for_training: bool,
    seed: int,
    dl_generator: th.Generator | None = None,
    num_workers: int | None = None,
) -> dict[int, th.utils.data.DataLoader]:
    """Loaders of both endpoints' datasets.

    dl_generator orders the batches and seeds the workers; torch seeds each
    worker's torch, numpy and random from it.
    """
    data_loaders = {}
    for i in [0, 1]:
        dataset = get_dataset(data_conf, i, train_data, for_training, seed)
        if len(dataset) == 0:
            raise ValueError(f"Dataset {data_conf[f'data{i}'].dataset} is empty.")
        loader_batch_size = batch_size
        if len(dataset) < loader_batch_size and not for_training:
            loader_batch_size = len(dataset)
        if len(dataset) < loader_batch_size:
            raise ValueError(
                f"Dataset {data_conf[f'data{i}'].dataset} size {len(dataset)} "
                f"is smaller than batch size {loader_batch_size}."
            )
        if num_workers is None:
            num_workers_int: int = data_conf.num_workers
        else:
            num_workers_int = num_workers
        data_loaders[i] = th.utils.data.DataLoader(
            dataset,
            batch_size=loader_batch_size,
            shuffle=for_training,
            num_workers=num_workers_int,
            pin_memory=data_conf.pin_memory,
            drop_last=for_training,
            generator=dl_generator,
        )
    return data_loaders


def rebatch_training_dataloaders(dataloaders, batch_size: int, generator):
    """Reuse training datasets with a new batch size and sampling stream."""
    return {
        direction: th.utils.data.DataLoader(
            loader.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=loader.num_workers,
            pin_memory=loader.pin_memory,
            drop_last=True,
            generator=generator,
        )
        for direction, loader in dataloaders.items()
    }


class PreloadedImageFolder(datasets.ImageFolder):
    """ImageFolder that decodes and resizes its images once, into memory.

    The transform must start with a deterministic Resize to `size`. The cached
    images are that Resize's output, which the full transform then resizes to
    their own size, a copy, so each item is the same tensor as from disk.
    """

    def preload(self, indices, size):
        resize = transforms.Resize(size)
        paths = [self.samples[i][0] for i in indices]
        file_loader = self.loader
        # PIL releases the GIL while decoding and resizing.
        with ThreadPoolExecutor(max_workers=min(32, os.cpu_count() or 1)) as pool:
            images = pool.map(lambda path: resize(file_loader(path)), paths)
            self.cache = dict(zip(paths, images))
        self.loader = self.cache.__getitem__


def _filter_dataset_by_labels(
    dataset, labels: list[str] | None
) -> th.utils.data.Dataset:
    if labels is None:
        return dataset
    label_indices = set(dataset.class_to_idx[label] for label in labels)
    indices = [
        i for i, target in enumerate(dataset.targets) if int(target) in label_indices
    ]
    return th.utils.data.Subset(dataset, indices)


TOY_DATASETS = {"checkerboard", "uniform_box", "image_density"}


def _toy_dataset_seed(seed: int, index: int, train_data: bool) -> int:
    return utils.derive_seed(seed, "dataset", index, "train" if train_data else "test")


def dataset_identity(data_conf, index: int, train_data: bool, seed: int) -> dict:
    """Everything that fixes the points of get_dataset(..., for_training=False):
    the endpoint's settings and split, and for synthetic sets their seed."""
    conf = OmegaConf.to_container(data_conf[f"data{index}"], resolve=True)
    identity = {"data": conf, "split": "train" if train_data else "test"}
    if conf["dataset"] in TOY_DATASETS:
        identity["seed"] = _toy_dataset_seed(seed, index, train_data)
    return identity


def get_dataset(
    data_conf, index: int, train_data: bool, for_training: bool, seed: int
) -> th.utils.data.Dataset:
    """Endpoint index's training or evaluation set.

    Synthetic sets are drawn from their own stream of the run seed, so every
    caller with the same seed gets the same points.
    """
    conf = data_conf[f"data{index}"]
    dataset_name = conf.dataset
    root = conf.get("root", "data")
    if dataset_name == "mnist":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5,), (0.5,)
                ),  # scale to ([0, 1] - .5) / .5 = [-1, 1]
            ]
        )
        dataset = datasets.MNIST(
            root=root, train=train_data, download=True, transform=transform
        )
        return _filter_dataset_by_labels(dataset, conf.labels)
    elif dataset_name == "emnist":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(_transpose_emnist),
                transforms.Normalize((0.5,), (0.5,)),  # scale to [-1, 1]
            ]
        )
        dataset = datasets.EMNIST(
            root=root,
            split="letters",
            train=train_data,
            download=True,
            transform=transform,
        )
        return _filter_dataset_by_labels(dataset, conf.labels)
    elif dataset_name == "afhq64":
        if for_training:
            transf_comp = transforms.Compose(
                [
                    transforms.Resize((64, 64)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,)),
                ]
            )
        else:
            transf_comp = transforms.Compose(
                [
                    transforms.Resize((64, 64)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,)),
                ]
            )
        folder = PreloadedImageFolder(
            root=os.path.join("data", "AFHQ", "train" if train_data else "test"),
            transform=transf_comp,
        )
        dataset = _filter_dataset_by_labels(folder, conf.labels)
        folder.preload(getattr(dataset, "indices", range(len(folder))), (64, 64))
        return dataset
    elif dataset_name in TOY_DATASETS:
        return get_toy_dataset(conf, _toy_dataset_seed(seed, index, train_data))
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


# Shared by checkerboard sampling and its deterministic target quadrature.
_CHECKERBOARD_CELLS_PER_AXIS = 4
_CHECKERBOARD_OCCUPIED_CELLS = np.asarray(
    [
        (ix, iy)
        for ix in range(_CHECKERBOARD_CELLS_PER_AXIS)
        for iy in range(_CHECKERBOARD_CELLS_PER_AXIS)
        if (ix + iy) % 2 == 1
    ],
    dtype=np.int64,
)


def checkerboard_cells(phase: int = 0) -> np.ndarray:
    """Occupied (ix, iy) cells; phase 1 is the complement of phase 0."""
    if phase == 0:
        return _CHECKERBOARD_OCCUPIED_CELLS
    if phase != 1:
        raise ValueError(f"Checkerboard phase must be 0 or 1, got {phase}.")
    occupied = {tuple(cell) for cell in _CHECKERBOARD_OCCUPIED_CELLS}
    return np.asarray(
        [
            (ix, iy)
            for ix in range(_CHECKERBOARD_CELLS_PER_AXIS)
            for iy in range(_CHECKERBOARD_CELLS_PER_AXIS)
            if (ix, iy) not in occupied
        ],
        dtype=np.int64,
    )


def make_checkerboard(n_samples: int, rng: np.random.Generator, phase: int = 0):
    """Sample the eight equally weighted occupied cells on the unit square."""
    cells = checkerboard_cells(phase)
    x = rng.random((n_samples, 2))
    occupied_y = np.asarray(
        [cells[cells[:, 0] == ix, 1] for ix in range(_CHECKERBOARD_CELLS_PER_AXIS)]
    )
    cell_x = np.floor(x[:, 0] * _CHECKERBOARD_CELLS_PER_AXIS).astype(np.int64)
    cell_choice = rng.integers(occupied_y.shape[1], size=n_samples)
    x[:, 1] = (x[:, 1] + occupied_y[cell_x, cell_choice]) / _CHECKERBOARD_CELLS_PER_AXIS
    return x


def sample_unit_square_toy(conf, n_samples: int, rng: np.random.Generator):
    """Draw a uniform_box or checkerboard endpoint in unit-square coordinates."""
    if conf.dataset == "checkerboard":
        return make_checkerboard(n_samples, rng, conf.get("phase", 0))
    if conf.dataset == "uniform_box":
        return rng.random((n_samples, 2))
    raise ValueError(f"No unit-square sampler for {conf.dataset}.")


@dataclass(frozen=True)
class KnownTargetGrid:
    """Unit-square quadrature and its W2 error bound for a known target."""

    points: np.ndarray
    target_discretization_error_upper_bound: float


def make_known_target_grid(
    dataset: str,
    subdivisions_per_checker_cell: int,
    phase: int = 0,
) -> KnownTargetGrid | None:
    """Discretize a uniform box or the sampler's checkerboard definition."""
    k = int(subdivisions_per_checker_cell)
    if k < 1:
        raise ValueError("subdivisions_per_checker_cell must be positive.")
    if dataset not in {"uniform_box", "checkerboard"}:
        return None

    cells_per_axis = _CHECKERBOARD_CELLS_PER_AXIS * k
    coordinates = (np.arange(cells_per_axis, dtype=np.float64) + 0.5) / cells_per_axis
    x, y = np.meshgrid(coordinates, coordinates, indexing="xy")
    points = np.column_stack([x.ravel(), y.ravel()])
    if dataset == "checkerboard":
        occupied = np.zeros(
            (_CHECKERBOARD_CELLS_PER_AXIS, _CHECKERBOARD_CELLS_PER_AXIS), dtype=bool
        )
        cells = checkerboard_cells(phase)
        occupied[cells[:, 1], cells[:, 0]] = True
        ix, iy = np.meshgrid(
            np.arange(cells_per_axis), np.arange(cells_per_axis), indexing="xy"
        )
        points = points[occupied[iy.ravel() // k, ix.ravel() // k]]

    return KnownTargetGrid(
        points=points,
        target_discretization_error_upper_bound=1.0 / (cells_per_axis * math.sqrt(6)),
    )


def known_target_cell_masses(conf, cells_per_axis: int) -> np.ndarray:
    """Exact target mass of each cell of a cells x cells grid on the unit square.

    Indexed [ix, iy], with ix along the first and iy along the second unit
    coordinate.
    """
    n = int(cells_per_axis)
    if n < 1:
        raise ValueError("cells_per_axis must be positive.")
    if conf.dataset == "uniform_box":
        return np.full((n, n), 1.0 / n**2)
    if conf.dataset == "checkerboard":
        if n % _CHECKERBOARD_CELLS_PER_AXIS:
            raise ValueError(
                f"cells_per_axis must be a multiple of {_CHECKERBOARD_CELLS_PER_AXIS}."
            )
        checker = np.zeros((_CHECKERBOARD_CELLS_PER_AXIS,) * 2)
        cells = checkerboard_cells(conf.get("phase", 0))
        checker[cells[:, 0], cells[:, 1]] = 1.0 / len(cells)
        k = n // _CHECKERBOARD_CELLS_PER_AXIS
        return np.kron(checker, np.ones((k, k))) / k**2
    if conf.dataset == "image_density":
        # make_image_density: uniform within pixels; column j covers
        # x in [j/w, (j+1)/w], row i (top first) covers y in [1-(i+1)/h, 1-i/h].
        pixels = _load_density_image(conf)
        height, width = pixels.shape

        def overlap(m):  # [pixel, cell]: the fraction of each pixel in each cell
            pixel = np.arange(m)[:, None] / m
            cell = np.arange(n)[None, :] / n
            length = np.minimum(pixel + 1 / m, cell + 1 / n) - np.maximum(pixel, cell)
            return np.clip(length, 0.0, None) * m

        return overlap(width).T @ pixels.T @ overlap(height)[::-1]
    raise ValueError(f"No cell masses for toy dataset {conf.dataset!r}.")


def make_uniform_box(conf, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    domain = conf.get("domain")
    if domain is None:
        raise ValueError("uniform_box requires domain.lower and domain.upper.")
    lower = np.asarray(domain.lower, dtype=np.float64)
    upper = np.asarray(domain.upper, dtype=np.float64)
    if lower.ndim != 1 or upper.shape != lower.shape or np.any(upper <= lower):
        raise ValueError("uniform_box domain bounds must be matching valid vectors.")
    return rng.uniform(lower, upper, size=(n_samples, lower.size))


def _resolve_repo_path(path: str | os.PathLike) -> Path:
    path = Path(path)
    if path.is_absolute():
        if path.exists():
            return path
        raise FileNotFoundError(f"Could not resolve path: {path}")

    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path

    repo_path = Path(__file__).resolve().parents[1] / path
    if repo_path.exists():
        return repo_path

    raise FileNotFoundError(f"Could not resolve path: {path}")


def _load_density_image(conf) -> np.ndarray:
    density = conf.get("density", "darkness")
    if density == "bridge_segmentation":
        return _load_bridge_segmentation_density(conf)
    if density != "darkness":
        raise ValueError(f"Unknown image density type: {density}")

    image_path = _resolve_repo_path(conf.image_path)
    image_size = int(conf.get("image_size", 256))
    image = Image.open(image_path).convert("L")

    preprocessing = conf.get("preprocessing", "resize")
    if preprocessing == "resize":
        image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
    elif preprocessing == "resize_aspect":
        image_height = int(conf.get("image_height", image_size))
        image = resize_aspect(image, image_height)
    elif preprocessing == "center_crop":
        centering = tuple(conf.get("centering", [0.5, 0.5]))
        image = ImageOps.fit(
            image,
            (image_size, image_size),
            method=Image.Resampling.LANCZOS,
            centering=centering,
        )
    else:
        raise ValueError(f"Unknown image preprocessing: {preprocessing}")

    if conf.get("autocontrast", True):
        image = ImageOps.autocontrast(image, cutoff=float(conf.get("cutoff", 1.0)))

    darkness = 1.0 - np.asarray(image, dtype=np.float32) / 255.0
    blur = float(conf.get("blur", 0.0))
    if blur > 0:
        darkness_image = Image.fromarray(np.uint8(np.clip(darkness, 0.0, 1.0) * 255))
        darkness_image = darkness_image.filter(ImageFilter.GaussianBlur(radius=blur))
        darkness = np.asarray(darkness_image, dtype=np.float32) / 255.0

    eps = float(conf.get("eps", 0.0))
    gamma = float(conf.get("gamma", 1.0))
    weights = eps + np.power(np.clip(darkness, 0.0, 1.0), gamma)
    weights = np.asarray(weights, dtype=np.float64)
    weights_sum = weights.sum(dtype=np.float64)
    if not np.isfinite(weights_sum) or weights_sum <= 0:
        raise ValueError(
            f"Image density weights must have positive finite sum, got {weights_sum}"
        )
    return weights / weights_sum


def _config_dict(conf_section) -> dict:
    if conf_section is None:
        return {}
    return {str(key): value for key, value in conf_section.items()}


def _load_bridge_segmentation_density(conf) -> np.ndarray:
    from .image_density_segmentation import (
        BridgeClassWeights,
        BridgeGradientConfig,
        BridgeThresholdMaskConfig,
        apply_class_gradients,
        bridge_classes,
        class_weight_map,
        load_aspect_image,
        probability_from_weights,
    )

    image_path = _resolve_repo_path(conf.image_path)
    image_height = int(conf.get("image_height", conf.get("image_size", 288)))
    image = load_aspect_image(image_path, image_height=image_height)

    threshold_config = BridgeThresholdMaskConfig(
        **_config_dict(conf.get("threshold_mask", None))
    )
    classes = bridge_classes(image, threshold_config)

    class_weights = BridgeClassWeights(**_config_dict(conf.get("class_weights", None)))
    weights = class_weight_map(classes, class_weights)
    gradient_config = BridgeGradientConfig(**_config_dict(conf.get("gradient", None)))
    weights = apply_class_gradients(weights, classes, gradient_config)

    return probability_from_weights(weights)


def make_image_density(conf, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    density = _load_density_image(conf)
    height, width = density.shape
    probabilities = density.ravel().astype(np.float64)
    probabilities /= probabilities.sum(dtype=np.float64)

    indices = rng.choice(height * width, size=n_samples, p=probabilities)
    pixel_y, pixel_x = np.divmod(indices, width)

    x = (pixel_x + rng.random(n_samples)) / width
    y = 1.0 - (pixel_y + rng.random(n_samples)) / height
    return np.column_stack([x, y])


def get_toy_dataset(conf, seed: int):
    dataset_name = conf.dataset
    # The training and evaluation sets both have n_samples points.
    n_samples = int(conf.get("n_samples", 0))
    if n_samples <= 0:
        raise ValueError(f"{dataset_name} requires a positive n_samples value.")

    if dataset_name == "checkerboard":
        x = make_checkerboard(n_samples, np.random.default_rng(seed), conf.get("phase", 0))
    elif dataset_name == "uniform_box":
        x = make_uniform_box(conf, n_samples, np.random.default_rng(seed))
    elif dataset_name == "image_density":
        x = make_image_density(conf, n_samples, np.random.default_rng(seed))
    else:
        raise NotImplementedError(f"Toy dataset {dataset_name} not implemented")
    # Checkerboards and image densities are drawn on the unit square and mapped
    # into their domain, which defaults to the unit square.
    if dataset_name in {"checkerboard", "image_density"} and conf.get("domain") is not None:
        x = postprocessing(conf).inv(th.from_numpy(x)).numpy()

    return th.utils.data.TensorDataset(th.from_numpy(x).to(th.float32))


def sample_timesteps(
    batch_size: int, device: th.device, generator: th.Generator, eps=1e-4
):
    return (
        th.rand(batch_size, device=device, generator=generator) * (1.0 - 2 * eps) + eps
    )


def get_data_normalization(data_conf) -> normalizations.AffineBox:
    """Use the common endpoint sampling domain for reflection."""
    box0 = postprocessing(data_conf.data0)
    box1 = postprocessing(data_conf.data1)
    if box0.lower != box1.lower or box0.upper != box1.upper:
        raise ValueError("Endpoint reflected-domain bounds must match exactly.")
    return box0
