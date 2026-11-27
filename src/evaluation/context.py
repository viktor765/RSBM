import json
import os
from dataclasses import dataclass, field

import torch as th

from .. import data_utils, utils
from .common import evaluation_dir, get_toy2d_eval_conf, shared_dir, step_dir
from .directions import DIRECTIONS, DirectionSpec
from .toy2d.plots import image_aspect, load_density_display_image


@dataclass
class EvaluationContext:
    data_conf: object
    eval_conf: object
    seed: int
    sde: object
    out_dir: str
    # Root of the evaluation-set plots that runs share.
    datasets_dir: str
    device: th.device
    postprocessing0: object
    postprocessing1: object
    # FID and the clipped MSD clip generated images to the unit cube.
    clip_generated: bool = True
    n_images: int = field(init=False)
    noise_draw: int = field(init=False)
    eval_batch_size: int = field(init=False)
    train_data: dict = field(init=False)
    test_data: dict = field(init=False)
    postprocessing: dict[int, object] = field(init=False)
    toy2d_conf: dict = field(init=False)
    _plot_aspect_cache: dict[int, float] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.n_images = self.eval_conf.get("n_samples", 64)
        self.noise_draw = self.eval_conf.get("noise_draw", 0)
        if (
            not isinstance(self.noise_draw, int)
            or isinstance(self.noise_draw, bool)
            or self.noise_draw < 0
        ):
            raise ValueError(
                f"eval.noise_draw must be a non-negative integer, got {self.noise_draw!r}."
            )
        self.train_data = data_utils.create_dataloaders(
            self.data_conf,
            1024,
            True,
            False,
            self.seed,
            dl_generator=None,
        )
        self.eval_batch_size = max(256, self.n_images)
        self.test_data = data_utils.create_dataloaders(
            self.data_conf,
            self.eval_batch_size,
            False,
            False,
            self.seed,
            dl_generator=None,
            num_workers=0,
        )
        for index, loader in self.test_data.items():
            if len(loader.dataset) < self.n_images:
                raise ValueError(
                    f"The data{index} evaluation set has {len(loader.dataset)} "
                    f"points, fewer than eval.n_samples = {self.n_images}."
                )
        self.postprocessing = {0: self.postprocessing0, 1: self.postprocessing1}
        self.toy2d_conf = get_toy2d_eval_conf(self.eval_conf)

    @property
    def is_2d(self) -> bool:
        sample = self.test_data[0].dataset[0][0]
        return sample.ndim == 1 and sample.shape[0] == 2

    @property
    def directions(self) -> tuple[DirectionSpec, ...]:
        return DIRECTIONS

    def lpips_supported(self) -> bool:
        return all(
            self.test_data[i].dataset[0][0].ndim == 3
            and self.test_data[i].dataset[0][0].shape[0] == 3
            for i in (0, 1)
        )

    def new_generator(self) -> th.Generator:
        """The evaluation noise of the run; every evaluation draws the same.

        eval.noise_draw r > 0 selects the run's r-th independent alternative,
        for repeated evaluations of one model; draw 0 is the training-time noise.
        """
        draw = () if self.noise_draw == 0 else (self.noise_draw,)
        return utils.create_generator(
            self.seed, "evaluation", *draw, device=self.device
        )

    def write_true_data_plots(self, plot) -> None:
        """Plot each endpoint's evaluation samples once for all runs sharing them.

        plot(dataset_index, unit-cube samples, directory) draws into a new
        directory under datasets_dir, keyed by the evaluation set and the number
        plotted. out_dir/true_data.json gives this run's directories.
        """
        directories = {}
        for dataset_index in (0, 1):
            identity = {
                **data_utils.dataset_identity(
                    self.data_conf, dataset_index, False, self.seed
                ),
                "n_plotted": self.n_images,
            }

            def write(directory, dataset_index=dataset_index):
                dataset = self.test_data[dataset_index].dataset
                data = th.stack([dataset[j][0] for j in range(self.n_images)])
                plot(dataset_index, self.postprocessing[dataset_index](data), directory)

            directories[f"ds{dataset_index}"] = shared_dir(
                self.datasets_dir, identity["data"]["dataset"], identity, write
            )
        os.makedirs(self.out_dir, exist_ok=True)
        with open(os.path.join(self.out_dir, "true_data.json"), "w", encoding="utf-8") as f:
            json.dump(directories, f, indent=2)
            f.write("\n")

    def true_eval_samples(self) -> dict[int, th.Tensor]:
        return {
            0: next(iter(self.test_data[0]))[0][: self.n_images],
            1: next(iter(self.test_data[1]))[0][: self.n_images],
        }

    def to_unit_cube_numpy(self, dataset_index: int, x: th.Tensor):
        """Map toy samples to unit-square coordinates for plots and metrics.

        Native-domain callers must supply the matching affine postprocessing
        map. This conversion does not clip out-of-domain samples.
        """
        return self.postprocessing[dataset_index](x).numpy()

    def plot_aspect(self, dataset_index: int) -> float:
        if dataset_index not in self._plot_aspect_cache:
            conf = self.data_conf[f"data{dataset_index}"]
            if conf.dataset == "image_density":
                image = load_density_display_image(conf)
                self._plot_aspect_cache[dataset_index] = image_aspect(image)
            else:
                self._plot_aspect_cache[dataset_index] = 1.0
        return self._plot_aspect_cache[dataset_index]

    def step_dir(self, step: int) -> str:
        return step_dir(self.out_dir, step)

    def evaluation_dir(self, step: int, weights: str) -> str:
        return evaluation_dir(self.out_dir, step, weights)
