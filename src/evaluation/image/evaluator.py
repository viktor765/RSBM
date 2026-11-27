import os

import torch as th

from ..common import write_json
from ..directions import DIRECTION_BY_NAME, DIRECTIONS
from ..retention import retention_conf, write_manifest
from ..sampling import sample_direction
from .cached_samples import write_cached_samples
from .metrics import ImageMetricEvaluator
from .plots import plot_images

# Per-direction entries of a metric record that go into the training curve.
CURVE_SCORES = (
    "fid",
    "kid",
    "lpips",
    "msd",
    "msd_clipped",
    "n_generated",
    "n_reference",
)
CURVE_OOB = ("total_oob_pixel_fraction", "fraction_images_with_any_oob")


class ImageEvaluator:
    def __init__(self, context):
        self.context = context
        retention_conf(context.eval_conf)
        self.metrics = ImageMetricEvaluator(context)

    def prepare(self) -> None:
        if self.context.eval_conf.get("plot_initial", True):
            self.write_initial_plots()
        self.metrics.prepare()

    def write_initial_plots(self) -> None:
        def plot(dataset_index, data, directory):
            plot_images(data, directory, "true", True)
            plot_images(data, directory, "true", False)

        self.context.write_true_data_plots(plot)

    def evaluate(
        self,
        model: th.nn.Module,
        step: int,
        *,
        weights: str,
        final_artifacts: bool = False,
    ) -> dict | None:
        """Plots, and the metric record of ImageMetricEvaluator.compute if any.

        model is the run's weights of kind `weights`; every call draws the same
        sources and noise.
        """
        generator = self.context.new_generator()
        model.eval()

        true_samples = self.context.true_eval_samples()
        x0_true = true_samples[0]
        x1_true = true_samples[1]
        x1_sampled = sample_direction(
            sde=self.context.sde,
            model=model,
            source=x0_true,
            direction=DIRECTION_BY_NAME["0_to_1"],
            device=self.context.device,
            generator=generator,
        )
        x0_sampled = sample_direction(
            sde=self.context.sde,
            model=model,
            source=x1_true,
            direction=DIRECTION_BY_NAME["1_to_0"],
            device=self.context.device,
            generator=generator,
        )
        samples = {0: x0_sampled, 1: x1_sampled}

        eval_dir = self.context.evaluation_dir(step, weights)
        os.makedirs(eval_dir, exist_ok=True)
        for dataset_index in (0, 1):
            plot_images(
                self.context.postprocessing[dataset_index](samples[dataset_index]),
                eval_dir,
                f"ds{dataset_index}_sampled",
                highlight_out_of_bounds=True,
            )
            plot_images(
                self.context.postprocessing[dataset_index](samples[dataset_index]),
                eval_dir,
                f"ds{dataset_index}_sampled",
                highlight_out_of_bounds=False,
            )

        metrics = None
        if self.metrics.enabled:
            metrics = self.metrics.compute(model, step)
            write_json(os.path.join(eval_dir, "eval_metrics.json"), metrics)
            write_json(
                os.path.join(eval_dir, "eval_protocol.json"), self.metrics.protocol
            )

        if final_artifacts:
            # Samples for figures; the metrics above hold the out-of-bounds records.
            write_cached_samples(self.context, model, eval_dir, weights)
        write_manifest(
            eval_dir, policy=retention_conf(self.context.eval_conf), step=step
        )
        return metrics

    def curve_fields(self, metrics: dict) -> dict:
        """The training-curve entries of evaluate's record, by direction name."""
        fields = {"protocol": self.metrics.protocol["id"]}
        for key in CURVE_SCORES + CURVE_OOB:
            values = {}
            for direction in DIRECTIONS:
                record = metrics[f"ds{direction.target_index}"]
                record = record["oob"] if key in CURVE_OOB else record
                if key in record:
                    values[direction.name] = record[key]
            if values:
                fields[key] = values
        return fields
