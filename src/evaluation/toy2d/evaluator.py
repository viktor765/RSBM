import importlib.metadata
import json
import logging
import os

import numpy as np
import torch as th

from ... import data_utils
from ...rsbm.sdes import ReflectedSDE
from ..common import protocol_with_id, solver_settings, write_json
from ..retention import retention_conf, write_manifest, write_toy_records
from ..sampling import sample_direction
from .metrics import (
    control_cost,
    empirical_w2,
    known_target_grid_w2,
    mean_squared_endpoint_displacement,
    region_masses,
    support_metrics,
)
from .paper_artifacts import (
    write_final_direction_artifacts,
    write_generation_procedure_artifact,
)
from .plots import plot_2d

logger = logging.getLogger(__name__)

TOY2D_METRIC_PROTOCOL_VERSION = 1
# Region masses compare generated and exact target masses on this grid of
# unit-square cells; it refines the 4 x 4 checkerboard.
REGION_CELLS_PER_AXIS = 8


def _known_target_grid_metrics(samples, target_conf, toy2d_conf):
    grid_subdivisions = toy2d_conf.get("known_target_grid_subdivisions")
    if grid_subdivisions is None:
        return None

    return known_target_grid_w2(
        samples,
        target_conf.dataset,
        subdivisions_per_checker_cell=grid_subdivisions,
        phase=target_conf.get("phase", 0),
    )


class Toy2DEvaluator:
    def __init__(self, context):
        self.context = context
        # A slim evaluation for sweeps: known-target grid W2 and control cost
        # only, without plots or final artifacts.
        self.sweep_metrics_only = context.toy2d_conf.get("sweep_metrics_only", False)
        if type(self.sweep_metrics_only) is not bool:
            raise ValueError("eval.toy2d.sweep_metrics_only must be boolean.")
        retention_conf(context.eval_conf)
        self.is_reflected = isinstance(context.sde, ReflectedSDE)
        self.target_cell_masses = {
            i: data_utils.known_target_cell_masses(
                context.data_conf[f"data{i}"], REGION_CELLS_PER_AXIS
            )
            for i in (0, 1)
        }
        self.protocol = self._protocol()

    def _protocol(self) -> dict:
        """What the metrics of evaluate measure; its id changes with any of it."""
        context = self.context
        toy2d = context.toy2d_conf

        def evaluation_set(i):
            return data_utils.dataset_identity(context.data_conf, i, False, context.seed)

        endpoint = {
            "coordinates": "unit cube, unclipped",
            "pairs": "each generated endpoint with its source",
        }
        w2 = {
            "reference": "the target evaluation set",
            "samples": "all generated endpoints and all target evaluation points",
            "cost": "squared Euclidean, exact optimal transport",
            "coordinates": endpoint["coordinates"],
        }
        metrics = {
            "w2": w2,
            "mean_squared_endpoint_displacement": {
                **endpoint,
                "mean": "over coordinates, then paths",
            },
            "final_support": {
                "boundary_eps": toy2d["boundary_eps"],
                "coordinates": endpoint["coordinates"],
            },
        }
        if toy2d.get("known_target_grid_subdivisions") is not None:
            metrics["known_target_grid"] = {
                "reference": "uniform weights on the cell centres of the known "
                "target, for uniform_box and checkerboard targets",
                "subdivisions_per_checker_cell": toy2d["known_target_grid_subdivisions"],
                "samples": "all generated endpoints",
                "cost": w2["cost"],
            }
        metrics["region_masses"] = {
            "cells_per_axis": REGION_CELLS_PER_AXIS,
            "reference": "exact target mass of each unit-square cell, for "
            "uniform_box, checkerboard and image_density targets",
            "total_variation": "half the L1 distance between generated and "
            "target cell masses; generated mass outside the unit square is excess",
            "outside_support_mass": "generated mass in cells of zero target "
            "mass or outside the unit square",
            "coordinates": endpoint["coordinates"],
        }
        if toy2d["control_cost"]:
            metrics["control_cost"] = {
                "definition": "mean over paths of the left Euler sum of "
                "|drift|^2 / (2 sigma^2) dt",
                "coordinates": "native",
            }
        if not self.is_reflected:
            metrics["projected"] = (
                "w2, known_target_grid, region_masses, final_support and mean_"
                "squared_endpoint_displacement of generated endpoints clipped to [0, 1]"
            )
        if self.sweep_metrics_only:
            metrics = {
                key: metrics[key]
                for key in ("known_target_grid", "control_cost")
                if key in metrics
            }
        return protocol_with_id(
            {
                "kind": "toy2d",
                "version": TOY2D_METRIC_PROTOCOL_VERSION,
                "generated": {
                    direction.name: {
                        "source": evaluation_set(direction.source_index),
                        "target": evaluation_set(direction.target_index),
                        "sampling_direction": direction.sampling_direction,
                    }
                    for direction in context.directions
                },
                "n_paths": context.n_images,
                "seed": context.seed,
                "noise_stream": "evaluation",
                "noise_draw": context.noise_draw,
                "solver": solver_settings(context.sde),
                "metrics": metrics,
                "libraries": {
                    name: importlib.metadata.version(name)
                    for name in ("numpy", "POT", "torch")
                },
            },
            device=str(context.device),
        )

    def prepare(self) -> None:
        if self.sweep_metrics_only:
            return
        if self.context.eval_conf.get("compute_fid", False):
            logger.warning("FID requested for 2D toy data; skipping FID.")
        if self.context.eval_conf.get("plot_initial", True):
            self.write_initial_plots()

    def write_initial_plots(self) -> None:
        def plot(dataset_index, data, directory):
            aspect = self.context.plot_aspect(dataset_index)
            plot_2d(data.numpy(), directory, "true", width_height_aspect=aspect)
            plot_2d(
                data.numpy(),
                directory,
                "true",
                extension="pdf",
                transparent=True,
                width_height_aspect=aspect,
            )

        self.context.write_true_data_plots(plot)

    def evaluate(
        self,
        model: th.nn.Module,
        step: int,
        *,
        weights: str,
        final_artifacts: bool = False,
    ) -> dict:
        """Score model, the run's weights of kind `weights`, at step.

        Every call draws the same sources and noise, so the weight sets of one
        evaluation are scored on the same paths' randomness.
        """
        if self.sweep_metrics_only:
            return self._evaluate_sweep_metrics(model, step, weights=weights)
        generator = self.context.new_generator()
        model.eval()

        true_samples = self.context.true_eval_samples()
        base_dir = self.context.evaluation_dir(step, weights)
        os.makedirs(base_dir, exist_ok=True)

        metrics = {}
        direction_artifacts = {}
        for direction in self.context.directions:
            direction_dir = os.path.join(base_dir, direction.name)
            os.makedirs(direction_dir, exist_ok=True)

            source = true_samples[direction.source_index]
            target = true_samples[direction.target_index]
            sampled_trajectory = sample_direction(
                sde=self.context.sde,
                model=model,
                source=source,
                direction=direction,
                device=self.context.device,
                generator=generator,
                return_trajectory=True,
            )
            trajectory = sampled_trajectory.states
            sampled = trajectory[:, -1]

            source_np = self.context.to_unit_cube_numpy(direction.source_index, source)
            target_np = self.context.to_unit_cube_numpy(direction.target_index, target)
            sampled_np = self.context.to_unit_cube_numpy(direction.target_index, sampled)
            # Plots use unit-square coordinates; control_cost keeps native ones.
            trajectory_np = self.context.to_unit_cube_numpy(
                direction.target_index, trajectory
            )
            direction_artifacts[direction.name] = {
                "sampled": sampled_np,
                "trajectory": trajectory_np,
            }

            target_aspect = self.context.plot_aspect(direction.target_index)
            plot_2d(
                sampled_np,
                direction_dir,
                "sampled_unprojected_w_ref",
                ref_x=target_np,
                width_height_aspect=target_aspect,
            )
            plot_2d(
                sampled_np,
                direction_dir,
                "sampled_unprojected",
                width_height_aspect=target_aspect,
            )

            direction_metrics = {
                "unprojected": {
                    **empirical_w2(
                        sampled_np,
                        target_np,
                    ),
                    "final_support": support_metrics(
                        sampled_np,
                        boundary_eps=self.context.toy2d_conf["boundary_eps"],
                    ),
                    "mean_squared_endpoint_displacement": (
                        mean_squared_endpoint_displacement(source_np, sampled_np)
                    ),
                }
            }

            target_masses = self.target_cell_masses[direction.target_index]
            direction_metrics["unprojected"]["region_masses"] = region_masses(
                sampled_np, target_masses
            )

            costs = None
            if self.context.toy2d_conf["control_cost"]:
                costs = control_cost(
                    self.context.sde,
                    model,
                    trajectory,
                    direction.sampling_direction,
                    self.context.device,
                    per_path=True,
                )
                direction_metrics["unprojected"]["control_cost"] = float(costs.mean())
                costs = costs.cpu().numpy()

            write_toy_records(
                self.context,
                direction_dir,
                direction,
                source_np,
                target_np,
                sampled_np,
                costs,
                target_cell_masses=target_masses,
            )

            if self.context.toy2d_conf.get("known_target_grid_subdivisions") is not None:
                target_conf = self.context.data_conf[f"data{direction.target_index}"]
                grid_metrics = _known_target_grid_metrics(
                    sampled_np,
                    target_conf,
                    self.context.toy2d_conf,
                )
                if grid_metrics is not None:
                    direction_metrics["known_target_grid"] = grid_metrics

            if not self.is_reflected:
                projected_np = np.clip(sampled_np, 0.0, 1.0)
                plot_2d(
                    projected_np,
                    direction_dir,
                    "sampled_projected",
                    ref_x=target_np,
                    width_height_aspect=target_aspect,
                )
                direction_metrics["projected"] = {
                    **empirical_w2(
                        projected_np,
                        target_np,
                    ),
                    "final_support": support_metrics(
                        projected_np,
                        boundary_eps=self.context.toy2d_conf["boundary_eps"],
                    ),
                    "mean_squared_endpoint_displacement": (
                        mean_squared_endpoint_displacement(source_np, projected_np)
                    ),
                }
                direction_metrics["projected"]["region_masses"] = region_masses(
                    projected_np, target_masses
                )
                if self.context.toy2d_conf.get("known_target_grid_subdivisions") is not None:
                    projected_grid_metrics = _known_target_grid_metrics(
                        projected_np,
                        target_conf,
                        self.context.toy2d_conf,
                    )
                    if projected_grid_metrics is not None:
                        direction_metrics["projected_known_target_grid"] = (
                            projected_grid_metrics
                        )

            if final_artifacts:
                write_final_direction_artifacts(
                    context=self.context,
                    model=model,
                    direction_dir=direction_dir,
                    direction_name=direction.name,
                    sampling_direction=direction.sampling_direction,
                    trajectory=trajectory_np,
                    domain=self.context.postprocessing[direction.target_index],
                )

            metrics_path = os.path.join(direction_dir, "metrics.json")
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(direction_metrics, f, indent=2, ensure_ascii=False)
            metrics[direction.name] = direction_metrics
            logger.info(
                "Step %s, %s weights: %s W2 %.6f",
                step,
                weights,
                direction.name,
                direction_metrics["unprojected"]["w2"],
            )

        if final_artifacts:
            write_generation_procedure_artifact(
                context=self.context,
                base_dir=base_dir,
                true_samples=true_samples,
                direction_artifacts=direction_artifacts,
            )

        metrics_path = os.path.join(base_dir, "eval_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        write_json(os.path.join(base_dir, "eval_protocol.json"), self.protocol)
        write_manifest(
            base_dir, policy=retention_conf(self.context.eval_conf), step=step
        )
        return metrics

    def _evaluate_sweep_metrics(self, model, step, *, weights):
        context = self.context
        generator = context.new_generator()
        model.eval()
        true_samples = context.true_eval_samples()
        base_dir = context.evaluation_dir(step, weights)
        os.makedirs(base_dir, exist_ok=True)
        metrics = {}
        for direction in context.directions:
            directory = os.path.join(base_dir, direction.name)
            os.makedirs(directory, exist_ok=True)
            source, target = (true_samples[i] for i in (direction.source_index, direction.target_index))
            path = sample_direction(
                sde=context.sde, model=model, source=source, direction=direction,
                device=context.device, generator=generator, return_trajectory=True,
            )
            sampled = context.to_unit_cube_numpy(direction.target_index, path.states[:, -1])
            record = {"unprojected": {}}
            costs = None
            if context.toy2d_conf["control_cost"]:
                costs = control_cost(context.sde, model, path.states, direction.sampling_direction,
                                     context.device, per_path=True)
                record["unprojected"]["control_cost"] = float(costs.mean())
                costs = costs.cpu().numpy()
            grid_metrics = _known_target_grid_metrics(
                sampled, context.data_conf[f"data{direction.target_index}"], context.toy2d_conf)
            if grid_metrics is not None:
                record["known_target_grid"] = grid_metrics
                logger.info("Step %s, %s weights: %s grid W2 %.6f", step, weights,
                            direction.name, grid_metrics["w2"])
            write_toy_records(
                context, directory, direction,
                context.to_unit_cube_numpy(direction.source_index, source),
                context.to_unit_cube_numpy(direction.target_index, target), sampled,
                costs, target_cell_masses=self.target_cell_masses[direction.target_index],
            )
            write_json(os.path.join(directory, "metrics.json"), record)
            metrics[direction.name] = record
        write_json(os.path.join(base_dir, "eval_metrics.json"), metrics)
        write_json(os.path.join(base_dir, "eval_protocol.json"), self.protocol)
        write_manifest(base_dir, policy=retention_conf(context.eval_conf), step=step)
        return metrics

    def curve_fields(self, metrics: dict) -> dict:
        """The training-curve entries of evaluate's record, by direction name."""
        if self.sweep_metrics_only:
            return {
                "protocol": self.protocol["id"],
                "known_target_grid_w2": {
                    name: m["known_target_grid"]["w2"]
                    for name, m in metrics.items()
                    if "known_target_grid" in m
                },
                "control_cost": {
                    name: m["unprojected"]["control_cost"]
                    for name, m in metrics.items()
                    if "control_cost" in m["unprojected"]
                },
            }
        return {
            "protocol": self.protocol["id"],
            "w2": {name: m["unprojected"]["w2"] for name, m in metrics.items()},
            "control_cost": {
                name: m["unprojected"]["control_cost"]
                for name, m in metrics.items()
                if "control_cost" in m["unprojected"]
            },
            "known_target_grid_w2": {
                name: m["known_target_grid"]["w2"]
                for name, m in metrics.items()
                if "known_target_grid" in m
            },
            "region_total_variation": {
                name: m["unprojected"]["region_masses"]["total_variation"]
                for name, m in metrics.items()
            },
            "outside_support_mass": {
                name: m["unprojected"]["region_masses"]["outside_support_mass"]
                for name, m in metrics.items()
            },
        }
