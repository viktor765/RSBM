import logging

import numpy as np
import torch as th

from ..common import select_evenly_spaced
from .plots import (
    load_density_display_image,
    plot_generation_procedure,
    plot_snapshots_and_vector_fields,
    plot_trajectories,
    plot_trajectory_frame,
    plot_trajectory_gif,
)

logger = logging.getLogger(__name__)


def write_final_direction_artifacts(
    *,
    context,
    model: th.nn.Module,
    direction_dir: str,
    direction_name: str,
    sampling_direction: int,
    trajectory: np.ndarray,
    domain,
) -> None:
    if not context.toy2d_conf["final_artifacts"]:
        return
    plot_trajectories(
        trajectory[:1],
        direction_dir,
        "single_trajectory",
        title=direction_name,
    )
    plot_trajectories(
        select_evenly_spaced(trajectory, 5),
        direction_dir,
        "five_trajectories",
        title=direction_name,
    )
    gif_paths = select_evenly_spaced(
        trajectory,
        context.toy2d_conf["n_gif_paths"],
    )
    plot_trajectory_gif(
        gif_paths,
        direction_dir,
        "paths",
        n_frames=context.toy2d_conf["n_gif_frames"],
    )
    plot_trajectory_frame(gif_paths, direction_dir, "paths_final_frame")
    plot_snapshots_and_vector_fields(
        trajectory,
        model=model,
        direction=sampling_direction,
        domain=domain,
        device=context.device,
        dir=direction_dir,
        file_name="snapshots_and_vector_field",
        n_times=context.toy2d_conf["n_snapshot_times"],
        n_particles=context.toy2d_conf["n_snapshot_particles"],
        vector_grid_size=context.toy2d_conf["vector_grid_size"],
    )


def write_generation_procedure_artifact(
    *,
    context,
    base_dir: str,
    true_samples: dict[int, th.Tensor],
    direction_artifacts: dict[str, dict[str, np.ndarray]],
) -> None:
    if not context.toy2d_conf["final_artifacts"]:
        return
    if not (
        context.data_conf.data0.dataset == "image_density"
        and context.data_conf.data1.dataset == "image_density"
    ):
        return
    if "0_to_1" not in direction_artifacts or "1_to_0" not in direction_artifacts:
        logger.warning("Skipping generation procedure figure; missing direction data.")
        return

    image0 = load_density_display_image(context.data_conf.data0, full_resolution=True)
    image1 = load_density_display_image(context.data_conf.data1, full_resolution=True)
    plot_generation_procedure(
        image0=image0,
        image1=image1,
        x0=context.to_unit_cube_numpy(0, true_samples[0]),
        x1=context.to_unit_cube_numpy(1, true_samples[1]),
        fake_01=direction_artifacts["0_to_1"]["sampled"],
        fake_10=direction_artifacts["1_to_0"]["sampled"],
        paths_01=direction_artifacts["0_to_1"]["trajectory"],
        paths_10=direction_artifacts["1_to_0"]["trajectory"],
        dir=base_dir,
        n_points=context.toy2d_conf["n_generation_points"],
        n_paths=context.toy2d_conf["n_generation_paths"],
    )
