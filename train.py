import json
import logging

import hydra
from accelerate import PartialState
from accelerate.utils import parse_flag_from_env
from omegaconf import DictConfig, OmegaConf
from tqdm.contrib.logging import logging_redirect_tqdm

from src import data_utils, models, run_dir, utils
from src.evaluation.common import training_summary_path
from src.rsbm import create_sde_and_bridge, trainers

logger = logging.getLogger(__name__)


def create_trainer(cfg: DictConfig):
    """Build the model and reference SDE, and the trainer of cfg.method."""
    model = models.create_model(cfg.model, cfg.sde.sigma)
    sde, bridge = create_sde_and_bridge(
        cfg.sde, data_utils.get_data_normalization(cfg.data)
    )
    if cfg.method == "rsb_fbsde":
        from src.rsb_fbsde.trainer import RsbFbsdeTrainer

        if PartialState().num_processes != 1:
            raise ValueError("The RSB-FBSDE baseline runs in one process.")
        # RSB-FBSDE simulates the reference SDE and has no use for its bridge.
        return RsbFbsdeTrainer(model=model, sde=sde, config=cfg)
    if cfg.method == "alpha_imf":
        trainer_class = trainers.get_trainer(model.__class__)
        return trainer_class(model=model, sde=sde, bridge=bridge, config=cfg)
    raise NotImplementedError(f"Method {cfg.method} not implemented.")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # PartialState does not read ACCELERATE_USE_CPU itself. Initialize it with
    # the same CPU request that Accelerator will honor later in this process.
    is_main_process = PartialState(
        cpu=parse_flag_from_env("ACCELERATE_USE_CPU")
    ).is_main_process
    run_dir.check_launch(cfg)

    with logging_redirect_tqdm():
        if is_main_process:
            logger.info(f"Output dir: {cfg.out_dir}")
            revision = utils.git_revision()
            logger.info(
                f"Git commit: {revision['commit']} (uncommitted changes: {revision['dirty']})"
            )
            record = run_dir.record_launch(cfg, revision)
            logger.info(f"Launch record: {record}")
            logger.info("Config:")
            logger.info(OmegaConf.to_yaml(cfg))

        utils.set_seed(cfg.seed)

        trainer = create_trainer(cfg)

        if is_main_process:
            logger.info(
                f"Model size (nbr of params): {sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)}"
            )

        trainer.train()

        if is_main_process:
            with open(
                training_summary_path(trainer.eval_dir), encoding="utf-8"
            ) as stream:
                completed = json.load(stream)["completed"]
            logger.info("Training completed" if completed else "Training interrupted")


if __name__ == "__main__":
    main()
