import os

from .. import data_utils
from ..rsbm.sdes import ReflectedSDE
from .context import EvaluationContext
from .image.evaluator import ImageEvaluator
from .toy2d.evaluator import Toy2DEvaluator


def create_evaluator(
    *,
    data_conf,
    eval_conf,
    seed,
    sde,
    out_dir,
    datasets_dir,
    device,
    postprocessing0,
    postprocessing1,
    clip_generated=True,
):
    context = EvaluationContext(
        data_conf=data_conf,
        eval_conf=eval_conf,
        seed=seed,
        sde=sde,
        out_dir=out_dir,
        datasets_dir=datasets_dir,
        device=device,
        postprocessing0=postprocessing0,
        postprocessing1=postprocessing1,
        clip_generated=clip_generated,
    )
    if context.is_2d:
        return Toy2DEvaluator(context)
    return ImageEvaluator(context)


def run_evaluator_arguments(config, sde) -> dict:
    """create_evaluator's arguments fixed by a run config, for training and eval.py.

    Reflected samples lie in the domain, so FID and MSD see them unclipped and
    there is no clipped MSD.
    """
    return {
        "data_conf": config.data,
        "eval_conf": config.eval,
        "seed": config.seed,
        "sde": sde,
        "datasets_dir": os.path.join(config.output_root, "datasets"),
        "postprocessing0": data_utils.postprocessing(config.data.data0),
        "postprocessing1": data_utils.postprocessing(config.data.data1),
        "clip_generated": not isinstance(sde, ReflectedSDE),
    }
