import argparse
import logging

from src.evaluation import posthoc

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a final model file, with final artifacts, using the run "
            "config stored in it. Needs no training checkpoint or run directory."
        )
    )
    parser.add_argument(
        "model",
        help="Path to final_model_<weights>.safetensors, or an evaluation's model.safetensors.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help=(
            "key=value evaluation settings, e.g. eval.n_samples=1024 or "
            f"sde.euler_steps=200. Allowed: {', '.join(posthoc.OVERRIDABLE)}."
        ),
    )
    parser.add_argument(
        "--out",
        default=None,
        help=(
            "New output directory; default <run>/posthoc/<launch time> for "
            "<run>/checkpoints/<model>, else posthoc/<launch time> beside the model."
        ),
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device; default the one training would use.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = posthoc.default_out_dir(args.model) if args.out is None else args.out
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
    )
    logger.info(f"Evaluating {args.model} into {out_dir}")
    posthoc.evaluate_final_model(args.model, out_dir, args.overrides, args.device)
    logger.info("Evaluation completed")


if __name__ == "__main__":
    main()
