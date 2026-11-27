"""Print the config train.py would run with the given overrides.

python -m scripts.print_config [OVERRIDE ...]

Composes the config without starting a Hydra job, so no run directory is
created; train.py refuses a run directory that holds anything else.
"""

import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def main():
    with initialize_config_dir(
        config_dir=str(Path(__file__).resolve().parents[1] / "conf"), version_base=None
    ):
        cfg = compose(config_name="config", overrides=sys.argv[1:])
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    main()
