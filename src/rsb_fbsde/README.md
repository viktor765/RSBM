# RSB-FBSDE baseline

A reimplementation of the reflected Schrödinger bridge method of
[Deng et al. (UAI 2024)](https://proceedings.mlr.press/v244/deng24b.html),
used as the baseline in the 2D experiments. The setup, hyperparameters and
implementation choices are described in the paper's appendix.

```bash
python train.py experiment=rsb_fbsde_uniform_to_checkerboard
python train.py experiment=rsb_fbsde_schrodinger_to_oresund
```

## Provenance and license

Adapted from [WayneDW/Reflected_Schrodinger_Bridge](https://github.com/WayneDW/Reflected_Schrodinger_Bridge)
(commit `908a509`, directory `rSB-FBSDE/`), which builds on
[SB-FBSDE](https://github.com/ghliu/SB-FBSDE). Both are licensed under
Apache-2.0, as are the adapted files below ([LICENSE](LICENSE)); each file's
header states what was changed. The rest of the repository is MIT.

| File | Upstream sources |
| --- | --- |
| `model.py` | `models/toy_model/Toy.py`, `models/utils.py`, `policy.py` |
| `loss.py` | `loss.py` |
| `trainer.py` | `runner.py`, `options.py`, `sde.py` |
| `conf/training/rsb_fbsde.yaml`, `conf/model/toy_policy.yaml`, `conf/experiment/rsb_fbsde_*.yaml` | `configs/default_checkerboard_config.py`, `options.py`, `models/toy_model/Toy.py` |
