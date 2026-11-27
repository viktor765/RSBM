# Reproducing the paper's runs

These are the commands that produced the paper's results.
Even with the same seed and software environment, GPU reruns are not
guaranteed to produce bitwise-identical weights or generated samples.

All runs used the Apptainer image built from [`my-pytorch.def`](my-pytorch.def)
(see the README), one process per run (no `accelerate launch`) and seeds 0, 1
and 2. A run writes to `outputs/<run_name>`; runs without `run_name` use the
default layout `outputs/<experiment>/<method>/<refl|non-refl>/sigma<σ>/seed<seed>`.
First create the subdir `logs` (`mkdir -p logs`).
Then run the following from the repository root. To use the published
weights and training records instead of training, see the last section.

## Two-dimensional experiments

One NVIDIA A100 40GB per job. Each job trains
seeds 0, 1 and 2 at once on its GPU, sharing it through MPS
([`scripts/run_concurrent.sh`](scripts/run_concurrent.sh)), and passes its
arguments on to `train.py`.

### Uniform ↔ checkerboard

```bash
sbatch slurm/checkerboard-rsbm.sh
sbatch slurm/checkerboard-fbsde.sh
```

Outputs: `uniform-to-checkerboard/{alpha_imf,rsb_fbsde}/refl/sigma0.5/seed{0,1,2}`.

Longer RSB-FBSDE runs, with 120 rather than 20 stages and a learning rate that
decays six times more slowly:

```bash
sbatch -t 15:00:00 slurm/checkerboard-fbsde.sh training.num_stages=120 training.lr_step=6000 eval.freq=300 training.checkpoint.frequency=16000 'run_name=uniform-to-checkerboard/rsb_fbsde-long/refl/sigma0.5/seed${seed}'
```

RSB-FBSDE ablations, each changing one setting:

```bash
sbatch -t 05:00:00 slurm/checkerboard-fbsde.sh training.divergence=exact 'run_name=uniform-to-checkerboard/rsb_fbsde-exact-divergence/refl/sigma0.5/seed${seed}'
sbatch slurm/checkerboard-fbsde.sh training.boundary_evaluation=crossing 'run_name=uniform-to-checkerboard/rsb_fbsde-boundary-crossing/refl/sigma0.5/seed${seed}'
sbatch slurm/checkerboard-fbsde.sh training.loss_quadrature=left 'run_name=uniform-to-checkerboard/rsb_fbsde-left-quadrature/refl/sigma0.5/seed${seed}'
sbatch slurm/checkerboard-fbsde.sh training.loss_quadrature=right 'run_name=uniform-to-checkerboard/rsb_fbsde-right-quadrature/refl/sigma0.5/seed${seed}'
```

Outputs: `uniform-to-checkerboard/rsb_fbsde-{long,exact-divergence,boundary-crossing,left-quadrature,right-quadrature}/refl/sigma0.5/seed{0,1,2}`.
The `-t` limits above replace the script's 3 hours for the longer jobs.

### Schrödinger ↔ Öresund

```bash
sbatch slurm/schrodinger-oresund-rsbm.sh
sbatch slurm/schrodinger-oresund-fbsde.sh
sbatch -t 15:00:00 slurm/schrodinger-oresund-fbsde.sh training.num_stages=120 training.lr_step=6000 eval.freq=300 training.checkpoint.frequency=16000 'run_name=schrodinger-to-oresund/rsb_fbsde-long/refl/sigma0.5/seed${seed}'
```

Outputs: `schrodinger-to-oresund/{alpha_imf,rsb_fbsde,rsb_fbsde-long}/refl/sigma0.5/seed{0,1,2}`.

### Selection sweep

The checkerboard recipes of both methods, which Schrödinger ↔ Öresund reuses,
were chosen on seeds 100, 101 and 102 (also on A100s).

1. `bash slurm/sweep-checkerboard-rsbm-pretrain.sh` sweeps RSBM's pretraining
   learning rate (15 min of pretraining). Write the chosen rate and update
   count into `conf/training/toy2dsb.yaml`; the paper's are 6e-4 and 20,000.
2. `bash slurm/sweep-checkerboard-rsbm.sh` sweeps RSBM's finetuning learning
   rate (one hour of training including pretraining).
   `bash slurm/sweep-checkerboard-fbsde.sh` RSB-FBSDE's learning rate, decay
   factor and cache size (two hours).
3. Plot each sweep and choose. The pretraining sweep goes first, on its own,
   since its runs are RSBM runs too:

   ```bash
   python -m scripts.plot_sweep outputs/sweep/uniform-to-checkerboard/rsbm-pretrain --out-dir outputs/figures/sweep/rsbm-pretrain
   python -m scripts.plot_sweep outputs/sweep/uniform-to-checkerboard/alpha_imf --out-dir outputs/figures/sweep/rsbm
   python -m scripts.plot_sweep outputs/sweep/uniform-to-checkerboard/rsb_fbsde --out-dir outputs/figures/sweep/fbsde --ceiling best
   ```

   `scripts/plot_sweep.py` finds the swept hyperparameters by diffing the runs'
   configurations. Each recipe is chosen by its lowest seed-mean control cost
   among evaluations, after pretraining, whose W2 is within the ceiling in both
   directions: the exact-sample ceiling for RSBM, and for RSB-FBSDE, which does
   not reach it, the best observed W2 plus two standard errors
   (`--ceiling best`). The chosen recipes, which the presets use, are RSBM
   finetuning at 3e-4 and RSB-FBSDE at learning rate 3e-4, decay factor 0.8 and
   cache size 262,144.


## Image experiments

One NVIDIA GH200 96GB per run.

### AFHQ 64, cat ↔ wild

Each script is an array job over seeds 0–2:

```bash
sbatch slurm/afhq-brownian.sh    # python train.py experiment=afhq64 sde=brownian seed=$SLURM_ARRAY_TASK_ID
sbatch slurm/afhq-reflected.sh   # python train.py experiment=afhq64 sde=reflected seed=$SLURM_ARRAY_TASK_ID
```

Outputs: `afhq64/alpha_imf/{non-refl,refl}/sigma0.75/seed{0,1,2}`.

### MNIST ↔ EMNIST

Each script is an array job over seeds 0–2:

```bash
sbatch slurm/mnist-brownian.sh     # python train.py experiment=mnist_to_emnist sde=brownian seed=$SLURM_ARRAY_TASK_ID
sbatch slurm/mnist-reflected.sh    # python train.py experiment=mnist_to_emnist sde=reflected seed=$SLURM_ARRAY_TASK_ID
```

Outputs:
`mnist-to-emnist/alpha_imf/{non-refl,refl}/sigma1.0/seed{0,1,2}`.


## Figures and tables

Draw the paper's figures and tables from the runs above:

```bash
python -m scripts.paper_artifacts_toy2d
python -m scripts.paper_artifacts_imgs
```

They write to `outputs/paper/toy2d/` and `outputs/paper/images/`.
`--runs-dir DIR` reads the runs from `DIR` instead of `outputs/`, `--out-dir`
writes elsewhere, and `--format png` writes PNG rather than PDF. The
Schrödinger ↔ Öresund figure, `generation_procedure.pdf`, is written by each
Schrödinger ↔ Öresund run into its final evaluation directory,
`eval/<step>/<weights>/`.


## Published weights and training records

The paper's trained models and training records are on Zenodo,
<https://doi.org/10.5281/zenodo.24574007>. Download them with

```bash
bash scripts/download_zenodo.sh
```

It downloads about 4.5 GB, mostly AFHQ weights, and checks every file against
its SHA-256 checksum. Patterns after the output directory download only the
matching weights; this one leaves out AFHQ:

```bash
bash scripts/download_zenodo.sh outputs/zenodo 'mnist_*' 'checkerboard_*' 'oresund_*'
```

The records unzip to `outputs/zenodo/training_records/`, which holds every run
behind the paper's figures and tables, including the long runs and ablations,
under the names of the commands above. Each run has its configuration,
`.hydra/config.yaml`, and its evaluation curve, `eval/curve.jsonl`; the image
runs also have their final evaluation metrics and seed-0 sample caches. The
weights are the final EMA weights of the seed 0–2 runs of the four
experiments. Each goes into its run, at `checkpoints/final_model_ema.safetensors`,
as training writes it. To download by hand from the record page instead, unzip
`training_records.zip` there and put each weight file where
`scripts/download_zenodo.sh` lists it.

Draw all the paper's figures and tables from the records, without the weights,
into `outputs/zenodo/paper/from_records/`:

```bash
bash scripts/paper_from_records.sh
```

Evaluate every downloaded weight file afresh with `eval.py`, and draw from those
evaluations alone what final weights reproduce, into
`outputs/zenodo/paper/from_weights/`:

```bash
bash scripts/paper_from_weights.sh
```

That is the final-evaluation rows of the two-dimensional comparison table
(`rsbm_vs_rsb_fbsde_final_table.tex`, which `paper_from_records.sh` also
writes), the image tables and comparison figures, and the Schrödinger ↔
Öresund generation figure. The training curves, the 6h row and the ablations
need the records. The evaluations go to
`outputs/zenodo/evals/<run>/eval/`; one that finished is kept when the script
runs again, so delete it to evaluate again. Arguments go to every `eval.py`
call, for example `--device cpu` on a machine without CUDA; RSBM would otherwise
use Accelerate's device, which on a Mac is MPS, untested with this code.