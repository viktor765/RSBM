# SPDX-License-Identifier: Apache-2.0
# Adapted from rSB-FBSDE/runner.py in WayneDW/Reflected_Schrodinger_Bridge.
# Modified for rsbm: rsbm's reflected SDE, time grid and per-direction
# networks, matched running and boundary quadrature and mirror Euler, paths started
# from RSBM's training sets, shuffled trajectory caches, shared evaluation,
# resource accounting and resumable state.
import copy
import json
import logging
import math
import time
from contextlib import contextmanager
from pathlib import Path

import torch
from omegaconf import OmegaConf

from .. import data_utils, models, utils
from ..evaluation import (
    create_evaluator,
    evaluation_dir,
    evaluation_weights,
    plot_directional_losses,
    run_evaluator_arguments,
    step_dir,
)
from ..evaluation.curve import append_curve_row, plot_curve
from ..evaluation.retention import retain_evaluated_model
from ..rsbm.sdes import ReflectedSDE
from ..run_dir import archive_abandoned_history
from ..training_diagnostics import (
    DIRECTION_GROUPS,
    accumulate_diagnostics,
    append_diagnostics,
    gradient_diagnostics,
    gradient_norm_limit,
    require_finite,
    summarize_diagnostics,
)
from .loss import alternate_loss

logger = logging.getLogger(__name__)


class ShuffledPoints:
    """Take arbitrary counts across shuffled source passes, retaining tails.

    The state is the generator state that drew the current permutation and
    the position in it, so a restored source continues the same pass.
    """

    def __init__(self, dataset, generator):
        self.points, self.generator = dataset.tensors[0], generator
        if not len(self.points):
            raise ValueError("A source dataset must not be empty.")
        self.pass_start = self.order = None
        self.position = 0

    def take(self, count):
        if type(count) is not int or count <= 0:
            raise ValueError("A source request must be a positive integer.")
        parts = []
        while count:
            if self.order is None or self.position == len(self.order):
                self._start_pass(self.generator.get_state())
            size = min(count, len(self.order) - self.position)
            parts.append(self.points[self.order[self.position:self.position + size]])
            self.position += size
            count -= size
        return torch.cat(parts)

    def _start_pass(self, generator_state):
        self.generator.set_state(generator_state)
        self.pass_start = generator_state
        self.order = torch.randperm(len(self.points), generator=self.generator)
        self.position = 0

    def state_dict(self):
        return {"pass_start": self.pass_start, "position": self.position}

    def load_state_dict(self, state):
        if state["pass_start"] is not None:
            self._start_pass(state["pass_start"].cpu())
            self.position = state["position"]


class RsbFbsdeTrainer:
    def __init__(self, model, sde, config):
        cfg = self.cfg = config
        self.conf = cfg.training
        # Training stops after num_stages or once training_seconds reaches
        # this, whichever comes first, checked between updates.
        self.max_training_seconds = self.conf.get("max_training_seconds")
        # eval.freq counts optimizer updates, or seconds of training_seconds.
        self.eval_freq_unit = cfg.eval.get("freq_unit", "iterations")
        self._validate(model, sde)
        self._next_evaluation_seconds = 0.0
        if self.conf.num_threads is not None:
            torch.set_num_threads(self.conf.num_threads)
        device = self.conf.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("RSB-FBSDE runs on CPU or CUDA devices.")
        self.out_dir = Path(cfg.out_dir)
        self.eval_dir = self.out_dir / "eval"
        self.checkpoint_dir = self.out_dir / "checkpoints"
        self.eval_dir.mkdir(parents=True, exist_ok=True)
        # Training caches and evaluation use the same configured sampler.
        self.sde = sde
        self.domain = sde.reflector.normalization
        self.model = model.to(self.device)
        self.ema_model = copy.deepcopy(self.model).requires_grad_(False).eval()
        self.optimizers, self.schedulers = {}, {}
        for direction in (1, -1):
            optimizer = torch.optim.AdamW(
                self.model.nets[direction].parameters(),
                lr=self.conf.lr,
                betas=tuple(self.conf.adam_betas),
                weight_decay=self.conf.weight_decay,
            )
            self.optimizers[direction] = optimizer
            self.schedulers[direction] = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=self.conf.lr_step, gamma=self.conf.lr_gamma
            )
        self.train_generator = utils.create_generator(
            cfg.seed, "training", device=self.device
        )
        # Paths start from RSBM's training sets (the same seed, so the same
        # points), reshuffled every pass as RSBM's loaders are. Checkpoints
        # keep each source's position, so a resume continues its passes.
        self.train_sources = {
            index: ShuffledPoints(
                data_utils.get_dataset(cfg.data, index, True, True, cfg.seed),
                utils.create_generator(cfg.seed, "path_starts", index),
            )
            for index in (0, 1)
        }
        self.stage = self.epoch = self.iteration = self.step = 0
        # The direction being trained. Each stage trains +1, then -1.
        self.direction = 1
        self.directional_updates = {1: 0, -1: 0}
        self.directional_losses = {1: [], -1: []}
        self._diagnostic_totals = {}
        self.fresh_paths = {1: 0, -1: 0}
        self.training_seconds = 0.0
        self._training_started_at = None
        # Part of training_seconds, and evaluation time, which it excludes.
        self.sampling_timer = utils.SegmentTimer(self.device)
        self.evaluation_timer = utils.SegmentTimer(self.device)
        self._cache = None
        self._path_order = None
        self._path_position = 0
        self.eval_weights = evaluation_weights(cfg.eval)
        # A fresh run first evaluates its initial weights, at step 0. A resumed
        # one already has: each scheduled evaluation precedes its checkpoint.
        self._evaluation_due_at_start = bool(cfg.eval.freq) and not self.conf.resume.dir
        if self.conf.resume.dir:
            self.load_checkpoint(self.conf.resume.dir)
        # The same evaluation sets and noise as RSBM's for this seed. Preparing
        # them leaves the global RNGs alone, as evaluating does.
        with utils.fork_global_rng(self.device):
            self.evaluator = create_evaluator(
                **run_evaluator_arguments(cfg, self.sde),
                out_dir=str(self.eval_dir),
                device=self.device,
            )
            self.evaluator.prepare()

    def _validate(self, model, sde):
        if self.cfg.method != "rsb_fbsde":
            raise ValueError("Use method=rsb_fbsde.")
        # Each direction has its own policy and optimizer, and its loss is in
        # the control Z.
        if self.cfg.model.output != "control" or not isinstance(
            model, models.MultiNet
        ):
            raise ValueError(
                "RSB-FBSDE trains one network per direction that outputs the "
                "control Z (model.directional=false, model.output=control)."
            )
        for index in (0, 1):
            if len(self.cfg.data[f"data{index}"].domain.lower) != 2:
                raise ValueError("RSB-FBSDE baseline requires 2D toy endpoints.")
        if self.max_training_seconds is not None and not self.max_training_seconds > 0:
            raise ValueError("training.max_training_seconds must be null or positive.")
        if self.conf.num_stages is None and self.max_training_seconds is None:
            raise ValueError(
                "training.num_stages may be null only with training.max_training_seconds."
            )
        if self.eval_freq_unit not in {"iterations", "seconds"}:
            raise ValueError("eval.freq_unit must be iterations or seconds.")
        for name in (
            "num_stages",
            "num_epochs",
            "iterations_per_epoch",
            "batch_size",
            "lr_step",
        ):
            if name == "num_stages" and self.conf[name] is None:
                continue
            if type(self.conf[name]) is not int or self.conf[name] < 1:
                raise ValueError(f"training.{name} must be a positive integer.")
        cache = self.conf.cache_size
        if type(cache) is not int or cache < self.conf.batch_size or cache % self.conf.batch_size:
            raise ValueError(
                "training.cache_size must be a positive multiple of training.batch_size."
            )
        if self.conf.lr <= 0 or not 0 < self.conf.lr_gamma <= 1:
            raise ValueError("Require positive lr and 0 < lr_gamma <= 1.")
        if self.conf.loss_quadrature not in {"left", "right", "trapezoidal"}:
            raise ValueError(
                "training.loss_quadrature must be left, right or trapezoidal."
            )
        if self.conf.boundary_evaluation not in {"projection", "crossing"}:
            raise ValueError(
                "training.boundary_evaluation must be projection or crossing."
            )
        if not 0 <= self.conf.ema_rate < 1 or self.conf.divergence not in {
            "exact",
            "hutchinson_gaussian",
            "hutchinson_rademacher",
        }:
            raise ValueError("Invalid EMA rate or divergence.")
        if not isinstance(sde, ReflectedSDE) or sde.sigma <= 0:
            raise ValueError("RSB-FBSDE requires a reflected SDE with positive sigma.")

    def _synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _start_timer(self):
        if self._training_started_at is None:
            self._synchronize()
            self._training_started_at = time.perf_counter()

    def _pause_timer(self):
        if self._training_started_at is not None:
            self._synchronize()
            self.training_seconds += time.perf_counter() - self._training_started_at
            self._training_started_at = None

    @contextmanager
    def _exclude_from_training(self):
        running = self._training_started_at is not None
        self._pause_timer()
        try:
            yield
        finally:
            if running:
                self._start_timer()

    def sample_training_data(self, sampling_direction):
        """Cache cache_size fresh paths of sampling_direction from its sources.

        The sampling policy does not train while the cache is in use; only
        the opposite direction does. As for RSBM, sampling_seconds times the
        path simulation, not the data loading.
        """
        # Direction -1 starts from data1.
        sources = self.train_sources[1 if sampling_direction == -1 else 0]
        source = sources.take(self.conf.cache_size).to(self.device)
        with self.sampling_timer.time():
            self._cache = self.simulate_paths(source, sampling_direction)
        self._path_order = None
        self._path_position = 0
        self.fresh_paths[sampling_direction] += len(source)

    @torch.no_grad()
    def simulate_paths(self, source, sampling_direction):
        """Grid states/controls plus interval boundary points and vectors.

        Point m of a path is its state X_m at the sampler's time m / n, with
        the sampler's control there. Boundary data are indexed separately
        by interval and, in crossing mode, edge. Their locations do not depend
        on the time quadrature.
        """
        drift = self.ema_model if self.conf.use_ema else self.model
        path = self.sde.forward_euler(
            drift,
            source,
            sampling_direction,
            self.train_generator,
            return_trajectory=True,
            boundary_evaluation=self.conf.boundary_evaluation,
        )
        final = path.states[:, -1]
        final_drift = drift(
            final, torch.ones_like(final[:, 0]), direction=sampling_direction
        )
        controls = torch.cat((path.drifts, final_drift[:, None]), 1)
        return {
            "states": path.states,
            "controls": controls / self.sde.sigma,
            "displacements": path.displacements,
            "boundary_points": path.boundary_points,
        }

    def sample_batch(self):
        """Draw cached paths, with all their intervals and matched running and
        boundary times.

        Each interval puts equal weights on its quadrature points: its start
        (left), end (right) or both (trapezoidal). A state shared by two
        intervals is evaluated once, with both weights, so trapezoidal gives
        a path's end states 1/2 and every other state 1. Boundary points are
        not shared: each interval evaluates its own at its quadrature times.
        """
        cache = self._cache
        n_steps = self.sde.euler_steps
        if self._path_order is None or self._path_position == len(self._path_order):
            self._path_order = torch.randperm(len(cache["states"]), device=self.device,
                                              generator=self.train_generator)
            self._path_position = 0
        start = self._path_position
        paths = self._path_order[start:start + self.conf.batch_size]
        self._path_position += self.conf.batch_size
        offsets = {"left": [0], "right": [1], "trapezoidal": [0, 1]}
        offsets = torch.tensor(offsets[self.conf.loss_quadrature], device=self.device)
        interval_points = torch.arange(n_steps, device=self.device)[:, None] + offsets
        interval_weights = torch.full(
            interval_points.shape, 1 / len(offsets), device=self.device
        )
        state_weights = torch.zeros(n_steps + 1, device=self.device).index_add_(
            0, interval_points.reshape(-1), interval_weights.reshape(-1)
        )
        grid = state_weights.nonzero().squeeze(1)

        def at_grid(key):
            return cache[key][paths[:, None], grid].flatten(0, 1)

        batch = {
            "states": at_grid("states"),
            "times": (1 - grid / n_steps).repeat(self.conf.batch_size),
            "importance_controls": at_grid("controls"),
            "weights": state_weights[grid].repeat(self.conf.batch_size),
            "displacements": cache["displacements"][paths].flatten(0, 1),
            "boundary_points": cache["boundary_points"][paths].flatten(0, 1),
            "boundary_times": (1 - interval_points / n_steps).repeat(
                self.conf.batch_size, 1
            ),
            "boundary_weights": interval_weights.repeat(self.conf.batch_size, 1),
        }
        shape = batch["states"].shape
        if self.conf.divergence == "hutchinson_gaussian":
            batch["trace_probes"] = torch.randn(
                shape, device=self.device, generator=self.train_generator
            )
        elif self.conf.divergence == "hutchinson_rademacher":
            batch["trace_probes"] = (
                torch.randint(
                    2, shape, device=self.device, generator=self.train_generator
                ).float()
                * 2
                - 1
            )
        return batch

    def update(self):
        batch = self.sample_batch()
        batch["states"] = batch["states"].detach().requires_grad_(True)
        optimizer = self.optimizers[self.direction]
        # The loss is in the control Z that the network outputs.
        policy = self.model.nets[self.direction].network
        optimizer.zero_grad(set_to_none=True)
        # The loss sums over paths; this scales it to the mean per-path loss.
        loss = (
            alternate_loss(
                policy, **batch, sigma=self.sde.sigma, dt=1 / self.sde.euler_steps
            )
            / self.conf.batch_size
        )
        loss.backward()
        # A nonfinite objective never reaches AdamW.
        total_loss = float(loss.detach())
        if not math.isfinite(total_loss):
            raise FloatingPointError(
                f"Nonfinite RSB-FBSDE loss at update {self.step + 1}."
            )
        norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(),
            gradient_norm_limit(self.conf.max_grad_norm),
        )
        require_finite(
            norm,
            f"RSB-FBSDE gradient norm at update {self.step + 1} direction {self.direction:+d}",
        )
        optimizer.step()
        self.schedulers[self.direction].step()
        self.directional_updates[self.direction] += 1
        self.directional_losses[self.direction].append(total_loss)
        # torch_ema's default use_num_updates=True, including its warmup.
        updates = self.directional_updates[self.direction]
        decay = min(self.conf.ema_rate, (1 + updates) / (10 + updates))
        with torch.no_grad():
            for shadow, parameter in zip(
                self.ema_model.nets[self.direction].parameters(),
                policy.parameters(),
                strict=True,
            ):
                shadow.sub_((shadow - parameter) * (1 - decay))
        self.step += 1
        accumulate_diagnostics(
            self._diagnostic_totals,
            {
                DIRECTION_GROUPS[self.direction]: {
                    "loss": norm.new_tensor(total_loss),
                    **gradient_diagnostics(norm, self.conf.max_grad_norm),
                }
            },
        )
        if (
            self.conf.log_freq and self.step % self.conf.log_freq == 0
        ) or self.step == self._update_limit():
            self._append_diagnostics()
        return total_loss

    def _update_limit(self):
        if self.conf.num_stages is None:
            return None
        return 2 * self.conf.num_stages * self.conf.num_epochs * self.conf.iterations_per_epoch

    def _append_diagnostics(self):
        append_diagnostics(
            self.out_dir,
            method="rsb_fbsde",
            step=self.step,
            phase="alternating",
            max_grad_norm=self.conf.max_grad_norm,
            summaries=summarize_diagnostics(self._diagnostic_totals),
        )

    def _advance(self):
        self.iteration += 1
        if self.iteration < self.conf.iterations_per_epoch:
            return
        self.iteration = 0
        self._cache = None
        self._path_order = None
        self._path_position = 0
        self.epoch += 1
        if self.epoch < self.conf.num_epochs:
            return
        self.epoch = 0
        # Unlike upstream, the last stage trains -1 as well.
        if self.direction == 1:
            self.direction = -1
        else:
            self.direction = 1
            self.stage += 1

    def _elapsed_training_seconds(self):
        self._synchronize()
        return self.training_seconds + (
            time.perf_counter() - self._training_started_at
            if self._training_started_at is not None
            else 0.0
        )

    def _stop_due(self):
        """Whether training ends at this update boundary."""
        return (self.conf.num_stages is not None and self.stage >= self.conf.num_stages) or (
            self.max_training_seconds is not None
            and self._elapsed_training_seconds() >= self.max_training_seconds
        )

    def _evaluation_scheduled(self):
        """Whether eval.freq schedules an evaluation at this update boundary."""
        frequency = self.cfg.eval.freq
        if not frequency:
            return False
        if self.eval_freq_unit == "seconds":
            return self._elapsed_training_seconds() >= self._next_evaluation_seconds
        return self.step % frequency == 0

    def train(self):
        completed = False
        # The training state is consistent between updates, so checkpoints
        # are written only there. A first Ctrl-C stops at the next one.
        between_updates = True
        self._start_timer()
        try:
            with utils.deferred_interrupt() as interrupt:
                if self._evaluation_due_at_start:
                    self.evaluate()
                    self._evaluation_due_at_start = False
                stopping = self._stop_due()
                while not stopping:
                    if interrupt.requested:
                        raise KeyboardInterrupt
                    between_updates = False
                    if self._cache is None:
                        # Direction d fits the time reversal of the -d process,
                        # whose paths start at the endpoint d ends at (a
                        # half-bridge step).
                        self.sample_training_data(-self.direction)
                    direction = self.direction
                    loss = self.update()
                    self._advance()
                    between_updates = True
                    stopping = self._stop_due()
                    if self.conf.log_freq and self.step % self.conf.log_freq == 0:
                        logger.info(
                            "RSB-FBSDE update %d direction %+d loss %.6g",
                            self.step,
                            direction,
                            loss,
                        )
                    # A time limit can stop between logging windows.
                    if stopping and self._diagnostic_totals:
                        self._append_diagnostics()
                    # Same eval.freq as RSBM; the final evaluation follows
                    # training.
                    if not stopping and self._evaluation_scheduled():
                        self.evaluate()
                    frequency = self.conf.checkpoint.frequency
                    if frequency and self.step % frequency == 0:
                        self.save_checkpoint(self.checkpoint_dir / "c_latest")
            completed = True
        except KeyboardInterrupt:
            logger.warning("RSB-FBSDE training interrupted after update %d.", self.step)
        finally:
            self._pause_timer()
            self.write_summary(completed)
            if between_updates:
                self.save_checkpoint(
                    self.checkpoint_dir / ("c_final" if completed else "c_latest")
                )
            else:
                logger.warning(
                    "RSB-FBSDE stopped during update %d, whose state cannot be "
                    "resumed; checkpoints/c_latest keeps the last periodic one.",
                    self.step + 1,
                )
        plots = not self.cfg.eval.get("toy2d", {}).get("sweep_metrics_only", False)
        if plots:
            plot_directional_losses(self.directional_losses, str(self.end_dir))
        if completed:
            self.save_final_model()
            self.evaluate(final_artifacts=True)
            # Include the final evaluation in evaluation_seconds.
            self.write_summary(completed)
        if plots:
            plot_curve(str(self.eval_dir), str(self.end_dir))

    @torch.no_grad()
    def evaluate(self, final_artifacts=False, record_curve=True):
        """Score each weight set of eval.weights at this step, in order."""
        with self._exclude_from_training(), self.evaluation_timer.time():
            if self.eval_freq_unit == "seconds" and self.cfg.eval.freq:
                # The next scheduled one follows the next multiple of eval.freq.
                self._next_evaluation_seconds = (
                    math.floor(self.training_seconds / self.cfg.eval.freq) + 1
                ) * self.cfg.eval.freq
            for weights in self.eval_weights:
                drift = self.ema_model if weights == "ema" else self.model
                with utils.fork_global_rng(self.device), utils.evaluation_mode(drift):
                    metrics = self.evaluator.evaluate(
                        drift,
                        self.step,
                        weights=weights,
                        final_artifacts=final_artifacts,
                    )
                retain_evaluated_model(
                    evaluation_dir(self.eval_dir, self.step, weights),
                    drift,
                    self.cfg,
                    step=self.step,
                    weights=weights,
                )
                if record_curve and isinstance(metrics, dict):
                    append_curve_row(
                        self.eval_dir,
                        step=self.step,
                        weights=weights,
                        training_seconds=self.training_seconds,
                        sampling_seconds=self.sampling_timer.seconds,
                        pretrain_seconds=0.0,
                        directional_updates=self.directional_updates,
                        fresh_paths=self.fresh_paths,
                        fields=self.evaluator.curve_fields(metrics),
                    )

    def _resume_contract(self):
        config = OmegaConf.to_container(self.cfg, resolve=True)
        training = config["training"]
        for key in (
            "resume",
            "device",
            "num_threads",
            "log_freq",
            "checkpoint",
            "max_training_seconds",
        ):
            training.pop(key, None)
        return {
            key: config[key]
            for key in ("method", "seed", "model", "sde", "data", "training")
        }

    def save_checkpoint(self, directory):
        with self._exclude_from_training():
            directory = Path(directory)
            directory.mkdir(parents=True, exist_ok=True)
            state = {
                "version": 1,
                "contract": self._resume_contract(),
                "model": self.model.state_dict(),
                "ema": self.ema_model.state_dict(),
                "optimizers": {
                    str(k): v.state_dict() for k, v in self.optimizers.items()
                },
                "schedulers": {
                    str(k): v.state_dict() for k, v in self.schedulers.items()
                },
                "position": [
                    self.stage,
                    self.direction,
                    self.epoch,
                    self.iteration,
                    self.step,
                ],
                "directional_updates": self.directional_updates,
                "directional_losses": self.directional_losses,
                "diagnostic_totals": self._diagnostic_totals,
                "fresh_paths": self.fresh_paths,
                "training_seconds": self.training_seconds,
                "sampling_seconds": self.sampling_timer.seconds,
                "evaluation_seconds": self.evaluation_timer.seconds,
                "cache": self._cache,
                "path_order": self._path_order,
                "path_position": self._path_position,
                "next_evaluation_seconds": self._next_evaluation_seconds,
                "train_rng": self.train_generator.get_state(),
                "path_starts": {
                    str(k): v.state_dict() for k, v in self.train_sources.items()
                },
                "train_rng_device_type": self.device.type,
            }
            temporary = directory / "rsb_fbsde.pt.tmp"
            torch.save(state, temporary)
            temporary.replace(directory / "rsb_fbsde.pt")

    def save_final_model(self):
        """Save each evaluated weight set alone, with the run config."""
        with self._exclude_from_training():
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            for weights in self.eval_weights:
                models.save_final_model(
                    str(self.checkpoint_dir / models.final_model_file_name(weights)),
                    self.ema_model if weights == "ema" else self.model,
                    self.cfg,
                    step=self.step,
                    weights=weights,
                )

    def load_checkpoint(self, directory):
        path = Path(directory)
        if path.is_dir():
            path = path / "rsb_fbsde.pt"
        state = torch.load(path, map_location=self.device, weights_only=True)
        if state["version"] != 1:
            raise ValueError(
                "RSB-FBSDE checkpoint uses an unsupported format."
            )
        if state["contract"] != self._resume_contract():
            raise ValueError(
                "RSB-FBSDE checkpoint experiment settings differ from this run."
            )
        if state["train_rng_device_type"] != self.device.type:
            raise ValueError("Resume requires the same RNG backend (CPU or CUDA).")
        self.model.load_state_dict(state["model"])
        self.ema_model.load_state_dict(state["ema"])
        for direction in (1, -1):
            self.optimizers[direction].load_state_dict(
                state["optimizers"][str(direction)]
            )
            self.schedulers[direction].load_state_dict(
                state["schedulers"][str(direction)]
            )
        self.stage, self.direction, self.epoch, self.iteration, self.step = state[
            "position"
        ]
        self.directional_updates = state["directional_updates"]
        self.directional_losses = state["directional_losses"]
        self._diagnostic_totals = state["diagnostic_totals"]
        self.fresh_paths = state["fresh_paths"]
        self.training_seconds = state["training_seconds"]
        self.sampling_timer.seconds = state["sampling_seconds"]
        self.evaluation_timer.seconds = state["evaluation_seconds"]
        self._cache = state["cache"]
        self._path_order = state["path_order"]
        self._path_position = state["path_position"]
        self._next_evaluation_seconds = state["next_evaluation_seconds"]
        self.train_generator.set_state(state["train_rng"].cpu())
        for index, source in self.train_sources.items():
            source.load_state_dict(state["path_starts"][str(index)])
        # A scheduled evaluation precedes its checkpoint; the final one follows.
        archive_abandoned_history(
            self.out_dir,
            self.step,
            include_step=self._stop_due(),
            resumed_from=path.parent,
        )

    def write_summary(self, completed):
        if self.device.type == "cuda":
            peak = torch.cuda.max_memory_reserved(self.device)
            total = torch.cuda.get_device_properties(self.device).total_memory
        else:
            peak = total = None
        summary = {
            "method": "rsb_fbsde",
            "completed": completed,
            "training_seconds": self.training_seconds,
            "sampling_seconds": self.sampling_timer.seconds,
            "pretrain_seconds": 0.0,
            "evaluation_seconds": self.evaluation_timer.seconds,
            "completed_updates": self.step,
            "directional_updates": {
                str(k): v for k, v in self.directional_updates.items()
            },
            "fresh_paths": {str(k): v for k, v in self.fresh_paths.items()},
            "configured_pretrain_updates": 0,
            "configured_finetune_updates": self._update_limit(),
            "trainable_parameters": sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            ),
            "euler_steps": self.sde.euler_steps,
            "n_reflections": None,
            "reflection": "mirror_euler",
            "loss_quadrature": self.conf.loss_quadrature,
            "boundary_evaluation": self.conf.boundary_evaluation,
            "noise_on_last_step": self.sde.noise_on_last_step,
            "project_last_step": self.sde.project_last_step,
            "sigma": self.sde.sigma,
            "device": str(self.device),
            "precision": "float32",
            "seed": self.cfg.seed,
            "num_threads": torch.get_num_threads(),
            "torch_version": str(torch.__version__),
            "peak_cuda_reserved_bytes": peak,
            "total_cuda_bytes": total,
            "peak_cuda_reserved_fraction": peak / total if peak is not None else None,
            "upstream_commit": "908a5093f7c33d1fcbb2d4c8733922564625aba8",
        }
        self.end_dir.mkdir(parents=True, exist_ok=True)
        (self.end_dir / "training_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )

    @property
    def end_dir(self):
        """Where training ending now writes its records; a resumed run ends elsewhere."""
        return Path(step_dir(self.eval_dir, self.step))
