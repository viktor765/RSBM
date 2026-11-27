import json
import logging
import math
import os
import random
import time
from contextlib import contextmanager

import accelerate
import numpy as np
import torch as th
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from omegaconf import OmegaConf
from torch.nn.functional import mse_loss
from torch.optim import Adam, AdamW, Optimizer
from tqdm import tqdm, trange

from .. import data_utils, models, utils
from ..evaluation import (
    create_evaluator,
    evaluation_dir,
    evaluation_weights,
    plot_directional_losses,
    plot_loss,
    run_evaluator_arguments,
    step_dir,
)
from ..evaluation.curve import append_curve_row, plot_curve
from ..evaluation.retention import (
    prune_training_checkpoints,
    retain_evaluated_model,
    validate_checkpoint_retention,
)
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
from . import bridges, sdes
from .continuation import TrainingSource, prepare_loaders

logger = logging.getLogger(__name__)

PRECISIONS = ("fp32", "bf16")


def autocast(device: th.device, precision: str):
    """bf16 autocast for network passes at precision bf16; a no-op at fp32."""
    return th.autocast(device.type, dtype=th.bfloat16, enabled=precision == "bf16")


def get_pretrain_lr(base_lr: float, step: int, lr_warmup_steps: int) -> float:
    if step >= lr_warmup_steps:
        return base_lr
    return base_lr * (step + 1) / lr_warmup_steps


def set_optimizer_lr(optimizer: Optimizer | dict[int, Optimizer], lr: float) -> None:
    if isinstance(optimizer, dict):
        for directional_optimizer in optimizer.values():
            set_optimizer_lr(directional_optimizer, lr)
        return
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def create_optimizer(model, training_conf, lr):
    optimizer_name = str(training_conf.optimizer).lower()
    optimizer_classes = {
        "adam": Adam,
        "adamw": AdamW,
    }
    if optimizer_name not in optimizer_classes:
        raise ValueError(
            f"Unknown optimizer {optimizer_name!r}; expected 'adam' or 'adamw'."
        )

    optimizer_class = optimizer_classes[optimizer_name]
    betas = tuple(training_conf.adam_betas)
    weight_decay = float(training_conf.weight_decay)

    def create(parameters):
        return optimizer_class(
            parameters,
            lr=float(lr),
            betas=betas,
            weight_decay=weight_decay,
        )

    if isinstance(model, models.MultiNet):
        return {key: create(net.parameters()) for key, net in model.nets.items()}
    return create(model.parameters())


def clip_optimizer_grad_norm(accelerator, optimizer, max_grad_norm):
    """Clip the parameters owned by the optimizer being stepped."""
    parameters = (
        parameter for group in optimizer.param_groups for parameter in group["params"]
    )
    return accelerator.clip_grad_norm_(parameters, max_grad_norm)


def clip_and_measure_grad_norm(
    accelerator: Accelerator,
    optimizer,
    max_grad_norm: float | None,
    *,
    device: th.device,
) -> th.Tensor:
    """Return the total gradient norm before optional clipping."""
    norm_limit = gradient_norm_limit(max_grad_norm)
    grad_norm = clip_optimizer_grad_norm(accelerator, optimizer, norm_limit)
    if grad_norm is None:
        return th.full((), float("nan"), device=device)
    return th.as_tensor(grad_norm, device=device).detach()


def prepare_training_components(
    accelerator,
    model,
    ema_model,
    optimizer_pretrain,
    optimizer_finetune,
    train_dataloaders,
    finetune_dataloaders,
):
    """Prepare single- or double-network training state with Accelerator."""
    if not isinstance(optimizer_pretrain, dict):
        prepared = accelerator.prepare(
            model,
            optimizer_pretrain,
            optimizer_finetune,
            train_dataloaders[0],
            train_dataloaders[1],
            finetune_dataloaders[0],
            finetune_dataloaders[1],
        )
        # Checkpoints save the EMA too; it is only evaluated, so not wrapped.
        ema_model = accelerator.prepare_model(ema_model, evaluation_mode=True)
        return prepared[0], ema_model, *prepared[1:]

    direction_keys = tuple(optimizer_pretrain)
    if tuple(optimizer_finetune) != direction_keys:
        raise ValueError(
            "Pretraining and finetuning optimizers have different directions."
        )

    pretrain_optimizers = [
        optimizer_pretrain[direction] for direction in direction_keys
    ]
    finetune_optimizers = [
        optimizer_finetune[direction] for direction in direction_keys
    ]
    prepared = accelerator.prepare(
        model,
        *pretrain_optimizers,
        *finetune_optimizers,
        train_dataloaders[0],
        train_dataloaders[1],
        finetune_dataloaders[0],
        finetune_dataloaders[1],
    )

    model, *prepared_rest = prepared
    ema_model = accelerator.prepare_model(ema_model, evaluation_mode=True)
    n_directions = len(direction_keys)
    prepared_pretrain = prepared_rest[:n_directions]
    prepared_finetune = prepared_rest[n_directions : 2 * n_directions]
    dataloader0, dataloader1, finetune0, finetune1 = prepared_rest[2 * n_directions :]

    return (
        model,
        ema_model,
        dict(zip(direction_keys, prepared_pretrain, strict=True)),
        dict(zip(direction_keys, prepared_finetune, strict=True)),
        dataloader0,
        dataloader1,
        finetune0,
        finetune1,
    )


class AlphaIMFTrainer:
    # A direction-conditioned network samples both finetuning directions at once.
    joint_finetune_sampling = True

    def __init__(
        self,
        model,
        sde: sdes.SDE,
        bridge: bridges.BrownianBridge,
        config,
    ):
        training_conf, eval_conf, data_conf = config.training, config.eval, config.data
        seed, out_dir = config.seed, config.out_dir
        dl_config = accelerate.DataLoaderConfiguration(split_batches=True)
        accelerator_kwargs = {}
        if isinstance(model, models.MultiNet):
            accelerator_kwargs["kwargs_handlers"] = [
                DistributedDataParallelKwargs(find_unused_parameters=True)
            ]
        self.accelerator = Accelerator(
            dataloader_config=dl_config,
            # Full precision, whatever accelerate launch or its config says.
            mixed_precision="no",
            **accelerator_kwargs,
        )

        self._safe_boundary = True
        self.completed = False
        self._interrupt = None
        self.step = 0
        self.start_step = 0
        self.losses = []
        self.directional_losses = {+1: [], -1: []}
        self._diagnostic_totals = {}
        self.training_seconds = 0.0
        self._training_started_at = None
        self.directional_updates = {+1: 0, -1: 0}
        self.fresh_paths = {+1: 0, -1: 0}
        # Parts of training_seconds, and evaluation time, which it excludes.
        self.sampling_timer = utils.SegmentTimer(self.accelerator.device)
        self.evaluation_timer = utils.SegmentTimer(self.accelerator.device)
        self.pretrain_seconds = 0.0

        self.config = config
        self.training_conf = training_conf
        self.n_pretrain_steps = training_conf.n_pretrain_steps
        self.n_finetune_steps = training_conf.n_finetune_steps
        self.pretrain_lr = training_conf.pretrain_lr
        self.finetune_lr = training_conf.finetune_lr
        self.lr_warmup_steps = training_conf.lr_warmup_steps
        self.use_ema = training_conf.use_ema
        self.ema_rate = training_conf.ema_rate
        self.compile_ema = training_conf.compile_ema
        self.precision = training_conf.precision
        self.finetune_sampling_precision = training_conf.finetune_sampling_precision
        for key in ("precision", "finetune_sampling_precision"):
            if training_conf[key] not in PRECISIONS:
                raise ValueError(f"training.{key} must be fp32 or bf16.")
        self.log_freq = training_conf.log_freq
        self.pretrain_checkpoint_freq = training_conf.checkpoint.pretrain_frequency
        self.finetune_checkpoint_freq = training_conf.checkpoint.finetune_frequency
        validate_checkpoint_retention(training_conf.checkpoint.get("keep_last"))
        self.max_grad_norm = training_conf.max_grad_norm
        if self.lr_warmup_steps < 0:
            raise ValueError("training.lr_warmup_steps must be nonnegative.")
        if self.lr_warmup_steps > self.n_pretrain_steps:
            logger.warning(
                "training.lr_warmup_steps=%d exceeds n_pretrain_steps=%d; pretraining "
                "ends during the warmup.",
                self.lr_warmup_steps,
                self.n_pretrain_steps,
            )
        # Training stops at the update limit or once training_seconds reaches
        # this, whichever comes first, checked between updates.
        self.max_training_seconds = training_conf.get("max_training_seconds")
        if self.max_training_seconds is not None and not self.max_training_seconds > 0:
            raise ValueError("training.max_training_seconds must be null or positive.")
        if self.n_finetune_steps is None and self.max_training_seconds is None:
            raise ValueError(
                "training.n_finetune_steps may be null only with "
                "training.max_training_seconds."
            )
        # Each refill simulates the endpoint pairs of cache_size / batch_size
        # finetuning updates, which then use them in order.
        cache_size = training_conf.get("cache_size", training_conf.batch_size)
        if (
            type(cache_size) is not int
            or cache_size < training_conf.batch_size
            or cache_size % training_conf.batch_size
        ):
            raise ValueError(
                "training.cache_size must be a positive multiple of training.batch_size."
            )
        self.cache_batches = cache_size // training_conf.batch_size
        # The endpoint pairs of the remaining updates, one tuple per update.
        self._endpoint_cache = []

        self.eval_freq = eval_conf.freq
        # eval.freq counts optimizer updates, or seconds of training_seconds.
        self.eval_freq_unit = eval_conf.get("freq_unit", "iterations")
        if self.eval_freq_unit not in {"iterations", "seconds"}:
            raise ValueError("eval.freq_unit must be iterations or seconds.")
        self._next_evaluation_seconds = 0.0
        # training.use_ema picks the finetune sampling weights, this the
        # evaluated ones; the EMA is updated every step either way.
        self.eval_weights = evaluation_weights(eval_conf)
        self.eval_dir = os.path.join(out_dir, "eval")
        self.checkpoint_dir = os.path.join(out_dir, "checkpoints")
        if self.accelerator.is_main_process:
            os.makedirs(self.eval_dir, exist_ok=True)
            os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.model = model
        self.ema_model = models.create_eval_copy(self.model)
        self._ema_inference_model = None
        self.sde = sde
        self.bridge = bridge

        self.optimizer_pretrain = create_optimizer(
            self.model, training_conf, self.pretrain_lr
        )
        self.optimizer_finetune = create_optimizer(
            self.model, training_conf, self.finetune_lr
        )
        # The data streams order the batches identically on every process;
        # each process draws its own times, bridge points and Euler paths.
        self.data_gen = utils.create_generator(seed, "pretrain_data")
        self.train_gen = utils.create_generator(
            seed,
            "training",
            self.accelerator.process_index,
            device=self.accelerator.device,
        )
        self.finetune_data_gen = utils.create_generator(seed, "finetune_data")

        self.train_dataloaders = data_utils.create_dataloaders(
            data_conf,
            training_conf.batch_size,
            True,
            True,
            seed,
            dl_generator=self.data_gen,
        )

        # A finetuning batch has as many endpoint pairs as each direction
        # trains on in a pretraining batch: half of it.
        self.finetune_dataloaders = data_utils.rebatch_training_dataloaders(
            self.train_dataloaders,
            utils.get_half_batch_size(training_conf.batch_size),
            self.finetune_data_gen,
        )

        self.train_dataloaders, self.pretrain_plans = prepare_loaders(
            self.train_dataloaders, self.data_gen
        )
        self.finetune_dataloaders, self.finetune_plans = prepare_loaders(
            self.finetune_dataloaders, self.finetune_data_gen
        )

        (
            self.model,
            self.ema_model,
            self.optimizer_pretrain,
            self.optimizer_finetune,
            self.train_dataloaders[0],
            self.train_dataloaders[1],
            self.finetune_dataloaders[0],
            self.finetune_dataloaders[1],
        ) = prepare_training_components(
            self.accelerator,
            self.model,
            self.ema_model,
            self.optimizer_pretrain,
            self.optimizer_finetune,
            self.train_dataloaders,
            self.finetune_dataloaders,
        )
        if training_conf.compile_model:
            # In place, after wrapping: the module, its parameters and its
            # state-dict keys stay the same; under DDP the wrapper is compiled
            # so gradient communication can overlap the backward pass.
            self.model.compile()
        self.optimizer = self.optimizer_pretrain
        self.device = self.accelerator.device
        self.pretrain_sources = {
            k: TrainingSource(loader, self.pretrain_plans[k], self.device)
            for k, loader in self.train_dataloaders.items()
        }
        self.finetune_sources = {
            k: TrainingSource(loader, self.finetune_plans[k], self.device)
            for k, loader in self.finetune_dataloaders.items()
        }

        self.load_checkpoint(training_conf.resume.dir)

        # After the checkpoint restores the global RNGs, which preparing the
        # evaluator (an Inception network, loader iterators) must not move.
        self.evaluator = None
        if self.accelerator.is_main_process:
            with utils.fork_global_rng(self.device):
                self.evaluator = create_evaluator(
                    **run_evaluator_arguments(config, self.sde),
                    out_dir=self.eval_dir,
                    device=self.accelerator.device,
                )
                self.evaluator.prepare()

    def _synchronize(self) -> None:
        if self.device.type == "cuda":
            th.cuda.synchronize(self.device)

    def _resume_training_timer(self) -> None:
        self._synchronize()
        self._training_started_at = time.perf_counter()

    def _pause_training_timer(self) -> None:
        if self._training_started_at is not None:
            self._synchronize()
            self.training_seconds += time.perf_counter() - self._training_started_at
            self._training_started_at = None

    @contextmanager
    def _exclude_from_training_time(self):
        """Pause at evaluation/checkpoint boundaries, without timing each update."""
        was_running = self._training_started_at is not None
        self._pause_training_timer()
        try:
            yield
        finally:
            if was_running:
                self._resume_training_timer()

    def _accumulate_step_diagnostics(self, step_diagnostics) -> None:
        accumulate_diagnostics(self._diagnostic_totals, step_diagnostics)

    def _summarize_and_reset_diagnostics(self) -> dict[str, dict[str, float]]:
        return summarize_diagnostics(self._diagnostic_totals, self.accelerator)

    def _log_diagnostics(self, summaries: dict[str, dict[str, float]]) -> None:
        parts = []
        for group in ("forward", "backward", "shared"):
            if group not in summaries:
                continue
            values = summaries[group]
            part = f"{group}: loss={values['loss']:.6g}"
            if "grad_norm" in values:
                clipped = (
                    f"{100.0 * values['clipped_fraction']:.1f}%"
                    if self.max_grad_norm is not None
                    else "disabled"
                )
                part += (
                    f", pre_clip_grad_norm={values['grad_norm']:.6g}, clipped={clipped}"
                )
            parts.append(part)
        if parts:
            steps = int(next(iter(summaries.values()))["steps"])
            logger.info("Diagnostics over %d steps | %s", steps, " | ".join(parts))

    def get_ema_inference_model(self):
        ema_model = self.accelerator.unwrap_model(self.ema_model)
        ema_model.eval()
        if not self.compile_ema:
            return ema_model
        if self._ema_inference_model is None:
            try:
                self._ema_inference_model = th.compile(ema_model)
            except Exception as exc:
                raise RuntimeError(
                    "Failed to compile the EMA model. Disable training.compile_ema "
                    "to fall back to the uncompiled EMA model."
                ) from exc
        self._ema_inference_model.eval()
        return self._ema_inference_model

    def _evaluated_weights(self, weights):
        """The unwrapped, uncompiled network of one weight set."""
        return self.accelerator.unwrap_model(
            self.ema_model if weights == "ema" else self.model
        )

    def evaluate(self, final_artifacts: bool = False, record_curve: bool = True):
        """Score each weight set of eval.weights at this step, in order."""
        with self._exclude_from_training_time(), self.evaluation_timer.time():
            if self.eval_freq_unit == "seconds" and self.eval_freq is not None:
                # The next scheduled one follows the next multiple of eval.freq.
                self._next_evaluation_seconds = (
                    math.floor(self.training_seconds / self.eval_freq) + 1
                ) * self.eval_freq
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                for weights in self.eval_weights:
                    if weights == "ema":
                        eval_model = self.get_ema_inference_model()
                    else:
                        eval_model = self._evaluated_weights(weights)
                    with (
                        th.inference_mode(),
                        utils.fork_global_rng(self.device),
                        utils.evaluation_mode(eval_model),
                    ):
                        metrics = self.evaluator.evaluate(
                            eval_model,
                            step=self.step,
                            weights=weights,
                            final_artifacts=final_artifacts,
                        )
                    retain_evaluated_model(
                        evaluation_dir(self.eval_dir, self.step, weights),
                        self._evaluated_weights(weights),
                        self.config,
                        step=self.step,
                        weights=weights,
                    )
                    # An image evaluation without metrics returns None.
                    if record_curve and isinstance(metrics, dict):
                        append_curve_row(
                            self.eval_dir,
                            step=self.step,
                            weights=weights,
                            training_seconds=self.training_seconds,
                            sampling_seconds=self.sampling_timer.seconds,
                            pretrain_seconds=self.pretrain_seconds,
                            directional_updates=self.directional_updates,
                            fresh_paths=self.fresh_paths,
                            fields=self.evaluator.curve_fields(metrics),
                        )
            self.accelerator.wait_for_everyone()

    def train(self):
        self._resume_training_timer()
        self.completed = False
        completed = False
        try:
            with utils.deferred_interrupt() as interrupt:
                self._interrupt = interrupt
                self._train()
            completed = self.completed = True
        except KeyboardInterrupt:
            if self.accelerator.is_main_process:
                logger.warning("Training interrupted after update %d.", self.step)
            if self._safe_boundary:
                self.save_checkpoint(
                    os.path.join(self.checkpoint_dir, f"c_{self.step}")
                )
            else:
                logger.warning(
                    "Stopped inside an update; keeping the last safe checkpoint."
                )
        finally:
            self._pause_training_timer()
            self._interrupt = None
            for source in (
                *self.pretrain_sources.values(),
                *self.finetune_sources.values(),
            ):
                source.close()
        if self.accelerator.is_main_process:
            if self.device.type == "cuda":
                peak_cuda_reserved_bytes = th.cuda.max_memory_reserved(self.device)
                total_cuda_bytes = th.cuda.get_device_properties(
                    self.device
                ).total_memory
            else:
                peak_cuda_reserved_bytes = None
                total_cuda_bytes = None
            # A resumed run ends elsewhere, keeping this end's records.
            end_dir = step_dir(self.eval_dir, self.step)
            os.makedirs(end_dir, exist_ok=True)
            if not self.config.eval.get("toy2d", {}).get("sweep_metrics_only", False):
                plot_loss(self.losses, end_dir)
                plot_directional_losses(self.directional_losses, end_dir)
                plot_curve(self.eval_dir, end_dir)
            with open(
                os.path.join(end_dir, "training_summary.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    {
                        "completed": completed,
                        "training_seconds": self.training_seconds,
                        "sampling_seconds": self.sampling_timer.seconds,
                        "pretrain_seconds": self.pretrain_seconds,
                        "evaluation_seconds": self.evaluation_timer.seconds,
                        "peak_cuda_reserved_bytes": peak_cuda_reserved_bytes,
                        "total_cuda_bytes": total_cuda_bytes,
                        "peak_cuda_reserved_fraction": (
                            peak_cuda_reserved_bytes / total_cuda_bytes
                            if peak_cuda_reserved_bytes is not None
                            else None
                        ),
                        "completed_updates": self.step,
                        "directional_updates": {
                            str(direction): count
                            for direction, count in self.directional_updates.items()
                        },
                        "fresh_paths": {
                            str(direction): count
                            for direction, count in self.fresh_paths.items()
                        },
                        "configured_pretrain_updates": self.n_pretrain_steps,
                        "configured_finetune_updates": self.n_finetune_steps,
                        "trainable_parameters": sum(
                            parameter.numel()
                            for parameter in self.accelerator.unwrap_model(
                                self.model
                            ).parameters()
                            if parameter.requires_grad
                        ),
                        "euler_steps": self.sde.euler_steps,
                        "noise_on_last_step": self.sde.noise_on_last_step,
                        "project_last_step": getattr(
                            self.sde, "project_last_step", None
                        ),
                        "n_reflections": getattr(
                            getattr(self.sde, "reflector", None), "n_reflections", None
                        ),
                    },
                    f,
                    indent=2,
                )

    def _train(self):
        self._stopped = self._stop_due()
        if getattr(self, "_evaluation_due_at_start", False):
            self.evaluate()
            self._evaluation_due_at_start = False
        self._pretrain()
        self._finetune()
        self.save_checkpoint(os.path.join(self.checkpoint_dir, "c_final"))
        self.save_final_model()
        # The last update skips its scheduled evaluation for this one.
        self.evaluate(final_artifacts=True)

    def _elapsed_training_seconds(self):
        self._synchronize()
        return self.training_seconds + (
            time.perf_counter() - self._training_started_at
            if self._training_started_at is not None
            else 0.0
        )

    def _update_limit_reached(self):
        return (
            self.n_finetune_steps is not None
            and self.step >= self.n_pretrain_steps + self.n_finetune_steps
        )

    def _stop_due(self):
        """Whether training ends at this update boundary."""
        return self._update_limit_reached() or (
            self.max_training_seconds is not None
            and self._elapsed_training_seconds() >= self.max_training_seconds
        )

    def _pretrain(self):
        dataloaders = self.pretrain_sources
        self.optimizer = self.optimizer_pretrain
        for step in trange(
            self.start_step,
            self.n_pretrain_steps,
            initial=min(self.start_step, self.n_pretrain_steps),
            total=self.n_pretrain_steps,
            desc="Pretraining updates",
            disable=not self.accelerator.is_main_process,
        ):
            if self._stopped:
                break
            self._check_interrupt()
            self._safe_boundary = False
            lr = get_pretrain_lr(self.pretrain_lr, step, self.lr_warmup_steps)
            set_optimizer_lr(self.optimizer_pretrain, lr)
            x0 = next(dataloaders[0])[0]
            x1 = next(dataloaders[1])[0]
            loss, step_diagnostics = self.pretrain_step(x0, x1)
            self._finish_update(
                loss,
                step_diagnostics,
                lr,
                checkpoint_due=self.pretrain_checkpoint_freq is not None
                and (
                    self.step + 1 == self.n_pretrain_steps
                    or (self.step + 1) % self.pretrain_checkpoint_freq == 0
                ),
            )
        for source in dataloaders.values():
            source.close()

    def _next_finetune_endpoints(self):
        """The next update's simulated endpoint pairs, refilling the cache when used up."""
        if not self._endpoint_cache:
            x0 = th.cat([next(self.finetune_sources[0])[0] for _ in range(self.cache_batches)])
            x1 = th.cat([next(self.finetune_sources[1])[0] for _ in range(self.cache_batches)])
            with self.sampling_timer.time():
                endpoints = self.sample_finetune_endpoints(x0, x1)
            size = len(x0) // self.cache_batches
            self._endpoint_cache = list(
                zip(*(value.split(size) for value in endpoints), strict=True)
            )
        return self._endpoint_cache.pop(0)

    def _finetune(self):
        self.optimizer = self.optimizer_finetune
        with tqdm(
            total=self.n_finetune_steps,
            initial=max(self.step - self.n_pretrain_steps, 0),
            desc="Finetuning updates",
            disable=not self.accelerator.is_main_process,
        ) as progress:
            while not self._stopped:
                self._check_interrupt()
                self._safe_boundary = False
                loss, step_diagnostics = self.finetune_step(
                    self._next_finetune_endpoints()
                )
                progress.update()
                self._finish_update(
                    loss,
                    step_diagnostics,
                    self.finetune_lr,
                    checkpoint_due=self.finetune_checkpoint_freq is not None
                    and (self.step + 1) % self.finetune_checkpoint_freq == 0,
                )
        for source in self.finetune_sources.values():
            source.close()

    def _check_interrupt(self):
        requested = bool(self._interrupt and self._interrupt.requested)
        # A signal on any rank requests a stop on every rank at this boundary.
        request = th.tensor(int(requested), device=self.device)
        if self.accelerator.reduce(request, reduction="sum").item():
            raise KeyboardInterrupt

    def _finish_update(self, loss, step_diagnostics, lr, *, checkpoint_due):
        # Each process supplies a scalar loss averaged over an equal-sized local
        # batch. Gather collects one scalar per process; their mean is the global
        # batch loss. With one process, this leaves the scalar unchanged.
        accelerated_loss = self.accelerator.gather(loss)
        mean_loss = accelerated_loss.mean().item()
        # Every process gets the same mean, so all stop together.
        if not math.isfinite(mean_loss):
            raise FloatingPointError(f"Nonfinite RSBM loss at update {self.step + 1}.")
        directions = tuple(
            direction
            for direction, group in DIRECTION_GROUPS.items()
            if group in step_diagnostics
        )
        if directions:
            local_directional_losses = th.stack(
                [
                    step_diagnostics[DIRECTION_GROUPS[direction]]["loss"]
                    .detach()
                    .float()
                    for direction in directions
                ]
            )
            # All processes participate so these histories use the same global
            # batch averaging as the combined loss history.
            mean_directional_losses = self.accelerator.reduce(
                local_directional_losses, reduction="mean"
            )
        if self.accelerator.is_main_process:
            self.losses.append(mean_loss)
            for index, direction in enumerate(directions):
                self.directional_losses[direction].append(
                    mean_directional_losses[index].item()
                )

        self._accumulate_step_diagnostics(step_diagnostics)

        self.step += 1

        self.accelerator.wait_for_everyone()
        models.update_ema(self.model, self.ema_model, self.ema_rate)

        self._safe_boundary = True
        is_final_step = self._stopped = self._stop_due()
        # A run that stops inside pretraining spent all its training there.
        if self.step == self.n_pretrain_steps or (
            is_final_step and self.step < self.n_pretrain_steps
        ):
            was_running = self._training_started_at is not None
            self._pause_training_timer()
            self.pretrain_seconds = self.training_seconds
            if was_running:
                self._resume_training_timer()

        if (self.log_freq and self.step % self.log_freq == 0) or is_final_step:
            diagnostic_summaries = self._summarize_and_reset_diagnostics()
            if self.accelerator.is_main_process:
                logger.info(
                    "Step: %d \t\tLoss: %.6f \t\tLR: %.8f", self.step, mean_loss, lr
                )
                self._log_diagnostics(diagnostic_summaries)
                append_diagnostics(
                    self.config.out_dir,
                    method="rsbm",
                    step=self.step,
                    phase=(
                        "pretrain"
                        if self.step <= self.n_pretrain_steps
                        else "pretrain+finetune"
                        if self.step
                        - max(
                            values["steps"] for values in diagnostic_summaries.values()
                        )
                        < self.n_pretrain_steps
                        else "finetune"
                    ),
                    max_grad_norm=self.max_grad_norm,
                    summaries=diagnostic_summaries,
                )
        if checkpoint_due:
            self.save_checkpoint(os.path.join(self.checkpoint_dir, f"c_{self.step}"))
        if not is_final_step and self._evaluation_scheduled():
            self.evaluate()
        self._check_interrupt()

    # Each direction's loss is the squared drift error over sigma^2, as in the
    # paper's objective, averaged over elements.
    # Only the network runs under training.precision; targets and losses are fp32.
    def fwd_loss_fn(self, x1, xt, t) -> th.Tensor:
        target = self.bridge.sigma**2 * self.bridge.score_1_given_t(x1, xt, t)
        with autocast(xt.device, self.precision):
            pred = self.model(xt, t, direction=1)
        return mse_loss(pred.to(target.dtype), target) / self.bridge.sigma**2

    def bwd_loss_fn(self, x0, xt, t) -> th.Tensor:
        target = self.bridge.sigma**2 * self.bridge.score_t_given_0(xt, x0, t)
        with autocast(xt.device, self.precision):
            pred = self.model(xt, 1 - t, direction=-1)
        return mse_loss(pred.to(target.dtype), target) / self.bridge.sigma**2

    def grad_step(
        self, fwd_kwargs, bwd_kwargs
    ) -> tuple[th.Tensor, dict[str, dict[str, th.Tensor]]]:
        self.model.zero_grad(set_to_none=True)
        fwd_loss = self.fwd_loss_fn(**fwd_kwargs)
        bwd_loss = self.bwd_loss_fn(**bwd_kwargs)
        # Directional losses are reporting only: the shared network still uses
        # their mean in one backward pass and one optimizer step.
        loss = 0.5 * (fwd_loss + bwd_loss)
        require_finite(
            th.stack([fwd_loss, bwd_loss, loss]),
            f"RSBM loss at update {self.step + 1}",
            self.accelerator,
        )
        self.accelerator.backward(loss)
        grad_norm = clip_and_measure_grad_norm(
            self.accelerator, self.optimizer, self.max_grad_norm, device=loss.device
        )
        require_finite(
            grad_norm, f"RSBM gradient norm at update {self.step + 1}", self.accelerator
        )
        self.optimizer.step()
        return loss.detach(), {
            "forward": {"loss": fwd_loss.detach()},
            "backward": {"loss": bwd_loss.detach()},
            "shared": {
                "loss": loss.detach(),
                **gradient_diagnostics(grad_norm, self.max_grad_norm),
            },
        }

    def pretrain_step(self, x0, x1):
        """One update on independent endpoint pairs, half per direction."""
        B = x0.shape[0]
        b = B // 2

        times = data_utils.sample_timesteps(B, self.device, self.train_gen)
        xt = self.bridge.sample_given_01(x0, x1, times, generator=self.train_gen)

        result = self.grad_step(
            fwd_kwargs={"x1": x1[:b], "xt": xt[:b], "t": times[:b]},
            bwd_kwargs={"x0": x0[b:], "xt": xt[b:], "t": times[b:]},
        )
        self.directional_updates[+1] += 1
        self.directional_updates[-1] += 1
        return result

    def sample_finetune_endpoints(self, x0, x1):
        if self.use_ema:
            sampling_model = self.get_ema_inference_model()
        else:
            sampling_model = self.model
        sampling_model.eval()
        # forward_euler casts each drift back, so under bf16 autocast Euler
        # states, noise and reflection stay fp32.
        with autocast(x0.device, self.finetune_sampling_precision):
            if self.joint_finetune_sampling:
                # One Euler pass over both directions: the network conditions
                # each sample on its direction, so x0 runs forward, x1 backward.
                directions = th.cat(
                    [
                        th.ones(len(x0), dtype=th.int64, device=x0.device),
                        th.full((len(x1),), -1, dtype=th.int64, device=x1.device),
                    ]
                )
                x_hat = self.sde.forward_euler(
                    sampling_model,
                    th.cat([x0, x1]),
                    direction=directions,
                    generator=self.train_gen,
                )
                x1_hat, x0_hat = x_hat[: len(x0)], x_hat[len(x0) :]
            else:
                x1_hat = self.sde.forward_euler(
                    sampling_model,
                    x0,
                    direction=1,
                    generator=self.train_gen,
                )
                x0_hat = self.sde.forward_euler(
                    sampling_model,
                    x1,
                    direction=-1,
                    generator=self.train_gen,
                )
        self.model.train()
        self.fresh_paths[+1] += x0.shape[0]
        self.fresh_paths[-1] += x1.shape[0]
        return x0, x1, x0_hat, x1_hat

    def finetune_step(self, endpoints):
        """One update on fresh bridge points between the simulated pairs."""
        x0, x1, x0_hat, x1_hat = endpoints

        times_fwd = data_utils.sample_timesteps(len(x1), self.device, self.train_gen)
        times_bwd = data_utils.sample_timesteps(len(x0), self.device, self.train_gen)
        xt_fwd = self.bridge.sample_given_01(
            x0_hat, x1, times_fwd, generator=self.train_gen
        )
        xt_bwd = self.bridge.sample_given_01(
            x0, x1_hat, times_bwd, generator=self.train_gen
        )

        result = self.grad_step(
            fwd_kwargs={"x1": x1, "xt": xt_fwd, "t": times_fwd},
            bwd_kwargs={"x0": x0, "xt": xt_bwd, "t": times_bwd},
        )
        self.directional_updates[+1] += 1
        self.directional_updates[-1] += 1
        return result

    def _resume_contract(self):
        config = OmegaConf.to_container(self.config, resolve=True)
        for key in (
            "resume",
            "checkpoint",
            "log_freq",
            "compile_ema",
            "compile_model",
            "max_training_seconds",
        ):
            config["training"].pop(key, None)
        return {
            **{
                key: config[key]
                for key in ("method", "seed", "model", "sde", "data", "training")
            },
            "device_type": self.device.type,
            "num_processes": self.accelerator.num_processes,
        }

    def _state_dict(self):
        numpy_state = np.random.get_state()
        state = {
            "version": 1,
            "contract": self._resume_contract(),
            "phase": ("pretrain" if self.step < self.n_pretrain_steps else "finetune"),
            "plans": {
                phase: {k: plan.state_dict() for k, plan in plans.items()}
                for phase, plans in (
                    ("pretrain", self.pretrain_plans),
                    ("finetune", self.finetune_plans),
                )
            },
            "diagnostic_totals": self._diagnostic_totals,
            "python_rng": random.getstate(),
            "numpy_rng": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
            "torch_rng": th.get_rng_state(),
            "cuda_rng": th.cuda.get_rng_state(self.device)
            if self.device.type == "cuda"
            else None,
            "mps_rng": th.mps.get_rng_state() if self.device.type == "mps" else None,
            "step": self.step,
            "losses": self.losses,
            "directional_losses": self.directional_losses,
            "training_seconds": self.training_seconds,
            "sampling_seconds": self.sampling_timer.seconds,
            "pretrain_seconds": self.pretrain_seconds,
            "evaluation_seconds": self.evaluation_timer.seconds,
            "directional_updates": self.directional_updates,
            "fresh_paths": self.fresh_paths,
            "data_gen": self.data_gen.get_state(),
            "train_gen": self.train_gen.get_state(),
            "finetune_data_gen": self.finetune_data_gen.get_state(),
        }
        # Endpoint pairs simulated for updates still to come.
        if self._endpoint_cache:
            state["endpoint_cache"] = [
                tuple(value.clone() for value in endpoints)
                for endpoints in self._endpoint_cache
            ]
        if self.eval_freq_unit == "seconds":
            state["next_evaluation_seconds"] = self._next_evaluation_seconds
        return state

    def _load_state_dict(self, state) -> None:
        self.step = state["step"]
        self.start_step = state["step"]
        self.losses = state["losses"]
        self.directional_losses = state["directional_losses"]
        self.training_seconds = state["training_seconds"]
        self.sampling_timer.seconds = state["sampling_seconds"]
        self.pretrain_seconds = state["pretrain_seconds"]
        self.evaluation_timer.seconds = state["evaluation_seconds"]
        self.directional_updates = state["directional_updates"]
        self.fresh_paths = state["fresh_paths"]
        self.data_gen.set_state(state["data_gen"].cpu())
        self.train_gen.set_state(state["train_gen"].cpu())
        self.finetune_data_gen.set_state(state["finetune_data_gen"].cpu())
        self._endpoint_cache = list(state.get("endpoint_cache", []))
        if self.eval_freq_unit == "seconds":
            self._next_evaluation_seconds = state["next_evaluation_seconds"]
        for phase, plans in (
            ("pretrain", self.pretrain_plans),
            ("finetune", self.finetune_plans),
        ):
            for k, plan in plans.items():
                plan.load_state_dict(state["plans"][phase][k])
        self._diagnostic_totals = state["diagnostic_totals"]
        random.setstate(state["python_rng"])
        kind, keys, *rest = state["numpy_rng"]
        np.random.set_state((kind, np.asarray(keys, dtype=np.uint32), *rest))
        th.set_rng_state(state["torch_rng"].cpu())
        if self.device.type == "cuda":
            th.cuda.set_rng_state(state["cuda_rng"].cpu(), self.device)
        if self.device.type == "mps":
            th.mps.set_rng_state(state["mps_rng"].cpu())

    @property
    def _state_file_name(self):
        return f"misc-{self.accelerator.process_index}.pt"

    def save_checkpoint(self, out_dir: str) -> None:
        if not self._safe_boundary:
            raise RuntimeError("RSBM checkpoints require an update boundary.")
        with self._exclude_from_training_time():
            self.accelerator.wait_for_everyone()
            marker = os.path.join(out_dir, "complete.json")
            if self.accelerator.is_main_process and os.path.exists(marker):
                os.remove(marker)
            self.accelerator.wait_for_everyone()
            self.accelerator.save_state(out_dir, safe_serialization=True)

            th.save(self._state_dict(), os.path.join(out_dir, self._state_file_name))
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                # Written last: incomplete checkpoint writes are rejected.
                with open(os.path.join(out_dir, "complete.json"), "w") as stream:
                    json.dump({"step": self.step}, stream)
                prune_training_checkpoints(
                    self.checkpoint_dir,
                    self.training_conf.checkpoint.get("keep_last"),
                    self.step,
                )
            self.accelerator.wait_for_everyone()

    def save_final_model(self) -> None:
        """Save each evaluated weight set alone, with the run config."""
        with self._exclude_from_training_time():
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                for weights in self.eval_weights:
                    models.save_final_model(
                        os.path.join(
                            self.checkpoint_dir, models.final_model_file_name(weights)
                        ),
                        self._evaluated_weights(weights),
                        self.config,
                        step=self.step,
                        weights=weights,
                    )

    def _evaluation_scheduled(self) -> bool:
        """Whether eval.freq schedules an evaluation at this update boundary."""
        if self.eval_freq is None or self._update_limit_reached():
            return False
        if self.eval_freq_unit == "seconds":
            return self._elapsed_training_seconds() >= self._next_evaluation_seconds
        return self.step % self.eval_freq == 0

    def load_checkpoint(self, dir: str) -> None:
        if dir is None:
            # A fresh run first evaluates its initial weights, at step 0.
            self._evaluation_due_at_start = self._evaluation_scheduled()
            return
        if not os.path.isfile(os.path.join(dir, "complete.json")):
            raise ValueError(
                "RSBM checkpoint is incomplete or uses an unsupported format."
            )
        state = th.load(
            os.path.join(dir, self._state_file_name),
            map_location=self.device,
            weights_only=True,
        )
        if state.get("version") != 1:
            raise ValueError("RSBM checkpoint uses an unsupported format.")
        if state["contract"] != self._resume_contract():
            raise ValueError(
                "RSBM checkpoint experiment settings differ from this run."
            )
        step = state["step"]
        with open(os.path.join(dir, "complete.json")) as stream:
            if json.load(stream)["step"] != step:
                raise ValueError(
                    "RSBM checkpoint completion record does not match its state."
                )
        self.accelerator.load_state(dir, {"weights_only": True})
        self._load_state_dict(state)
        if self.accelerator.is_main_process:
            # A boundary checkpoint precedes that step's scheduled evaluation.
            archive_abandoned_history(
                self.config.out_dir, step, include_step=True, resumed_from=dir
            )
        self.accelerator.wait_for_everyone()
        # Recreate an evaluation that was scheduled just after the checkpoint.
        self._evaluation_due_at_start = self._evaluation_scheduled()


class AlphaIMFTrainerDoubleNet(AlphaIMFTrainer):
    # Separate networks cannot share a forward pass across directions.
    joint_finetune_sampling = False

    def grad_step(
        self, fwd_kwargs, bwd_kwargs
    ) -> tuple[th.Tensor, dict[str, dict[str, th.Tensor]]]:
        losses = []
        step_diagnostics = {}
        pending_gradients = {}

        for direction, loss_fn, kwargs in [
            (1, self.fwd_loss_fn, fwd_kwargs),
            (-1, self.bwd_loss_fn, bwd_kwargs),
        ]:
            optimizer = self.optimizer[direction]
            self.model.zero_grad(set_to_none=True)
            loss = loss_fn(**kwargs)
            require_finite(
                loss,
                f"RSBM loss at update {self.step + 1} direction {direction:+d}",
                self.accelerator,
            )
            self.accelerator.backward(loss)
            grad_norm = clip_and_measure_grad_norm(
                self.accelerator,
                optimizer,
                self.max_grad_norm,
                device=loss.device,
            )
            require_finite(
                grad_norm,
                f"RSBM gradient norm at update {self.step + 1} direction {direction:+d}",
                self.accelerator,
            )
            # Retain gradient references while clearing the model for the next
            # DDP backward. Neither directional optimizer steps until both pass.
            pending_gradients[direction] = [
                (parameter, parameter.grad)
                for group in optimizer.param_groups
                for parameter in group["params"]
            ]
            losses.append(loss.detach())
            step_diagnostics[DIRECTION_GROUPS[direction]] = {
                "loss": loss.detach(),
                **gradient_diagnostics(grad_norm, self.max_grad_norm),
            }

        mean_loss = th.stack(losses).mean()
        require_finite(
            mean_loss, f"RSBM loss at update {self.step + 1}", self.accelerator
        )
        for direction, gradients in pending_gradients.items():
            for parameter, grad in gradients:
                parameter.grad = grad
            self.optimizer[direction].step()
            if direction == 1:
                self.optimizer[direction].zero_grad(set_to_none=True)

        return mean_loss, step_diagnostics


def get_trainer(model_class):
    if model_class is models.MultiNet:
        return AlphaIMFTrainerDoubleNet
    else:
        return AlphaIMFTrainer
