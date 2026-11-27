import hashlib
import json
import logging
import random
import signal
import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch as th

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]


def git_revision(repo_dir: Path = REPO_ROOT) -> dict:
    """Return the checked-out commit and whether the work tree has changes.

    Uses the git command when available. Without it (e.g. in a container
    without git), reads the commit from .git directly and reports dirty as None.
    Both are None outside a git checkout.
    """
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir, capture_output=True, text=True, check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=repo_dir, capture_output=True, text=True, check=True,
        ).stdout
        return {"commit": commit, "dirty": bool(status.strip())}
    except (OSError, subprocess.CalledProcessError):
        pass
    git_dir = Path(repo_dir) / ".git"
    try:
        head = (git_dir / "HEAD").read_text().strip()
        if not head.startswith("ref: "):
            return {"commit": head, "dirty": None}
        ref = head.removeprefix("ref: ")
        ref_file = git_dir / ref
        if ref_file.is_file():
            return {"commit": ref_file.read_text().strip(), "dirty": None}
        for line in (git_dir / "packed-refs").read_text().splitlines():
            if line.endswith(" " + ref):
                return {"commit": line.split()[0], "dirty": None}
    except OSError:
        pass
    return {"commit": None, "dirty": None}


class SegmentTimer:
    """Accumulate the wall time spent in repeated code segments.

    On CUDA the segments are timed with events on the current stream, so
    timing adds no host synchronization; pending events are resolved when
    ``seconds`` is read. On CPU, segments are timed with ``time.perf_counter``.
    """

    def __init__(self, device: th.device):
        self.device = th.device(device)
        self._seconds = 0.0
        self._pending = []

    @contextmanager
    def time(self):
        if self.device.type == "cuda":
            start = th.cuda.Event(enable_timing=True)
            end = th.cuda.Event(enable_timing=True)
            start.record()
            try:
                yield
            finally:
                end.record()
                self._pending.append((start, end))
        else:
            started_at = time.perf_counter()
            try:
                yield
            finally:
                self._seconds += time.perf_counter() - started_at

    @property
    def seconds(self) -> float:
        if self._pending:
            self._pending[-1][1].synchronize()
            self._seconds += sum(s.elapsed_time(e) for s, e in self._pending) / 1000
            self._pending.clear()
        return self._seconds

    @seconds.setter
    def seconds(self, value: float) -> None:
        self._pending.clear()
        self._seconds = float(value)


def set_seed(seed: int):
    """Seed the global RNGs (model initialization); torch seeds every device."""
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)


def derive_seed(seed: int, *stream: str | int, bits: int = 32) -> int:
    """The seed of the random stream named by stream, for a run seed.

    Hashing the run seed with the stream's name keeps every stream of every
    run apart; offsets such as seed + k would let run 1's first stream
    replay run 0's second. Torch CPU generators use the low 32 bits.
    """
    payload = json.dumps(
        {"base_seed": int(seed), "parts": stream}, sort_keys=True, separators=(",", ":")
    )
    value = int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")
    return value % (2**bits)


def create_generator(
    seed: int, *stream: str | int, device: th.device | str = "cpu"
) -> th.Generator:
    """A torch generator for the named random stream of a run seed."""
    return th.Generator(device).manual_seed(derive_seed(seed, *stream))


@contextmanager
def fork_global_rng(device: th.device | str):
    """Restore the global RNGs on exit: Python's, NumPy's, and torch's on the
    CPU and on device.

    Evaluators are built, prepared and run inside it, so they never change the
    random numbers of training: dropout draws from torch's global RNG, and
    initializing a metric network or building a DataLoader iterator draws
    from it too.
    """
    device = th.device(device)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    if device.type == "cpu":
        torch_rng = th.random.fork_rng(devices=[])
    else:
        torch_rng = th.random.fork_rng(devices=[device], device_type=device.type)
    try:
        with torch_rng:
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


@contextmanager
def evaluation_mode(module: th.nn.Module):
    """Put module in eval mode, restoring each submodule's mode on exit."""
    modes = [(submodule, submodule.training) for submodule in module.modules()]
    module.eval()
    try:
        yield module
    finally:
        for submodule, training in modes:
            submodule.training = training


@contextmanager
def deferred_interrupt():
    """Turn a first Ctrl-C into a request the caller polls (`requested`) at a
    safe point; a second one raises KeyboardInterrupt at once. Outside the
    main thread, Ctrl-C is left alone."""
    request = SimpleNamespace(requested=False)
    if threading.current_thread() is not threading.main_thread():
        yield request
        return

    def handler(signum, frame):
        if request.requested:
            raise KeyboardInterrupt
        request.requested = True
        logger.warning(
            "Interrupt requested: stopping at the next safe point. "
            "Interrupt again to stop at once."
        )

    previous = signal.signal(signal.SIGINT, handler)
    try:
        yield request
    finally:
        signal.signal(signal.SIGINT, previous)


def get_half_batch_size(batch_size: int) -> int:
    """Return the per-direction half-batch size used by the RSBM objective."""
    if batch_size % 2 != 0:
        raise ValueError(
            "The bidirectional training batch must be even so it can be split "
            f"equally between directions, got {batch_size}."
        )
    return batch_size // 2


def make_broadcastable(tensor: th.Tensor, other: th.Tensor) -> th.Tensor:
    return tensor.view(*tensor.shape, *([1] * (other.dim() - tensor.dim())))
