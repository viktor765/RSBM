"""Data defined as normalized when transformed to unit cube"""

from abc import ABC, abstractmethod

import torch as th


class Normalization(ABC):
    @abstractmethod
    def __call__(self, x: th.Tensor) -> th.Tensor:
        raise NotImplementedError

    @abstractmethod
    def inv(self, x: th.Tensor) -> th.Tensor:
        raise NotImplementedError


class AffineBox(Normalization):
    """Affine map between an axis-aligned box and the unit cube."""

    def __init__(self, lower, upper):
        lower = th.as_tensor(lower, dtype=th.float64)
        upper = th.as_tensor(upper, dtype=th.float64)
        if lower.ndim != 1 or upper.shape != lower.shape:
            raise ValueError("AffineBox bounds must be one-dimensional and match.")
        if not th.all(upper > lower):
            raise ValueError("AffineBox upper bounds must exceed lower bounds.")
        self.lower = tuple(float(value) for value in lower)
        self.upper = tuple(float(value) for value in upper)

    def _bounds(self, x: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        lower = th.as_tensor(self.lower, dtype=x.dtype, device=x.device)
        upper = th.as_tensor(self.upper, dtype=x.dtype, device=x.device)
        if x.ndim > 0 and x.shape[-1] != lower.numel():
            if x.shape[-1] % lower.numel() != 0:
                raise ValueError(
                    "The flattened state size is not divisible by the "
                    "AffineBox dimension."
                )
            repeats = x.shape[-1] // lower.numel()
            lower = lower.repeat(repeats)
            upper = upper.repeat(repeats)
        return lower, upper

    def __call__(self, x: th.Tensor) -> th.Tensor:
        lower, upper = self._bounds(x)
        return (x - lower) / (upper - lower)

    def inv(self, x: th.Tensor) -> th.Tensor:
        lower, upper = self._bounds(x)
        return lower + (upper - lower) * x
