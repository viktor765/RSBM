import math

import torch as th

from ..utils import make_broadcastable
from .normalizations import Normalization


def recommended_n_reflections(total_standard_deviation: float) -> int:
    """Choose an odd image count for a unit-cube diffusion over [0, 1].

    Pass max_i(sigma / box_width_i), NOT sigma in native coordinates. The
    omitted probability and unit-coordinate score error are bounded by 1e-10
    per coordinate, uniformly over endpoints and positive time intervals <= 1.
    """
    sigma = float(total_standard_deviation)
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("Automatic reflection truncation requires finite positive diffusion.")
    if sigma > 2:
        raise ValueError(
            "Automatic reflection truncation is validated only through unit-cube "
            "standard deviation 2.0. Set sde.n_reflections explicitly."
        )
    # m=2 is already sufficient at sigma=0.1 and below. This branch also
    # avoids underflow of sigma**2 for very small positive diffusion.
    if sigma <= 0.1:
        return 5
    variance = sigma**2
    m = 2
    while True:
        # This makes both tail bounds monotone in variance up to sigma**2.
        if m**2 >= 1 + 2 * variance:
            factor = 2 * math.exp(-(m**2 - 1) / (2 * variance))
            omitted_mass = factor * (1 + variance / m)
            score_error = factor * ((2 * m + 1) / variance + 2 + 1 / m)
            if max(omitted_mass, score_error) <= 1e-10:
                return 2 * m + 1
        m += 1


class Reflector:
    def __init__(self, n_reflections: int, normalization: Normalization):
        if n_reflections < 1:
            raise ValueError(f"n_reflections must be >= 1, got {n_reflections}")
        self.n_reflections = n_reflections
        self.normalization = normalization

    def reflect(self, x: th.Tensor) -> th.Tensor:
        """
        Reflect the input tensor x into the unit cube [0, 1]^d.
        """
        x = self.normalization(x)
        x = x % 2
        over_one = x > 1
        x[over_one] = 2 - x[over_one]
        return self.normalization.inv(x)

    def project(self, x: th.Tensor) -> th.Tensor:
        """
        Project the input tensor x into the unit cube [0, 1]^d.
        """
        x = self.normalization(x)
        x = th.clamp(x, 0.0, 1.0)
        return self.normalization.inv(x)

    def drefl_dx(self, device: th.device | None = None) -> th.Tensor:
        """
        Compute the derivative of the reflection function with respect to x.
        """
        n = self.n_reflections
        rng = th.arange(n, device=device) + ((n // 2) % 2)
        return 1 - 2 * (rng % 2)

    def get_reflections(self, x: th.Tensor) -> th.Tensor:
        """
        Generate reflected versions of x. Output shape: (num_reflections, *x.shape).

        Example n=7:
        -2 - x
        -2 + x
         0 - x
         0 + x
         2 - x
         2 + x
         4 - x
        """
        n = self.n_reflections
        rng = th.arange(n, device=x.device) - n // 2 + 1
        offsets = rng - (rng % 2)

        x = x.unsqueeze(0)
        offsets = make_broadcastable(offsets, x)
        signs = make_broadcastable(self.drefl_dx(x.device), x)

        reflections = self.normalization.inv(offsets + signs * self.normalization(x))
        return reflections
