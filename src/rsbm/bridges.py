import numpy as np
import torch as th

from ..utils import make_broadcastable
from .reflectors import Reflector


class BrownianBridge:
    # x_t^{0, 1} = (1 - t) x0 + t x1 + sigma (B_t - t B_1)
    def __init__(self, sigma: float):
        self.sigma = sigma

    def _variance(self, t_start, t_end):
        """Variance the reference accumulates from t_start to t_end."""
        return self.sigma**2 * (t_end - t_start)

    def _bridge_statistics(self, x0, x1, t):
        """Mean and standard deviation of x_t given x0 and x1."""
        t = make_broadcastable(t, x0)
        bridge_mean = (1 - t) * x0 + t * x1
        bridge_std = self.sigma * th.sqrt(t * (1 - t))
        return bridge_mean, bridge_std

    def sample_given_01(self, x0, x1, t, generator: th.Generator | None = None):
        bridge_mean, bridge_std = self._bridge_statistics(x0, x1, t)
        noise = th.randn(
            x0.shape, dtype=x0.dtype, device=x0.device, generator=generator
        )
        return bridge_mean + bridge_std * noise

    def score_1_given_t(self, x1, xt, t):
        # Gradient is with respect to xt.
        variance_t_to_1 = self._variance(make_broadcastable(t, x1), 1)
        return (x1 - xt) / variance_t_to_1

    def score_t_given_0(self, xt, x0, t):
        # Gradient is with respect to xt.
        variance_0_to_t = self._variance(0, make_broadcastable(t, x0))
        return -(xt - x0) / variance_0_to_t


class ReflectedBrownianBridge(BrownianBridge):
    def __init__(self, sigma: float, reflector: Reflector):
        super().__init__(sigma)
        self.reflector = reflector

    def _unfolded_endpoint_probabilities(self, x0, x1_reflections):
        """Probability of each unreflected endpoint, given x0, over [0, 1]."""
        variance_0_to_1 = self._variance(0, 1)
        return th.softmax(-((x1_reflections - x0) ** 2) / (2 * variance_0_to_1), dim=0)

    def sample_given_01(self, x0, x1, t, generator: th.Generator | None = None):
        batch_size = x0.shape[0]
        signal_shape = x0.shape[1:]
        signal_size = np.prod(signal_shape)

        # Sample an unreflected endpoint independently in every dimension.
        x1 = x1.reshape(-1)
        x1_reflections = self.reflector.get_reflections(x1)
        x0 = x0.reshape(-1).unsqueeze(0)
        probabilities = self._unfolded_endpoint_probabilities(x0, x1_reflections)
        indices = th.multinomial(
            probabilities.mT, 1, generator=generator, replacement=True
        ).squeeze(-1)
        x1_sampled = x1_reflections[
            indices, th.arange(batch_size * signal_size, device=x0.device)
        ].reshape(batch_size, *signal_shape)
        x0 = x0.squeeze(0).reshape(batch_size, *signal_shape)

        bridge_mean, bridge_std = self._bridge_statistics(x0, x1_sampled, t)
        noise = th.randn(
            x0.shape, dtype=x0.dtype, device=x0.device, generator=generator
        )
        return self.reflector.reflect(bridge_mean + bridge_std * noise)

    def score_1_given_t(self, x1, xt, t):
        # Gradient is with respect to xt.
        variance_t_to_1 = self._variance(make_broadcastable(t, x1).unsqueeze(0), 1)
        x1_reflections = self.reflector.get_reflections(x1)
        xt = xt.unsqueeze(0)
        exponents = -((x1_reflections - xt) ** 2) / (2 * variance_t_to_1)
        exponent_gradients = (x1_reflections - xt) / variance_t_to_1
        weights = th.exp(exponents - th.max(exponents, dim=0, keepdim=True).values)
        return th.sum(exponent_gradients * weights, dim=0) / th.sum(weights, dim=0)

    def score_t_given_0(self, xt, x0, t):
        # Gradient is with respect to xt.
        variance_0_to_t = self._variance(0, make_broadcastable(t, x0).unsqueeze(0))
        x0_reflections = self.reflector.get_reflections(x0)
        xt = xt.unsqueeze(0)
        exponents = -((xt - x0_reflections) ** 2) / (2 * variance_0_to_t)
        exponent_gradients = -(xt - x0_reflections) / variance_0_to_t
        weights = th.exp(exponents - th.max(exponents, dim=0, keepdim=True).values)
        return th.sum(exponent_gradients * weights, dim=0) / th.sum(weights, dim=0)
