import torch as th
import torch.nn as nn


class ControlDrift(nn.Module):
    """SDE drift sigma Z of a network that outputs the control Z.

    The reference is driftless, so the drift is the scaled control alone.
    sigma is the reference's diffusion, not a parameter: the state dict holds
    only the network's weights.
    """

    def __init__(self, network: nn.Module, sigma: float):
        super().__init__()
        self.network, self.sigma = network, sigma

    def forward(self, x: th.Tensor, t: th.Tensor, y=None) -> th.Tensor:
        return self.sigma * self.network(x, t, y=y)
