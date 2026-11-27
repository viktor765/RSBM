import torch as th

from ..rsb_fbsde.model import ToyPolicy


class ToyPolicyNet(ToyPolicy):
    """RSB-FBSDE's toy network with rsbm's model call signature.

    Like upstream, the time embedding sees t * interval. Here t is rsbm's
    direction-relative integration time in [0, 1]. See src/rsb_fbsde/README.md
    for the network's provenance and license.
    """

    def __init__(self, interval=100, **kwargs):
        super().__init__(interval, T=1.0, **kwargs)

    def forward(self, x: th.Tensor, t: th.Tensor, y=None) -> th.Tensor:
        if y is not None:
            raise NotImplementedError("Conditional ToyPolicyNet not implemented.")
        return super().forward(x, t)
