# SPDX-License-Identifier: Apache-2.0
# Adapted from rSB-FBSDE/models/toy_model/Toy.py, models/utils.py and policy.py
# in WayneDW/Reflected_Schrodinger_Bridge (see README.md).
# Modified for rsbm: local imports, configurable widths and a
# zero-initialized output layer. rsbm wraps it as src/models/toy_policy.py,
# one network per direction.
import math

import torch
from torch import nn


class SiLU(nn.Module):
    def forward(self, x):
        # Keep upstream's expression, including its second derivatives.
        return x * torch.sigmoid(x)


def timestep_embedding(t, dimension):
    half = dimension // 2
    frequencies = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device).float() / half
    )
    phase = t[:, None].float() * frequencies[None]
    embedding = torch.cat((phase.cos(), phase.sin()), dim=-1)
    if dimension % 2:
        embedding = torch.cat((embedding, torch.zeros_like(embedding[:, :1])), -1)
    return embedding


class ResNetFC(nn.Module):
    def __init__(self, hidden_dim, num_res_blocks):
        super().__init__()
        self.map = nn.Linear(2, hidden_dim)
        self.res_blocks = nn.ModuleList(
            nn.Sequential(
                *[
                    layer
                    for _ in range(3)
                    for layer in (nn.Linear(hidden_dim, hidden_dim), SiLU())
                ]
            )
            for _ in range(num_res_blocks)
        )

    def forward(self, x):
        h = self.map(x)
        for block in self.res_blocks:
            h = (h + block(h)) / math.sqrt(2)
        return h


class ToyPolicy(nn.Module):
    """Upstream toy network; predicts Z, so the SDE drift adds g(t) Z."""

    def __init__(
        self,
        interval,
        T=1.0,
        hidden_dim=256,
        time_embed_dim=128,
        num_res_blocks=3,
    ):
        super().__init__()
        self.interval, self.T = interval, T
        self.time_embed_dim = time_embed_dim
        self.t_module = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim),
            SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.x_module = ResNetFC(hidden_dim, num_res_blocks)
        self.out_module = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), SiLU(), nn.Linear(hidden_dim, 2)
        )
        # Z = 0 at initialization; the output layer still gets gradients.
        nn.init.zeros_(self.out_module[-1].weight)
        nn.init.zeros_(self.out_module[-1].bias)

    def forward(self, x, t):
        t = torch.as_tensor(t, dtype=x.dtype, device=x.device).reshape(-1)
        if t.numel() == 1:
            t = t.expand(x.shape[0])
        embedding = timestep_embedding(t / self.T * self.interval, self.time_embed_dim)
        return self.out_module(self.x_module(x) + self.t_module(embedding))
