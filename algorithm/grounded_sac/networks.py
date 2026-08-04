from __future__ import annotations

import torch
from torch import nn


class FrozenObservationNormalizer(nn.Module):
    """A non-trainable normalizer fitted once on the initial demonstrations."""

    def __init__(self, observation_dim: int) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(observation_dim))
        self.register_buffer("std", torch.ones(observation_dim))
        self.register_buffer("fitted", torch.tensor(False))

    @torch.no_grad()
    def fit(self, observations: torch.Tensor) -> None:
        if bool(self.fitted):
            raise RuntimeError("observation normalizer is frozen")
        self.mean.copy_(observations.mean(dim=0))
        self.std.copy_(observations.std(dim=0).clamp_min(1e-3))
        self.fitted.fill_(True)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return (observation - self.mean) / self.std


def _trunk(input_dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
        nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
    )


class VectorActor(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.trunk = _trunk(observation_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, action_dim)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.output(self.trunk(observation)))


class VectorCritic(nn.Module):
    """Direct-state vector critic with task-valid per-horizon bounds."""

    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int, lower_bound: torch.Tensor) -> None:
        super().__init__()
        horizons = len(lower_bound)
        self.register_buffer("lower_bound", lower_bound.detach().clone())
        self.trunk = _trunk(observation_dim + action_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, horizons)

    def forward(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # Every return in this task lies between the all-step-penalty return
        # and the immediate-success return of one.  Bounding the head blocks
        # invalid optimistic values from entering the bootstrap loop.
        unit = torch.sigmoid(self.output(self.trunk(torch.cat((observation, action), dim=-1))))
        return self.lower_bound + (1.0 - self.lower_bound) * unit
