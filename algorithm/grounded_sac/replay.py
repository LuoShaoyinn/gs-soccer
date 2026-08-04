from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ReplayBatch:
    observation: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    next_observation: torch.Tensor
    success: torch.Tensor
    terminal: torch.Tensor
    human_suffix: torch.Tensor

    def to(self, device: torch.device | str) -> "ReplayBatch":
        return ReplayBatch(**{name: getattr(self, name).to(device) for name in self.__dataclass_fields__})


class VectorReplayBuffer:
    """One physical replay with index views for human suffixes.

    Every row stores its own next observation, so vector-environment resets do
    not corrupt terminal transitions. `human_suffix` is an index-only view in
    spirit: it is a tag on the sole physical transition row, not copied data.
    Runtime controller/familiarity diagnostics are TensorBoard-only and are
    intentionally not retained in replay.
    """

    def __init__(self, capacity: int, observation_dim: int, action_dim: int, *, device: str = "cpu") -> None:
        self.capacity = int(capacity)
        self.device = torch.device(device)
        self.observation = torch.empty((capacity, observation_dim), dtype=torch.float32, device=self.device)
        self.action = torch.empty((capacity, action_dim), dtype=torch.float32, device=self.device)
        self.reward = torch.empty(capacity, dtype=torch.float32, device=self.device)
        self.next_observation = torch.empty((capacity, observation_dim), dtype=torch.float32, device=self.device)
        self.success = torch.empty(capacity, dtype=torch.bool, device=self.device)
        self.terminal = torch.empty(capacity, dtype=torch.bool, device=self.device)
        self.human_suffix = torch.zeros(capacity, dtype=torch.bool, device=self.device)
        self.position = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    @torch.no_grad()
    def add_batch(self, batch: ReplayBatch) -> torch.Tensor:
        batch = batch.to(self.device)
        n = len(batch.reward)
        if n > self.capacity:
            raise ValueError("one vector step exceeds replay capacity")
        indices = (torch.arange(n, device=self.device) + self.position) % self.capacity
        for name in ReplayBatch.__dataclass_fields__:
            getattr(self, name)[indices] = getattr(batch, name)
        self.human_suffix[indices] = False
        self.position = (self.position + n) % self.capacity
        self.size = min(self.capacity, self.size + n)
        return indices

    @torch.no_grad()
    def mark_human_suffix(self, indices: list[int]) -> None:
        if indices:
            self.human_suffix[torch.as_tensor(indices, device=self.device)] = True

    def _sample_indices(self, count: int, *, human_suffix: bool) -> torch.Tensor:
        valid = self.human_suffix[:self.size] if human_suffix else torch.ones(self.size, dtype=torch.bool, device=self.device)
        candidates = torch.nonzero(valid, as_tuple=False).squeeze(1)
        if len(candidates) == 0:
            raise RuntimeError("requested replay view has no rows")
        return candidates[torch.randint(len(candidates), (count,), device=self.device)]

    def sample(self, count: int, target_device: torch.device | str, *, human_suffix: bool = False) -> ReplayBatch:
        idx = self._sample_indices(count, human_suffix=human_suffix)
        return ReplayBatch(**{name: getattr(self, name)[idx].to(target_device, non_blocking=True) for name in ReplayBatch.__dataclass_fields__})

    def all_human_observations(self) -> torch.Tensor:
        ids = torch.nonzero(self.human_suffix[:self.size], as_tuple=False).squeeze(1)
        if len(ids) == 0:
            raise RuntimeError("no successful human suffixes have been recorded")
        return self.observation[ids]
