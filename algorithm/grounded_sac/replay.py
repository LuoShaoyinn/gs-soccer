from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterator

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
        self.total_transitions = 0

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
        self.total_transitions += n
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

    def all_human_actions(self) -> torch.Tensor:
        ids = torch.nonzero(self.human_suffix[:self.size], as_tuple=False).squeeze(1)
        if len(ids) == 0:
            raise RuntimeError("no successful human suffixes have been recorded")
        return self.action[ids]

    @torch.no_grad()
    def retain_human_suffix(self) -> int:
        """Discard non-human rows and compact successful suffixes in-place."""
        indices = torch.nonzero(self.human_suffix[:self.size], as_tuple=False).squeeze(1)
        if len(indices) == 0:
            raise RuntimeError("cannot retain an empty successful human-suffix replay")
        for name in ReplayBatch.__dataclass_fields__:
            source = getattr(self, name)[indices].clone()
            getattr(self, name)[: len(indices)] = source
        self.size = len(indices)
        self.position = self.size % self.capacity
        self.human_suffix[:self.size] = True
        return self.size

    @torch.no_grad()
    def state_dict(self) -> dict[str, object]:
        """Serialize only populated rows, compacted into chronological order.

        This keeps a full replay checkpoint correct after wraparound without
        writing unused capacity.  Sampling does not depend on physical row
        order, so loading the compact form into rows ``[0:size]`` is lossless.
        """
        if self.size < self.capacity:
            indices = torch.arange(self.size, device=self.device)
        else:
            indices = (torch.arange(self.capacity, device=self.device) + self.position) % self.capacity
        return {
            "capacity": self.capacity,
            "size": self.size,
            "total_transitions": self.total_transitions,
            "fields": {
                name: getattr(self, name)[indices].detach().cpu()
                for name in ReplayBatch.__dataclass_fields__
            },
        }

    @torch.no_grad()
    def iter_state_shards(
        self, shard_count: int, *, human_suffix_only: bool = False
    ) -> tuple[dict[str, int], Iterator[dict[str, object]]]:
        """Return a small manifest and lazily materialized CPU replay shards."""
        if shard_count <= 0:
            raise ValueError("shard_count must be positive")
        if human_suffix_only:
            indices = torch.nonzero(
                self.human_suffix[:self.size], as_tuple=False
            ).squeeze(1)
        elif self.size < self.capacity:
            indices = torch.arange(self.size, device=self.device)
        else:
            indices = (
                torch.arange(self.capacity, device=self.device) + self.position
            ) % self.capacity
        size = int(len(indices))
        if size == 0:
            raise RuntimeError("cannot save an empty replay snapshot")
        actual_shards = min(int(shard_count), size)
        boundaries = torch.linspace(
            0, size, actual_shards + 1, dtype=torch.int64
        ).tolist()

        def shards() -> Iterator[dict[str, object]]:
            for shard_id, (begin, end) in enumerate(zip(boundaries[:-1], boundaries[1:], strict=True)):
                shard_indices = indices[begin:end]
                yield {
                    "shard_id": shard_id,
                    "start": begin,
                    "end": end,
                    "fields": {
                        name: getattr(self, name)[shard_indices].detach().cpu()
                        for name in ReplayBatch.__dataclass_fields__
                    },
                }

        manifest = {
            "capacity": self.capacity,
            "size": size,
            "total_transitions": size if human_suffix_only else self.total_transitions,
            "shard_count": actual_shards,
        }
        return manifest, shards()

    @torch.no_grad()
    def load_state_shards(
        self, manifest: dict[str, object], shards: Iterator[dict[str, object]]
    ) -> None:
        """Load shards directly into replay storage without joining them in RAM."""
        size = int(manifest["size"])
        if size > self.capacity:
            raise ValueError(f"checkpoint replay has {size} rows but capacity is {self.capacity}")
        expected_start = 0
        for expected_id, shard in enumerate(shards):
            if int(shard["shard_id"]) != expected_id:
                raise ValueError(f"unexpected replay shard id {shard['shard_id']}")
            begin, end = int(shard["start"]), int(shard["end"])
            if begin != expected_start or end <= begin or end > size:
                raise ValueError(f"invalid replay shard range [{begin}, {end})")
            fields = shard["fields"]
            if not isinstance(fields, dict):
                raise TypeError("checkpoint replay shard fields must be a dictionary")
            for name in ReplayBatch.__dataclass_fields__:
                value = fields.get(name)
                if not isinstance(value, torch.Tensor) or len(value) != end - begin:
                    raise ValueError(f"invalid replay shard field: {name}")
                getattr(self, name)[begin:end] = value.to(self.device)
            expected_start = end
        if expected_start != size:
            raise ValueError(f"replay shards cover {expected_start} rows, expected {size}")
        self.size = size
        self.position = size % self.capacity
        self.total_transitions = int(manifest.get("total_transitions", size))

    @torch.no_grad()
    def load_state_dict(self, state: dict[str, object]) -> None:
        size = int(state["size"])
        if size > self.capacity:
            raise ValueError(f"checkpoint replay has {size} rows but capacity is {self.capacity}")
        fields = state["fields"]
        if not isinstance(fields, dict):
            raise TypeError("checkpoint replay fields must be a dictionary")
        for name in ReplayBatch.__dataclass_fields__:
            value = fields.get(name)
            if not isinstance(value, torch.Tensor) or len(value) != size:
                raise ValueError(f"invalid checkpoint replay field: {name}")
            getattr(self, name)[:size] = value.to(self.device)
        self.size = size
        self.position = size % self.capacity
        self.total_transitions = int(state.get("total_transitions", size))
