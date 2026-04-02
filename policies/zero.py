from __future__ import annotations

import torch


class ZeroPolicy:
    def __init__(self, *, device: torch.device | str = "cpu"):
        self.device = torch.device(device)

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.zeros((obs.shape[0], 3), dtype=torch.float, device=obs.device)


Policy = ZeroPolicy
