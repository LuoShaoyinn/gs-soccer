from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from models.model import Model


class GameModel(Model, ABC):
    @property
    @abstractmethod
    def observation_space(self) -> list:
        pass

    @property
    @abstractmethod
    def action_space(self) -> list:
        pass

    @abstractmethod
    def preprocess_action(self, action: list[torch.Tensor]
                          ) -> list[torch.Tensor]: # type:ignore[override]
        pass

    @abstractmethod
    def build_observation(self, envs_idx: torch.Tensor, **kwargs
                          ) -> list[torch.Tensor]: # type:ignore[override]
        pass

    @abstractmethod
    def build_reward(self, envs_idx: torch.Tensor, **kwargs
                     ) -> list[torch.Tensor]: # type:ignore[override]
        pass

    @abstractmethod
    def build_terminated(self, envs_idx: torch.Tensor, **kwargs
                         ) -> list[torch.Tensor]: # type:ignore[override]
        pass

    @abstractmethod
    def build_truncated(self, envs_idx: torch.Tensor, **kwargs
                        ) -> list[torch.Tensor]: # type:ignore[override]
        pass
