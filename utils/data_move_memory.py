import numpy as np
import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union

from skrl.memories.torch import RandomMemory

class DataMoveMemory(RandomMemory):
    def __init__(self, target_device: str | None = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._target_device = target_device
    def add_samples(self, **tensors: torch.Tensor) -> None:
        for name, tensor in tensors.items():
            tensor.to(self.device)
        super().add_samples(**tensors) 
    def sample_by_index(self, *args, **kwargs):
        kwargs["indexes"] = kwargs["indexes"].to(self.device, non_blocking=True)
        return super().sample_by_index(*args, **kwargs)
    def sample(self, *args, **kwargs):
        batch = super().sample(*args, **kwargs)
        for i in range(len(batch)):
            for j in range(len(batch[i])):
                batch[i][j] = batch[i][j].to(self._target_device, 
                                             non_blocking=True)
        return batch


