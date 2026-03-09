# model.py
#   Based class for modeling

import torch
import genesis as gs
import gymnasium as gym
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

@dataclass(kw_only = True)
class ModelConfig:
    pass

class Model(ABC):
    def __init__(self, cfg: ModelConfig, scene: gs.Scene):
        self.cfg = cfg
        self.scene = scene

    def build(self):
        pass

    def config(self):
        pass
    
    def reset(self, envs_idx: torch.Tensor):
        pass
    
    def preprocess_action(self, action: torch.Tensor) -> torch.Tensor:
        return action
 
    @property
    @abstractmethod
    def observation_space(self) -> gym.spaces.Box:
        pass
    
    @property
    @abstractmethod
    def action_space(self) -> gym.spaces.Box:
        pass 
 
    @abstractmethod
    def build_observation(self, envs_idx, **kwargs) -> torch.Tensor:
        pass
    
    def build_reward(self, envs_idx, body_pos, **kwargs) -> torch.Tensor:
        return torch.zeros((envs_idx.shape[0], 1), dtype=torch.float, device=gs.device)
    
    def build_terminated(self, envs_idx, **kwargs) -> torch.Tensor:
        return torch.zeros((envs_idx.shape[0], 1), dtype=torch.bool, device=gs.device)
    
    def build_truncated(self, envs_idx,  **kwargs) -> torch.Tensor:
        return torch.zeros((envs_idx.shape[0], 1), dtype=torch.bool, device=gs.device)
    
    def build_info(self, envs_idx, **kwargs) -> dict[str, torch.Tensor]:
        return {}
    
