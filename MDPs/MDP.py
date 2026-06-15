# MDP.py
#   Base class for the MDP

import torch
import genesis as gs
import gymnasium as gym
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass(kw_only = True)
class MDPConfig:
    pass

class MDP(ABC):
    def __init__(self, cfg: MDPConfig, scene: gs.Scene):
        self.cfg = cfg
        self.scene = scene

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def config(self):
        pass
    
    @abstractmethod
    def reset(self, envs_idx: torch.Tensor, robot_reset_fn, field_reset_fn):
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
    
    @abstractmethod
    def build_reward(self, envs_idx, **kwargs) -> torch.Tensor:
        pass
    
    @abstractmethod
    def build_terminated(self, envs_idx, **kwargs) -> torch.Tensor:
        pass
    
    @abstractmethod
    def build_truncated(self, envs_idx,  **kwargs) -> torch.Tensor:
        pass
    
    @abstractmethod
    def build_info(self, envs_idx, **kwargs) -> dict[str, torch.Tensor]:
        pass
    
