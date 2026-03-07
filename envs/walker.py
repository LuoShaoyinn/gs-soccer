# single_walker.py
#   Build up a single walker env
#

import torch
import numpy as np
import genesis as gs
import gymnasium as gym
import random
from dataclasses import dataclass, field
from typing import Optional, Callable

from .env import Env, EnvConfig
from robots.robot import RobotConfig, Robot
from fields.field import FieldConfig, Field
from models.model import ModelConfig, Model

@dataclass(kw_only = True)
class WalkEnvConfig(EnvConfig):
    num_envs:       int     = 1
    field_range:    float   = 1.0
    rl_dt:          float   = 0.02
    substeps:       int     = 10
    show_viewer:    bool    = False


class WalkEnv(Env): 
    cfg: WalkEnvConfig

    def config(self):
        super().config()
        self.cmd_vel = torch.rand((self.num_envs, 3)) * 2.0 - 1.0
     
    def reset(self, envs_idx: torch.Tensor | None = None
              ) -> tuple[torch.Tensor, dict]:
        if envs_idx is None:
            envs_idx = self.all_envs_idx
        self.cmd_vel[envs_idx] = (torch.rand((envs_idx.shape[0], 3), 
                                             dtype=torch.float, 
                                             device=gs.device) * 2.0 - 1.0)
        return super().reset(envs_idx)
    

    def get_state(self, envs_idx: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"cmd_vel": self.cmd_vel[envs_idx],
                **super().get_state(envs_idx)}
