# dribble.py
#   Build up a dribble env
#

import torch
import numpy as np
import genesis as gs
import gymnasium as gym
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, Callable

from .env import EnvConfig, Env
from robots.controlled_robot import ControlledRobotWrapperConfig, ControlledRobotWrapper
from fields.field            import FieldConfig, Field


@dataclass(kw_only = True)
class DribbleEnvConfig(EnvConfig):
    robot_cfg:          ControlledRobotWrapperConfig
    robot_class:        type[ControlledRobotWrapper]
    field_cfg:          FieldConfig
    field_class:        type[Field]
    ctrl_freq_ratio:    int         = 10
    target_pos:         np.ndarray  = field(default_factory= \
            lambda: np.array([5.0, 0.0], dtype=np.float32))
    robot_reset_pos:    np.ndarray  = field(default_factory= \
            lambda: np.array([-0.8, 0.0], dtype=np.float32))
    robot_reset_noise:  float       = 0.2
    ball_reset_pos:     np.ndarray  = field(default_factory= \
            lambda: np.array([0.0, 0.0], dtype=np.float32))
    ball_reset_noise:   float       = 0.1


class DribbleEnv(Env):
    cfg: DribbleEnvConfig

    def config(self):
        super().config()
        self.all_envs_idx = torch.arange(self.num_envs, 
                                         dtype=torch.long, 
                                         device=gs.device)
        self.ball_reset_pos  = torch.from_numpy( self.cfg.ball_reset_pos).to(gs.device)
        self.robot_reset_pos = torch.from_numpy(self.cfg.robot_reset_pos).to(gs.device)
        self.target_pos      = torch.from_numpy(self.cfg.target_pos) \
                                    .to(gs.device) \
                                    .broadcast_to((self.num_envs, 2))
    
    def step(self, action: torch.Tensor
             ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                        dict[str,torch.Tensor]]: 
        action = self.model.preprocess_action(action)
        for i in range(self.cfg.ctrl_freq_ratio):
            self.robot.step(action=action)
            self.scene.step()
        kwargs = self.get_state(envs_idx=self.all_envs_idx)
        next_observation    = self.model.build_observation(**kwargs)
        reward              = self.model.build_reward(**kwargs)
        terminated          = self.model.build_terminated(**kwargs)
        truncated           = self.model.build_truncated(**kwargs)
        info                = self.model.build_info(**kwargs)
        need_reset = torch.logical_or(terminated, truncated)
        if need_reset.any():
            reset_idx = torch.nonzero(need_reset)
            reset_observation, reset_info = self.reset(reset_idx)
            next_observation[reset_idx] = reset_observation
        return (next_observation, reward, terminated, truncated, info)
     
    def get_state(self, envs_idx: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"target_pos": self.target_pos[envs_idx], 
                 **super().get_state(envs_idx=envs_idx)}
