# dribble.py
#   Build up a dribble env
#

import torch
import numpy as np
import genesis as gs
import gymnasium as gym
import random
from dataclasses import dataclass, field
from typing import Optional, Callable

from .env import EnvConfig, Env
from robots.controlled_robot import ControlledRobotWrapperConfig, ControlledRobotWrapper
from fields.field            import FieldConfig, Field


@dataclass(kw_only = True)
class DribbleEnvConfig(EnvConfig):
    robot_cfg:      ControlledRobotWrapperConfig
    robot_class:    type[ControlledRobotWrapper]
    field_cfg:      FieldConfig
    field_class:    type[Field]
    num_envs:       int     = 1
    field_range:    float   = 1.0
    rl_dt:          float   = 0.02
    substeps:       int     = 10
    show_viewer:    bool    = False


class DribbleEnv(Env):
    cfg: DribbleEnvConfig
    
    def build(self): 
        self.robot = self.cfg.robot_class(cfg=self.cfg.robot_cfg, scene=self.scene)
        self.field = self.cfg.field_class(cfg=self.cfg.field_cfg, scene=self.scene)
        self.robot.build()
        self.field.build()

    def config(self):
        self.observation_space = self.robot.observation_space
        self.action_space = self.robot.action_space
        self.robot.config()
        self.field.config() 
        self.all_envs_idx = torch.arange(self.num_envs, 
                                         dtype=torch.long, 
                                         device=gs.device)

    def step(self, action: torch.Tensor, envs_idx: torch.Tensor | None = None
             ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                        dict[str,torch.Tensor]]: 
        self.robot.step(action=action, envs_idx=self.all_envs_idx)
        self.scene.step()
        kwargs = self.get_state(envs_idx=self.all_envs_idx)
        next_observation    = self.build_observation(**kwargs)
        reward              = self.build_reward(**kwargs)
        terminated          = self.build_terminated(**kwargs)
        truncated           = self.build_truncated(**kwargs)
        info                = self.build_info(**kwargs)
        need_reset = torch.logical_or(terminated, truncated)
        if need_reset.any():
            reset_idx = torch.nonzero(need_reset)
            reset_observation, reset_info = self.reset(reset_idx)
            next_observation[reset_idx] = reset_observation
        return (next_observation, reward, terminated, truncated, info)
    
    def reset(self, envs_idx: torch.Tensor | None = None
              ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        envs_idx = envs_idx or self.all_envs_idx
        self.robot.reset(envs_idx=envs_idx)
        self.field.reset(envs_idx=envs_idx)
        kwargs = self.get_state(envs_idx=envs_idx)
        return (self.build_observation(**kwargs), self.build_info(**kwargs))

    def get_state(self, envs_idx: torch.Tensor) -> dict[str, torch.Tensor]:
        return {**self.robot.get_state(envs_idx=envs_idx), 
                **self.field.get_state(envs_idx=envs_idx)}
    
    def build_observation(self, **kwargs):
        return self.robot.get_observation(**kwargs)

    def build_terminated(self, body_pos: torch.Tensor, **kwargs
                         ) -> torch.Tensor: # type: ignore[override]
        return body_pos[:, 2] < 0.3 # terminated if fall
    
    def build_truncated(self, **kwargs) -> torch.Tensor:
        return torch.zeros((self.cfg.num_envs, ), dtype=torch.bool, device=gs.device)
    
    def build_reward(self, **kwargs) -> torch.Tensor:
        return torch.zeros((self.cfg.num_envs, ), dtype=torch.float, device=gs.device)
    
    def build_info(self, **kwargs) -> dict[str, torch.Tensor]:
        return {}
