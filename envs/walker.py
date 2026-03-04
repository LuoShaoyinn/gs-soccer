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


@dataclass
class WalkEnvConfig(EnvConfig):
    robot_cfg:      RobotConfig = field(default_factory=lambda: RobotConfig)
    robot_class:    type[Robot] = field(default_factory=lambda: Robot)
    field_cfg:      FieldConfig = field(default_factory=lambda: FieldConfig)
    field_class:    type[Field] = field(default_factory=lambda: Field)
    num_envs:       int     = 1
    field_range:    float   = 1.0
    rl_dt:          float   = 0.02
    substeps:       int     = 10
    show_viewer:    bool    = False


class WalkEnv(Env): 
    cfg: WalkEnvConfig

    @torch.no_grad()
    @torch.compile()
    def build(self):
        self.field = self.cfg.field_class(self.cfg.field_cfg, self.scene)
        self.robot = self.cfg.robot_class(self.cfg.robot_cfg, self.scene)
        self.field.build()
        self.robot.build()

    @torch.no_grad()
    @torch.compile()
    def config(self):
        self.observation_space = self.robot.observation_space
        self.action_space = self.robot.action_space
        self.field.config()
        self.robot.config()
        self.all_envs_idx = torch.arange(self.num_envs, 
                                         dtype=torch.long, 
                                         device=gs.device)
        self.cmd_vel = torch.rand((self.num_envs, 3)) * 2.0 - 1.0
    
    @torch.no_grad()
    @torch.compiler.disable
    def step(self, action: torch.Tensor, envs_idx: torch.Tensor | None = None
             ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                        dict[str,torch.Tensor]]: 
        envs_idx = envs_idx or self.all_envs_idx
        self.robot.step(action=action, envs_idx=self.all_envs_idx)
        self.scene.step()
        kwargs = self.get_state(envs_idx=envs_idx)
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
    
    @torch.no_grad()
    @torch.compile()
    def reset(self, envs_idx: torch.Tensor | None = None
              ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        envs_idx = envs_idx or self.all_envs_idx
        reset_n = envs_idx.shape[0]
        self.cmd_vel[envs_idx] = (torch.rand((reset_n, 3), 
                                             dtype=torch.float, 
                                             device=gs.device) * 2.0 - 1.0)
        self.robot.reset(envs_idx=envs_idx)
        self.field.reset(envs_idx=envs_idx)
        kwargs = self.get_state(envs_idx=envs_idx)
        return (self.build_observation(**kwargs), self.build_info(**kwargs))
    

    @torch.no_grad()
    @torch.compile()
    def get_state(self, envs_idx: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"cmd_vel": self.cmd_vel[envs_idx], 
                **self.robot.get_state(envs_idx = envs_idx), 
                **self.field.get_state(envs_idx = envs_idx)}
 
 
    @torch.no_grad()
    @torch.compile()
    def build_observation(self, **kwargs):
        return self.robot.build_observation(**kwargs)

    @torch.no_grad()
    @torch.compile()
    def build_terminated(self, **kwargs) -> torch.Tensor:
        return torch.zeros((self.cfg.num_envs, ), dtype=torch.bool, device=gs.device)
    
    @torch.no_grad()
    @torch.compile()
    def build_truncated(self, **kwargs) -> torch.Tensor:
        return torch.zeros((self.cfg.num_envs, ), dtype=torch.bool, device=gs.device)
    
    @torch.no_grad()
    @torch.compile()
    def build_reward(self, **kwargs) -> torch.Tensor:
        return torch.zeros((self.cfg.num_envs, ), dtype=torch.float, device=gs.device)
    
    @torch.no_grad()
    @torch.compile()
    def build_info(self, **kwargs) -> dict[str, torch.Tensor]:
        return {}
