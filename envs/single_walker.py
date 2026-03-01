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

from .robot import RobotConfig, Robot


@dataclass
class SingleWalkerEnvConfig():
    robot_cfg:      RobotConfig
    num_envs:       int     = 1
    field_range:    float   = 1.0
    rl_dt:          float   = 0.02
    substeps:       int     = 10
    show_viewer:    bool    = False


class SingleWalkerEnv():
    def __init__(self, cfg: SingleWalkerEnvConfig):
        super().__init__()
        self.cfg = cfg
        self.num_envs = cfg.num_envs
        self.num_agents = 1
        self.is_vector_env = True
        self.gs_build() 
        self.gs_config()
    
    #@torch.no_grad()
    #@torch.compiler.disable # prevent torch from compiling underlying gs
    def gs_build(self): # scene = field + robot
        self.scene = gs.Scene(
            viewer_options = gs.options.ViewerOptions(
                camera_pos    = (0, -3.5, 2.5),
                camera_lookat = (0.0, 0.0, 0.5),
                camera_fov    = 30,
                res           = (960, 640),
                max_FPS       = 60,
            ),
            sim_options = gs.options.SimOptions(
                dt = self.cfg.rl_dt,
                substeps = self.cfg.substeps,
            ),
            rigid_options=gs.options.RigidOptions(
                enable_self_collision=True,
                tolerance=1e-6,
                max_collision_pairs=20,
            ),
            show_viewer = self.cfg.show_viewer,
        )
        self.plane = self.scene.add_entity(gs.morphs.Plane())
        self.robot = Robot(self.cfg.robot_cfg, self.scene)
        self.robot.gs_build()
        self.scene.build(n_envs=self.cfg.num_envs, \
                env_spacing=(self.cfg.field_range, self.cfg.field_range))

    #@torch.no_grad()
    #@torch.compiler.disable # prevent torch from compiling underlying gs
    def gs_config(self):
        self.observation_space = self.robot.observation_space
        self.action_space = self.robot.action_space
        self.cmd_vel = torch.rand((self.num_envs, 3)) * 2.0 - 1.0
        self.robot.gs_config()
        self.all_envs_idx = torch.arange(self.num_envs, 
                                         dtype=torch.long, 
                                         device=gs.device)
 
    #@torch.no_grad()
    #@torch.compile()
    def step(self, action: torch.Tensor):
        self.robot.step(action=action, envs_idx=self.all_envs_idx)
        self.scene.step()
        kwargs = self.robot.get_state(cmd_vel=self.cmd_vel, 
                                      envs_idx=self.all_envs_idx)
        next_observation    = self.get_observation(**kwargs)
        reward              = self.get_reward(**kwargs)
        terminated          = self.get_terminated(**kwargs)
        truncated           = self.get_truncated(**kwargs)
        info                = self.get_info(**kwargs)
        need_reset = torch.logical_or(terminated, truncated)
        if need_reset.any():
            reset_idx = torch.nonzero(need_reset)
            reset_observation, reset_info = self.reset(reset_idx)
            next_observation[reset_idx] = reset_observation
        return (next_observation, reward, terminated, truncated, info)
    
    #@torch.no_grad()
    #@torch.compile()
    def reset(self, envs_idx: torch.Tensor | None = None
              ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        envs_idx = envs_idx or self.all_envs_idx
        reset_n = envs_idx.shape[0]
        self.cmd_vel[envs_idx] = (torch.rand((reset_n, 3), 
                                             dtype=torch.float, 
                                             device=gs.device) * 2.0 - 1.0)
        self.robot.reset(envs_idx=envs_idx)
        kwargs = self.robot.get_state(envs_idx=envs_idx)
        return (self.get_observation(**kwargs), self.get_info(**kwargs))

    #@torch.no_grad()
    #@torch.compile()
    def get_observation(self, **kwargs):
        return self.robot.get_observation(**kwargs)

    #@torch.no_grad()
    #@torch.compile()
    def get_terminated(self, **kwargs) -> torch.Tensor:
        return torch.zeros((self.cfg.num_envs, ), dtype=torch.bool, device=gs.device)
    
    #@torch.no_grad()
    #@torch.compile()
    def get_truncated(self, **kwargs) -> torch.Tensor:
        return torch.zeros((self.cfg.num_envs, ), dtype=torch.bool, device=gs.device)
    
    #@torch.no_grad()
    #@torch.compile()
    def get_reward(self, **kwargs) -> torch.Tensor:
        return torch.zeros((self.cfg.num_envs, ), dtype=torch.float, device=gs.device)
    
    #@torch.no_grad()
    #@torch.compile()
    def get_info(self, **kwargs) -> dict[str, torch.Tensor]:
        return {}

    #@torch.no_grad()
    def render(self):
        pass

    #@torch.no_grad()
    def close(self):
        pass
