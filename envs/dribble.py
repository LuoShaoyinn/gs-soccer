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

from .controlled_robot import ControlledRobotConfig, ControlledRobot
from .field import FieldConfig, Field


@dataclass
class DribbleEnvConfig():
    robot_cfg:      ControlledRobotConfig
    field_cfg:      FieldConfig
    num_envs:       int     = 1
    field_range:    float   = 1.0
    rl_dt:          float   = 0.02
    substeps:       int     = 10
    show_viewer:    bool    = False


class DribbleEnv():
    def __init__(self, cfg: DribbleEnvConfig):
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
        self.robot = ControlledRobot(cfg=self.cfg.robot_cfg, scene=self.scene)
        self.field = Field(cfg=self.cfg.field_cfg, scene=self.scene)
        self.robot.gs_build()
        self.field.gs_build()
        self.scene.build(n_envs=self.cfg.num_envs, \
                env_spacing=(self.cfg.field_range, self.cfg.field_range))

    #@torch.no_grad()
    #@torch.compiler.disable # prevent torch from compiling underlying gs
    def gs_config(self):
        self.observation_space = self.robot.observation_space
        self.action_space = self.robot.action_space
        self.cmd_vel = torch.rand((self.num_envs, 3), device=gs.device) * 2.0 - 1.0
        self.robot.gs_config()
        self.field.gs_config()
 
    #@torch.no_grad()
    #@torch.compile()
    def step(self, action : torch.Tensor):
        ret = self.robot.step(action=action, cmd_vel=self.cmd_vel)
        self.scene.step()
        return ret
    
    #@torch.no_grad()
    #@torch.compile()
    def reset(self, envs_idx: torch.Tensor | None = None
              ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if envs_idx is None:
            envs_idx = torch.arange(self.cfg.num_envs, 
                                    dtype=torch.long, 
                                    device=gs.device)
        reset_n = envs_idx.shape[0]
        self.cmd_vel[envs_idx] = (torch.rand((reset_n, 3), 
                                             dtype=torch.float, 
                                             device=gs.device) * 2.0 - 1.0)
        return self.robot.reset(envs_idx=envs_idx)

    #@torch.no_grad()
    def render(self):
        self.robot.render()

    #@torch.no_grad()
    def close(self):
        self.robot.close()
