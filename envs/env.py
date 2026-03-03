# envs.py
#   Abstract envs


import torch
import numpy as np
import genesis as gs
import gymnasium as gym
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from robots.robot import RobotConfig, Robot
from fields.field import FieldConfig, Field


@dataclass
class EnvConfig():
    num_envs:       int     = 1
    field_range:    float   = 1.0
    rl_dt:          float   = 0.02
    substeps:       int     = 10
    show_viewer:    bool    = False


class Env(ABC):
    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        self.num_envs = cfg.num_envs
        self.num_agents = 1
        self.is_vector_env = True
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
        self.build()
        self.scene.build(n_envs=self.cfg.num_envs, \
                env_spacing=(self.cfg.field_range, self.cfg.field_range))
        self.config()
    
    @abstractmethod
    def build(self): 
        ''' Is called after the scene is created, but before the scene is built. '''
        pass

    @abstractmethod
    def config(self):
        ''' Is called after the scene is built, but before the first reset. '''
        pass
 
    
    @abstractmethod
    def get_state(self, envs_idx: torch.Tensor, **kwargs) -> dict[str, torch.Tensor]:
        pass

    @abstractmethod
    def build_observation(self, **kwargs):
        pass

    @abstractmethod
    def build_terminated(self, **kwargs) -> torch.Tensor:
        pass
    
    @abstractmethod
    def build_truncated(self, **kwargs) -> torch.Tensor:
        pass
    
    @abstractmethod
    def build_reward(self, **kwargs) -> torch.Tensor:
        pass
    
    @abstractmethod
    def build_info(self, **kwargs) -> dict[str, torch.Tensor]:
        pass

    def render(self):
        pass

    def close(self):
        pass
