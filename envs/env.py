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
from models.model import ModelConfig, Model


@dataclass(kw_only = True)
class EnvConfig():
    robot_cfg:      RobotConfig 
    robot_class:    type[Robot]
    field_cfg:      FieldConfig
    field_class:    type[Field]
    model_cfg:      ModelConfig
    model_class:    type[Model]
    policy_freq:    int
    sim_freq:       int     = 500
    env_spacing:    float   = 0.0
    num_envs:       int     = 1
    show_viewer:    bool    = True
    self_collision: bool    = True


class Env(ABC):
    @torch.compiler.disable
    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        self.num_envs = cfg.num_envs
        self.num_agents = 1
        self.is_vector_env = True
        self.all_envs_idx = torch.arange(self.num_envs, 
                                         dtype=torch.long, 
                                         device=gs.device)
        assert self.cfg.sim_freq % self.cfg.policy_freq == 0,  \
               "sim_freq must be divisible by policy_freq"
        self.scene = gs.Scene(
            viewer_options = gs.options.ViewerOptions(
                camera_pos    = (0, -3.5, 2.5),
                camera_lookat = (0.0, 0.0, 0.5),
                camera_fov    = 30,
                res           = (960, 640),
                max_FPS       = 60,
            ),
            sim_options = gs.options.SimOptions(
                dt = 1.0 / self.cfg.policy_freq,
                substeps = self.cfg.sim_freq // self.cfg.policy_freq,
            ),
            rigid_options=gs.options.RigidOptions(
                enable_self_collision=self.cfg.self_collision,
                tolerance=1e-6,
                max_collision_pairs=20,
            ),
            show_viewer = self.cfg.show_viewer,
        )
        self.build()
        self.scene.build(n_envs=self.cfg.num_envs, \
                env_spacing=(self.cfg.env_spacing, self.cfg.env_spacing))
        self.config()
    
    def build(self):
        ''' Is called after the scene is created, but before the scene is built. '''
        self.field = self.cfg.field_class(self.cfg.field_cfg, self.scene)
        self.robot = self.cfg.robot_class(self.cfg.robot_cfg, self.scene)
        self.model = self.cfg.model_class(self.cfg.model_cfg, self.scene)
        self.field.build()
        self.robot.build()
        self.model.build()

    def config(self):
        ''' Is called after the scene is built, but before the first reset. '''
        self.field.config()
        self.robot.config()
        self.model.config()
        self.observation_space = self.model.observation_space
        self.action_space = self.model.action_space
    
    def step(self, action: torch.Tensor
             ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                        dict[str,torch.Tensor]]: 
        action = self.model.preprocess_action(action)
        self.robot.step(action=action)
        self.scene.step()
        kwargs = self.get_state(envs_idx=self.all_envs_idx)
        next_observation    = self.model.build_observation(envs_idx=self.all_envs_idx, **kwargs)
        reward              = self.model.build_reward(envs_idx=self.all_envs_idx, **kwargs)
        terminated          = self.model.build_terminated(envs_idx=self.all_envs_idx, **kwargs)
        truncated           = self.model.build_truncated(envs_idx=self.all_envs_idx, **kwargs)
        info                = self.model.build_info(envs_idx=self.all_envs_idx, **kwargs)
        need_reset = torch.logical_or(terminated, truncated).squeeze(1)
        if need_reset.any():
            reset_idx = torch.nonzero(need_reset).squeeze(1)
            reset_observation, reset_info = self.reset(reset_idx)
            next_observation[reset_idx] = reset_observation
        return (next_observation, reward, terminated, truncated, info)
    
    def reset(self, envs_idx: torch.Tensor | None = None
             ) -> tuple[torch.Tensor, dict]: 
        if envs_idx is None:
            envs_idx = self.all_envs_idx
        self.robot.reset(envs_idx=envs_idx)
        self.field.reset(envs_idx=envs_idx)
        self.model.reset(envs_idx=envs_idx)
        kwargs = self.get_state(envs_idx=envs_idx)
        return (self.model.build_observation(envs_idx=envs_idx, **kwargs), 
                self.model.build_info(envs_idx=envs_idx, **kwargs))
  
    def get_state(self, envs_idx: torch.Tensor) -> dict[str, torch.Tensor]:
        return {**self.robot.get_state(envs_idx = envs_idx), 
                **self.field.get_state(envs_idx = envs_idx)} 
    
    def close(self):
        pass

    def render(self):
        pass
