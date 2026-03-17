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
    self_collision:     bool        = False
    target_pos:         np.ndarray  = field(default_factory= \
            lambda: np.array([4.5, 0.0], dtype=np.float32))
    robot_reset_dis:    float       = 1.2
    robot_reset_ang:    float       = np.pi
    robot_reset_dis_noise:  float   = 0.6
    robot_reset_ang_noise:  float   = np.pi
    ball_reset_pos:     np.ndarray  = field(default_factory= \
            lambda: np.array([0.0, 0.0], dtype=np.float32))
    ball_reset_noise:   float       = 0.5
    action_noise:       float       = 0.1
    ball_mass:          float       = 0.180
    ball_mass_noise:    float       = 0.050
    ball_damping:       float       = 5e-4
    ball_damping_noise: float       = 0.1   # add noise in log space


class DribbleEnv(Env):
    cfg: DribbleEnvConfig

    def config(self):
        super().config()
        self.all_envs_idx = torch.arange(self.num_envs, 
                                         dtype=torch.long, 
                                         device=gs.device)
        self.ball_reset_pos  = torch.from_numpy(self.cfg.ball_reset_pos).to(gs.device)
        self.target_pos      = torch.from_numpy(self.cfg.target_pos) \
                                    .to(gs.device) \
                                    .broadcast_to((self.num_envs, 2))
        self.ball_damping_log = torch.log(torch.tensor(self.cfg.ball_damping, device=gs.device))

    @torch.compiler.disable
    def __gs_step(self):
        self.scene.step()
    
    @torch.no_grad()
    @torch.compile()
    def step(self, action: torch.Tensor
             ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                        dict[str,torch.Tensor]]: 
        action = self.model.preprocess_action(action)
        for i in range(self.cfg.ctrl_freq_ratio):
            action_noise = (torch.randn_like(action) - 0.5)
            self.robot.step(action=action + action_noise * 2.0 * self.cfg.action_noise)
            self.__gs_step()
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
        B = envs_idx.shape[0]
        def rand(pos: torch.Tensor | float, noise: float, dim):
            return pos + 2.0 * noise * (torch.rand((B, dim), \
                        dtype=torch.float, device=gs.device) - 0.5)
        # Domain randomization
        ball_mass_shift = rand(0.0, self.cfg.ball_mass_noise, 1).squeeze(1)
        ball_damping = torch.exp(self.ball_damping_log + \
                self.cfg.ball_damping_noise * torch.rand((3, ), dtype=torch.float, device=gs.device))
        # randomize reset position
        ball_pos =      rand(self.ball_reset_pos, self.cfg.ball_reset_noise, 2)
        self.field.reset(envs_idx=envs_idx, reset_pos=ball_pos, 
                         ball_mass_shift=ball_mass_shift, 
                         ball_damping=ball_damping)
        robot_pos_r =   rand(self.cfg.robot_reset_dis, self.cfg.robot_reset_dis_noise, 1)
        robot_pos_ang = rand(self.cfg.robot_reset_ang, self.cfg.robot_reset_ang_noise, 1)
        robot_pos = ball_pos + robot_pos_r * \
                                torch.cat([torch.cos(robot_pos_ang), 
                                           torch.sin(robot_pos_ang)], dim=1)
        robot_yaw =     rand(0.0, torch.pi, 1).squeeze(1)
        robot_quat =    torch.zeros((B, 4), device=gs.device, dtype=torch.float)
        robot_quat[:, 0] = torch.cos(robot_yaw / 2)
        robot_quat[:, 3] = torch.sin(robot_yaw / 2)
        self.robot.reset(envs_idx=envs_idx, 
                         reset_pos=robot_pos,
                         reset_quat=robot_quat)
        self.model.reset(envs_idx=envs_idx)
        kwargs = self.get_state(envs_idx=envs_idx)
        return (self.model.build_observation(envs_idx=envs_idx, **kwargs), 
                self.model.build_info(envs_idx=envs_idx, **kwargs))
     
    def get_state(self, envs_idx: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"target_pos": self.target_pos[envs_idx], 
                 **super().get_state(envs_idx=envs_idx)}
