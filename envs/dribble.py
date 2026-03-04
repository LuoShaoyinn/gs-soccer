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
    robot_reset_pos:   np.ndarray  = field(default_factory= \
            lambda: np.array([-0.5, 0.0], dtype=np.float32))
    robot_reset_noise:  float       = 0.1
    ball_reset_pos:     np.ndarray  = field(default_factory= \
            lambda: np.array([0.0, 0.0], dtype=np.float32))
    ball_reset_noise:   float       = 0.1
    ball_range:         np.ndarray  = field(default_factory= \
            lambda: np.array([12, 9], dtype=np.float32))
    terminate_step_limit:   int     = 2000
    freq_ratio:         int         = 2
    self_collision:     bool        = False
    ball_loss_distance: float       = 1.5


class DribbleEnv(Env):
    cfg: DribbleEnvConfig
    
    @torch.no_grad()
    @torch.compile()
    def build(self): 
        self.robot = self.cfg.robot_class(cfg=self.cfg.robot_cfg, scene=self.scene)
        self.field = self.cfg.field_class(cfg=self.cfg.field_cfg, scene=self.scene)
        self.robot.build()
        self.field.build()

    @torch.no_grad()
    @torch.compile()
    def config(self):
        self.observation_space = self.robot.observation_space
        self.action_space = self.robot.action_space
        self.robot.config()
        self.field.config() 
        self.all_envs_idx = torch.arange(self.num_envs, 
                                         dtype=torch.long, 
                                         device=gs.device)
        self.step_count = torch.zeros((self.num_envs, ), 
                                      dtype=torch.long, 
                                      device=gs.device)
        self.ball_reset_pos = torch.from_numpy(self.cfg.ball_reset_pos).to(gs.device)
        self.robot_pos = torch.from_numpy(self.cfg.robot_reset_pos).to(gs.device)


    @torch.no_grad()
    @torch.compiler.disable
    def step(self, action: torch.Tensor, envs_idx: torch.Tensor | None = None
             ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                        dict[str,torch.Tensor]]: 
        self.step_count[envs_idx] += 1
        for i in range(self.cfg.freq_ratio):
            self.robot.step(action=action, envs_idx=self.all_envs_idx)
            self.scene.step()
        kwargs = self.get_state(envs_idx=self.all_envs_idx)
        kwargs["cmd_vel"] = action
        next_observation    = self.build_observation(**kwargs)
        reward              = self.build_reward(**kwargs)
        terminated          = self.build_terminated(**kwargs)
        truncated           = self.build_truncated(**kwargs)
        info                = self.build_info(**kwargs)
        need_reset = torch.logical_or(terminated, truncated)
        if need_reset.any():
            reset_idx = torch.nonzero(need_reset.squeeze(1)).squeeze(1)
            reset_observation, reset_info = self.reset(reset_idx)
            next_observation[reset_idx] = reset_observation
        return (next_observation, reward, terminated, truncated, info)
    
    @torch.no_grad()
    @torch.compile()
    def reset(self, envs_idx: torch.Tensor | None = None
              ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if envs_idx is None:
            envs_idx = self.all_envs_idx
        self.step_count[envs_idx] = 0
        robot_pos = self.robot_pos.broadcast_to((len(envs_idx), 2)).clone()
        robot_pos += self.cfg.robot_reset_noise * torch.randn((len(envs_idx), 2), device=gs.device)
        robot_pos = F.pad(robot_pos, (0, 1), value=0.5)
        self.robot.reset(envs_idx=envs_idx, robot_pos=robot_pos)
        ball_pos = self.ball_reset_pos.broadcast_to((len(envs_idx), 2)).clone()
        ball_pos += self.cfg.ball_reset_noise * torch.randn((len(envs_idx), 2), device=gs.device)
        ball_pos = F.pad(ball_pos, (0, 1), value=0.03)
        self.field.reset(envs_idx=envs_idx, ball_pos=ball_pos)
        kwargs = self.get_state(envs_idx=envs_idx)
        kwargs["cmd_vel"] = torch.zeros((len(envs_idx), 3), device=gs.device)
        return (self.build_observation(**kwargs), self.build_info(**kwargs))

    @torch.no_grad()
    @torch.compile()
    def get_state(self, envs_idx: torch.Tensor) -> dict[str, torch.Tensor]:
        return {**self.robot.get_state(envs_idx=envs_idx), 
                **self.field.get_state(envs_idx=envs_idx)}
    
    @torch.no_grad()
    @torch.compile()
    def build_observation(self, **kwargs):
        return self.robot.build_observation(**kwargs)

    @torch.no_grad()
    @torch.compile()
    def build_terminated(self, body_pos: torch.Tensor, ball_pos: torch.Tensor, **kwargs
                         ) -> torch.Tensor: # type: ignore[override]
        robot_fall = body_pos[:, 2] < 0.3
        ball_out_of_range = torch.logical_or(
            torch.abs(ball_pos[:, 0]) > self.cfg.ball_range[0], 
            torch.abs(ball_pos[:, 1]) > self.cfg.ball_range[1]) 
        robot_out_of_range = torch.logical_or(
            torch.abs(body_pos[:, 0]) > self.cfg.ball_range[0],
            torch.abs(body_pos[:, 1]) > self.cfg.ball_range[1])
        ball_too_far_from_robot = torch.linalg.norm(\
                ball_pos[:, 0:2] - body_pos[:, 0:2], dim=-1) > self.cfg.ball_loss_distance
        step_limit_reached = self.step_count >= self.cfg.terminate_step_limit
        return (robot_fall 
                | ball_out_of_range 
                | step_limit_reached 
                | robot_out_of_range 
                | ball_too_far_from_robot).unsqueeze(1)
    
    @torch.no_grad()
    @torch.compile()
    def build_truncated(self, **kwargs) -> torch.Tensor:
        return torch.zeros((self.cfg.num_envs, 1), dtype=torch.bool, device=gs.device)
    
    @torch.no_grad()
    @torch.compile()
    def build_reward(self, cmd_vel, ball_pos, ball_vel, body_pos, **kwargs) -> torch.Tensor: # type: ignore[override]
        rew_ball_vel_x = torch.clip(1.0 - torch.exp(-ball_vel[:, 0] / 0.1), -1.0, 1.0)
        rew_ball_pos_y = torch.exp(-torch.abs(ball_pos[:, 1] / self.cfg.ball_range[1])) - 1.0
        rew_fall = torch.where(body_pos[:, 2] < 0.3, -1.0, 0.0)
        rew_close_to_ball = torch.exp(-torch.linalg.norm(ball_pos[:, 0:2] - body_pos[:, 0:2], dim=1))
        rew_action = -torch.exp(-cmd_vel[:, 0] + torch.abs(cmd_vel[:, 1]))
        return ((rew_ball_vel_x
                + rew_ball_pos_y * 0.03
                + rew_fall * 5.0
                + rew_action * 0.1
                + rew_close_to_ball * 0.5) / (1.0 + 0.03 + 0.1 + 0.5)) .unsqueeze(1)
    
    @torch.no_grad()
    @torch.compile()
    def build_info(self, ball_pos: torch.Tensor, ball_vel: torch.Tensor, 
                   body_pos: torch.Tensor, cmd_vel: torch.Tensor, 
                   **kwargs) -> dict[str, dict[str, torch.Tensor]]: # type: ignore[override]
        rew_ball_vel_x = 1.0 - torch.exp(-ball_vel[:, 0])
        rew_ball_pos_y = torch.exp(-torch.abs(ball_pos[:, 1] / self.cfg.ball_range[1])) - 1.0
        rew_close_to_ball = torch.exp(-torch.linalg.norm(ball_pos[:, 0:2] - body_pos[:, 0:2], dim=1))
        return {"extra": 
                {"ball_pos_x": ball_pos[:, 0].mean().detach(),
                 "ball_pos_y": ball_pos[:, 1].mean().detach(),
                 "ball_vel_x": ball_pos[:, 0].mean().detach(),
                 "rew_ball_vel_x": rew_ball_vel_x.mean().detach(),
                 "rew_ball_pos_y": rew_ball_pos_y.mean().detach(),
                 "rew_close_to_ball": rew_close_to_ball.mean().detach(),
                 "cmd_vel": cmd_vel.mean().detach()
                 }}
