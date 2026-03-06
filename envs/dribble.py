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
            lambda: np.array([-0.8, 0.0], dtype=np.float32))
    robot_reset_noise:  float       = 0.2
    ball_reset_pos:     np.ndarray  = field(default_factory= \
            lambda: np.array([0.0, 0.0], dtype=np.float32))
    ball_reset_noise:   float       = 0.1
    terminate_range:    np.ndarray  = field(default_factory= \
            lambda: np.array([12, 9], dtype=np.float32))
    truncated_step_limit:   int     = 400
    freq_ratio:         int         = 5
    self_collision:     bool        = False
    ball_loss_distance: float       = 2.0


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
        self.__cmd_vel = torch.zeros((self.num_envs, 5, 3), device=gs.device)
        tensor_zero = torch.tensor([0.0], dtype=torch.float, device=gs.device)
        self.__reward_info = {
            "rew_ball_x": tensor_zero.mean().detach(),
            "rew_ball_y": tensor_zero.mean().detach(),
            "rew_cmd_smooth": tensor_zero.mean().detach(),
            "rew_close_to_ball": tensor_zero.mean().detach(),
            "rew_rel_y": tensor_zero.mean().detach(),
            "rew_action": tensor_zero.mean().detach(),
            "rew_heading": tensor_zero.mean().detach(),
        }


    @torch.no_grad()
    @torch.compiler.disable
    def step(self, action: torch.Tensor, envs_idx: torch.Tensor | None = None
             ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                        dict[str,torch.Tensor]]: 
        self.__cmd_vel = torch.roll(self.__cmd_vel, shifts=-1, dims=1)
        self.__cmd_vel[:, -1, :] = action
        self.step_count[envs_idx] += 1
        for i in range(self.cfg.freq_ratio):
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
        self.__cmd_vel[envs_idx] = 0.0
        # reset robot and ball position with some noise
        robot_pos = self.robot_pos.broadcast_to((len(envs_idx), 2)).clone()
        robot_pos += self.cfg.robot_reset_noise * torch.randn((len(envs_idx), 2), device=gs.device)
        robot_pos = F.pad(robot_pos, (0, 1), value=0.5)
        self.robot.reset(envs_idx=envs_idx, reset_pos=robot_pos)
        ball_pos = self.ball_reset_pos.broadcast_to((len(envs_idx), 2)).clone()
        ball_pos += self.cfg.ball_reset_noise * torch.randn((len(envs_idx), 2), device=gs.device)
        ball_pos = F.pad(ball_pos, (0, 1), value=0.06)
        self.field.reset(envs_idx=envs_idx, ball_pos=ball_pos)
        kwargs = self.get_state(envs_idx=envs_idx)
        return (self.build_observation(**kwargs), self.build_info(**kwargs))

    @torch.no_grad()
    @torch.compile()
    def get_state(self, envs_idx: torch.Tensor) -> dict[str, torch.Tensor]:
        state = {
            "cmd_vel": self.__cmd_vel[envs_idx][:, -1, :], 
            "last_cmd_vel": self.__cmd_vel[envs_idx][:, -2, :],
            **self.robot.get_state(envs_idx=envs_idx), 
            **self.field.get_state(envs_idx=envs_idx) }
        def get_continuous_heading(q: torch.Tensor) -> torch.Tensor:
            # q: (N, 4) -> (w, x, y, z)
            w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
            vx = 1.0 - 2.0 * (y**2 + z**2)
            vy = 2.0 * (x * y + w * z)
            norm = torch.sqrt(vx**2 + vy**2 + 1e-8)
            return torch.stack([vx / norm, vy / norm], dim=-1)
        def get_ball_relative_pos(body_pos, body_heading, ball_pos, **kwargs):
            rel_tx = ball_pos[:, 0] - body_pos[:, 0]
            rel_ty = ball_pos[:, 1] - body_pos[:, 1]
            hx, hy = body_heading[:, 0], body_heading[:, 1]
            pos_rel_x = rel_tx * hx + rel_ty * hy
            pos_rel_y = -rel_tx * hy + rel_ty * hx
            return torch.stack([pos_rel_x, pos_rel_y], dim=-1)
        def get_ball_relative_vel(body_pos, body_heading, body_lin_vel, body_ang_vel, 
                                  ball_pos, ball_vel, **kwargs):
            v_rel_world_x = ball_vel[:, 0] - body_lin_vel[:, 0]
            v_rel_world_y = ball_vel[:, 1] - body_lin_vel[:, 1]
            hx, hy = body_heading[:, 0], body_heading[:, 1]
            vel_rel_x = v_rel_world_x * hx + v_rel_world_y * hy
            vel_rel_y = -v_rel_world_x * hy + v_rel_world_y * hx
            return torch.stack([vel_rel_x, vel_rel_y], dim=-1)
        state["body_heading"] = get_continuous_heading(state["body_quat"])
        state["ball_pos_rel"] = get_ball_relative_pos(**state)
        state["ball_vel_rel"] = get_ball_relative_vel(**state)
        state["ball_distance"] = torch.norm(state["ball_pos_rel"], dim=1, keepdim=True)     
        return state
    
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
            torch.abs(ball_pos[:, 0]) > self.cfg.terminate_range[0], 
            torch.abs(ball_pos[:, 1]) > self.cfg.terminate_range[1]) 
        robot_out_of_range = torch.logical_or(
            torch.abs(body_pos[:, 0]) > self.cfg.terminate_range[0],
            torch.abs(body_pos[:, 1]) > self.cfg.terminate_range[1])
        ball_too_far_from_robot = torch.linalg.norm(\
                ball_pos[:, 0:2] - body_pos[:, 0:2], dim=1) > self.cfg.ball_loss_distance
        return (robot_fall 
                | ball_out_of_range 
                | robot_out_of_range 
                | ball_too_far_from_robot).unsqueeze(1)
    
    @torch.no_grad()
    @torch.compile()
    def build_truncated(self, **kwargs) -> torch.Tensor:
        return (self.step_count >= self.cfg.truncated_step_limit).unsqueeze(1)
    
    @torch.no_grad()
    @torch.compile()
    def build_reward(self, cmd_vel, last_cmd_vel, body_pos, body_lin_vel, body_heading,
                     ball_pos, ball_vel, ball_pos_rel, ball_vel_rel, 
                     **kwargs) -> torch.Tensor: # type: ignore[override]
        rew_ball_x = torch.clip(1.0 - torch.exp(-ball_vel[:, 0] / 0.1), -1.0, 1.0)
        rew_ball_y = torch.clip(torch.exp(-ball_pos[:, 1] * ball_vel[:, 1] 
                                          / self.cfg.terminate_range[1]) - 1.0, 
                                -1.0, 1.0)
        rew_cmd_smooth = torch.exp(-torch.linalg.norm(cmd_vel - last_cmd_vel, dim=1))
        ball_pos_rel_unprojected = ball_pos[:, 0:2] - body_pos[:, 0:2]
        ball_distance = torch.linalg.norm(ball_pos_rel_unprojected, dim=1) + 1e-5
        rew_close_to_ball = (ball_pos_rel_unprojected * body_lin_vel[:, 0:2]).sum(dim=1) / ball_distance
        rew_rel_y = -(ball_pos_rel[:, 1] / (ball_distance + 0.1)) ** 2
        rew_action = 1.0 - torch.exp(-torch.linalg.norm(cmd_vel, dim=1) / 0.5)
        rew_heading = body_heading[:, 0] - 1.0  # encourage facing the goal
        self.__reward_info = {
            "rew_ball_x": rew_ball_x.mean().detach(),
            "rew_ball_y": rew_ball_y.mean().detach(),
            "rew_cmd_smooth": rew_cmd_smooth.mean().detach(),
            "rew_close_to_ball": rew_close_to_ball.mean().detach(),
            "rew_rel_y": rew_rel_y.mean().detach(),
            "rew_action": rew_action.mean().detach(),
            "rew_heading": rew_heading.mean().detach(),
        }
        return ((rew_ball_x
                + rew_ball_y * 0.03
                + rew_cmd_smooth * 0.2
                + rew_close_to_ball * 1.0
                + rew_rel_y * 4.0
                + rew_action * 0.5
                + rew_heading * 1.0)
                / (1.0 + 0.03 + 0.2 + 1.0 + 4.0 + 0.5 + 1.0)).unsqueeze(1)
    
    @torch.no_grad()
    @torch.compile()
    def build_info(self, ball_pos, ball_vel, body_pos, cmd_vel, ball_pos_rel, ball_vel_rel,
                   **kwargs) -> dict[str, dict[str, torch.Tensor]]: # type: ignore[override]
        return {"extra": 
                {"ball_pos_x": ball_pos[:, 0].mean().detach(),
                 "ball_pos_y": ball_pos[:, 1].mean().detach(),
                 "ball_vel_x": ball_vel[:, 0].mean().detach(),
                 "ball_vel_y": ball_vel[:, 1].mean().detach(),
                 "body_pos_x": body_pos[:, 0].mean().detach(),
                 "body_pos_y": body_pos[:, 1].mean().detach(),
                 "ball_rel_pos_x": ball_pos_rel[:, 0].mean().detach(),
                 "ball_rel_pos_y": ball_pos_rel[:, 1].mean().detach(),
                 "cmd_vel": cmd_vel.mean().detach(),
                 **self.__reward_info, }}
