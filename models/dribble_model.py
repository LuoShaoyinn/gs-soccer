# dribble_model.py
#   Define dribble model

import torch
import numpy as np
import genesis as gs
import gymnasium as gym
import torch.nn.functional as F
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from .model import ModelConfig, Model

@dataclass(kw_only = True)
class DribbleModelConfig(ModelConfig):
    terminate_range:        np.ndarray  = field(default_factory= \
            lambda: np.array([4.5, 3], dtype=np.float32))
    timeout_steps_limit:    int         = 40
    control_y:              float       = 0.1
    finished_distance:      float       = 1.0
    action_scale:           float       = 2.0


class DribbleModel(Model):
    cfg: DribbleModelConfig 

    def config(self):
        super().config()
        num_envs = self.scene.n_envs
        self.time_steps = torch.zeros((num_envs, 1), 
                                      dtype=torch.float, 
                                      device=gs.device)
        self.cmd_vel = torch.zeros((self.scene.n_envs, 2, 3), 
                                   dtype=torch.float,
                                   device=gs.device)
        self.terminate_range = torch.from_numpy(self.cfg.terminate_range) \
                                    .to(gs.device)
        self.cache_valid = torch.zeros((self.scene.n_envs,), dtype=torch.bool, device=gs.device)
        self.cache = {
            "body_pos_2D":              torch.zeros((num_envs, 2), dtype=torch.float, device=gs.device),
            "body_vel_2D":              torch.zeros((num_envs, 2), dtype=torch.float, device=gs.device),
            "ball_pos_2D":              torch.zeros((num_envs, 2), dtype=torch.float, device=gs.device),
            "ball_vel_2D":              torch.zeros((num_envs, 2), dtype=torch.float, device=gs.device),
            "body_omega":               torch.zeros((num_envs, 1), dtype=torch.float, device=gs.device),
            "ball_pos_rel":             torch.zeros((num_envs, 2), dtype=torch.float, device=gs.device),
            "ball_vel_rel":             torch.zeros((num_envs, 2), dtype=torch.float, device=gs.device),
            "ball_dis":                 torch.zeros((num_envs, 1), dtype=torch.float, device=gs.device),
            "body_heading":             torch.zeros((num_envs, 2), dtype=torch.float, device=gs.device),
            "ball_pos_proj":            torch.zeros((num_envs, 2), dtype=torch.float, device=gs.device),
            "ball_vel_proj":            torch.zeros((num_envs, 2), dtype=torch.float, device=gs.device),
            "ball_to_target":           torch.zeros((num_envs, 2), dtype=torch.float, device=gs.device),
            "ball_to_target_dis":       torch.zeros((num_envs, 1), dtype=torch.float, device=gs.device),
            "ball_to_target_unit":      torch.zeros((num_envs, 2), dtype=torch.float, device=gs.device),
            "failed":                   torch.zeros((num_envs, 1), dtype=torch.bool,  device=gs.device),
            "finished":                 torch.zeros((num_envs, 1), dtype=torch.bool,  device=gs.device),
            "timeout":                  torch.zeros((num_envs, 1), dtype=torch.bool,  device=gs.device),
        }
        self.rewards = {}

    
    def reset(self, envs_idx: torch.Tensor): 
        self.cache_valid[envs_idx] = False
        self.time_steps[envs_idx] = 0.0
        self.cmd_vel[envs_idx] = 0.0

    def preprocess_action(self, action: torch.Tensor) -> torch.Tensor:
        self.cache_valid = torch.zeros_like(self.cache_valid)
        self.cmd_vel = torch.roll(self.cmd_vel, shifts=-1, dims=1)
        self.cmd_vel[:, -1, :] = torch.clip(action, -1.0, 1.0)
        self.time_steps += 1.0
        return self.cmd_vel.mean(dim=1) * self.cfg.action_scale

    @property
    def observation_space(self) -> gym.spaces.Box:
        # body_pos[0:2]
        # body_heading[2:4] (sin_yaw, cos_yaw)          Do not use angle
        # body_vel[4:7]     (lin_x, lin_y, ang_z)
        # ball_pos_proj[7:9] (ball_rel_x, ball_rel_y)
        # ball_vel_proj[9:11]
        # cmd_vel[11:14]    (lin_x, lin_y, ang_z)       last control command
        # target_pos[14:16]
        return gym.spaces.Box(low   = -10.0, high  =  10.0, 
                              shape = (16,), dtype = np.float32)
    
    @property
    def action_space(self) -> gym.spaces.Box:
        # lin_x, lin_y, ang_z
        return gym.spaces.Box(low   = -1.0, high  =  1.0, 
                              shape = (3,), dtype = np.float32) 

    def build_observation(self, envs_idx, **kwargs) -> torch.Tensor: # type: ignore[override]
        self.build_cache(envs_idx=envs_idx, **kwargs)
        return torch.cat((self.cache["body_pos_2D"][envs_idx], 
                          self.cache["body_heading"][envs_idx], 
                          self.cache["body_vel_2D"][envs_idx], 
                          self.cache["body_omega"][envs_idx],
                          self.cache["ball_pos_proj"][envs_idx], 
                          self.cache["ball_vel_proj"][envs_idx], 
                          self.cmd_vel[envs_idx, -1, :],
                          kwargs["target_pos"]), dim=1)

    def build_terminated(self, envs_idx, **kwargs) -> torch.Tensor: # type: ignore[override]
        self.build_cache(envs_idx=envs_idx, **kwargs)
        return self.cache["failed"][envs_idx] | self.cache["finished"][envs_idx]
    
    def build_truncated(self, envs_idx, **kwargs) -> torch.Tensor:
        return torch.zeros((envs_idx.shape[0], 1), dtype=torch.bool, device=gs.device)
    
    def build_reward(self, envs_idx, **kwargs) -> torch.Tensor: # type: ignore[override]
        self.build_cache(envs_idx=envs_idx, **kwargs)

        body_pos_2D   = self.cache["body_pos_2D"][envs_idx]
        body_vel_2D   = self.cache["body_vel_2D"][envs_idx]
        body_heading  = self.cache["body_heading"][envs_idx]
        ball_pos_2D   = self.cache["ball_pos_2D"][envs_idx]
        ball_vel_2D   = self.cache["ball_vel_2D"][envs_idx]
        ball_pos_rel  = self.cache["ball_pos_rel"][envs_idx]
        ball_pos_proj = self.cache["ball_pos_proj"][envs_idx]
        ball_dis      = self.cache["ball_dis"][envs_idx]
        target_pos    = kwargs["target_pos"]
        cmd_now       = self.cmd_vel[envs_idx, -1, :]
        cmd_prev      = self.cmd_vel[envs_idx,  0, :]
        ball_to_target_unit = self.cache["ball_to_target_unit"][envs_idx]
        def dot_2D(a, b):
            return (a * b).sum(dim=1, keepdim=True)
        def cross_2D(a, b):
            return a[:, 0:1] * b[:, 1:2] - a[:, 1:2] * b[:, 0:1]

        # ball movement reward
        rew_ball_toward_target = dot_2D(ball_to_target_unit, ball_vel_2D)
        rew_ball_offset_target = -(cross_2D(ball_to_target_unit, ball_vel_2D) ** 2)

        # cmd_vel reward
        reward_smooth_action = torch.exp(-torch.norm(cmd_now - cmd_prev, dim=1, keepdim=True))
        reward_larger_action = torch.exp(-(1.0 - (cmd_now ** 2).sum(dim=1, keepdim=True)) ** 2)

        # owning ball reward
        rew_close_to_ball  = dot_2D(ball_pos_rel, body_vel_2D) / (ball_dis + 0.1)
        rew_not_lost_ball  = -(ball_pos_proj[:, 1].unsqueeze(1) / (ball_dis + 0.1)) ** 2
        rew_facing_to_ball = dot_2D(body_heading, ball_pos_rel) / (ball_dis + 0.1)
        rew_facing_target  = torch.where(abs(ball_pos_proj[:,1:]) < self.cfg.control_y, 
                                        dot_2D(body_heading, ball_to_target_unit), 
                                        0.0)

        # finishing reward
        rew_finished    = self.cache["finished"][envs_idx].float()
        rew_failed      = self.cache["failed"][envs_idx].float()

        self.rewards = {
            "rew_ball_toward_target":   rew_ball_toward_target * 6.0,
            "rew_ball_offset_target":   rew_ball_offset_target * 1.0,
            "reward_smooth_action":     reward_smooth_action * 0.2,
            "reward_larger_action":     reward_larger_action * 2.0,
            "rew_close_to_ball":        rew_close_to_ball * 4.0,
            "rew_not_lost_ball":        rew_not_lost_ball * 0.5,
            "rew_facing_target":        rew_facing_target * 1.0,
            "rew_facing_to_ball":       rew_facing_to_ball * 0.3,
            "rew_finished":             rew_finished * (60.0 - self.time_steps.float()), 
            "rew_ball_out_of_range":    rew_failed * -10.0, 
        }
    
        return sum(self.rewards.values()) \
                / (6.0 + 1.0 + 0.2 + 2.0 + 4.0 + 0.5 + 1.0 + 0.3) # type: ignore[operator]
 
    @torch.compiler.disable
    def build_info(self, envs_idx, **kwargs
                   ) -> dict[str, dict[str, torch.Tensor]]: # type: ignore[override]
        return {"extra": {
            k: v.detach().mean().cpu()
            for k, v in self.rewards.items()
        }} # type: ignore[return]
    
    def build_cache(self, envs_idx, body_pos, body_quat, body_lin_vel, body_ang_vel, 
                    ball_pos, ball_vel, **kwargs):
        if self.cache_valid[envs_idx].all():
            return
        self.cache_valid[envs_idx] = True
        def get_continuous_heading(q: torch.Tensor) -> torch.Tensor:
            # q: (N, 4) -> (w, x, y, z)
            w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
            vx = 1.0 - 2.0 * (y**2 + z**2)
            vy = 2.0 * (x * y + w * z)
            norm = torch.sqrt(vx**2 + vy**2 + 1e-8)
            return torch.stack([vx / norm, vy / norm], dim=-1)
        def get_ball_relative_pos(body_heading, ball_pos_rel):
            rel_tx = ball_pos_rel[:, 0]
            rel_ty = ball_pos_rel[:, 1]
            hx, hy = body_heading[:, 0], body_heading[:, 1]
            pos_rel_x = rel_tx * hx + rel_ty * hy
            pos_rel_y = -rel_tx * hy + rel_ty * hx
            return torch.stack([pos_rel_x, pos_rel_y], dim=-1)
        def get_ball_relative_vel(body_heading, ball_vel_rel):
            v_rel_world_x = ball_vel_rel[:, 0]
            v_rel_world_y = ball_vel_rel[:, 1]
            hx, hy = body_heading[:, 0], body_heading[:, 1]
            vel_rel_x = v_rel_world_x * hx + v_rel_world_y * hy
            vel_rel_y = -v_rel_world_x * hy + v_rel_world_y * hx
            return torch.stack([vel_rel_x, vel_rel_y], dim=-1)
        self.cache["body_pos_2D"][envs_idx]   = body_pos[:, 0:2].contiguous()
        self.cache["body_vel_2D"][envs_idx]   = body_lin_vel[:, 0:2].contiguous()
        self.cache["ball_pos_2D"][envs_idx]   = ball_pos[:, 0:2].contiguous()
        self.cache["ball_vel_2D"][envs_idx]   = ball_vel[:, 0:2].contiguous()
        self.cache["body_omega"][envs_idx]    = body_ang_vel[:, 2].unsqueeze(1).contiguous()
        self.cache["ball_pos_rel"][envs_idx]  = self.cache["ball_pos_2D"][envs_idx] \
                                                - self.cache["body_pos_2D"][envs_idx]
        self.cache["ball_vel_rel"][envs_idx]  = self.cache["ball_vel_2D"][envs_idx] \
                                                - self.cache["body_vel_2D"][envs_idx]
        self.cache["ball_dis"][envs_idx]      = torch.norm(self.cache["ball_pos_rel"][envs_idx], 
                                                           dim=1, keepdim=True)     
        self.cache["body_heading"][envs_idx]  = get_continuous_heading(q = body_quat)
        self.cache["ball_pos_proj"][envs_idx] = get_ball_relative_pos(
                body_heading = self.cache["body_heading"][envs_idx], 
                ball_pos_rel = self.cache["ball_pos_rel"][envs_idx])
        self.cache["ball_vel_proj"][envs_idx] = get_ball_relative_vel(
                body_heading = self.cache["body_heading"][envs_idx], 
                ball_vel_rel = self.cache["ball_vel_rel"][envs_idx])
        self.cache["ball_to_target"][envs_idx] = \
                kwargs["target_pos"] - self.cache["ball_pos_2D"][envs_idx]
        self.cache["ball_to_target_dis"][envs_idx] = \
                self.cache["ball_to_target"][envs_idx].norm(dim=1, keepdim=True) + 1e-3 
        self.cache["ball_to_target_unit"][envs_idx] = \
                self.cache["ball_to_target"][envs_idx] / self.cache["ball_to_target_dis"][envs_idx]
        # reuse failed
        ball_pos_2D       = self.cache["ball_pos_2D"][envs_idx]
        body_pos_2D       = self.cache["body_pos_2D"][envs_idx]
        ball_dis          = self.cache["ball_dis"][envs_idx]
        robot_fall        = (body_pos[:, 2] < 0.3).unsqueeze(1)
        ball_out_of_range = (torch.abs(ball_pos_2D) > self.terminate_range).any(dim=1).unsqueeze(1)
        body_out_of_range = (torch.abs(body_pos_2D) > self.terminate_range).any(dim=1).unsqueeze(1)
        finished          = self.cache["ball_to_target_dis"][envs_idx] < self.cfg.finished_distance
        timeout           = self.time_steps[envs_idx] >= self.cfg.timeout_steps_limit
        self.cache["failed"][envs_idx]   = body_out_of_range | ball_out_of_range | robot_fall | timeout
        self.cache["finished"][envs_idx] = finished
