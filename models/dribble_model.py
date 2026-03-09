# dribble_model.py
#   Define dribble model

import torch
import numpy as np
import genesis as gs
import gymnasium as gym
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from .model import ModelConfig, Model

@dataclass(kw_only = True)
class DribbleModelConfig(ModelConfig):
    terminate_range:        np.ndarray  = field(default_factory= \
            lambda: np.array([12, 9], dtype=np.float32))
    truncated_step_limit:   int         = 400
    self_collision:         bool        = False
    ball_loss_distance:     float       = 2.0


class DribbleModel(Model):
    cfg: DribbleModelConfig 
    def config(self):
        super().config()
        self.time_steps = torch.zeros((self.scene.n_envs, 1), 
                                      dtype=torch.float, 
                                      device=gs.device)
        self.cmd_vel = torch.zeros((self.scene.n_envs, 2, 3), 
                                   dtype=torch.float,
                                   device=gs.device)
        self.terminate_range = torch.from_numpy(self.cfg.terminate_range) \
                                    .to(gs.device)
        self.cache = dict()
    
    def reset(self, envs_idx: torch.Tensor): 
        self.cache = dict()   # reset will invalid cache
        self.time_steps[envs_idx] = 0
        self.cmd_vel[envs_idx] = 0.0

    def preprocess_action(self, action: torch.Tensor) -> torch.Tensor:
        self.cache = dict()   # action will invalid cache
        self.cmd_vel = torch.roll(self.cmd_vel, shifts=-1, dims=1)
        self.cmd_vel[:, -1, :] = action
        self.time_steps += 1.0
        return action

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
        self.build_cache(**kwargs)
        return torch.cat((self.cache["body_pos_2D"][envs_idx], 
                          self.cache["body_heading"][envs_idx], 
                          self.cache["body_vel_2D"][envs_idx], 
                          self.cache["body_omega"][envs_idx],
                          self.cache["ball_pos_proj"][envs_idx], 
                          self.cache["ball_vel_proj"][envs_idx], 
                          self.cmd_vel[envs_idx, -1, :],
                          kwargs["target_pos"]), dim=1)

    def build_terminated(self, envs_idx, **kwargs) -> torch.Tensor: # type: ignore[override]
        self.build_cache(**kwargs)
        ball_pos_2D       = self.cache["ball_pos_2D"][envs_idx]
        body_pos_2D       = self.cache["body_pos_2D"][envs_idx]
        ball_dis          = self.cache["ball_dis"][envs_idx]
        robot_fall        = (kwargs["body_pos"][:, 2] < 0.3).unsqueeze(1)
        ball_out_of_range = (torch.abs(ball_pos_2D) > self.terminate_range).any(dim=1).unsqueeze(1)
        body_out_of_range = (torch.abs(body_pos_2D) > self.terminate_range).any(dim=1).unsqueeze(1)
        ball_lost         = ball_dis > self.cfg.ball_loss_distance
        return robot_fall | ball_out_of_range | body_out_of_range | ball_lost
    
    def build_truncated(self, envs_idx, **kwargs) -> torch.Tensor:
        return self.time_steps >= self.cfg.truncated_step_limit
    
    def build_reward(self, envs_idx, **kwargs) -> torch.Tensor: # type: ignore[override]
        return torch.zeros((envs_idx.shape[0], 1), dtype=torch.float, device=gs.device)
    
    def build_info(self, envs_idx, **kwargs) -> dict[str, dict[str, torch.Tensor]]: # type: ignore[override]
        return {"extra": {}}
    
    def build_cache(self, body_pos, body_quat, body_lin_vel, body_ang_vel, 
                    ball_pos, ball_vel, **kwargs):
        if self.cache:
            return 
        def get_continuous_heading(q: torch.Tensor) -> torch.Tensor:
            # q: (N, 4) -> (w, x, y, z)
            w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
            vx = 1.0 - 2.0 * (y**2 + z**2)
            vy = 2.0 * (x * y + w * z)
            norm = torch.sqrt(vx**2 + vy**2 + 1e-8)
            return torch.stack([vx / norm, vy / norm], dim=-1)
        def get_ball_relative_pos(body_heading, ball_pos_rel):
            rel_tx = ball_pos[:, 0] - body_pos[:, 0]
            rel_ty = ball_pos[:, 1] - body_pos[:, 1]
            hx, hy = body_heading[:, 0], body_heading[:, 1]
            pos_rel_x = rel_tx * hx + rel_ty * hy
            pos_rel_y = -rel_tx * hy + rel_ty * hx
            return torch.stack([pos_rel_x, pos_rel_y], dim=-1)
        def get_ball_relative_vel(body_heading, ball_vel_rel):
            v_rel_world_x = ball_vel[:, 0] - body_lin_vel[:, 0]
            v_rel_world_y = ball_vel[:, 1] - body_lin_vel[:, 1]
            hx, hy = body_heading[:, 0], body_heading[:, 1]
            vel_rel_x = v_rel_world_x * hx + v_rel_world_y * hy
            vel_rel_y = -v_rel_world_x * hy + v_rel_world_y * hx
            return torch.stack([vel_rel_x, vel_rel_y], dim=-1)
        self.cache = {}
        self.cache["body_pos_2D"]   = body_pos[:, 0:2].contiguous()
        self.cache["body_vel_2D"]   = body_lin_vel[:, 0:2].contiguous()
        self.cache["ball_pos_2D"]   = ball_pos[:, 0:2].contiguous()
        self.cache["ball_vel_2D"]   = ball_vel[:, 0:2].contiguous()
        self.cache["body_omega"]    = body_ang_vel[:, 2].unsqueeze(1).contiguous()
        self.cache["ball_pos_rel"]  = self.cache["ball_pos_2D"] - self.cache["body_pos_2D"]
        self.cache["ball_vel_rel"]  = self.cache["ball_vel_2D"] - self.cache["body_vel_2D"]
        self.cache["ball_dis"]      = torch.norm(self.cache["ball_pos_rel"], dim=1, keepdim=True)     
        self.cache["body_heading"]  = get_continuous_heading(q = body_quat)
        self.cache["ball_pos_proj"] = get_ball_relative_pos(body_heading = self.cache["body_heading"],
                                                            ball_pos_rel = self.cache["ball_pos_rel"])
        self.cache["ball_vel_proj"] = get_ball_relative_vel(body_heading = self.cache["body_heading"],
                                                            ball_vel_rel = self.cache["ball_vel_rel"])
