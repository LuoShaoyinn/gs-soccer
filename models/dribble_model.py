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
        zero = torch.tensor(0.0, dtype=torch.float, device=gs.device)
        self.rewards = {
            "reward_target": zero,
            "reward_behind": zero,
            "reward_front": zero,
            "reward_center": zero,
            "heading_align": zero,
            "collinear": zero,
            "ball_vel_toward_tar": zero,
            "reward_ball_speed": zero,
            "penalty_ball_lost": zero,
            "penalty_boundary": zero,
            "penalty_action": zero,
            "penalty_smooth": zero,
            "reward_success": zero,
        }

    
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
        self.build_cache(**kwargs)

        body_pos_2D   = self.cache["body_pos_2D"][envs_idx]
        body_heading  = self.cache["body_heading"][envs_idx]
        ball_pos_2D   = self.cache["ball_pos_2D"][envs_idx]
        ball_vel_2D   = self.cache["ball_vel_2D"][envs_idx]
        ball_pos_proj = self.cache["ball_pos_proj"][envs_idx]
        ball_dis      = self.cache["ball_dis"][envs_idx]

        target_pos    = kwargs["target_pos"]
        cmd_now       = self.cmd_vel[envs_idx, -1, :]
        cmd_prev      = self.cmd_vel[envs_idx,  0, :]

        eps = 1e-6

        # ball -> target
        ball_to_target = target_pos - ball_pos_2D
        ball_to_target_dis = torch.norm(ball_to_target, dim=1, keepdim=True)
        ball_to_target_dir = ball_to_target / (ball_to_target_dis + eps)

        # robot desired position: slightly behind the ball w.r.t. target direction
        behind_distance = 0.25
        body_pos_desired = ball_pos_2D - behind_distance * ball_to_target_dir
        body_to_desired_dis = torch.norm(body_pos_2D - body_pos_desired, dim=1, keepdim=True)

        # ball position in robot local frame
        ball_front = ball_pos_proj[:, 0:1]
        ball_lateral = ball_pos_proj[:, 1:2]

        # robot heading alignment with target direction
        heading_align = torch.sum(body_heading * ball_to_target_dir, dim=1, keepdim=True)
        heading_align = torch.clamp(heading_align, min=0.0)

        # robot-ball-target collinearity: robot should stay behind the ball
        robot_to_ball = ball_pos_2D - body_pos_2D
        robot_to_ball_dir = robot_to_ball / (ball_dis + eps)
        collinear = torch.sum(robot_to_ball_dir * ball_to_target_dir, dim=1, keepdim=True)
        collinear = torch.clamp(collinear, min=0.0)

        # ball velocity toward target
        ball_vel_toward_target = torch.sum(ball_vel_2D * ball_to_target_dir, dim=1, keepdim=True)

        # desired ball speed band for stable dribbling
        ball_speed_target = 0.5
        reward_ball_speed = torch.exp(-((ball_vel_toward_target - ball_speed_target) ** 2) / (0.3 ** 2))

        # shaping rewards
        reward_target = torch.exp(-(ball_to_target_dis ** 2) / (2.0 ** 2))
        reward_behind = torch.exp(-(body_to_desired_dis ** 2) / (0.20 ** 2))
        reward_front  = torch.exp(-((ball_front - 0.22) ** 2) / (0.10 ** 2))
        reward_center = torch.exp(-(ball_lateral ** 2) / (0.12 ** 2))

        # penalties
        penalty_ball_lost = torch.clamp(ball_dis - 0.70, min=0.0)

        d_edge_ball = torch.minimum(self.terminate_range[0] - torch.abs(ball_pos_2D[:, 0:1]),
                                    self.terminate_range[1] - torch.abs(ball_pos_2D[:, 1:2]))
        penalty_boundary = torch.clamp(0.30 - d_edge_ball, min=0.0)

        penalty_action = torch.sum(cmd_now ** 2, dim=1, keepdim=True)
        penalty_smooth = torch.sum((cmd_now - cmd_prev) ** 2, dim=1, keepdim=True)

        # success bonus
        reward_success = (ball_to_target_dis < 0.25).float()

        self.rewards = {
            "target": 0.60 * reward_target,
            "behind": 0.60 * reward_behind,
            "front": 0.40 * reward_front,
            "center": 0.40 * reward_center,
            "heading": 0.30 * heading_align,
            "collinear": 0.50 * collinear,
            "ball_vel_target": 0.20 * ball_vel_toward_target,
            "ball_speed": 0.20 * reward_ball_speed,
            "ball_lost": -0.25 * penalty_ball_lost,
            "boundary": -0.50 * penalty_boundary,
            "action": -0.05 * penalty_action,
            "smooth": -0.10 * penalty_smooth,
            "success": 30.0 * reward_success,
        }

        return sum(self.rewards.values())  # type: ignore[operator]
 
    @torch.compiler.disable
    def build_info(self, envs_idx, **kwargs
                   ) -> dict[str, dict[str, torch.Tensor]]: # type: ignore[override]
        return {"extra": {
            k: v.detach().mean().cpu()
            for k, v in self.rewards.items()
        }} # type: ignore[return]
    
    def build_cache(self, body_pos, body_quat, body_lin_vel, body_ang_vel, 
                    ball_pos, ball_vel, **kwargs):
        if self.cache:
            return 
        @torch.no_grad()
        @torch.compile()
        def get_continuous_heading(q: torch.Tensor) -> torch.Tensor:
            # q: (N, 4) -> (w, x, y, z)
            w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
            vx = 1.0 - 2.0 * (y**2 + z**2)
            vy = 2.0 * (x * y + w * z)
            norm = torch.sqrt(vx**2 + vy**2 + 1e-8)
            return torch.stack([vx / norm, vy / norm], dim=-1)
        @torch.no_grad()
        @torch.compile()
        def get_ball_relative_pos(body_heading, ball_pos_rel):
            rel_tx = ball_pos[:, 0] - body_pos[:, 0]
            rel_ty = ball_pos[:, 1] - body_pos[:, 1]
            hx, hy = body_heading[:, 0], body_heading[:, 1]
            pos_rel_x = rel_tx * hx + rel_ty * hy
            pos_rel_y = -rel_tx * hy + rel_ty * hx
            return torch.stack([pos_rel_x, pos_rel_y], dim=-1)
        @torch.no_grad()
        @torch.compile()
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
