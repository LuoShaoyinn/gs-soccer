# walk_model.py
#   Define walking model

import torch
import numpy as np
import genesis as gs
import gymnasium as gym
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from .model import ModelConfig, Model

@dataclass(kw_only = True)
class MOS9WalkModelConfig(ModelConfig):
    n_dofs:             int
    target_q_offset:    np.ndarray
    step_dt:            float       = 0.02
    history_frames:     int         = 5
    cycle_time:         float       = 0.8
    step_ewma_factor:   float       = 0.2
    dofs_pos_scale:     float       = 1.0
    dofs_vel_scale:     float       = 0.05
    action_scale:       float       = 0.25
    body_ang_vel_noise: float       = 0.12 * 0.6
    proj_gravity_noise: float       = 0.12 * 0.6
    obs_clip:           tuple       = (-18.0, 18.0)

    tracking_sigma:         float   = 0.25
    tracking_lin_vel_weight: float  = 1.5
    tracking_ang_vel_weight: float  = 0.75
    lin_vel_z_penalty_weight: float = 1.0
    ang_vel_xy_penalty_weight: float = 0.05
    orientation_penalty_weight: float = 1.0
    torque_penalty_weight:  float   = 2e-5
    dof_vel_penalty_weight: float   = 1e-4
    dof_acc_penalty_weight: float   = 2.5e-7
    action_rate_penalty_weight: float = 0.01
    alive_reward_weight:    float   = 0.2
    termination_body_height: float  = 0.2
    termination_link_height: float  = 0.05
    max_episode_length:     int     = 500
    speed_chunk_length:     int     = 10


class MOS9WalkModel(Model):
    cfg: MOS9WalkModelConfig
    def config(self):
        self.target_q_offset = torch.from_numpy(self.cfg.target_q_offset).to(gs.device)
        self.dim_observations = 11 + 3 * self.cfg.n_dofs
        self.time_steps = torch.zeros((self.scene.n_envs, ), 
                                      dtype=torch.float, 
                                      device=gs.device)
        self.last_obs = torch.zeros((self.scene.n_envs, 
                                     self.cfg.history_frames, 
                                     self.dim_observations), 
                                    dtype=torch.float,
                                    device=gs.device)
        self.last_action = torch.zeros((self.scene.n_envs, self.cfg.n_dofs),
                                      dtype=torch.float, 
                                      device=gs.device)
        self.ewma_action = torch.zeros((self.scene.n_envs, self.cfg.n_dofs),
                                      dtype=torch.float, 
                                      device=gs.device)
        self.speed_buffer = torch.zeros(
            (self.scene.n_envs, self.cfg.speed_chunk_length),
            dtype=torch.float, device=gs.device,
        )
        self.episode_steps = torch.zeros(
            (self.scene.n_envs,), dtype=torch.long, device=gs.device,
        )
        self._log: dict[str, torch.Tensor] = {}
        self.prev_action = torch.zeros(
            (self.scene.n_envs, self.cfg.n_dofs), dtype=torch.float, device=gs.device,
        )
        self.last_dofs_vel = torch.zeros(
            (self.scene.n_envs, self.cfg.n_dofs), dtype=torch.float, device=gs.device,
        )

    def reset(self, envs_idx: torch.Tensor):
        self.time_steps[envs_idx]   = 0.0
        self.last_action[envs_idx]  = 0.0
        self.ewma_action[envs_idx]  = 0.0
        self.last_obs[envs_idx]     = 0.0
        self.speed_buffer[envs_idx] = 0.0
        self.episode_steps[envs_idx] = 0
        self.prev_action[envs_idx] = 0.0
        self.last_dofs_vel[envs_idx] = 0.0
    
    def preprocess_action(self, action: torch.Tensor):
        self.prev_action.copy_(self.last_action)
        self.last_action = action
        self.time_steps += 1.0
        self.episode_steps += 1
        self.ewma_action *= self.cfg.step_ewma_factor
        self.ewma_action += self.cfg.action_scale * (1.0 - self.cfg.step_ewma_factor) * action
        return self.ewma_action + self.target_q_offset


    @property
    def observation_space(self) -> gym.spaces.Box:
        # [sin, cos, vx, vy, az,
        #  dof_pos(12), dof_vel(12), action(12),
        #  body_ang_vel(3), projected_gravity(3)]
        return gym.spaces.Box(
            low   = self.cfg.obs_clip[0], high = self.cfg.obs_clip[1], 
            shape = (self.dim_observations * self.cfg.history_frames,),
            dtype = np.float32
        )
    
    @property
    def action_space(self) -> gym.spaces.Box:
        return gym.spaces.Box(low   = -1.0, high  = 1.0, 
                              shape = (self.cfg.n_dofs,), 
                              dtype = np.float32)


    def build_observation(self, envs_idx, body_lin_vel, body_ang_vel, body_quat, 
                          cmd_vel, dofs_pos, dofs_vel, **kwargs) -> torch.Tensor: # type: ignore
        def project_gravity(quat: torch.Tensor) -> torch.Tensor:
            n = quat.shape[0]
            gravity = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float, device=gs.device)
            gravity = gravity.broadcast_to((n, 3))
            q_w = quat[:, 0:1]
            q_vec = quat[:, 1:4]
            return (
                gravity * (2.0 * q_w * q_w - 1.0)
                - torch.cross(q_vec, gravity, dim=1) * (2.0 * q_w)
                + q_vec * (2.0 * (q_vec * gravity).sum(dim=1, keepdim=True))
            )
        def add_noise(x: torch.Tensor, scale: float):
            return x + (torch.randn_like(x) * scale)
        n = envs_idx.shape[0]

        phase = self.time_steps[envs_idx] * (2.0 * torch.pi * self.cfg.step_dt / self.cfg.cycle_time)
        obs_sin_phase = torch.sin(phase).reshape((n, 1))
        obs_cos_phase = torch.cos(phase).reshape((n, 1))

        obs_cmd_vel = cmd_vel.reshape((n, 3))

        obs_dofs_pos = (dofs_pos - self.target_q_offset) * self.cfg.dofs_pos_scale
        
        obs_dofs_vel = dofs_vel * self.cfg.dofs_vel_scale

        obs_last_action = self.last_action[envs_idx]

        obs_body_ang_vel = add_noise(body_ang_vel, self.cfg.body_ang_vel_noise)

        obs_proj_gravity = add_noise(project_gravity(body_quat), self.cfg.proj_gravity_noise)

        obs_single_frame = torch.cat((
            obs_sin_phase,      
            obs_cos_phase,      
            obs_cmd_vel,        
            obs_dofs_pos,       
            obs_dofs_vel,       
            obs_last_action,    
            obs_body_ang_vel,   
            obs_proj_gravity
        ), dim=-1)
        obs_single_frame = torch.clip(obs_single_frame.float(), 
                                      self.cfg.obs_clip[0], 
                                      self.cfg.obs_clip[1])

        self.last_obs[envs_idx] = torch.roll(self.last_obs[envs_idx], shifts=-1, dims=1)
        self.last_obs[envs_idx, -1, :] = obs_single_frame
        return self.last_obs[envs_idx].reshape(n, -1)

    def build_reward(
        self, envs_idx, body_lin_vel, body_ang_vel, body_quat, dofs_torque, dofs_vel, cmd_vel, **kwargs
    ) -> torch.Tensor:
        def project_gravity(quat: torch.Tensor) -> torch.Tensor:
            n = quat.shape[0]
            gravity = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float, device=gs.device)
            gravity = gravity.broadcast_to((n, 3))
            q_w = quat[:, 0:1]
            q_vec = quat[:, 1:4]
            return (
                gravity * (2.0 * q_w * q_w - 1.0)
                - torch.cross(q_vec, gravity, dim=1) * (2.0 * q_w)
                + q_vec * (2.0 * (q_vec * gravity).sum(dim=1, keepdim=True))
            )

        forward_speed = body_lin_vel[:, 0:1]
        lateral_speed = body_lin_vel[:, 1:2]
        lin_vel_z = body_lin_vel[:, 2:3]
        yaw_rate = body_ang_vel[:, 2:3]
        ang_vel_xy_sq = (body_ang_vel[:, 0:2] ** 2).sum(dim=1, keepdim=True)
        proj_gravity = project_gravity(body_quat)
        orientation_sq = (proj_gravity[:, 0:2] ** 2).sum(dim=1, keepdim=True)

        self.speed_buffer = torch.roll(self.speed_buffer, shifts=-1, dims=1)
        self.speed_buffer[:, -1:] = forward_speed
        chunk_speed = self.speed_buffer.mean(dim=1, keepdim=True)

        lin_err_sq = (chunk_speed - cmd_vel[:, 0:1]) ** 2 + (lateral_speed - cmd_vel[:, 1:2]) ** 2
        tracking_lin_vel_reward = torch.exp(-lin_err_sq / self.cfg.tracking_sigma)
        ang_err_sq = (yaw_rate - cmd_vel[:, 2:3]) ** 2
        tracking_ang_vel_reward = torch.exp(-ang_err_sq / self.cfg.tracking_sigma)

        torque_penalty = (dofs_torque ** 2).sum(dim=1, keepdim=True)
        dof_vel_penalty = (dofs_vel ** 2).sum(dim=1, keepdim=True)
        dof_acc = (dofs_vel - self.last_dofs_vel) / self.cfg.step_dt
        dof_acc_penalty = (dof_acc ** 2).sum(dim=1, keepdim=True)
        self.last_dofs_vel.copy_(dofs_vel)
        action_rate_penalty = ((self.last_action - self.prev_action) ** 2).sum(dim=1, keepdim=True)
        alive_reward = torch.ones_like(tracking_lin_vel_reward)

        tracking_lin_term = self.cfg.tracking_lin_vel_weight * tracking_lin_vel_reward
        tracking_ang_term = self.cfg.tracking_ang_vel_weight * tracking_ang_vel_reward
        lin_vel_z_term = self.cfg.lin_vel_z_penalty_weight * lin_vel_z * lin_vel_z
        ang_vel_xy_term = self.cfg.ang_vel_xy_penalty_weight * ang_vel_xy_sq
        orientation_term = self.cfg.orientation_penalty_weight * orientation_sq
        torque_term = self.cfg.torque_penalty_weight * torque_penalty
        dof_vel_term = self.cfg.dof_vel_penalty_weight * dof_vel_penalty
        dof_acc_term = self.cfg.dof_acc_penalty_weight * dof_acc_penalty
        action_rate_term = self.cfg.action_rate_penalty_weight * action_rate_penalty
        alive_term = self.cfg.alive_reward_weight * alive_reward
        total_reward = (
            tracking_lin_term
            + tracking_ang_term
            + alive_term
            - torque_term
            - lin_vel_z_term
            - ang_vel_xy_term
            - orientation_term
            - dof_vel_term
            - dof_acc_term
            - action_rate_term
        )
        self._log = {
            "+tracking_lin_vel": tracking_lin_term,
            "+tracking_ang_vel": tracking_ang_term,
            "+alive": alive_term,
            "-lin_vel_z": -lin_vel_z_term,
            "-ang_vel_xy": -ang_vel_xy_term,
            "-orientation": -orientation_term,
            "-torque": -torque_term,
            "-dof_vel": -dof_vel_term,
            "-dof_acc": -dof_acc_term,
            "-action_rate": -action_rate_term,
            "+total": total_reward,
            "state/forward_speed": forward_speed,
            "state/lateral_speed": lateral_speed,
            "state/lin_vel_z": lin_vel_z,
            "state/yaw_rate": yaw_rate,
            "state/cmd_forward_speed": cmd_vel[:, 0:1],
            "state/cmd_lateral_speed": cmd_vel[:, 1:2],
            "state/cmd_yaw_rate": cmd_vel[:, 2:3],
            "state/chunk_speed": chunk_speed,
            "state/torque_sum_sq": torque_penalty,
            "state/proj_gravity_xy_sq": orientation_sq,
        }
        return total_reward

    def build_terminated(
        self, envs_idx, body_pos, non_foot_heights, **kwargs
    ) -> torch.Tensor:
        body_down = body_pos[:, 2:3] < self.cfg.termination_body_height
        link_down = (non_foot_heights < self.cfg.termination_link_height).any(
            dim=1, keepdim=True,
        )
        return body_down | link_down

    def build_truncated(self, envs_idx, **kwargs) -> torch.Tensor:
        return (
            self.episode_steps[envs_idx] >= self.cfg.max_episode_length
        ).unsqueeze(1)

    @torch.compiler.disable
    def build_info(self, envs_idx, **kwargs) -> dict[str, dict[str, torch.Tensor]]:
        return {"extra": {
            f"Reward / {k}": v.detach().mean().cpu()
            for k, v in self._log.items()
        }}
