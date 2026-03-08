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
class WalkModelConfig(ModelConfig):
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
    body_eu_ang_noise:  float       = 0.12 * 0.6
    obs_clip:           tuple       = (-18.0, 18.0)


class WalkModel(Model):
    cfg: WalkModelConfig
    def config(self):
        self.target_q_offset = torch.from_numpy(self.cfg.target_q_offset).to(gs.device)
        self.dim_observations = 11 + 3 * self.cfg.n_dofs
        self.time_steps = torch.zeros((self.scene.n_envs, 1), 
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

    def reset(self, envs_idx: torch.Tensor):
        self.time_steps[envs_idx]   = 0.0
        self.last_action[envs_idx]  = 0.0
        self.ewma_action[envs_idx]  = 0.0
        self.last_obs[envs_idx]     = 0.0
    
    def preprocess_action(self, action: torch.Tensor):
        self.last_action = action
        self.time_steps += 1.0
        self.ewma_action *= self.cfg.step_ewma_factor
        self.ewma_action += self.cfg.action_scale * (1.0 - self.cfg.step_ewma_factor) * action
        return self.ewma_action + self.target_q_offset


    @property
    def observation_space(self) -> gym.spaces.Box:
        # [sin, cos, vx, vy, az, 
        #  dof_pos(12), dof_vel(12), action(12), 
        #  omega(3), rpy(3)]
        return gym.spaces.Box(
            low   = self.cfg.obs_clip[0], high = self.cfg.obs_clip[1], 
            shape = (self.dim_observations * self.cfg.history_frames,),
            dtype = np.float32
        )
    
    @property
    def action_space(self) -> gym.spaces.Box:
        return gym.spaces.Box(low   = -np.pi, high  = np.pi, 
                              shape = (self.cfg.n_dofs,), 
                              dtype = np.float32)


    def build_observation(self, body_lin_vel, body_ang_vel, body_quat, 
                          cmd_vel, dofs_pos, dofs_vel, **kwargs) -> torch.Tensor: # type: ignore
        def quaternion_to_euler_array(quat):
            w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
            t0 = +2.0 * (w * x + y * z)
            t1 = +1.0 - 2.0 * (x * x + y * y)
            roll = torch.atan2(t0, t1)
            t2 = +2.0 * (w * y - z * x)
            t2 = torch.clamp(t2, -1.0, 1.0)
            pitch = torch.asin(t2)
            t3 = +2.0 * (w * z + x * y)
            t4 = +1.0 - 2.0 * (y * y + z * z)
            yaw = torch.atan2(t3, t4) 
            return torch.stack([roll, pitch, yaw], dim=-1)
        def add_noise(x: torch.Tensor, scale: float):
            return x + (torch.randn_like(x) * scale)
        n_envs = self.scene.n_envs

        # 1. Phase Observations (Sin/Cos)
        phase = self.time_steps * (2.0 * torch.pi * self.cfg.step_dt / self.cfg.cycle_time)
        obs_sin_phase = torch.sin(phase).reshape((n_envs, 1))
        obs_cos_phase = torch.cos(phase).reshape((n_envs, 1))

        # 2. Command Velocity (vx, vy, az)
        obs_cmd_vel = cmd_vel.reshape((n_envs, 3))

        # 3. DOF Position: (q - offset) * scale
        # The snippet uses 1.0 as the scale for dof_pos
        obs_dofs_pos = (dofs_pos - self.target_q_offset) * self.cfg.dofs_pos_scale
        
        # 4. DOF Velocity: dq * scale
        # The snippet uses 0.05 as the scale for dof_vel
        obs_dofs_vel = dofs_vel * self.cfg.dofs_vel_scale

        # 5. Last Action (n_dofs dims)
        obs_last_action = self.last_action

        # 6. Base Angular Velocity + Noise
        # The snippet uses noise scale: 0.12 * 0.6
        obs_body_ang_vel = add_noise(body_ang_vel, self.cfg.body_ang_vel_noise)

        # 7. Euler Angles + Noise 
        eu_ang = quaternion_to_euler_array(body_quat)
        # Normalize euler angles if they exceed pi
        eu_ang = torch.where(eu_ang > np.pi, eu_ang - 2 * np.pi, eu_ang)
        obs_eu_ang = add_noise(eu_ang, self.cfg.body_eu_ang_noise)

        # Concatenate in the exact order found in the sim-to-sim script:
        # [sin, cos, vx, vy, az, dof_pos, dof_vel, action, omega, rpy]
        obs_single_frame = torch.cat((
            obs_sin_phase,      # 1
            obs_cos_phase,      # 1
            obs_cmd_vel,        # 3
            obs_dofs_pos,       # 12
            obs_dofs_vel,       # 12
            obs_last_action,    # 12
            obs_body_ang_vel,   # 3
            obs_eu_ang          # 3
        ), dim=-1) # Total = 47
        obs_single_frame = torch.clip(obs_single_frame, 
                                      self.cfg.obs_clip[0], 
                                      self.cfg.obs_clip[1])

        self.last_obs = torch.roll(self.last_obs, shifts=-1, dims=1)
        self.last_obs[:, -1, :] = obs_single_frame
        return self.last_obs.reshape(n_envs, -1)
