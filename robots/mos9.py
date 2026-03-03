# robot.py
#   Inherit from robot and build up mos9
#

import torch
import numpy as np
import genesis as gs
import gymnasium as gym
from dataclasses import dataclass, field

from .robot import Robot, RobotConfig


@dataclass(kw_only = True)
class MOS9Config(RobotConfig):
    robot_URDF:     str         = 'assets/MOS9/MOS9_walk.urdf'
    base_link_name: str         = "body"
    joint_names:    list[str]   = field(default_factory=\
            lambda: ['b_Lh','Lh_Ll','Ll_Ll1','Ll1_Ll2','Ll2_La','La_Lf', 
                   'b_Rh','Rh_Rl','Rl_Rl1','Rl1_Rl2','Rl2_Ra','Ra_Rf'])
    kp:             np.ndarray  = field(default_factory=\
            lambda: np.array([100.0, 100.0,100.0, 100.0, 50.0, 24.0,
                              100.0, 100.0,100.0, 100.0, 50.0, 24.0], 
                             dtype=np.float32))
    kv:             np.ndarray  = field(default_factory=\
            lambda: np.array([2.0, 2.0, 2.0, 2.0,1.5,0.3,
                              2.0, 2.0, 2.0, 2.0,1.5,0.3], 
                             dtype=np.float32))
    initial_pos:    np.ndarray  = field(default_factory=\
            lambda: np.array([0.0, 0.0, 0.5], dtype=np.float32))
    force_range:    np.ndarray  = field(default_factory=\
            lambda: np.array([[-100] * 12, [100] * 12], dtype=np.float32))
    velocity_range: np.ndarray  = field(default_factory=\
            lambda: np.array([[-100] * 12, [100] * 12], dtype=np.float32)) 
    target_q_offset: np.ndarray  = field(default_factory=\
            lambda: np.array([0.0] * 12, dtype=np.float32))
    cycle_time:    float        = 0.8
    step_ewma_factor: float     = 0.2


class MOS9(Robot):
    cfg: MOS9Config

    def config(self) -> None:
        super().config()
        self.target_q_offset = torch.tensor(self.cfg.target_q_offset,  device=gs.device)
        self.__last_action   = torch.zeros((self.n_envs, self.n_dofs), device=gs.device)
        self.__last_target_q = torch.zeros((self.n_envs, self.n_dofs), device=gs.device)
        self.__last_obs      = torch.zeros((self.scene.n_envs, 5, 47), device=gs.device)

    #@torch.no_grad()
    #@torch.compile()
    def step(self, action: torch.Tensor, envs_idx: torch.Tensor) -> None:
        '''
        Here action is target_q
        '''
        self.__last_action = action
        self.__last_target_q[envs_idx] *= self.cfg.step_ewma_factor
        self.__last_target_q[envs_idx] += 0.25 * action * (1.0 - self.cfg.step_ewma_factor)
        super().step(action=self.__last_target_q[envs_idx] + self.target_q_offset,
                     envs_idx=envs_idx)

    #@torch.no_grad()
    #@torch.compile()
    def reset(self, envs_idx: torch.Tensor, **kwargs) -> None: # type: ignore[override]
        super().reset(envs_idx=envs_idx, **kwargs)
        self.__last_action[envs_idx] = 0.0
        self.__last_target_q[envs_idx] = self.target_q_offset
        self.__last_obs[envs_idx] = 0.0
    

    # --------------------------------------
    # Observation, actions
    # The functions you may want to override
    # --------------------------------------
    @property
    def observation_space(self) -> gym.spaces.Box:
        # Each single frame is 47 dimensions:
        # [sin, cos, vx, vy, az, dof_pos(12), dof_vel(12), action(12), omega(3), rpy(3)]
        return gym.spaces.Box(
            low   = -18.0, 
            high  = 18.0, 
            shape = (47 * 5,),
            dtype = np.float32
        )
    
    @property
    def action_space(self) -> gym.spaces.Box:
        N = len(self.cfg.joint_names)
        pos_limits = self.robot.get_dofs_limit(dofs_idx_local=self.dofs_idx_local)
        return gym.spaces.Box(low   = pos_limits[0].cpu().numpy(), 
                              high  = pos_limits[1].cpu().numpy(), 
                              shape = (N,), 
                              dtype = np.float32)
    
    #@torch.no_grad()
    #@torch.compile()
    def build_observation(self, body_lin_vel, body_ang_vel, body_quat, \
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

        # 1. Phase Observations (Sin/Cos)
        phase = 2 * np.pi * self.scene.t * self.scene.dt / self.cfg.cycle_time
        obs_sin_phase = torch.sin(torch.tensor([phase], device=gs.device)) \
                             .repeat(self.n_envs, 1)
        obs_cos_phase = torch.cos(torch.tensor([phase], device=gs.device)) \
                             .repeat(self.n_envs, 1)

        # 2. Command Velocity (vx, vy, az)
        obs_cmd_vel = cmd_vel # Assumed [batch, 3]

        # 3. DOF Position: (q - offset) * scale
        # The snippet uses 1.0 as the scale for dof_pos
        obs_dofs_pos = (dofs_pos - self.target_q_offset) * 1.0
        
        # 4. DOF Velocity: dq * scale
        # The snippet uses 0.05 as the scale for dof_vel
        obs_dofs_vel = dofs_vel * 0.05

        # 5. Last Action (12 dims)
        obs_last_action = self.__last_action

        # 6. Base Angular Velocity + Noise
        # The snippet uses noise scale: 0.12 * 0.6
        obs_body_ang_vel = add_noise(body_ang_vel, 0.12 * 0.6)

        # 7. Euler Angles + Noise (Replaces Projected Gravity)
        eu_ang = quaternion_to_euler_array(body_quat)
        # Normalize euler angles if they exceed pi (as seen in snippet)
        eu_ang = torch.where(eu_ang > np.pi, eu_ang - 2 * np.pi, eu_ang)
        obs_eu_ang = add_noise(eu_ang, 0.12 * 0.6)

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
        obs_single_frame = torch.clip(obs_single_frame, -18.0, 18.0)

        # Final clipping (Snippet uses 18.0)
        self.__last_obs = torch.roll(self.__last_obs, shifts=-1, dims=1)
        self.__last_obs[:, -1, :] = obs_single_frame
        return self.__last_obs.reshape(self.scene.n_envs, -1)
