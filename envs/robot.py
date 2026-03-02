# robot.py
#   Build up a robot
#

import torch
import numpy as np
import genesis as gs
import gymnasium as gym
from dataclasses import dataclass, field


@dataclass
class RobotConfig:
    robot_URDF:     str                 # urdf path
    joint_names:    list[str]           # all joint names
    kp:             np.ndarray          # kp
    kv:             np.ndarray          # kv
    velocity_range: np.ndarray          # joint velocity range
    force_range:    np.ndarray          # joint force(torque) range, 2xN
    initial_pos:    np.ndarray  = field(default_factory=\
            lambda: np.array([0.0, 0.0, 0.0], dtype=np.float32))
    initial_quat:   np.ndarray  = field(default_factory=\
            lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    target_q_offset:np.ndarray  = field(default_factory=\
            lambda: np.array([0.0] * 12, dtype=np.float32))
    cycle_time:    float        = 0.8
    step_ewma_factor: float     = 0.2
    decimate_aggressiveness: int = 10    # convexify
    base_link_name: str = "base"        # base link_name


class Robot:
    def __init__(self, cfg: RobotConfig, scene: gs.Scene):
        self.cfg = cfg
        self.scene = scene
        assert type(cfg.robot_URDF) == str, \
            f"URDF filename should be a str, got {cfg.robot_URDF}"
        assert type(cfg.joint_names) == list, \
            f"joint_names should be a list[str], got{cfg.joint_names}"


    # ------------------------
    # Genesis helper functions
    # ------------------------
    #@torch.no_grad()
    #@torch.compiler.disable # prevent torch from compiling underlying gs
    def gs_build(self) -> None:
        self.robot = self.scene.add_entity(gs.morphs.URDF( \
            file = self.cfg.robot_URDF, \
            pos  = self.cfg.initial_pos, \
            quat = self.cfg.initial_quat, \
            decimate_aggressiveness = self.cfg.decimate_aggressiveness, \
            requires_jac_and_IK = False, \
        ), vis_mode='collision')
        self.robot_base = self.robot.get_link(self.cfg.base_link_name)
        self.dofs_idx_local = [ self.robot.get_joint(name).dofs_idx_local[0] \
                                for name in self.cfg.joint_names]
        self.dofs_idx_local = torch.tensor( self.dofs_idx_local, \
                                            device=gs.device)
        self.imu = self.scene.add_sensor( \
            gs.sensors.IMU( \
                entity_idx=self.robot.idx,                  # type: ignore
                link_idx_local=self.robot_base.idx_local,   # type: ignore
                pos_offset=(0.0, 0.0, 0.0),                 # type: ignore
            )
        )

    #@torch.no_grad()
    #@torch.compiler.disable # prevent torch from compiling underlying gs
    def gs_config(self) -> None:
        self.robot.set_dofs_kp(
            kp             = self.cfg.kp, 
            dofs_idx_local = self.dofs_idx_local,
        )
        self.robot.set_dofs_kv(
            kv             = self.cfg.kv, 
            dofs_idx_local = self.dofs_idx_local,
        )
        self.robot.set_dofs_force_range(
            lower          = self.cfg.force_range[0], 
            upper          = self.cfg.force_range[1], 
            dofs_idx_local = self.dofs_idx_local,
        )

        # initialize some constants for quick refering
        # we have to do it here after the scene is initialized
        self.n_envs = self.scene.n_envs
        self.n_dofs = len(self.cfg.joint_names)
        self.initial_pos  = torch.from_numpy(self.cfg.initial_pos).to(gs.device)
        self.initial_quat = torch.from_numpy(self.cfg.initial_quat).to(gs.device)
        self.target_q_offset = torch.from_numpy(self.cfg.target_q_offset).to(gs.device)
        self.__last_action = torch.zeros((self.n_envs, self.n_dofs),
                                       dtype=torch.float,
                                       device=gs.device)
        self.__last_target_q = torch.zeros((self.n_envs, self.n_dofs),
                                           dtype=torch.float,
                                           device=gs.device)
        self.__last_obs = torch.zeros((self.n_envs, 5, 47),
                                       dtype=torch.float,
                                       device=gs.device)

    #@torch.no_grad()
    #@torch.compiler.disable # prevent torch from compiling underlying gs
    def gs_step(self, action: torch.Tensor, envs_idx: torch.Tensor) -> None:
        self.robot.control_dofs_position(action, 
                                         dofs_idx_local=self.dofs_idx_local, 
                                         envs_idx=envs_idx)

    #@torch.no_grad()
    #@torch.compiler.disable # prevent torch from compiling underlying gs
    def gs_reset(self, 
                 reset_pos:     torch.Tensor, 
                 reset_quat:    torch.Tensor, 
                 envs_idx:      torch.Tensor) -> None:
        self.robot.set_dofs_position(torch.zeros((len(self.dofs_idx_local)), 
                                                 device=gs.device), 
                                     dofs_idx_local=self.dofs_idx_local, 
                                     envs_idx=envs_idx)
        self.robot.set_pos(pos=reset_pos, envs_idx=envs_idx)
        self.robot.set_quat(quat=reset_quat, envs_idx=envs_idx)
    
    #@torch.no_grad()
    #@torch.compiler.disable # prevent torch from compiling underlying gs
    def gs_state(self, envs_idx: torch.Tensor) -> dict[str, torch.Tensor]:
        robot = self.robot
        robot_base = self.robot_base
        dofs_idx_local = self.dofs_idx_local
        return {
            "dofs_pos"  : robot.get_dofs_position(dofs_idx_local=dofs_idx_local, envs_idx=envs_idx), 
            "dofs_vel"  : robot.get_dofs_velocity(dofs_idx_local=dofs_idx_local, envs_idx=envs_idx),
            "body_pos"  : robot_base.get_pos(envs_idx=envs_idx), 
            "body_quat" : robot_base.get_quat(envs_idx=envs_idx), 
            "body_lin_vel"  : robot_base.get_vel(envs_idx=envs_idx), 
            "body_ang_vel"  : self.imu.read(envs_idx=envs_idx).ang_vel, 
            "last_action"   : self.__last_action,
        }


    # ---------------
    # Steps and reset
    # ---------------
    #@torch.no_grad()
    #@torch.compile()
    def step(self, action: torch.Tensor, envs_idx: torch.Tensor) -> None:
        '''
        Here action is target_q
        '''
        self.__last_action = action
        self.__last_target_q[envs_idx] *= self.cfg.step_ewma_factor
        self.__last_target_q[envs_idx] += 0.25 * action * (1.0 - self.cfg.step_ewma_factor)
        self.gs_step(action=self.__last_target_q[envs_idx] + self.target_q_offset,
                     envs_idx=envs_idx)

    #@torch.no_grad()
    #@torch.compile()
    def reset(self, envs_idx: torch.Tensor) -> None:
        reset_n = envs_idx.shape[0]
        reset_pos  = torch.broadcast_to(self.initial_pos,  (reset_n, 3))
        reset_quat = torch.broadcast_to(self.initial_quat, (reset_n, 4))
        self.gs_reset(reset_pos=reset_pos, reset_quat=reset_quat, envs_idx=envs_idx)
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
    def get_state(self, envs_idx: torch.Tensor, **kwargs) -> dict[str, torch.Tensor]:
        return {**self.gs_state(envs_idx), **kwargs}

    #@torch.no_grad()
    #@torch.compile()
    def get_observation(self, body_lin_vel, body_ang_vel, body_quat, \
                        cmd_vel, dofs_pos, dofs_vel, **kwargs) -> torch.Tensor: 
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
