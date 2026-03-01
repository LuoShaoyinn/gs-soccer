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
    initial_pos:    np.ndarray = field(default_factory=\
            lambda: np.array([0.0, 0.0, 0.0], dtype=np.float32))
    initial_quat:   np.ndarray = field(default_factory=\
            lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    decimate_aggressiveness: int = 5    # convexify
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
        self.initial_pos = torch.from_numpy(self.cfg.initial_pos).to(gs.device)
        self.initial_quat = torch.from_numpy(self.cfg.initial_quat).to(gs.device)
        self.__last_action = torch.zeros((self.scene.n_envs, len(self.cfg.joint_names)),
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
        self.gs_step(action, envs_idx)
        self.__last_action = action

    #@torch.no_grad()
    #@torch.compile()
    def reset(self, envs_idx: torch.Tensor) -> None:
        reset_n = envs_idx.shape[0]
        reset_pos  = torch.broadcast_to(self.initial_pos,  (reset_n, 3))
        reset_quat = torch.broadcast_to(self.initial_quat, (reset_n, 4))
        self.gs_reset(reset_pos=reset_pos, reset_quat=reset_quat, envs_idx=envs_idx)
        self.__last_action[envs_idx] = 0.0
    

    # --------------------------------------
    # Observation, actions
    # The functions you may want to override
    # --------------------------------------
    @property
    def observation_space(self) -> gym.spaces.Box:
        N = len(self.cfg.joint_names)
        return gym.spaces.Box(low   = -100.0, 
                              high  =  100.0, 
                              shape = (12 + 3 * N,), 
                              dtype = np.float32)
    
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
            cmd_vel, dofs_pos, dofs_vel, last_action, **kwargs) -> torch.Tensor:
        def compute_projected_gravity(quat):
            w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
            # Pre-calculate common terms for speed
            xw = x * w
            yw = y * w
            yz = y * z
            xz = x * z
            # Calculate components
            # These represent the local-frame gravity vector
            gx = 2 * (xz - yw)
            gy = 2 * (yz + xw)
            gz = w**2 - x**2 - y**2 + z**2
            return torch.stack([gx, gy, gz], dim=-1)
        def add_noise(x: torch.Tensor, scale: float):
            return x + (torch.rand(x.shape, device=gs.device) * 2.0 - 1.0) * scale
        obs_body_lin_vel  = torch.clip(body_lin_vel, min=-100.0, max=100.0)
        obs_body_ang_vel  = add_noise(body_ang_vel * 0.2, 0.2)
        obs_body_ang_vel  = torch.clip(obs_body_ang_vel, min=-100.0, max=100.0)
        obs_proj_gravity  = add_noise(compute_projected_gravity(body_quat), 0.05)
        obs_cmd_vel       = cmd_vel
        obs_dofs_pos      = add_noise(dofs_pos, 0.1)
        obs_dofs_pos      = torch.clip(obs_dofs_pos, min=-100.0, max=100.0)
        obs_dofs_vel      = add_noise(dofs_vel * 0.05, 1.5)
        obs_dofs_vel      = torch.clip(obs_dofs_vel, min=-100.0, max=100.0)
        obs_last_action   = last_action
        return torch.cat((obs_body_lin_vel, 
                          obs_body_ang_vel, 
                          obs_proj_gravity,
                          obs_cmd_vel,
                          obs_dofs_pos,
                          obs_dofs_vel,
                          obs_last_action), dim=1) 
