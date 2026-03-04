# robot.py
#   Build up a robot
#

import torch
import numpy as np
import genesis as gs
import gymnasium as gym
from typing import TypeVar, Generic
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

@dataclass(kw_only = True)
class RobotConfig:
    robot_URDF:     str                 # urdf path
    base_link_name: str                 # base link_name
    joint_names:    list[str]           # all joint names
    kp:             np.ndarray          # kp
    kv:             np.ndarray          # kv
    velocity_range: np.ndarray          # joint velocity range
    force_range:    np.ndarray          # joint force(torque) range, 2xN
    initial_pos:    np.ndarray  = field(default_factory=\
            lambda: np.array([0.0, 0.0, 0.0], dtype=np.float32))
    initial_quat:   np.ndarray  = field(default_factory=\
            lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    decimate_aggressiveness: int = 10   # convexify
    self_collision: bool        = True


class Robot(ABC):
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
    @torch.no_grad()
    @torch.compiler.disable 
    def __gs_build(self) -> None:
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

    @torch.no_grad()
    @torch.compiler.disable 
    def __gs_config(self) -> None:
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


    @torch.no_grad()
    @torch.compiler.disable
    def __gs_step(self, action: torch.Tensor, envs_idx: torch.Tensor) -> None:
        self.robot.control_dofs_position(action, 
                                         dofs_idx_local=self.dofs_idx_local, 
                                         envs_idx=envs_idx)

    @torch.no_grad()
    @torch.compiler.disable
    def __gs_reset(self, 
                   reset_pos:     torch.Tensor, 
                   reset_quat:    torch.Tensor, 
                   envs_idx:      torch.Tensor) -> None:
        self.robot.set_dofs_position(torch.zeros((len(self.dofs_idx_local)), 
                                                 device=gs.device), 
                                     dofs_idx_local=self.dofs_idx_local, 
                                     envs_idx=envs_idx)
        self.robot.set_pos(pos=reset_pos, envs_idx=envs_idx)
        self.robot.set_quat(quat=reset_quat, envs_idx=envs_idx)
    
    @torch.no_grad()
    @torch.compiler.disable 
    def __gs_state(self, envs_idx: torch.Tensor) -> dict[str, torch.Tensor]:
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
        }


    # -------------------------------------------------
    # Steps and reset and state
    # They're here to be the final of the compute graph
    # -------------------------------------------------
    @torch.no_grad()
    @torch.compile()
    def build(self) -> None:
        self.__gs_build()

    @torch.no_grad()
    @torch.compile()
    def config(self) -> None:
        self.__gs_config()
        self.n_envs = self.scene.n_envs
        self.n_dofs = len(self.cfg.joint_names)
        self.__init_pos  = torch.from_numpy(self.cfg.initial_pos).to(gs.device)
        self.__init_quat = torch.from_numpy(self.cfg.initial_quat).to(gs.device)

    @torch.no_grad()
    @torch.compile()
    def step(self, action: torch.Tensor, envs_idx: torch.Tensor) -> None:
        self.__gs_step(action=action, envs_idx=envs_idx)

    @torch.no_grad()
    @torch.compile()
    def reset(self, envs_idx: torch.Tensor, 
              reset_pos: torch.Tensor | None = None,
              reset_quat: torch.Tensor | None = None, **kwargs) -> None:
        n_reset_envs = envs_idx.shape[0]
        reset_pos = reset_pos or self.__init_pos.broadcast_to((n_reset_envs, 3))
        reset_quat = reset_quat or self.__init_quat.broadcast_to((n_reset_envs, 4))
        self.__gs_reset(reset_pos=reset_pos, reset_quat=reset_quat, 
                        envs_idx=envs_idx)
    
    @torch.no_grad()
    @torch.compile()
    def get_state(self, envs_idx: torch.Tensor) -> dict[str, torch.Tensor]:
        return {**self.__gs_state(envs_idx=envs_idx)}
    

    # --------------------------------------
    # Observation, actions
    # The functions you may want to override
    # --------------------------------------
    @property
    @abstractmethod
    def observation_space(self) -> gym.spaces.Box:
        pass
    
    @property
    @abstractmethod
    def action_space(self) -> gym.spaces.Box:
        pass
    
    @abstractmethod
    def build_observation(self, **kwargs) -> torch.Tensor:
        '''
        This method is to be called by the env which often requires 
        extra kwargs that the robot itself doesn't know. 
        '''
        pass 
