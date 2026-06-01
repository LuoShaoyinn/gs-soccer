# robot.py
#   Build up a robot
#

import torch
import numpy as np
import genesis as gs
import gymnasium as gym
from typing import Any, TypeVar, Generic
from torch.nn import functional as F
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
    initial_pos:    np.ndarray   = field(default_factory=\
            lambda: np.array([0.0, 0.0, 0.0], dtype=np.float32))
    initial_quat:   np.ndarray   = field(default_factory=\
            lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    head_link_name: str | None = None
    head_camera_res: tuple[int, int] = (320, 240)
    head_camera_fov: float = 70.0
    head_camera_near: float = 0.01
    head_camera_far: float = 20.0
    head_camera_lookahead: float = 1.0
    head_camera_pos_offset: np.ndarray = field(default_factory=\
            lambda: np.array([0.08, 0.0, 0.0], dtype=np.float32))
    head_camera_forward: np.ndarray = field(default_factory=\
            lambda: np.array([1.0, 0.0, 0.0], dtype=np.float32))
    head_camera_up: np.ndarray = field(default_factory=\
            lambda: np.array([0.0, 0.0, 1.0], dtype=np.float32))
    decimate: bool = True
    decimate_face_num: int = 100
    decimate_aggressiveness: int = 8


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
    @torch.compiler.disable 
    def __gs_build(self) -> None:
        self.robot = self.scene.add_entity(gs.morphs.URDF( \
            file = self.cfg.robot_URDF, \
            pos  = self.cfg.initial_pos, \
            quat = self.cfg.initial_quat, \
            decimate = self.cfg.decimate, \
            decimate_face_num = self.cfg.decimate_face_num, \
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
        self.head_camera = None
        self.head_camera_link = None

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


    @staticmethod
    def __rotate_by_quat(vec: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
        if vec.ndim == 1:
            vec = vec.broadcast_to((quat.shape[0], 3))
        q_xyz = quat[:, 1:4]
        q_w = quat[:, 0:1]
        t = 2.0 * torch.cross(q_xyz, vec, dim=1)
        return vec + q_w * t + torch.cross(q_xyz, t, dim=1)

    @torch.compiler.disable
    def __head_camera_pose(self, env_idx: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.head_camera_link is None:
            raise RuntimeError("Head camera has not been built. Call build_head_camera() before scene.build().")

        envs_idx = torch.tensor([env_idx], dtype=torch.long, device=gs.device)
        head_pos = self.head_camera_link.get_pos(envs_idx=envs_idx)
        head_quat = self.head_camera_link.get_quat(envs_idx=envs_idx)
        offset = torch.as_tensor(self.cfg.head_camera_pos_offset, dtype=gs.tc_float, device=gs.device)
        forward = torch.as_tensor(self.cfg.head_camera_forward, dtype=gs.tc_float, device=gs.device)
        up = torch.as_tensor(self.cfg.head_camera_up, dtype=gs.tc_float, device=gs.device)

        forward = forward / torch.clamp(torch.linalg.norm(forward), min=1e-6)
        up = up / torch.clamp(torch.linalg.norm(up), min=1e-6)
        camera_pos = head_pos + self.__rotate_by_quat(offset, head_quat)
        camera_forward = self.__rotate_by_quat(forward, head_quat)
        camera_up = self.__rotate_by_quat(up, head_quat)
        camera_lookat = camera_pos + self.cfg.head_camera_lookahead * camera_forward
        return camera_pos[0], camera_lookat[0], camera_up[0]

    @torch.compiler.disable
    def __gs_update_head_camera_pose(self, env_idx: int = 0) -> None:
        if self.head_camera is None:
            raise RuntimeError("Head camera has not been built. Call build_head_camera() before scene.build().")
        pos, lookat, up = self.__head_camera_pose(env_idx=env_idx)
        self.head_camera.set_pose(pos=pos, lookat=lookat, up=up)

    @torch.compiler.disable
    def __gs_step(self, action: torch.Tensor) -> None:
        self.robot.control_dofs_position(action, 
                                         dofs_idx_local=self.dofs_idx_local)

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
    def build(self) -> None:
        self.__gs_build()

    def build_head_camera(
        self,
        *,
        res: tuple[int, int] | None = None,
        fov: float | None = None,
        GUI: bool = False,
        env_idx: int | None = None,
    ) -> Any:
        if self.cfg.head_link_name is None:
            raise ValueError("RobotConfig.head_link_name must be set before building a head camera.")
        self.head_camera_link = self.robot.get_link(self.cfg.head_link_name)
        res = res or self.cfg.head_camera_res
        fov = fov or self.cfg.head_camera_fov
        self.head_camera = self.scene.add_camera(
            res=res,
            pos=(0.0, 0.0, 1.0),
            lookat=(1.0, 0.0, 1.0),
            fov=fov,
            GUI=GUI,
            near=self.cfg.head_camera_near,
            far=self.cfg.head_camera_far,
            env_idx=env_idx,
        )
        return self.head_camera

    def config(self) -> None:
        self.__gs_config()
        self.n_envs = self.scene.n_envs
        self.n_dofs = len(self.cfg.joint_names)
        self.init_pos  = torch.from_numpy(self.cfg.initial_pos).to(gs.device)
        self.init_quat = torch.from_numpy(self.cfg.initial_quat).to(gs.device)

    def step(self, action: torch.Tensor) -> None:
        self.__gs_step(action=action)

    def update_head_camera_pose(self, env_idx: int = 0) -> None:
        self.__gs_update_head_camera_pose(env_idx=env_idx)

    def render_head_camera(self, env_idx: int = 0, **kwargs) -> Any:
        self.update_head_camera_pose(env_idx=env_idx)
        return self.head_camera.render(**kwargs)

    def reset(self, envs_idx: torch.Tensor, 
              reset_pos: torch.Tensor | None = None,
              reset_quat: torch.Tensor | None = None, **kwargs) -> None:
        n_reset_envs = envs_idx.shape[0]
        if reset_pos is None:
            reset_pos = self.init_pos.broadcast_to((n_reset_envs, 3))
        elif reset_pos.shape[1] == 2:
            reset_pos = F.pad(reset_pos, (0, 1), value=self.cfg.initial_pos[2].item())
        if reset_quat is None:
            reset_quat = self.init_quat.broadcast_to((n_reset_envs, 4))
        self.__gs_reset(reset_pos=reset_pos, reset_quat=reset_quat, 
                        envs_idx=envs_idx)
    
    def get_state(self, envs_idx: torch.Tensor) -> dict[str, torch.Tensor]:
        return {**self.__gs_state(envs_idx=envs_idx)}
