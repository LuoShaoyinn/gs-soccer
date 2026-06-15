# robot.py
#   Build up a robot
#

import torch
import numpy as np
import genesis as gs
from dataclasses import dataclass, field
from abc import ABC

@dataclass(kw_only = True)
class RobotConfig:
    robot_URDF:     str                 # urdf path
    base_link_name: str                 # base link_name
    joint_names:    list[str]           # all joint names
    foot_link_names: list[str]          # foot link names for contact/position sensing
    kp:             np.ndarray          # kp
    kv:             np.ndarray          # kv
    armature:       np.ndarray          # per-joint armature
    damping:        np.ndarray          # per-joint damping
    velocity_range: np.ndarray          # joint velocity range
    force_range:    np.ndarray          # joint force(torque) range, 2xN
    initial_pos:    np.ndarray   = field(default_factory=\
            lambda: np.array([0.0, 0.0, 0.0], dtype=np.float32))
    initial_quat:   np.ndarray   = field(default_factory=\
            lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    vis_mode:           str             = "visual"
    decimate:           bool            = True
    decimate_face_num:  int             = 100
    decimate_aggressiveness: int        = 8
    friction_rnd:       float           = 0.0
    mass_shift_rnd:     float           = 0.0
    com_shift_rnd:      float           = 0.0
    kp_ratio_rnd:       float           = 0.0
    kv_ratio_rnd:       float           = 0.0
    armature_ratio_rnd: float           = 0.0
    damping_ratio_rnd:  float           = 0.0


class Robot(ABC):
    def __init__(self, cfg: RobotConfig, scene: gs.Scene):
        self.cfg = cfg
        self.scene = scene
        assert isinstance(cfg.robot_URDF, str), \
            f"URDF filename should be a str, got {cfg.robot_URDF}"
        assert isinstance(cfg.joint_names, list), \
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
            file_meshes_are_zup = True, \
            requires_jac_and_IK = False, \
        ), vis_mode=self.cfg.vis_mode)
        self.robot_base = self.robot.get_link(self.cfg.base_link_name)
        self.dofs_idx_local = torch.tensor(\
                [ self.robot.get_joint(name).dofs_idx_local[0] for name in self.cfg.joint_names], \
                device=gs.device)
        self.foot_links_idx_local = torch.tensor( \
                [ self.robot.get_link(name).idx_local for name in self.cfg.foot_link_names], \
                device=gs.device)
        self.imu = self.scene.add_sensor( \
            gs.sensors.IMU( \
                entity_idx=self.robot.idx,
                link_idx_local=self.robot_base.idx_local,
                pos_offset=(0.0, 0.0, 0.0),               
            )
        )
    @torch.compiler.disable 
    def __gs_config(self) -> None:
        n_dofs = len(self.dofs_idx_local)
        def rnd(scale: float, shape: tuple):
            if scale == 0.0:
                return torch.zeros(*shape, device=gs.device)
            return scale * (torch.rand(self.scene.n_envs, *shape) - 0.5)
        kp = torch.as_tensor(self.cfg.kp, dtype=torch.float32, device=gs.device)
        kv = torch.as_tensor(self.cfg.kv, dtype=torch.float32, device=gs.device)
        armature = torch.as_tensor(self.cfg.armature, dtype=torch.float32, device=gs.device)
        damping = torch.as_tensor(self.cfg.damping, dtype=torch.float32, device=gs.device)
        self.robot.set_dofs_kp(
            kp             = kp + rnd(self.cfg.kp_ratio_rnd, shape=(n_dofs,)),
            dofs_idx_local = self.dofs_idx_local,
        )
        self.robot.set_dofs_kv(
            kv             = kv + rnd(self.cfg.kv_ratio_rnd, shape=(n_dofs,)),
            dofs_idx_local = self.dofs_idx_local,
        )
        self.robot.set_dofs_force_range(
            lower          = self.cfg.force_range[0],
            upper          = self.cfg.force_range[1],
            dofs_idx_local = self.dofs_idx_local,
        )
        self.robot.set_dofs_armature(
            armature       = armature + rnd(self.cfg.armature_ratio_rnd, shape=(n_dofs,)),
            dofs_idx_local = self.dofs_idx_local,
        )
        self.robot.set_dofs_damping(
            damping        = damping + rnd(self.cfg.damping_ratio_rnd, shape=(n_dofs,)),
            dofs_idx_local = self.dofs_idx_local,
        )
        if self.cfg.mass_shift_rnd != 0.0:
            self.robot.set_mass_shift(
                mass_shift      = rnd(self.cfg.mass_shift_rnd, shape=(self.robot.n_links,)),
                links_idx_local = range(self.robot.n_links),
            )
        if self.cfg.com_shift_rnd != 0.0:
            self.robot.set_COM_shift(
                com_shift       = rnd(self.cfg.com_shift_rnd, shape=(self.robot.n_links, 3)),
                links_idx_local = range(self.robot.n_links),
            )
        if self.cfg.friction_rnd != 0.0:
            self.robot.set_friction_ratio(
                friction_ratio  = rnd(self.cfg.friction_rnd, shape=(self.robot.n_links,)) + 1.0,
                links_idx_local = range(self.robot.n_links),
            )

    @torch.compiler.disable
    def __gs_step(self, action: torch.Tensor) -> None:
        self.robot.control_dofs_position(action, 
                                         dofs_idx_local=self.dofs_idx_local)

    @torch.compiler.disable
    def __gs_reset(self, 
                   envs_idx:    torch.Tensor,
                   joint_pos:   torch.Tensor,
                   reset_pos:   torch.Tensor, 
                   reset_quat:  torch.Tensor, **kwargs) -> None:
        self.robot.set_dofs_position(joint_pos,
                                     dofs_idx_local=self.dofs_idx_local, 
                                     envs_idx=envs_idx)
        self.robot.set_pos(pos=reset_pos, envs_idx=envs_idx)
        self.robot.set_quat(quat=reset_quat, envs_idx=envs_idx)
    
    @torch.compiler.disable 
    def __gs_state(self, envs_idx: torch.Tensor) -> dict[str, torch.Tensor]:
        robot = self.robot
        robot_base = self.robot_base
        dofs_idx_local = self.dofs_idx_local
        state = {
            "dofs_pos"  :       robot.get_dofs_position(dofs_idx_local=dofs_idx_local, envs_idx=envs_idx), 
            "dofs_vel"  :       robot.get_dofs_velocity(dofs_idx_local=dofs_idx_local, envs_idx=envs_idx),
            "dofs_force":       robot.get_dofs_control_force(dofs_idx_local=dofs_idx_local, envs_idx=envs_idx),
            "body_pos"  :       robot_base.get_pos(envs_idx=envs_idx), 
            "body_quat" :       robot_base.get_quat(envs_idx=envs_idx), 
            "body_lin_vel"  :   robot_base.get_vel(envs_idx=envs_idx), 
            "body_ang_vel"  :   self.imu.read(envs_idx=envs_idx).ang_vel, 
            "link_contact_forces": robot.get_links_net_contact_force(envs_idx=envs_idx),
            "foot_pos"  :       robot.get_links_pos(links_idx_local=self.foot_links_idx_local, envs_idx=envs_idx),
            "foot_vel"  :       robot.get_links_vel(links_idx_local=self.foot_links_idx_local, envs_idx=envs_idx),
            "foot_idx_local":   self.foot_links_idx_local,
        }
        return state


    # -------------------------------------------------
    # Steps and reset and state
    # They're here to be the final of the compute graph
    # -------------------------------------------------
    def build(self) -> None:
        self.__gs_build()

    def config(self) -> None:
        self.__gs_config()
        self.n_envs = self.scene.n_envs
        self.n_dofs = len(self.cfg.joint_names)
        self.init_pos  = torch.from_numpy(self.cfg.initial_pos).to(gs.device)
        self.init_quat = torch.from_numpy(self.cfg.initial_quat).to(gs.device)

    def step(self, action: torch.Tensor) -> None:
        self.__gs_step(action=action)

    def reset(self, envs_idx: torch.Tensor, **kwargs) -> None:
        self.__gs_reset(envs_idx=envs_idx, **kwargs)
    
    def get_state(self, envs_idx: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.__gs_state(envs_idx=envs_idx)
