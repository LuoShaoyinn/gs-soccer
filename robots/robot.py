# robot.py
#   Build up a robot
#

import torch
import numpy as np
import genesis as gs
from torch.nn import functional as F
from dataclasses import dataclass, field
from abc import ABC

@dataclass(kw_only = True)
class RobotConfig:
    robot_URDF:     str                 # urdf path
    base_link_name: str                 # base link_name
    joint_names:    list[str]           # all joint names
    foot_link_names: list[str] | None = None  # foot link names for contact/position sensing
    kp:             np.ndarray          # kp
    kv:             np.ndarray          # kv
    armature:       np.ndarray | None = None  # per-joint armature
    damping:        np.ndarray | float | None = None  # per-joint damping
    velocity_range: np.ndarray          # joint velocity range
    force_range:    np.ndarray          # joint force(torque) range, 2xN
    initial_pos:    np.ndarray   = field(default_factory=\
            lambda: np.array([0.0, 0.0, 0.0], dtype=np.float32))
    initial_quat:   np.ndarray   = field(default_factory=\
            lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    vis_mode: str = "visual"
    decimate: bool = True
    decimate_face_num: int = 100
    decimate_aggressiveness: int = 8
    file_meshes_are_zup: bool = True
    friction_ratio_range: tuple[float, float] | None = None
    mass_shift_range:     tuple[float, float] | None = None
    com_shift_range:      tuple[float, float] | None = None
    kp_ratio_range:       tuple[float, float] | None = None
    kv_ratio_range:       tuple[float, float] | None = None
    armature_ratio_range: tuple[float, float] | None = None


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
            file_meshes_are_zup = self.cfg.file_meshes_are_zup, \
            requires_jac_and_IK = False, \
        ), vis_mode=self.cfg.vis_mode)
        self.robot_base = self.robot.get_link(self.cfg.base_link_name)
        self.dofs_idx_local = [ self.robot.get_joint(name).dofs_idx_local[0] \
                                for name in self.cfg.joint_names]
        self.dofs_idx_local = torch.tensor( self.dofs_idx_local, \
                                            device=gs.device)
        self.imu = self.scene.add_sensor( \
            gs.sensors.IMU( \
                entity_idx=self.robot.idx,
                link_idx_local=self.robot_base.idx_local,
                pos_offset=(0.0, 0.0, 0.0),               
            )
        )
        if self.cfg.foot_link_names is not None:
            self._foot_links = [self.robot.get_link(name) \
                                for name in self.cfg.foot_link_names]
            self._foot_links_idx_local = [link.idx_local \
                                          for link in self._foot_links]
        else:
            self._foot_links_idx_local = None
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
        if self.cfg.armature is not None:
            self.robot.set_dofs_armature(
                armature       = self.cfg.armature,
                dofs_idx_local = self.dofs_idx_local,
            )
        if self.cfg.damping is not None:
            self.robot.set_dofs_damping(
                damping        = self.cfg.damping,
                dofs_idx_local = self.dofs_idx_local,
            )

    @torch.compiler.disable
    def __gs_randomize(self) -> None:
        robot = self.robot
        n_envs = self.n_envs
        n_links = robot.n_links
        n_dofs = self.n_dofs
        cfg = self.cfg

        def uniform(lo: float, hi: float, *shape: int) -> torch.Tensor:
            return lo + (hi - lo) * torch.rand(shape, device=gs.device)

        if cfg.friction_ratio_range is not None:
            robot.set_friction_ratio(
                uniform(*cfg.friction_ratio_range, n_envs, n_links),
                links_idx_local=range(n_links),
            )
        if cfg.mass_shift_range is not None:
            robot.set_mass_shift(
                uniform(*cfg.mass_shift_range, n_envs, n_links),
                links_idx_local=range(n_links),
            )
        if cfg.com_shift_range is not None:
            robot.set_COM_shift(
                uniform(*cfg.com_shift_range, n_envs, n_links, 3),
                links_idx_local=range(n_links),
            )
        nominal_kp = torch.as_tensor(cfg.kp, dtype=torch.float32, device=gs.device)
        nominal_kv = torch.as_tensor(cfg.kv, dtype=torch.float32, device=gs.device)
        if cfg.kp_ratio_range is not None:
            robot.set_dofs_kp(
                nominal_kp * uniform(*cfg.kp_ratio_range, n_envs, n_dofs),
                dofs_idx_local=self.dofs_idx_local,
            )
        if cfg.kv_ratio_range is not None:
            robot.set_dofs_kv(
                nominal_kv * uniform(*cfg.kv_ratio_range, n_envs, n_dofs),
                dofs_idx_local=self.dofs_idx_local,
            )
        if cfg.armature is not None and cfg.armature_ratio_range is not None:
            nominal_armature = torch.as_tensor(cfg.armature, dtype=torch.float32, device=gs.device)
            robot.set_dofs_armature(
                nominal_armature * uniform(*cfg.armature_ratio_range, n_envs, n_dofs),
                dofs_idx_local=self.dofs_idx_local,
            )

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
        state = {
            "dofs_pos"  : robot.get_dofs_position(dofs_idx_local=dofs_idx_local, envs_idx=envs_idx), 
            "dofs_vel"  : robot.get_dofs_velocity(dofs_idx_local=dofs_idx_local, envs_idx=envs_idx),
            "dofs_force": robot.get_dofs_control_force(dofs_idx_local=dofs_idx_local, envs_idx=envs_idx),
            "body_pos"  : robot_base.get_pos(envs_idx=envs_idx), 
            "body_quat" : robot_base.get_quat(envs_idx=envs_idx), 
            "body_lin_vel"  : robot_base.get_vel(envs_idx=envs_idx), 
            "body_ang_vel"  : self.imu.read(envs_idx=envs_idx).ang_vel, 
            "link_contact_forces": robot.get_links_net_contact_force(envs_idx=envs_idx),
        }
        if self._foot_links_idx_local is not None:
            state["foot_pos"] = robot.get_links_pos(
                links_idx_local=self._foot_links_idx_local, envs_idx=envs_idx,
            )
            state["foot_vel"] = robot.get_links_vel(
                links_idx_local=self._foot_links_idx_local, envs_idx=envs_idx,
            )
            all_cf = state["link_contact_forces"]
            state["foot_contact_forces"] = all_cf[:, self._foot_idx_tensor, :]
            body_cf = all_cf[:, self._body_link_mask, :].norm(dim=-1)
            state["body_contact_force"] = body_cf.max(dim=-1).values
        if self._dofs_pos_limit is not None:
            state["dofs_pos_limit"] = self._dofs_pos_limit
            state["dofs_vel_limit"] = self._dofs_vel_limit
            state["dofs_effort_limit"] = self._dofs_effort_limit
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
        self.__gs_randomize()

        if self._foot_links_idx_local is not None:
            self._foot_idx_tensor = torch.tensor(
                self._foot_links_idx_local, dtype=torch.long, device=gs.device)
            n_links = self.robot.n_links
            self._body_link_mask = torch.ones(n_links, dtype=torch.bool, device=gs.device)
            for fi in self._foot_links_idx_local:
                self._body_link_mask[fi] = False
        else:
            self._foot_idx_tensor = None
            self._body_link_mask = None

        pos_lo, pos_hi, vel_lim, eff_lim = [], [], [], []
        for name in self.cfg.joint_names:
            joint = self.robot.get_joint(name)
            lim = joint.dofs_limit[0]
            pos_lo.append(float(lim[0]))
            pos_hi.append(float(lim[1]))
            vel_lim.append(float(joint.dofs_velocity_limit[0]))
            eff_lim.append(float(joint.dofs_force_range[0][1]))
        self._dofs_pos_limit = torch.tensor(
            [pos_lo, pos_hi], dtype=torch.float32, device=gs.device)
        self._dofs_vel_limit = torch.tensor(
            vel_lim, dtype=torch.float32, device=gs.device)
        self._dofs_effort_limit = torch.tensor(
            eff_lim, dtype=torch.float32, device=gs.device)

    def step(self, action: torch.Tensor) -> None:
        self.__gs_step(action=action)

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
