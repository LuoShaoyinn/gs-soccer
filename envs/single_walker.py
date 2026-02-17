# single_walker.py
#   Build up a single walker env
#

import torch
import numpy as np
import genesis as gs
import gymnasium as gym
import random
from dataclasses import dataclass, field
from typing import Optional, Callable


@dataclass
class SingleWalkerEnvConfig():
    # Robot
    robot_URDF: str
    joint_names: list[str]
    # Function
    reward_fn:      Callable
    terminated_fn:  Callable
    truncated_fn:   Callable
    gen_cmd_fn:     Callable
    action_scale: np.ndarray = field(default_factory=lambda: np.ones(12))
    # Simulation setting
    device = "cuda"
    num_envs: int = 1
    field_range: float = 1.0
    rl_dt: float = 0.02
    step_simulation_cnt: int = 10
    step_simulation_rnd_cnt: int = 2
    show_viewer: bool = False
    robot_initial_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    robot_initial_quat: np.ndarray = field(default_factory=
                                           lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    # Control parameters
    kp: np.ndarray = field(default_factory=\
            lambda: np.array([30.0] * 12, dtype=np.float32)) 
    kv: np.ndarray = field(default_factory=\
            lambda: np.array([3.0] * 12, dtype=np.float32))
    # Limits
    velocity_range: np.ndarray = field(default_factory=\
            lambda: np.array([[-10.0] * 12, [10.0] * 12], dtype=np.float32))
    force_range: np.ndarray = field(default_factory=\
            lambda: np.array([[-100.0] * 12, [100.0] * 12], dtype=np.float32))

class SingleWalkerEnv(gym.vector.VectorEnv):
    def __init__(self, cfg: SingleWalkerEnvConfig):
        super().__init__()
        assert type(cfg.robot_URDF) == str, \
            f"URDF filename should be a str, got {cfg.robot_URDF}"
        assert type(cfg.joint_names) == list, \
            f"joint_names should be a list[str], got{cfg.joint_names}"
        self.cfg = cfg
        self.num_envs = cfg.num_envs
        self.num_agents = 1
        self.is_vector_env = True
        self.command = torch.zeros((self.num_envs, 3))

        self._gs_build() 
        self._gs_config()
        self._build_spacebox()
    
    @torch.no_grad()
    def _gs_build(self):
        self.scene = gs.Scene(
            viewer_options = gs.options.ViewerOptions(
                camera_pos    = (0, -3.5, 2.5),
                camera_lookat = (0.0, 0.0, 0.5),
                camera_fov    = 30,
                res           = (960, 640),
                max_FPS       = 60,
            ),
            sim_options = gs.options.SimOptions(
                dt = self.cfg.rl_dt,
                substeps = 5
            ),
            show_viewer = self.cfg.show_viewer,
        )
        self.plane = self.scene.add_entity(gs.morphs.Plane())
        self._gs_build_robot()
        self.scene.build(n_envs=self.cfg.num_envs, \
                env_spacing=(self.cfg.field_range, self.cfg.field_range))

    @torch.no_grad()
    def _gs_build_robot(self):
        self.robot = self.scene.add_entity(gs.morphs.URDF(
            file = self.cfg.robot_URDF,
            pos  = self.cfg.robot_initial_pos, 
            quat = self.cfg.robot_initial_quat,
            decimate_aggressiveness = 5,
            requires_jac_and_IK = False, 
        ), vis_mode='collision')
        self.robot_base = self.robot.get_link('base')
        self.dofs_idx_local = [self.robot.get_joint(name).dofs_idx_local[0]
                         for name in self.cfg.joint_names]
        self.dofs_idx_local = torch.as_tensor(self.dofs_idx_local)\
                                 .to(self.cfg.device)
        self.action_scale = torch.from_numpy(self.cfg.action_scale)\
                                 .to(self.cfg.device)
        self.imu = self.scene.add_sensor(
            gs.sensors.IMU(
                entity_idx=self.robot.idx, # type: ignore
                link_idx_local=self.robot_base.idx_local, # type: ignore
                pos_offset=(0.0, 0.0, 0.0), # type: ignore
            )
        )

    @torch.no_grad()
    def _gs_config(self):
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
    def _build_spacebox(self) -> None:
        cfg          = self.cfg
        n_joints     = len(self.dofs_idx_local)
        pos_limits   = self.robot.get_dofs_limit(dofs_idx_local=self.dofs_idx_local)
        # [qpos, qvel, lin_acc, ang_vel, project_grav, prev_action, cmd]
        # stack 6 frames.
        shapes = n_joints + n_joints + 3 + 3 + 3 + n_joints + 3
        std_range = torch.tensor([[-1.0] * 3, [1.0] * 3], device=cfg.device)
        self._obs_range = torch.zeros((3, shapes), device=cfg.device)
        self._obs_stack = torch.zeros((cfg.num_envs, 6, shapes), device=cfg.device)
        for _ in range(2):
            self._obs_range[_] = torch.cat((
                pos_limits[_],          # qpos
                torch.from_numpy(cfg.velocity_range[_]).to(cfg.device), # qvel
                std_range[_] * 10.0,    # lin_acc
                std_range[_] * 20.0,    # ang_vel
                std_range[_],           # project_grav
                pos_limits[_],          # prev_action
                std_range[_]))          # action
        self._obs_range[2] = self._obs_range[1] - self._obs_range[0]
        self.observation_space = gym.spaces.Box(
                low=self._obs_range[0].repeat(6).cpu().numpy(), 
                high=self._obs_range[1].repeat(6).cpu().numpy(), 
                shape=(shapes * 6, ), 
                dtype=np.float32)
        self.action_space = gym.spaces.Box(
                low=pos_limits[0].cpu().numpy(),
                high=pos_limits[1].cpu().numpy(),
                shape=(n_joints, ),
                dtype=np.float32)

    @torch.no_grad()
    #@torch.compiler.disable()
    def _gs_step(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        robot = self.robot
        robot_base = self.robot_base
        dofs_idx_local = self.dofs_idx_local
        dofs_pos = robot.get_dofs_position(dofs_idx_local=dofs_idx_local)
        robot.control_dofs_position(
                dofs_pos + action * self.action_scale,
                dofs_idx_local=self.dofs_idx_local)
        self.scene.step()
        return {
            "imu_data"  : self.imu.read(), 
            "dofs_pos"  : dofs_pos, 
            "dofs_vel"  : robot.get_dofs_velocity(dofs_idx_local=dofs_idx_local), 
            # "dofs_force": robot.get_dofs_force(dofs_idx_local=dofs_idx_local), 
            "body_quat" : robot_base.get_quat(), 
            "body_vel"  : robot_base.get_vel(), 
            "body_pos"  : robot_base.get_pos(), 
            "cmd_vel"   : self.command, 
            "net_contact_force" : torch.sum( torch.square(
                robot.get_links_net_contact_force()[:, :-2, :]), dim=2)
        }

    #@torch.compile()
    def _get_projected_gravity(self, quat: torch.Tensor) -> torch.Tensor: 
        w = quat[:, 0:1]
        q_vec = quat[:, 1:]
        v = torch.tensor([0.0, 0.0, -1.0], device=quat.device)
        a = torch.cross(q_vec, v.expand_as(q_vec), dim=-1) + w * v
        res = v + 2.0 * torch.cross(q_vec, a, dim=-1)
        return res
 
    @torch.no_grad()
    # #@torch.compile()
    def step(self, action : torch.Tensor): # type: ignore
        kwargs = self._gs_step(action)
        observation = self._get_observation(prev_action = action, **kwargs)
        reward, info_rew = self.cfg.reward_fn(**kwargs)
        terminated, info_tre = self.cfg.terminated_fn(**kwargs)
        truncated, info_tru = self.cfg.truncated_fn(**kwargs)
        self.reset_batch(envs_idx = torch.logical_or(terminated, truncated))
        return (observation, 
                reward.unsqueeze(1), 
                terminated.unsqueeze(1), 
                truncated.unsqueeze(1), 
                {"extra": {**info_rew, **info_tre, **info_tru}})
    
    @torch.no_grad()
    #@torch.compiler.disable()
    def reset_batch(self, envs_idx: torch.Tensor):
        if not envs_idx.any():
            return
        zeros = torch.zeros((len(self.dofs_idx_local)), 
                             device=self.cfg.device)
        self.robot.set_dofs_position(zeros, 
                                     dofs_idx_local=self.dofs_idx_local,
                                     envs_idx=envs_idx)
        self.robot.set_pos(pos=self.cfg.robot_initial_pos, envs_idx=envs_idx)
        self.robot.set_quat(quat=self.cfg.robot_initial_quat, envs_idx=envs_idx)
        self.command[envs_idx] = self.cfg.gen_cmd_fn()

    @torch.no_grad()
    #@torch.compile
    def _get_observation(self, dofs_pos, dofs_vel, \
            imu_data, body_quat, prev_action, cmd_vel, **kwargs):
        projected_gravity = self._get_projected_gravity(body_quat)
        new_observation = torch.cat((dofs_pos,      # qpos
                                     dofs_vel,      # qvel 
                                     imu_data[0],   # lin_acc
                                     imu_data[1],   # ang_vel
                                     projected_gravity, 
                                     prev_action, 
                                     cmd_vel), dim=1)
        new_observation = torch.clamp(new_observation, \
                self._obs_range[0], self._obs_range[1])
        new_observation = 2.0 * (new_observation - self._obs_range[0])\
                              / self._obs_range[2] - 1.0
        self._obs_stack = torch.roll(self._obs_stack, shifts=-1, dims=1)
        self._obs_stack[:, -1, :] = new_observation
        return self._obs_stack.view(self.cfg.num_envs, -1)
    
    @torch.no_grad()
    #@torch.compile()
    def reset(self, seed=None, options=None
            ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        self.reset_batch(envs_idx = torch.arange(self.num_envs, \
                                                device=self.cfg.device))
        res = self.step(torch.zeros((self.num_envs, len(self.cfg.joint_names)),\
                                       device = self.cfg.device))
        return res[0], res[-1] # observation, info

    @torch.no_grad()
    def render(self):
        pass

