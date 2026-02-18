import numpy as np
import torch
import genesis as gs
import gymnasium as gym
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class SingleWalkerEnvConfig:
    # Robot
    robot_URDF: str
    joint_names: list[str]
    # External callback functions
    reward_fn: Callable
    terminated_fn: Callable
    truncated_fn: Callable
    gen_cmd_fn: Callable
    # Control
    action_scale: np.ndarray = field(default_factory=lambda: np.ones(12, dtype=np.float32))
    default_joint_angles: np.ndarray = field(default_factory=lambda: np.zeros(12, dtype=np.float32))
    kp: np.ndarray = field(default_factory=lambda: np.array([30.0] * 12, dtype=np.float32))
    kv: np.ndarray = field(default_factory=lambda: np.array([3.0] * 12, dtype=np.float32))
    force_range: np.ndarray = field(
        default_factory=lambda: np.array([[-100.0] * 12, [100.0] * 12], dtype=np.float32)
    )
    # Simulation
    device: str = "cuda"
    num_envs: int = 1
    field_range: float = 2.0
    sim_dt: float = 0.005
    control_decimation: int = 4
    command_resample_time_s: float = 10.0
    episode_length_s: float = 20.0
    gait_period_s: float = 0.8
    show_viewer: bool = False
    robot_initial_pos: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    robot_initial_quat: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    )
    # Observation/action clipping
    observation_clip: float = 100.0
    action_clip: float = 100.0
    command_zero_threshold: float = 0.2
    soft_dof_pos_limit: float = 0.9
    # Contact groups
    foot_link_names: list[str] = field(default_factory=lambda: ["Lfoot", "Rfoot"])
    penalized_contact_link_names: list[str] = field(
        default_factory=lambda: ["Rhip", "Lhip", "Rleg1", "Lleg1"]
    )
    termination_contact_link_names: list[str] = field(default_factory=lambda: ["base"])


class SingleWalkerEnv(gym.vector.VectorEnv):
    def __init__(self, cfg: SingleWalkerEnvConfig):
        assert isinstance(cfg.robot_URDF, str), f"URDF filename should be a str, got {cfg.robot_URDF}"
        assert isinstance(cfg.joint_names, list), f"joint_names should be a list[str], got {cfg.joint_names}"
        self.cfg = cfg
        self.num_envs = cfg.num_envs
        self.num_agents = 1
        self.is_vector_env = True

        self.policy_dt = cfg.sim_dt * cfg.control_decimation
        self.max_episode_length = max(1, int(np.ceil(cfg.episode_length_s / self.policy_dt)))
        self.command_resample_interval = max(1, int(np.ceil(cfg.command_resample_time_s / self.policy_dt)))

        self._gs_build()
        self._gs_config()
        self._build_spacebox()
        super().__init__()

        n_joints = len(self.dofs_idx_local)
        self.command = torch.zeros((self.num_envs, 3), device=self.cfg.device)
        self.command_scale = torch.tensor([2.0, 2.0, 0.25], device=self.cfg.device)
        self.last_actions = torch.zeros((self.num_envs, n_joints), device=self.cfg.device)
        self.last_dofs_vel = torch.zeros((self.num_envs, n_joints), device=self.cfg.device)
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.cfg.device)
        self.gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self.cfg.device).repeat(self.num_envs, 1)

    @torch.no_grad()
    def _to_env_ids(self, envs_idx: torch.Tensor | np.ndarray | list[int] | None) -> torch.Tensor:
        if envs_idx is None:
            return torch.arange(self.num_envs, device=self.cfg.device, dtype=torch.long)
        if isinstance(envs_idx, torch.Tensor):
            if envs_idx.dtype == torch.bool:
                return torch.nonzero(envs_idx, as_tuple=False).flatten().to(dtype=torch.long)
            return envs_idx.to(device=self.cfg.device, dtype=torch.long).flatten()
        envs_idx_t = torch.as_tensor(envs_idx, device=self.cfg.device)
        if envs_idx_t.dtype == torch.bool:
            return torch.nonzero(envs_idx_t, as_tuple=False).flatten().to(dtype=torch.long)
        return envs_idx_t.to(dtype=torch.long).flatten()

    @torch.no_grad()
    def _quat_rotate_inverse(self, quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        # quat: [w, x, y, z], vec in world frame.
        w = quat[:, 0:1]
        q_vec = quat[:, 1:]
        a = vec * (2.0 * w * w - 1.0)
        b = 2.0 * w * torch.cross(q_vec, vec, dim=1)
        c = 2.0 * q_vec * torch.sum(q_vec * vec, dim=1, keepdim=True)
        return a - b + c

    @torch.no_grad()
    def _quat_to_roll_pitch(self, quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        w = quat[:, 0]
        x = quat[:, 1]
        y = quat[:, 2]
        z = quat[:, 3]
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = torch.atan2(sinr_cosp, cosr_cosp)
        sinp = torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0)
        pitch = torch.asin(sinp)
        return roll, pitch

    @torch.no_grad()
    def _compute_phase(self) -> torch.Tensor:
        t = self.episode_length_buf.float() * self.policy_dt
        return (t % self.cfg.gait_period_s) / self.cfg.gait_period_s

    @torch.no_grad()
    def _resolve_link_indices(self, link_names: list[str]) -> torch.Tensor:
        idx_local = [self.robot.get_link(name).idx_local for name in link_names]
        return torch.as_tensor(idx_local, dtype=torch.long, device=self.cfg.device)

    @torch.no_grad()
    def _gs_build(self):
        self.scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(0, -3.5, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=30,
                res=(960, 640),
                max_FPS=60,
            ),
            sim_options=gs.options.SimOptions(dt=self.cfg.sim_dt, substeps=1),
            show_viewer=self.cfg.show_viewer,
        )
        self.scene.add_entity(gs.morphs.Plane())
        self._gs_build_robot()
        self.scene.build(
            n_envs=self.cfg.num_envs,
            env_spacing=(self.cfg.field_range, self.cfg.field_range),
        )

    @torch.no_grad()
    def _gs_build_robot(self):
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file=self.cfg.robot_URDF,
                pos=self.cfg.robot_initial_pos,
                quat=self.cfg.robot_initial_quat,
                decimate_aggressiveness=5,
                requires_jac_and_IK=False,
            ),
            vis_mode="collision",
        )
        self.robot_base = self.robot.get_link("base")

        dofs_idx = [self.robot.get_joint(name).dofs_idx_local[0] for name in self.cfg.joint_names]
        self.dofs_idx_local = torch.as_tensor(dofs_idx, dtype=torch.long, device=self.cfg.device)

        self.action_scale = torch.as_tensor(self.cfg.action_scale, device=self.cfg.device, dtype=torch.float32)
        self.default_dofs_pos = torch.as_tensor(
            self.cfg.default_joint_angles, device=self.cfg.device, dtype=torch.float32
        )
        if self.default_dofs_pos.ndim != 1 or self.default_dofs_pos.numel() != len(self.dofs_idx_local):
            raise ValueError(
                f"default_joint_angles must have {len(self.dofs_idx_local)} elements, "
                f"got shape {tuple(self.default_dofs_pos.shape)}"
            )

        self.feet_idx_local = self._resolve_link_indices(self.cfg.foot_link_names)
        self.penalized_idx_local = self._resolve_link_indices(self.cfg.penalized_contact_link_names)
        self.termination_idx_local = self._resolve_link_indices(self.cfg.termination_contact_link_names)

    @torch.no_grad()
    def _gs_config(self):
        dof_lower, dof_upper = self.robot.get_dofs_limit(dofs_idx_local=self.dofs_idx_local)
        self.dof_pos_limits = torch.stack((dof_lower, dof_upper), dim=0)
        dof_mid = 0.5 * (dof_lower + dof_upper)
        dof_half_range = 0.5 * (dof_upper - dof_lower) * self.cfg.soft_dof_pos_limit
        self.soft_dof_pos_limits = torch.stack((dof_mid - dof_half_range, dof_mid + dof_half_range), dim=0)

        self.robot.set_dofs_kp(kp=self.cfg.kp, dofs_idx_local=self.dofs_idx_local)
        self.robot.set_dofs_kv(kv=self.cfg.kv, dofs_idx_local=self.dofs_idx_local)
        self.robot.set_dofs_force_range(
            lower=self.cfg.force_range[0],
            upper=self.cfg.force_range[1],
            dofs_idx_local=self.dofs_idx_local,
        )

    @torch.no_grad()
    def _build_spacebox(self) -> None:
        n_joints = len(self.dofs_idx_local)
        # Unitree-style obs: base_ang_vel(3), projected_gravity(3), cmd(3),
        # dof_pos_rel(12), dof_vel(12), actions(12), phase sin/cos(2) => 47.
        obs_dim = 3 + 3 + 3 + n_joints + n_joints + n_joints + 2
        obs_bound = self.cfg.observation_clip * np.ones(obs_dim, dtype=np.float32)
        act_bound = self.cfg.action_clip * np.ones(n_joints, dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-obs_bound, high=obs_bound, dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-act_bound, high=act_bound, dtype=np.float32)

    @torch.no_grad()
    def _resample_commands(self, envs_idx: torch.Tensor | np.ndarray | list[int] | None):
        env_ids = self._to_env_ids(envs_idx)
        if env_ids.numel() == 0:
            return
        cmd = self.cfg.gen_cmd_fn(num_envs=int(env_ids.numel()))
        cmd_t = torch.as_tensor(cmd, device=self.cfg.device, dtype=torch.float32)
        if cmd_t.ndim == 1:
            cmd_t = cmd_t.unsqueeze(0)
        if cmd_t.shape[0] == 1 and env_ids.numel() > 1:
            cmd_t = cmd_t.repeat(env_ids.numel(), 1)
        self.command[env_ids] = cmd_t
        keep_xy = (torch.norm(self.command[env_ids, :2], dim=1) > self.cfg.command_zero_threshold).float()
        self.command[env_ids, :2] *= keep_xy.unsqueeze(1)

    @torch.no_grad()
    def _collect_state(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        dofs_pos = self.robot.get_dofs_position(dofs_idx_local=self.dofs_idx_local)
        dofs_vel = self.robot.get_dofs_velocity(dofs_idx_local=self.dofs_idx_local)
        body_pos = self.robot_base.get_pos()
        body_quat = self.robot_base.get_quat()
        body_vel_world = self.robot_base.get_vel()
        body_ang_world = self.robot_base.get_ang()

        base_lin_vel = self._quat_rotate_inverse(body_quat, body_vel_world)
        base_ang_vel = self._quat_rotate_inverse(body_quat, body_ang_world)
        projected_gravity = self._quat_rotate_inverse(body_quat, self.gravity_vec)
        roll, pitch = self._quat_to_roll_pitch(body_quat)

        link_contact_force = self.robot.get_links_net_contact_force()
        net_contact_force = torch.linalg.norm(link_contact_force, dim=2)

        feet_pos = self.robot.get_links_pos(links_idx_local=self.feet_idx_local)
        feet_vel = self.robot.get_links_vel(links_idx_local=self.feet_idx_local)
        feet_contact_force = link_contact_force[:, self.feet_idx_local, :]
        penalized_contact_force = link_contact_force[:, self.penalized_idx_local, :]
        termination_contact_force = link_contact_force[:, self.termination_idx_local, :]

        return {
            "action": action,
            "last_action": self.last_actions,
            "dofs_pos": dofs_pos,
            "dofs_vel": dofs_vel,
            "last_dofs_vel": self.last_dofs_vel,
            "dof_pos_limits": self.soft_dof_pos_limits,
            "body_pos": body_pos,
            "body_quat": body_quat,
            "body_vel": body_vel_world,
            "base_lin_vel": base_lin_vel,
            "base_ang_vel": base_ang_vel,
            "projected_gravity": projected_gravity,
            "roll": roll,
            "pitch": pitch,
            "cmd_vel": self.command,
            "phase": self._compute_phase(),
            "feet_pos": feet_pos,
            "feet_vel": feet_vel,
            "feet_contact_force": feet_contact_force,
            "penalized_contact_force": penalized_contact_force,
            "termination_contact_force": termination_contact_force,
            "net_contact_force": net_contact_force,
            "episode_length_buf": self.episode_length_buf,
            "max_episode_length": torch.tensor(self.max_episode_length, device=self.cfg.device),
        }

    @torch.no_grad()
    def _gs_step(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        clipped_action = torch.clamp(action, -self.cfg.action_clip, self.cfg.action_clip)
        dof_target = self.default_dofs_pos.unsqueeze(0) + clipped_action * self.action_scale
        dof_target = torch.clamp(dof_target, self.dof_pos_limits[0], self.dof_pos_limits[1])

        for _ in range(self.cfg.control_decimation):
            self.robot.control_dofs_position(dof_target, dofs_idx_local=self.dofs_idx_local)
            self.scene.step()
        return self._collect_state(clipped_action)

    @torch.no_grad()
    def _get_observation(self, base_ang_vel, projected_gravity, cmd_vel, dofs_pos, dofs_vel, action, phase, **kwargs):
        sin_phase = torch.sin(2.0 * torch.pi * phase).unsqueeze(1)
        cos_phase = torch.cos(2.0 * torch.pi * phase).unsqueeze(1)
        obs = torch.cat(
            (
                base_ang_vel * 0.25,
                projected_gravity,
                cmd_vel * self.command_scale,
                (dofs_pos - self.default_dofs_pos.unsqueeze(0)),
                dofs_vel * 0.05,
                action,
                sin_phase,
                cos_phase,
            ),
            dim=1,
        )
        return torch.clamp(obs, -self.cfg.observation_clip, self.cfg.observation_clip)

    @torch.no_grad()
    def step(self, action: torch.Tensor):  # type: ignore[override]
        kwargs = self._gs_step(action)
        self.episode_length_buf += 1

        resample_mask = self.episode_length_buf % self.command_resample_interval == 0
        self._resample_commands(resample_mask)
        kwargs["cmd_vel"] = self.command
        kwargs["phase"] = self._compute_phase()
        kwargs["episode_length_buf"] = self.episode_length_buf

        observation = self._get_observation(**kwargs)
        reward, info_rew = self.cfg.reward_fn(**kwargs)
        terminated, info_tre = self.cfg.terminated_fn(**kwargs)
        truncated, info_tru = self.cfg.truncated_fn(**kwargs)

        done = torch.logical_or(terminated, truncated)
        self.last_actions[:] = kwargs["action"]
        self.last_dofs_vel[:] = kwargs["dofs_vel"]
        self.reset_batch(done)

        return (
            observation,
            reward.unsqueeze(1),
            terminated.unsqueeze(1),
            truncated.unsqueeze(1),
            {"extra": {**info_rew, **info_tre, **info_tru}},
        )

    @torch.no_grad()
    def reset_batch(self, envs_idx: torch.Tensor | np.ndarray | list[int]):
        env_ids = self._to_env_ids(envs_idx)
        if env_ids.numel() == 0:
            return

        n = int(env_ids.numel())
        dof_pos = self.default_dofs_pos.unsqueeze(0).repeat(n, 1)
        self.robot.set_dofs_position(dof_pos, dofs_idx_local=self.dofs_idx_local, envs_idx=env_ids)
        self.robot.set_pos(pos=self.cfg.robot_initial_pos, envs_idx=env_ids)
        self.robot.set_quat(quat=self.cfg.robot_initial_quat, envs_idx=env_ids)
        self._resample_commands(env_ids)

        self.last_actions[env_ids] = 0.0
        self.last_dofs_vel[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0

    @torch.no_grad()
    def reset(self, seed=None, options=None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:  # type: ignore[override]
        del seed, options
        all_envs = torch.arange(self.num_envs, device=self.cfg.device, dtype=torch.long)
        self.reset_batch(all_envs)
        zero_action = torch.zeros((self.num_envs, len(self.dofs_idx_local)), device=self.cfg.device)
        kwargs = self._collect_state(zero_action)
        kwargs["cmd_vel"] = self.command
        kwargs["phase"] = self._compute_phase()
        observation = self._get_observation(**kwargs)
        return observation, {"extra": {}}

    @torch.no_grad()
    def render(self):
        pass
