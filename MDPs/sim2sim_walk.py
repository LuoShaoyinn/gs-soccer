import torch
import gymnasium as gym
import numpy as np
from dataclasses import dataclass

import genesis as gs

from .MDP import MDP, MDPConfig
from algorithm.pi_visual_walk.teacher import (
    HISTORY_LEN,
    NUM_GENESIS_JOINTS,
    POLICY_ACTION_SCALE,
    POLICY_DEFAULT_POS,
    POLICY_TO_GENESIS,
    WALK_OBS_DIM,
)

ANG_VEL_SCALE = 0.25
JOINT_VEL_SCALE = 0.05


@dataclass(kw_only=True)
class WalkConfig(MDPConfig):
    vel_cmd: tuple[float, float, float] = (0.5, 0.0, 0.0)
    max_episode_steps: int = 500
    base_height_min: float = 0.2
    init_height: float = 0.50
    termination_grace_steps: int = 50
    target_body_height: float = 0.45
    dof_vel_max: float = 30.0
    dof_acc_max: float = 1500.0
    dof_force_max: float = 100.0
    upright_thresh: float = 0.7
    contact_force_thresh: float = 1.0
    termination_penalty: float = 2.0


class WalkMDP(MDP):
    cfg: WalkConfig

    def build(self):
        pass

    def config(self):
        dev = gs.device
        num_envs = self.scene.n_envs

        self.policy_to_genesis = torch.tensor(POLICY_TO_GENESIS, dtype=torch.long, device=dev)
        default22 = torch.tensor(POLICY_DEFAULT_POS, dtype=torch.float32, device=dev)
        scale22 = torch.tensor(POLICY_ACTION_SCALE, dtype=torch.float32, device=dev)
        self.default_pos = default22[self.policy_to_genesis]
        self.action_scale = scale22[self.policy_to_genesis]

        self._base_pos = torch.tensor([0.0, 0.0, self.cfg.init_height], dtype=torch.float32, device=dev)
        self._base_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=dev)
        self._vel_cmd = torch.tensor(self.cfg.vel_cmd, dtype=torch.float32, device=dev)

        self._buf_ang_vel = torch.zeros((num_envs, HISTORY_LEN, 3), dtype=torch.float32, device=dev)
        self._buf_grav = torch.zeros((num_envs, HISTORY_LEN, 3), dtype=torch.float32, device=dev)
        self._buf_cmd = torch.zeros((num_envs, HISTORY_LEN, 3), dtype=torch.float32, device=dev)
        self._buf_jpos = torch.zeros((num_envs, HISTORY_LEN, NUM_GENESIS_JOINTS), dtype=torch.float32, device=dev)
        self._buf_jvel = torch.zeros((num_envs, HISTORY_LEN, NUM_GENESIS_JOINTS), dtype=torch.float32, device=dev)
        self._buf_act = torch.zeros((num_envs, HISTORY_LEN, NUM_GENESIS_JOINTS), dtype=torch.float32, device=dev)
        self._default_gravity = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=dev)
        self._last_action = torch.zeros((num_envs, NUM_GENESIS_JOINTS), dtype=torch.float32, device=dev)
        self._pending_action = None
        self._prev_dofs_vel = torch.zeros((num_envs, NUM_GENESIS_JOINTS), dtype=torch.float32, device=dev)
        self._episode_step = torch.zeros(num_envs, dtype=torch.long, device=dev)

    def reset(self, envs_idx, robot_reset_fn, field_reset_fn):
        for buf in (self._buf_ang_vel, self._buf_grav, self._buf_cmd, self._buf_jpos, self._buf_jvel, self._buf_act):
            buf[envs_idx] = 0.0
        self._buf_grav[envs_idx] = self._default_gravity
        self._buf_cmd[envs_idx] = self._vel_cmd
        self._last_action[envs_idx] = 0.0
        self._pending_action = None
        self._prev_dofs_vel[envs_idx] = 0.0
        self._episode_step[envs_idx] = 0

        n = envs_idx.shape[0]
        robot_reset_fn(
            joint_pos=self.default_pos.unsqueeze(0).expand(n, -1),
            reset_pos=self._base_pos.unsqueeze(0).expand(n, -1),
            reset_quat=self._base_quat.unsqueeze(0).expand(n, -1),
        )
        field_reset_fn()

    def build_observation(self, envs_idx, **kwargs):
        dofs_pos = kwargs["dofs_pos"]
        dofs_vel = kwargs["dofs_vel"]
        body_quat = kwargs["body_quat"]
        ang_vel = kwargs["body_ang_vel"]
        batch = envs_idx.shape[0]

        for buf in (self._buf_ang_vel, self._buf_grav, self._buf_cmd, self._buf_jpos, self._buf_jvel, self._buf_act):
            buf[envs_idx, :-1] = buf[envs_idx, 1:].clone()

        self._buf_ang_vel[envs_idx, -1] = ang_vel * ANG_VEL_SCALE
        self._buf_grav[envs_idx, -1] = self._quat_to_projected_gravity(body_quat)
        self._buf_cmd[envs_idx, -1] = self._vel_cmd
        self._buf_jpos[envs_idx, -1] = dofs_pos - self.default_pos
        self._buf_jvel[envs_idx, -1] = dofs_vel * JOINT_VEL_SCALE
        self._buf_act[envs_idx, -1] = self._last_action[envs_idx]

        obs = torch.cat(
            [
                self._buf_ang_vel[envs_idx].reshape(batch, -1),
                self._buf_grav[envs_idx].reshape(batch, -1),
                self._buf_cmd[envs_idx].reshape(batch, -1),
                self._buf_jpos[envs_idx].reshape(batch, -1),
                self._buf_jvel[envs_idx].reshape(batch, -1),
                self._buf_act[envs_idx].reshape(batch, -1),
            ],
            dim=-1,
        )

        if self._pending_action is not None:
            self._last_action[envs_idx] = self._pending_action[envs_idx]
            if envs_idx.shape[0] == self._last_action.shape[0]:
                self._pending_action = None
        return obs

    def preprocess_action(self, action):
        self._pending_action = action.clone()
        return self.default_pos + action * self.action_scale

    def build_reward(self, envs_idx, **kwargs):
        body_pos = kwargs["body_pos"]
        body_quat = kwargs["body_quat"]
        body_lin_vel = kwargs["body_lin_vel"]
        body_ang_vel = kwargs["body_ang_vel"]
        dofs_vel = kwargs["dofs_vel"]
        dofs_force = kwargs["dofs_force"]
        proj_grav = self._quat_to_projected_gravity(body_quat)

        w, x, y, z = body_quat[:, 0], body_quat[:, 1], body_quat[:, 2], body_quat[:, 3]
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)
        vx_body = cos_y * body_lin_vel[:, 0] + sin_y * body_lin_vel[:, 1]
        vy_body = -sin_y * body_lin_vel[:, 0] + cos_y * body_lin_vel[:, 1]

        reward = torch.zeros(envs_idx.shape[0], dtype=torch.float32, device=gs.device)
        terms = {}

        term = 0.3 * torch.exp(-((body_pos[:, 2] - self.cfg.target_body_height) ** 2) / 0.04**2)
        reward += term
        terms["body_height"] = term

        term = -0.02 * (proj_grav[:, 0] ** 2 + proj_grav[:, 1] ** 2)
        reward += term
        terms["upright"] = term

        lin_err = (self._vel_cmd[0] - vx_body) ** 2 + (self._vel_cmd[1] - vy_body) ** 2
        term = 0.4 * torch.exp(-lin_err / 0.05)
        reward += term
        terms["track_lin"] = term

        ang_err = (self._vel_cmd[2] - body_ang_vel[:, 2]) ** 2
        term = 0.2 * torch.exp(-ang_err / 0.05) * (abs(self.cfg.vel_cmd[2]) > 0.1)
        reward += term
        terms["track_ang"] = term

        term = -0.0002 * (body_ang_vel[:, 0] ** 2 + body_ang_vel[:, 1] ** 2)
        reward += term
        terms["ang_vel_xy"] = term

        term = -1e-6 * (dofs_vel ** 2).sum(-1)
        reward += term
        terms["dof_vel"] = term

        term = -1e-6 * (dofs_force ** 2).sum(-1)
        reward += term
        terms["dof_force"] = term

        term = -1e-5 * ((dofs_force * dofs_vel).abs()).sum(-1).clamp(max=50.0)
        reward += term
        terms["energy"] = term

        reward = reward.clamp(-0.01, 0.01)
        term_pen = self._check_term_conditions(envs_idx, **kwargs).float() * -self.cfg.termination_penalty
        reward += term_pen
        terms["term_pen"] = term_pen

        self._reward_terms = {f"r_{name}": value.detach() for name, value in terms.items()}
        return reward.unsqueeze(1)

    def build_terminated(self, envs_idx, **kwargs):
        term = self._check_term_conditions(envs_idx, **kwargs)
        self._prev_dofs_vel[envs_idx] = kwargs["dofs_vel"].clone()
        return term.unsqueeze(1)

    def build_truncated(self, envs_idx, **kwargs):
        self._episode_step[envs_idx] += 1
        return (self._episode_step[envs_idx] >= self.cfg.max_episode_steps).unsqueeze(1)

    def build_info(self, envs_idx, **kwargs):
        info = {}
        if hasattr(self, "_reward_terms"):
            for key, value in self._reward_terms.items():
                info[key] = value[envs_idx]
        info["body_pos_z"] = kwargs["body_pos"][:, 2].detach()
        info["walk_vx_body"] = self._body_vx(kwargs["body_lin_vel"], kwargs["body_quat"]).detach()
        if hasattr(self, "_term_info"):
            for key, value in self._term_info.items():
                info[key] = value[envs_idx]
        return info

    def _check_term_conditions(self, envs_idx, **kwargs):
        dt = 1.0 / 50.0
        body_pos = kwargs["body_pos"]
        body_quat = kwargs["body_quat"]
        dofs_vel = kwargs["dofs_vel"]
        dofs_force = kwargs["dofs_force"]

        term_height = body_pos[:, 2] < self.cfg.base_height_min
        term = term_height
        body_cf = self._body_contact_force(kwargs)
        term_contact = torch.zeros_like(term)
        if body_cf is not None:
            term_contact = body_cf > self.cfg.contact_force_thresh
            term = term | term_contact

        proj_grav = self._quat_to_projected_gravity(body_quat)
        tilt = (proj_grav[:, 0] ** 2 + proj_grav[:, 1] ** 2).sqrt()
        dof_acc = ((dofs_vel - self._prev_dofs_vel[envs_idx]) / dt).abs()
        term_dof_vel = (dofs_vel.abs() > self.cfg.dof_vel_max).any(-1)
        term_dof_acc = (dof_acc > self.cfg.dof_acc_max).any(-1)
        term_dof_force = (dofs_force.abs() > self.cfg.dof_force_max).any(-1)
        term_tilt = tilt > self.cfg.upright_thresh
        term = term | (
            term_dof_vel
            | term_dof_acc
            | term_dof_force
            | term_tilt
        )
        grace = self._episode_step[envs_idx] < self.cfg.termination_grace_steps
        term = term & (~grace)
        self._term_info = {
            "term/height": term_height.float().detach(),
            "term/contact": term_contact.float().detach(),
            "term/dof_vel": term_dof_vel.float().detach(),
            "term/dof_acc": term_dof_acc.float().detach(),
            "term/dof_force": term_dof_force.float().detach(),
            "term/tilt": term_tilt.float().detach(),
            "term/grace": grace.float().detach(),
            "term/any": term.float().detach(),
        }
        return term

    @staticmethod
    def _body_contact_force(kwargs):
        link_cf = kwargs.get("link_contact_forces")
        foot_idx = kwargs.get("foot_idx_local")
        if link_cf is None or foot_idx is None:
            return None
        mask = torch.ones(link_cf.shape[1], dtype=torch.bool, device=link_cf.device)
        mask[foot_idx] = False
        return link_cf[:, mask, :].norm(dim=-1).max(dim=-1).values

    @staticmethod
    def _quat_to_projected_gravity(quat):
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        return torch.stack(
            [
                -2.0 * (x * z - w * y),
                -2.0 * (y * z + w * x),
                -(1.0 - 2.0 * (x * x + y * y)),
            ],
            dim=-1,
        )

    @staticmethod
    def _body_vx(lin_vel, quat):
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return torch.cos(yaw) * lin_vel[:, 0] + torch.sin(yaw) * lin_vel[:, 1]

    @property
    def observation_space(self):
        return gym.spaces.Box(-float("inf"), float("inf"), (WALK_OBS_DIM,), dtype=np.float32)

    @property
    def action_space(self):
        return gym.spaces.Box(-float("inf"), float("inf"), (NUM_GENESIS_JOINTS,), dtype=np.float32)
