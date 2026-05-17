# walk_model.py
#   Unitree-like walking model (obs/action/reward style)

import math
import torch
import numpy as np
import genesis as gs
import gymnasium as gym
from dataclasses import dataclass
from .model import ModelConfig, Model


@dataclass(kw_only=True)
class MOS9WalkModelConfig(ModelConfig):
    n_dofs: int
    target_q_offset: np.ndarray
    step_dt: float = 0.02
    history_frames: int = 5
    action_scale: float = 0.25
    obs_clip: tuple = (-18.0, 18.0)

    # Unitree-like reward scales
    reward_tracking_sigma: float = 0.25
    reward_tracking_lin_weight: float = 1.0
    reward_tracking_ang_weight: float = 0.5
    reward_alive_weight: float = 0.15
    reward_lin_vel_z_weight: float = -2.0
    reward_ang_vel_xy_weight: float = -0.05
    reward_joint_vel_weight: float = -0.001
    reward_joint_acc_weight: float = -2.5e-7
    reward_action_rate_weight: float = -0.05
    reward_energy_weight: float = -2e-5
    reward_flat_orientation_weight: float = -5.0
    reward_base_height_weight: float = -10.0
    reward_dof_pos_limits_weight: float = -5.0
    reward_hip_pos_weight: float = -1.0
    reward_undesired_contact_weight: float = -1.0

    base_height_target: float = 0.5
    dof_pos_soft_limit: float = 0.9
    termination_body_height: float = 0.2
    termination_link_height: float = 0.05
    termination_tilt_limit_rad: float = 0.8
    max_episode_length: int = 1000
    only_positive_rewards: bool = True


class MOS9WalkModel(Model):
    cfg: MOS9WalkModelConfig

    @staticmethod
    def _project_gravity(quat: torch.Tensor) -> torch.Tensor:
        n = quat.shape[0]
        gravity = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float, device=gs.device).broadcast_to((n, 3))
        q_w = quat[:, 0:1]
        q_vec = quat[:, 1:4]
        return (
            gravity * (2.0 * q_w * q_w - 1.0)
            - torch.cross(q_vec, gravity, dim=1) * (2.0 * q_w)
            + q_vec * (2.0 * (q_vec * gravity).sum(dim=1, keepdim=True))
        )

    def config(self):
        self.target_q_offset = torch.from_numpy(self.cfg.target_q_offset).to(gs.device)
        self.dim_observations = 9 + 3 * self.cfg.n_dofs
        self.time_steps = torch.zeros((self.scene.n_envs,), dtype=torch.float, device=gs.device)
        self.episode_steps = torch.zeros((self.scene.n_envs,), dtype=torch.long, device=gs.device)
        self.last_obs = torch.zeros(
            (self.scene.n_envs, self.cfg.history_frames, self.dim_observations),
            dtype=torch.float,
            device=gs.device,
        )
        self.last_action = torch.zeros((self.scene.n_envs, self.cfg.n_dofs), dtype=torch.float, device=gs.device)
        self.prev_action = torch.zeros((self.scene.n_envs, self.cfg.n_dofs), dtype=torch.float, device=gs.device)
        self.last_dofs_vel = torch.zeros((self.scene.n_envs, self.cfg.n_dofs), dtype=torch.float, device=gs.device)
        self.target_q = self.target_q_offset.unsqueeze(0).repeat(self.scene.n_envs, 1).clone()
        # compatibility with walker infra
        self.ewma_action = torch.zeros((self.scene.n_envs, self.cfg.n_dofs), dtype=torch.float, device=gs.device)
        self._cache_proj_gravity = torch.zeros((self.scene.n_envs, 3), dtype=torch.float, device=gs.device)
        self._log: dict[str, torch.Tensor] = {}

    def reset(self, envs_idx: torch.Tensor):
        self.time_steps[envs_idx] = 0.0
        self.episode_steps[envs_idx] = 0
        self.last_obs[envs_idx] = 0.0
        self.last_action[envs_idx] = 0.0
        self.prev_action[envs_idx] = 0.0
        self.last_dofs_vel[envs_idx] = 0.0
        self.ewma_action[envs_idx] = 0.0
        self.target_q[envs_idx] = self.target_q_offset
        self._cache_proj_gravity[envs_idx] = 0.0

    def preprocess_action(self, action: torch.Tensor):
        self.prev_action.copy_(self.last_action)
        self.last_action = action
        self.time_steps += 1.0
        self.episode_steps += 1
        # Unitree-like action mapping: default offset + scaled action
        self.ewma_action = self.cfg.action_scale * action
        self.target_q = self.target_q_offset + self.ewma_action
        return self.target_q

    @property
    def observation_space(self) -> gym.spaces.Box:
        return gym.spaces.Box(
            low=self.cfg.obs_clip[0],
            high=self.cfg.obs_clip[1],
            shape=(self.dim_observations * self.cfg.history_frames,),
            dtype=np.float32,
        )

    @property
    def action_space(self) -> gym.spaces.Box:
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(self.cfg.n_dofs,), dtype=np.float32)

    def build_observation(self, envs_idx, body_ang_vel, body_quat, cmd_vel, dofs_pos, dofs_vel, **kwargs):
        n = envs_idx.shape[0]
        proj_gravity = self._project_gravity(body_quat)
        self._cache_proj_gravity[envs_idx] = proj_gravity
        obs_single = torch.cat(
            (
                body_ang_vel * 0.2,
                proj_gravity,
                cmd_vel.reshape((n, 3)),
                (dofs_pos - self.target_q_offset),
                dofs_vel * 0.05,
                self.last_action[envs_idx],
            ),
            dim=-1,
        )
        obs_single = torch.clamp(obs_single, self.cfg.obs_clip[0], self.cfg.obs_clip[1])
        self.last_obs[envs_idx] = torch.roll(self.last_obs[envs_idx], shifts=-1, dims=1)
        self.last_obs[envs_idx, -1, :] = obs_single
        return self.last_obs[envs_idx].reshape(n, -1)

    def build_reward(
        self,
        envs_idx,
        body_lin_vel,
        body_ang_vel,
        body_pos,
        body_quat,
        cmd_vel,
        dofs_pos,
        dofs_vel,
        dofs_torque,
        non_foot_heights,
        **kwargs,
    ):
        proj_gravity = self._cache_proj_gravity[envs_idx]
        if torch.count_nonzero(proj_gravity).item() == 0:
            proj_gravity = self._project_gravity(body_quat)

        lin_err_sq = torch.sum((cmd_vel[:, :2] - body_lin_vel[:, :2]) ** 2, dim=1, keepdim=True)
        tracking_lin = torch.exp(-lin_err_sq / self.cfg.reward_tracking_sigma)
        yaw_err_sq = (cmd_vel[:, 2:3] - body_ang_vel[:, 2:3]) ** 2
        tracking_ang = torch.exp(-yaw_err_sq / self.cfg.reward_tracking_sigma)

        lin_vel_z_l2 = body_lin_vel[:, 2:3] ** 2
        ang_vel_xy_l2 = torch.sum(body_ang_vel[:, :2] ** 2, dim=1, keepdim=True)
        joint_vel_l2 = torch.sum(dofs_vel ** 2, dim=1, keepdim=True)
        joint_acc_l2 = torch.sum(((dofs_vel - self.last_dofs_vel) / self.cfg.step_dt) ** 2, dim=1, keepdim=True)
        self.last_dofs_vel.copy_(dofs_vel)
        action_rate_l2 = torch.sum((self.last_action - self.prev_action) ** 2, dim=1, keepdim=True)
        energy = torch.sum(dofs_torque ** 2, dim=1, keepdim=True)
        flat_orientation_l2 = torch.sum(proj_gravity[:, :2] ** 2, dim=1, keepdim=True)
        base_height_l2 = (body_pos[:, 2:3] - self.cfg.base_height_target) ** 2

        dof_pos_rel = dofs_pos - self.target_q_offset
        dof_pos_limits = torch.sum(torch.relu(torch.abs(dof_pos_rel) - self.cfg.dof_pos_soft_limit), dim=1, keepdim=True)
        hip_idx = torch.tensor([0, 6], dtype=torch.long, device=gs.device)
        hip_pos = torch.sum(torch.abs(dof_pos_rel.index_select(dim=1, index=hip_idx)), dim=1, keepdim=True)
        undesired_contacts = (non_foot_heights < self.cfg.termination_link_height).float().mean(dim=1, keepdim=True)

        total = (
            self.cfg.reward_tracking_lin_weight * tracking_lin
            + self.cfg.reward_tracking_ang_weight * tracking_ang
            + self.cfg.reward_alive_weight
            + self.cfg.reward_lin_vel_z_weight * lin_vel_z_l2
            + self.cfg.reward_ang_vel_xy_weight * ang_vel_xy_l2
            + self.cfg.reward_joint_vel_weight * joint_vel_l2
            + self.cfg.reward_joint_acc_weight * joint_acc_l2
            + self.cfg.reward_action_rate_weight * action_rate_l2
            + self.cfg.reward_energy_weight * energy
            + self.cfg.reward_flat_orientation_weight * flat_orientation_l2
            + self.cfg.reward_base_height_weight * base_height_l2
            + self.cfg.reward_dof_pos_limits_weight * dof_pos_limits
            + self.cfg.reward_hip_pos_weight * hip_pos
            + self.cfg.reward_undesired_contact_weight * undesired_contacts
        )
        if self.cfg.only_positive_rewards:
            total = torch.clamp(total, min=0.0)

        self._log = {
            "+track_lin_vel_xy": self.cfg.reward_tracking_lin_weight * tracking_lin,
            "+track_ang_vel_z": self.cfg.reward_tracking_ang_weight * tracking_ang,
            "+alive": torch.ones_like(total) * self.cfg.reward_alive_weight,
            "-lin_vel_z": self.cfg.reward_lin_vel_z_weight * lin_vel_z_l2,
            "-ang_vel_xy": self.cfg.reward_ang_vel_xy_weight * ang_vel_xy_l2,
            "-joint_vel": self.cfg.reward_joint_vel_weight * joint_vel_l2,
            "-joint_acc": self.cfg.reward_joint_acc_weight * joint_acc_l2,
            "-action_rate": self.cfg.reward_action_rate_weight * action_rate_l2,
            "-energy": self.cfg.reward_energy_weight * energy,
            "-flat_orientation": self.cfg.reward_flat_orientation_weight * flat_orientation_l2,
            "-base_height": self.cfg.reward_base_height_weight * base_height_l2,
            "-dof_pos_limits": self.cfg.reward_dof_pos_limits_weight * dof_pos_limits,
            "-hip_pos": self.cfg.reward_hip_pos_weight * hip_pos,
            "-undesired_contacts": self.cfg.reward_undesired_contact_weight * undesired_contacts,
            "+total": total,
            "state/body_height": body_pos[:, 2:3],
            "state/cmd_vx": cmd_vel[:, 0:1],
            "state/base_vx": body_lin_vel[:, 0:1],
        }
        return total

    def build_terminated(self, envs_idx, body_pos, body_quat, non_foot_heights, **kwargs):
        body_down = body_pos[:, 2:3] < self.cfg.termination_body_height
        link_down = (non_foot_heights < self.cfg.termination_link_height).any(dim=1, keepdim=True)
        # projected gravity xy norm ~= sin(tilt), terminate if tilt > 0.8 rad
        proj = self._project_gravity(body_quat)
        bad_orientation = torch.norm(proj[:, :2], dim=1, keepdim=True) > math.sin(self.cfg.termination_tilt_limit_rad)
        return body_down | link_down | bad_orientation

    def build_truncated(self, envs_idx, **kwargs):
        return (self.episode_steps[envs_idx] >= self.cfg.max_episode_length).unsqueeze(1)

    @torch.compiler.disable
    def build_info(self, envs_idx, **kwargs):
        return {"extra": {f"Reward / {k}": v.detach().mean().cpu() for k, v in self._log.items()}}

