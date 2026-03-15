# walk_model.py
#   Define walking model for pi

import torch
import numpy as np
import genesis as gs
import gymnasium as gym
from dataclasses import dataclass

from .model import ModelConfig, Model


@dataclass(kw_only=True)
class PIWalkModelConfig(ModelConfig):
    n_dofs: int
    target_q_offset: np.ndarray
    history_frames: int = 5
    dofs_pos_scale: float = 1.0
    dofs_vel_scale: float = 1.0
    action_scale: float = 0.25
    obs_clip: tuple = (-100.0, 100.0)
    action_clip: tuple = (-100.0, 100.0)


class PIWalkModel(Model):
    cfg: PIWalkModelConfig

    def config(self):
        self.target_q_offset = torch.from_numpy(self.cfg.target_q_offset).to(gs.device)
        self.dim_observations = 9 + 3 * self.cfg.n_dofs
        self.last_obs = torch.zeros(
            (self.scene.n_envs, self.cfg.history_frames, self.dim_observations),
            dtype=torch.float,
            device=gs.device,
        )
        self.last_action = torch.zeros((self.scene.n_envs, self.cfg.n_dofs), dtype=torch.float, device=gs.device)

    def reset(self, envs_idx: torch.Tensor):
        self.last_obs[envs_idx] = 0.0
        self.last_action[envs_idx] = 0.0

    def preprocess_action(self, action: torch.Tensor):
        self.last_action = torch.clip(action, self.cfg.action_clip[0], self.cfg.action_clip[1])
        return self.last_action * self.cfg.action_scale + self.target_q_offset

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
        return gym.spaces.Box(
            low=self.cfg.action_clip[0],
            high=self.cfg.action_clip[1],
            shape=(self.cfg.n_dofs,),
            dtype=np.float32,
        )

    def build_observation(
        self, envs_idx, body_ang_vel, body_quat, cmd_vel, dofs_pos, dofs_vel, **kwargs
    ) -> torch.Tensor:  # type: ignore
        n_envs = self.scene.n_envs
        gravity = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float, device=gs.device).broadcast_to(
            (n_envs, 3)
        )
        q_w = body_quat[:, 0:1]
        q_vec = body_quat[:, 1:4]
        proj_gravity = (
            gravity * (2.0 * q_w * q_w - 1.0)
            - torch.cross(q_vec, gravity, dim=1) * (2.0 * q_w)
            + q_vec * (2.0 * (q_vec * gravity).sum(dim=1, keepdim=True))
        )
        obs_single_frame = torch.cat(
            (
                body_ang_vel,
                proj_gravity,
                cmd_vel.reshape((n_envs, 3)),
                (dofs_pos - self.target_q_offset) * self.cfg.dofs_pos_scale,
                dofs_vel * self.cfg.dofs_vel_scale,
                self.last_action[envs_idx],
            ),
            dim=-1,
        )
        obs_single_frame = torch.clip(obs_single_frame, self.cfg.obs_clip[0], self.cfg.obs_clip[1])

        self.last_obs = torch.roll(self.last_obs, shifts=-1, dims=1)
        self.last_obs[:, -1, :] = obs_single_frame
        return self.last_obs.reshape(n_envs, -1)
