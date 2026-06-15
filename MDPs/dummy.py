# dummy.py
#   A dummy standing MDP. Shows how to inherit from MDP.

import numpy as np
import torch
import genesis as gs
import gymnasium as gym
from dataclasses import dataclass

from .MDP import MDP, MDPConfig


@dataclass(kw_only=True)
class DummyMDPConfig(MDPConfig):
    action_dim: int
    home_pose: np.ndarray
    base_pos: np.ndarray
    base_quat: np.ndarray
    ball_pos: np.ndarray


class DummyMDP(MDP):
    cfg: DummyMDPConfig

    def build(self):
        pass

    def config(self):
        self._home_pose = torch.from_numpy(self.cfg.home_pose).to(gs.device)
        self._base_pos = torch.from_numpy(self.cfg.base_pos).to(gs.device)
        self._base_quat = torch.from_numpy(self.cfg.base_quat).to(gs.device)
        self._ball_pos = torch.from_numpy(self.cfg.ball_pos).to(gs.device)

    def reset(self, envs_idx, robot_reset_fn, field_reset_fn):
        n = envs_idx.shape[0]
        robot_reset_fn(
            joint_pos=self._home_pose.broadcast_to((n, self.cfg.action_dim)),
            reset_pos=self._base_pos.broadcast_to((n, 3)),
            reset_quat=self._base_quat.broadcast_to((n, 4)),
        )
        field_reset_fn(ball_pos=self._ball_pos.broadcast_to((n, 3)))

    @property
    def observation_space(self) -> gym.spaces.Box:
        return gym.spaces.Box(
            low=-1e6, high=1e6, shape=(self.cfg.action_dim * 2,), dtype=np.float32
        )

    @property
    def action_space(self) -> gym.spaces.Box:
        return gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.cfg.action_dim,), dtype=np.float32
        )

    def build_observation(self, envs_idx, dofs_pos=None, dofs_vel=None, **kwargs):
        if dofs_pos is None or dofs_vel is None:
            n = envs_idx.shape[0]
            return torch.zeros(
                (n, self.cfg.action_dim * 2), dtype=torch.float32, device=gs.device
            )
        return torch.cat([dofs_pos, dofs_vel], dim=-1)

    def build_reward(self, envs_idx, **kwargs):
        return torch.zeros((envs_idx.shape[0], 1), dtype=torch.float, device=gs.device)

    def build_terminated(self, envs_idx, **kwargs):
        return torch.zeros((envs_idx.shape[0], 1), dtype=torch.bool, device=gs.device)

    def build_truncated(self, envs_idx, **kwargs):
        return torch.zeros((envs_idx.shape[0], 1), dtype=torch.bool, device=gs.device)

    def build_info(self, envs_idx, **kwargs):
        return {}
