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

from .env import Env, EnvConfig
from robots.robot import RobotConfig, Robot
from fields.field import FieldConfig, Field
from models.model import ModelConfig, Model

NON_FOOT_LINK_NAMES = [
    "Rhip", "Lhip", "Rlap", "Llap",
    "Rleg1", "Lleg1", "Rleg2", "Lleg2",
    "Rankle", "Lankle",
]

@dataclass(kw_only = True)
class WalkEnvConfig(EnvConfig):
    num_envs:       int     = 1
    field_range:    float   = 1.0
    rl_dt:          float   = 0.02
    show_viewer:    bool    = False


class WalkEnv(Env): 
    cfg: WalkEnvConfig

    def _sample_cmd_vel(self, n_envs: int) -> torch.Tensor:
        # Easier curriculum: forward-only command. Lateral and yaw are zero.
        cmd_vel = torch.zeros((n_envs, 3), dtype=torch.float, device=gs.device)
        cmd_vel[:, 0] = 0.2 + 0.8 * torch.rand((n_envs,), dtype=torch.float, device=gs.device)
        return cmd_vel

    def config(self):
        super().config()
        self.cmd_vel = self._sample_cmd_vel(self.num_envs)
        self._non_foot_links = [
            self.robot.robot.get_link(n) for n in NON_FOOT_LINK_NAMES
        ]
        self._kp = torch.from_numpy(self.robot.cfg.kp).to(gs.device)
        self._kv = torch.from_numpy(self.robot.cfg.kv).to(gs.device)

    def reset(self, envs_idx: torch.Tensor | None = None
              ) -> tuple[torch.Tensor, dict]:
        if envs_idx is None:
            envs_idx = self.all_envs_idx
        self.cmd_vel[envs_idx] = self._sample_cmd_vel(envs_idx.shape[0])
        return super().reset(envs_idx)

    @torch.compiler.disable
    def _get_non_foot_heights(self, envs_idx):
        parts = []
        for link in self._non_foot_links:
            pos = link.get_pos(envs_idx=envs_idx)
            parts.append(pos[:, 2:3])
        return torch.cat(parts, dim=1)

    def get_state(self, envs_idx: torch.Tensor) -> dict[str, torch.Tensor]:
        state = {"cmd_vel": self.cmd_vel[envs_idx],
                 **super().get_state(envs_idx)}
        state["non_foot_heights"] = self._get_non_foot_heights(envs_idx)
        target_q = self.model.ewma_action[envs_idx] + self.model.target_q_offset
        state["dofs_torque"] = (
            self._kp * (target_q - state["dofs_pos"])
            - self._kv * state["dofs_vel"]
        )
        return state
