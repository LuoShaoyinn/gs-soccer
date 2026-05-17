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
FOOT_LINK_NAMES = ["Rf", "Lf"]

@dataclass(kw_only = True)
class WalkEnvConfig(EnvConfig):
    num_envs:       int     = 1
    field_range:    float   = 1.0
    rl_dt:          float   = 0.02
    show_viewer:    bool    = False
    foot_contact_height: float = 0.03


class WalkEnv(Env): 
    cfg: WalkEnvConfig

    def config(self):
        super().config()
        self.cmd_vel = torch.rand((self.num_envs, 3)) * 2.0 - 1.0
        self._non_foot_links = [
            self.robot.robot.get_link(n) for n in NON_FOOT_LINK_NAMES
        ]
        self._foot_links = [
            self.robot.robot.get_link(n) for n in FOOT_LINK_NAMES
        ]
        self._kp = torch.from_numpy(self.robot.cfg.kp).to(gs.device)
        self._kv = torch.from_numpy(self.robot.cfg.kv).to(gs.device)

    def reset(self, envs_idx: torch.Tensor | None = None
              ) -> tuple[torch.Tensor, dict]:
        if envs_idx is None:
            envs_idx = self.all_envs_idx
        self.cmd_vel[envs_idx] = (torch.rand((envs_idx.shape[0], 3), 
                                             dtype=torch.float, 
                                             device=gs.device) * 2.0 - 1.0)
        return super().reset(envs_idx)

    @torch.compiler.disable
    def _get_non_foot_heights(self, envs_idx):
        parts = []
        for link in self._non_foot_links:
            pos = link.get_pos(envs_idx=envs_idx)
            parts.append(pos[:, 2:3])
        return torch.cat(parts, dim=1)

    @torch.compiler.disable
    def _get_foot_heights(self, envs_idx):
        parts = []
        for link in self._foot_links:
            pos = link.get_pos(envs_idx=envs_idx)
            parts.append(pos[:, 2:3])
        return torch.cat(parts, dim=1)

    @torch.compiler.disable
    def _get_foot_lin_speeds(self, envs_idx):
        parts = []
        for link in self._foot_links:
            vel = link.get_vel(envs_idx=envs_idx)
            parts.append(torch.norm(vel, dim=1, keepdim=True))
        return torch.cat(parts, dim=1)

    def get_state(self, envs_idx: torch.Tensor) -> dict[str, dict[str, torch.Tensor]]:
        state = super().get_state(envs_idx)
        commands = {"cmd_vel": self.cmd_vel[envs_idx]}
        contacts = {
            "non_foot_heights": self._get_non_foot_heights(envs_idx),
            "foot_heights": self._get_foot_heights(envs_idx),
            "foot_lin_speeds": self._get_foot_lin_speeds(envs_idx),
        }
        contacts["foot_contacts"] = (contacts["foot_heights"] < self.cfg.foot_contact_height).float()
        target_q = self.model.ewma_action[envs_idx] + self.model.target_q_offset
        robot_state = state["robot"]
        actuation = {
            "target_q": target_q,
            "dofs_torque": (
                self._kp * (target_q - robot_state["dofs_pos"])
                - self._kv * robot_state["dofs_vel"]
            ),
        }
        state["commands"] = commands
        state["contacts"] = contacts
        state["actuation"] = actuation
        return state
