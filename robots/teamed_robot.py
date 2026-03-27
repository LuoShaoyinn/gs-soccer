# teamed_robot.py
#   TeamedRobot wrapper to group same-class robots as one team.
#

import torch
import numpy as np
import genesis as gs
import gymnasium as gym
from typing import Any
from dataclasses import dataclass, field
from robots.robot import Robot, RobotConfig

@dataclass(kw_only=True)
class TeamedRobotConfig(RobotConfig):
    robot_cfgs: list[Any]
    robot_class: type[Robot]
    # Override the following fields from RobotConfig
    robot_URDF:     str | None          = None
    base_link_name: str | None          = None
    joint_names:    list[str] | None    = None
    kp:             np.ndarray | None   = None
    kv:             np.ndarray | None   = None
    velocity_range: np.ndarray | None   = None
    force_range:    np.ndarray | None   = None


class TeamedRobot(Robot):
    cfg: TeamedRobotConfig

    def __init__(self, cfg: TeamedRobotConfig, scene: gs.Scene):
        self.cfg = cfg
        self.scene = scene
        self.robots = [self.cfg.robot_class(robot_cfg, scene) for robot_cfg in self.cfg.robot_cfgs]
        if len(self.robots) == 0:
            raise ValueError("TeamedRobot requires at least one robot config")

    def build(self) -> None:
        for robot in self.robots:
            robot.build()

    def config(self) -> None:
        for robot in self.robots:
            robot.config()

    def step(self, action: torch.Tensor) -> None:
        action = action.view(-1, len(self.robots), -1)
        for idx, robot in enumerate(self.robots):
            robot.step(action[:, idx, :])

    def reset(self, envs_idx: torch.Tensor, 
              reset_pos:  torch.Tensor | None = None,
              reset_quat: torch.Tensor | None = None, **kwargs) -> None:
        for idx, robot in enumerate(self.robots):
            robot.reset(envs_idx=envs_idx, 
                        reset_pos=reset_pos, 
                        reset_quat=reset_quat, **kwargs)

    def get_state(self, envs_idx: torch.Tensor) -> dict[str, torch.Tensor]:
        state_list = [robot.get_state(envs_idx=envs_idx) for robot in self.robots]
        keys = state_list[0].keys()
        return {key: torch.stack([state[key] for state in state_list], dim=1) for key in keys}
