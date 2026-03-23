from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import genesis as gs
import torch

from robots.robot import Robot, RobotConfig


@dataclass(kw_only=True)
class TeamedRobotConfig(RobotConfig):
    robot_cfgs: list[Any]
    robot_class: type[Robot]

    # Unused placeholders to satisfy RobotConfig schema.
    robot_URDF: str = ""
    base_link_name: str = ""
    joint_names: list[str] = field(default_factory=list)
    kp: Any = None
    kv: Any = None
    velocity_range: Any = None
    force_range: Any = None


class TeamedRobot(Robot):
    """Robot-compatible wrapper that groups same-class robots as one team."""

    cfg: TeamedRobotConfig

    def __init__(self, cfg: TeamedRobotConfig, scene: gs.Scene):
        # Intentionally bypass Robot.__init__: this wrapper does not own one URDF.
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
        if action.ndim == 2:
            action = action.unsqueeze(1)
        if action.shape[1] != len(self.robots):
            raise ValueError(f"Expected team action dim=1 to be {len(self.robots)}, got {action.shape[1]}")
        for idx, robot in enumerate(self.robots):
            robot.step(action[:, idx, :])

    def reset(self, envs_idx: torch.Tensor, **kwargs) -> None:
        reset_pos = kwargs.pop("reset_pos", None)
        reset_quat = kwargs.pop("reset_quat", None)
        for idx, robot in enumerate(self.robots):
            local_kwargs = dict(kwargs)
            if reset_pos is not None:
                local_kwargs["reset_pos"] = reset_pos[:, idx, :]
            if reset_quat is not None:
                local_kwargs["reset_quat"] = reset_quat[:, idx, :]
            robot.reset(envs_idx=envs_idx, **local_kwargs)

    def get_state(self, envs_idx: torch.Tensor) -> dict[str, torch.Tensor]:
        state_list = [robot.get_state(envs_idx=envs_idx) for robot in self.robots]
        keys = state_list[0].keys()
        return {key: torch.stack([state[key] for state in state_list], dim=1) for key in keys}
