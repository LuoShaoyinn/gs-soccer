# robot.py
#   Build up a robot
#

import torch
import numpy as np
import genesis as gs
import gymnasium as gym
from dataclasses import dataclass, field

from .robot import RobotConfig, Robot
from models.model import ModelConfig, Model

@dataclass(kw_only = True)
class ControlledRobotWrapperConfig():
    robot_cfg:          RobotConfig
    robot_class:        type[Robot]
    ctrl_model_cfg:     ModelConfig
    ctrl_model_class:   type[Model]
    ctrl_policy_path:   str


class ControlledRobotWrapper():
    def __init__(self, cfg: ControlledRobotWrapperConfig, scene: gs.Scene):
        self.cfg = cfg
        self.scene = scene
        self.robot = self.cfg.robot_class(self.cfg.robot_cfg, scene)
        self.model = self.cfg.ctrl_model_class(self.cfg.ctrl_model_cfg, scene)
        self.policy = torch.jit.load(self.cfg.ctrl_policy_path).to(gs.device)

    def build(self):
        self.robot.build()

    def config(self):
        self.robot.config()
        self.all_envs_idx = torch.arange(self.scene.n_envs, 
                                         dtype=torch.long, 
                                         device=gs.device)
 
    def step(self, action: torch.Tensor) -> None:
        # action: lin_x, lin_y, ang_z
        for i in range(self.ctrl_substeps):
            state = self.robot.get_state(envs_idx=self.all_envs_idx)
            policy_obs = self.model.build_observation(cmd_vel=action,
                                                      **state)
            policy_action = self.policy(policy_obs)
            self.robot.step(action=policy_action)

    def reset(self, envs_idx: torch.Tensor, **kwargs) -> None: # type: ignore
        self.robot.reset(envs_idx=envs_idx, **kwargs)
 
    def get_state(self, envs_idx: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.robot.get_state(envs_idx=envs_idx)
