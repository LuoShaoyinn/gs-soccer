# robot.py
#   Build up a robot
#

import torch
import numpy as np
import genesis as gs
import gymnasium as gym
from dataclasses import dataclass, field

from .robot import RobotConfig, Robot


@dataclass
class ControlledRobotConfig(RobotConfig):
    policy_path:        str = ""


class ControlledRobot(Robot):
    @property
    def observation_space(self) -> gym.spaces.Box:
        N = len(self.cfg.joint_names)
        return gym.spaces.Box(low   = -100.0, 
                              high  =  100.0, 
                              shape = (10,), 
                              dtype = np.float32)
    
    @property
    def action_space(self) -> gym.spaces.Box:
        return gym.spaces.Box(low   = -1.0, 
                              high  =  1.0, 
                              shape = (3,), 
                              dtype = np.float32)

    def gs_config(self) -> None:
        super().gs_config()
        self.move_dofs_idx = torch.tensor([0, 1, 5], 
                                          dtype=torch.long,
                                          device=gs.device)

    def gs_step(self, action: torch.Tensor, envs_idx: torch.Tensor) -> None:
        self.robot.control_dofs_position(action, 
                                         dofs_idx_local=self.move_dofs_idx, 
                                         envs_idx=envs_idx)
