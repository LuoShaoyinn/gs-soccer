# robot.py
#   Build up a robot
#

import torch
import numpy as np
import genesis as gs
import gymnasium as gym
from dataclasses import dataclass, field

from .robot import RobotConfig, Robot


@dataclass(kw_only = True)
class ControlledRobotWrapperConfig():
    robot_cfg:          RobotConfig
    robot_class:        type[Robot]
    policy_path:        str
    # the ratio of policy frequency to control frequency, 
    # e.g., freq_ratio=2 means the policy is called every 2 control steps


class ControlledRobotWrapper():
    def __init__(self, cfg: ControlledRobotWrapperConfig, scene: gs.Scene):
        self.cfg = cfg
        self.scene = scene
        self.robot = self.cfg.robot_class(self.cfg.robot_cfg, scene)
        self.policy = torch.jit.load(self.cfg.policy_path).to(gs.device)

    def build(self):
        self.robot.build()

    def config(self):
        self.robot.config()
 

    def step(self, envs_idx: torch.Tensor, action: torch.Tensor) -> None:
        # action: lin_x, lin_y, ang_z
        state = self.get_state(envs_idx=envs_idx)
        policy_obs = self.robot.build_observation(cmd_vel=action,
                                                  **state)
        policy_action = self.policy(policy_obs)
        self.robot.step(action=policy_action, envs_idx=envs_idx)

    def reset(self, envs_idx: torch.Tensor, **kwargs) -> None:
        self.robot.reset(envs_idx=envs_idx, **kwargs)
 

    @property
    @torch.compiler.disable
    def observation_space(self) -> gym.spaces.Box:
        # body_pos[0:2]
        # body_heading[2:4] (sin_yaw, cos_yaw)          Do not use angle
        # body_vel[4:7]     (lin_x, lin_y, ang_z)
        # ball_pos_rel[7:9] (ball_rel_x, ball_rel_y)
        # ball_vel_rel[9:11]
        # cmd_vel[11:14]    (lin_x, lin_y, ang_z)       last control command
        # target_pos[14:16]
        return gym.spaces.Box(low   = -10.0, 
                              high  =  10.0, 
                              shape = (16,), 
                              dtype = np.float32)
    
    @property
    @torch.compiler.disable
    def action_space(self) -> gym.spaces.Box:
        # lin_x, lin_y, ang_z
        return gym.spaces.Box(low   = -1.0, 
                              high  =  1.0, 
                              shape = (3,), 
                              dtype = np.float32)
    
    def get_state(self, envs_idx: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.robot.get_state(envs_idx=envs_idx)

    def build_observation(self, 
                        body_pos_2D: torch.Tensor, 
                        body_lin_vel: torch.Tensor,
                        body_ang_vel: torch.Tensor,
                        body_heading: torch.Tensor,
                        ball_pos_rel: torch.Tensor,
                        cmd_vel: torch.Tensor,
                        **kwargs) -> torch.Tensor:
        return torch.cat((body_pos[:, 0:2], 
                          body_heading,
                          body_lin_vel[:, 0:2], 
                          body_ang_vel[:, 2].unsqueeze(-1),
                          ball_pos_rel,
                          cmd_vel), dim=1)
