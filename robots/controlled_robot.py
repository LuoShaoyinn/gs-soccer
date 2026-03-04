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

    @torch.no_grad()
    @torch.compile()
    def build(self):
        self.robot.build()

    @torch.no_grad()
    @torch.compile()
    def config(self):
        self.robot.config()
        self.__last_cmd_vel = torch.zeros((self.scene.n_envs, 3), device=gs.device)
 

    @torch.no_grad()
    @torch.compile()
    def step(self, envs_idx: torch.Tensor, action: torch.Tensor) -> None:
        # action: lin_x, lin_y, ang_z
        state = self.get_state(envs_idx=envs_idx)
        policy_obs = self.robot.build_observation(cmd_vel=action,
                                                  **state)
        policy_action = self.policy(policy_obs)
        self.robot.step(action=policy_action, envs_idx=envs_idx)

    @torch.no_grad()
    @torch.compile()
    def reset(self, envs_idx: torch.Tensor, **kwargs) -> None:
        self.__last_cmd_vel[envs_idx] = 0.0
        self.robot.reset(envs_idx=envs_idx, **kwargs)
 

    @property
    @torch.no_grad()
    @torch.compiler.disable
    def observation_space(self) -> gym.spaces.Box:
        # body_pos[0:3]
        # body_vel[3:6] (lin_x, lin_y, ang_z)
        # ball_rel[6:8]
        return gym.spaces.Box(low   = -10.0, 
                              high  =  10.0, 
                              shape = (9,), 
                              dtype = np.float32)
    
    @property
    @torch.no_grad()
    @torch.compiler.disable
    def action_space(self) -> gym.spaces.Box:
        # lin_x, lin_y, ang_z
        return gym.spaces.Box(low   = -1.0, 
                              high  =  1.0, 
                              shape = (3,), 
                              dtype = np.float32)
    
    @torch.no_grad()
    @torch.compile()
    def get_state(self, envs_idx: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"last_cmd_vel": self.__last_cmd_vel[envs_idx], 
                **self.robot.get_state(envs_idx=envs_idx)}

    @torch.no_grad()
    @torch.compile()
    def build_observation(self, 
                        body_pos: torch.Tensor, 
                        body_lin_vel: torch.Tensor,
                        body_ang_vel: torch.Tensor,
                        ball_pos: torch.Tensor,
                        **kwargs) -> torch.Tensor:
        return torch.cat((body_pos, 
                          body_lin_vel[:, 0:2], 
                          body_ang_vel[:, 2].unsqueeze(-1),
                          ball_pos - body_pos), dim=1)
