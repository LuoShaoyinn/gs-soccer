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
        self.__last_cmd_vel = torch.zeros((self.scene.n_envs, 3), device=gs.device)
        self.__last_state = {}
 
    def step(self, envs_idx: torch.Tensor, action: torch.Tensor) -> None:
        # action: lin_x, lin_y, ang_z
        self.__last_cmd_vel[envs_idx] = action
        policy_obs = self.robot.build_observation(cmd_vel=action, **self.__last_state)
        policy_action = self.policy(policy_obs)
        self.robot.step(action=policy_action, envs_idx=envs_idx)

    def reset(self, envs_idx: torch.Tensor, **kwargs) -> None:
        self.__last_cmd_vel[envs_idx] = 0.0
        self.robot.reset(envs_idx=envs_idx, **kwargs)
 
    @property
    def observation_space(self) -> gym.spaces.Box:
        # body_pos[0:3]
        # body_vel[3:6] (lin_x, lin_y, ang_z)
        # ball_rel[6:8]
        return gym.spaces.Box(low   = -10.0, 
                              high  =  10.0, 
                              shape = (8,), 
                              dtype = np.float32)
    
    @property
    def action_space(self) -> gym.spaces.Box:
        # lin_x, lin_y, ang_z
        return gym.spaces.Box(low   = -1.0, 
                              high  =  1.0, 
                              shape = (3,), 
                              dtype = np.float32)
    
    def get_state(self, envs_idx: torch.Tensor) -> dict[str, torch.Tensor]:
        new_obs = self.robot.get_state(envs_idx=envs_idx)
        if envs_idx.shape[0] == self.scene.n_envs:
            self.__last_state = new_obs
        else:
            for (k, v) in new_obs.items():
                self.__last_state[k][envs_idx] = v[envs_idx]
        return {"last_cmd_vel": self.__last_cmd_vel, **self.__last_state}

    def get_observation(self, 
                        body_pos: torch.Tensor, 
                        body_lin_vel: torch.Tensor,
                        body_ang_vel: torch.Tensor,
                        ball_pos: torch.Tensor,
                        **kwargs) -> torch.Tensor: # type: ignore
        return torch.cat((body_pos, 
                          body_lin_vel[:, 0:2], 
                          body_ang_vel[:, 2].unsqueeze(-1),
                          ball_pos - body_pos), dim=1)
