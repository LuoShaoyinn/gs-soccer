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
    policy_path:        str = ""    # in child class there should be a default


class ControlledRobot(Robot):
    def __init__(self, cfg: ControlledRobotConfig, scene: gs.Scene):
        super().__init__(cfg, scene)

 
    # ------------------------
    # Genesis helper functions
    # ------------------------
    def gs_config(self) -> None:
        super().gs_config()
        self.policy_observation = None
#        self.policy_actor = torch.jit.load(self.cfg.policy_path) # type: ignore
        self.__last_action = torch.zeros((self.scene.n_envs, 3),
                                         dtype=torch.float,
                                         device=gs.device)
        self.__last_state = super().get_state(torch.arange(self.scene.n_envs, 
                                                           dtype=torch.long,
                                                           device=gs.device))
 
 
    # ---------------
    # Steps and reset
    # ---------------
    def step(self, action: torch.Tensor,  # action is cmd_vel here
             envs_idx: torch.Tensor) -> None:
        # inference with policy
        policy_observation = super().get_observation(cmd_vel=action, 
                                                     **self.__last_state)
        policy_action = torch.zeros((self.scene.n_envs, 12), dtype=torch.float, device=gs.device) # self.policy_actor(self.policy_observation)
        super().step(policy_action, envs_idx)
        self.__last_action = action


    # --------------------------------------
    # Observation, actions
    # The functions you may want to override
    # --------------------------------------
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
    
    #@torch.no_grad()
    #@torch.compile()
    def get_state(self, envs_idx: torch.Tensor, **kwargs) -> dict[str, torch.Tensor]:
        ret = super().get_state(envs_idx=envs_idx)
        ret["last_cmd_vel"] = self.__last_action
        if envs_idx.shape[0] == self.scene.n_envs:
            for (k, v) in ret.items():
                if k in self.__last_state:
                    self.__last_state[k][envs_idx] = v.clone()
        else:
            self.__last_state = ret
        return ret

    #@torch.no_grad()
    #@torch.compile()
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
