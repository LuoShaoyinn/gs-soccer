# game.py
#   Build up a game env
#

import copy
import torch
import numpy as np
import genesis as gs
from envs.env import Env, EnvConfig
from dataclasses import dataclass, field
from robots.robot import RobotConfig, Robot

@dataclass(kw_only=True)
class GameEnvConfig(EnvConfig):
    robot_cfg:          dict[str, RobotConfig]
    robot_class:        dict[str, type[Robot]]
    robot_reset_pos:    dict[str, np.ndarray] # should be (x, y, yaw)
    ball_reset_pos:     np.ndarray  = field(default_factory=\
            lambda: np.array([0.0, 0.0], dtype=np.float32))
    reset_pos_noise:    float       = 0.5
    reset_yaw_noise:    float       = 0.1
    action_noise:       float       = 0.1
    ctrl_freq_ratio:    int         = 10


class GameEnv(Env):
    cfg: GameEnvConfig
    def __init__(self, cfg: GameEnvConfig):
        assert cfg.robot_class.keys()      == {"red", "blue"}, \
                "robot name must be 'red' and 'blue'"
        assert cfg.robot_cfg.keys()        == {"red", "blue"}, \
                "robot cfg must be 'red' and 'blue'"
        assert cfg.robot_reset_pos.keys()  == {"red", "blue"}, \
                "robot reset pos must be 'red' and 'blue'"
        self.team_names = ("red", "blue")
        super().__init__(cfg)


    def build(self):
        self.field = self.cfg.field_class(self.cfg.field_cfg, self.scene)
        self.model = self.cfg.model_class(self.cfg.model_cfg, self.scene)
        self.robots = {}
        for name in self.team_names:
            robot_class = self.cfg.robot_class[name]
            self.robots[name] = robot_class(self.cfg.robot_cfg[name], self.scene)
        self.field.build()
        for robot in self.robots.values():
            robot.build()
        self.model.build()


    def config(self):
        self.field.config()
        for robot in self.robots.values():
            robot.config()
        self.model.config()

        self.observation_space  = self.model.observation_space
        self.action_space       = self.model.action_space
        self.num_agents         = len(self.robots)
        self.all_envs_idx       = torch.arange(self.num_envs, dtype=torch.long, device=gs.device)
        self.robot_reset_pos    = {name: torch.from_numpy(pos) \
                                              .to(gs.device) \
                                              .broadcast_to((self.num_envs, 3))
                                      for name, pos in self.cfg.robot_reset_pos.items()}
        self.ball_reset_pos     = torch.from_numpy(self.cfg.ball_reset_pos) \
                                        .to(gs.device) \
                                        .broadcast_to((self.num_envs, 2))


    def step(self, action: dict[str, torch.Tensor]): # type: ignore[override]
        action = self.model.preprocess_action(action) # type: ignore[arg-type]

        for _ in range(self.cfg.ctrl_freq_ratio):
            for name in self.team_names:
                action_noise = 2.0 * (torch.rand_like(action[name]) - 0.5)
                action_noise = 1.0 + self.cfg.action_noise * action_noise
                self.robots[name].step(action[name] * action_noise)
            self.gs_step()

        kwargs              = self.get_state(envs_idx=self.all_envs_idx)
        next_observation    = self.model.build_observation(envs_idx=self.all_envs_idx, **kwargs)
        reward              = self.model.build_reward(envs_idx=self.all_envs_idx, **kwargs)
        terminated          = self.model.build_terminated(envs_idx=self.all_envs_idx, **kwargs)
        truncated           = self.model.build_truncated(envs_idx=self.all_envs_idx, **kwargs)
        info                = self.model.build_info(envs_idx=self.all_envs_idx, **kwargs)

        need_reset = torch.zeros((self.num_envs,), dtype=torch.bool, device=gs.device)
        for name in self.team_names:
            need_reset |= torch.logical_or(terminated[name], truncated[name]).squeeze(1)
        if need_reset.any():
            reset_idx = torch.nonzero(need_reset).squeeze(1)
            reset_observation, _ = self.reset(reset_idx)
            for name in self.team_names:
                next_observation[name][reset_idx] = reset_observation[name]
        return (next_observation, reward, terminated, truncated, info)
    

    def reset(self, envs_idx: torch.Tensor | None = None):
        if envs_idx is None:
            envs_idx = self.all_envs_idx
        n_envs = envs_idx.shape[0]
    
        def randomize(x: torch.Tensor, noise: float) -> torch.Tensor:
            return x + 2.0 * (torch.rand_like(x) - 0.5) * noise

        def yaw_to_quat(yaw: torch.Tensor) -> torch.Tensor:
            half_yaw = yaw / 2.0
            return torch.stack((torch.cos(half_yaw),
                                torch.zeros_like(half_yaw), 
                                torch.zeros_like(half_yaw), 
                                torch.sin(half_yaw)), dim=-1)
        
        # Reset ball and robot position
        ball_reset_pos = self.ball_reset_pos[envs_idx]
        self.field.reset(envs_idx=envs_idx, 
                         ball_pos=randomize(x=ball_reset_pos, 
                                                  noise=self.cfg.reset_pos_noise))
        for name in self.team_names:
            robot = self.robots[name]
            robot_reset = self.robot_reset_pos[name][envs_idx]
            robot.reset(envs_idx=envs_idx, 
                        reset_pos=randomize(x=robot_reset[:, :2], 
                                            noise=self.cfg.reset_pos_noise), 
                        reset_quat=yaw_to_quat(\
                                randomize(x=robot_reset[:, 2:], 
                                          noise=self.cfg.reset_yaw_noise)))

        # Don't forget to reset the model
        self.model.reset(envs_idx=envs_idx)
        kwargs = self.get_state(envs_idx=envs_idx)
        return  self.model.build_observation(envs_idx=envs_idx, **kwargs), \
                self.model.build_info(envs_idx=envs_idx, **kwargs)


    def get_state(self, envs_idx: torch.Tensor) -> dict[str, object]: # type: ignore[override]
        ret = {name: self.robots[name].get_state(envs_idx=envs_idx)
                for name in self.team_names}
        return ret | self.field.get_state(envs_idx=envs_idx)
