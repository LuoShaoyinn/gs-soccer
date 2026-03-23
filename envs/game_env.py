from __future__ import annotations

import copy
from dataclasses import dataclass, field

import genesis as gs
import numpy as np
import torch

from envs.env import Env, EnvConfig
from robots.teamed_robot import TeamedRobot


@dataclass(kw_only=True)
class GameEnvConfig(EnvConfig):
    num_teams: int = 2
    ctrl_freq_ratio: int = 10
    team_reset_pos: list[np.ndarray] = field(
        default_factory=lambda: [np.array([-1.5, 0.0], dtype=np.float32), np.array([1.5, 0.0], dtype=np.float32)]
    )
    ball_reset_pos: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0], dtype=np.float32))
    reset_pos_noise: float = 0.5
    reset_yaw_noise: float = float(np.pi)
    action_noise: float = 0.1


class GameEnv(Env):
    cfg: GameEnvConfig

    def build(self):
        self.field = self.cfg.field_class(self.cfg.field_cfg, self.scene)
        self.robots = [self.cfg.robot_class(copy.deepcopy(self.cfg.robot_cfg), self.scene) for _ in range(self.cfg.num_teams)]
        self.model = self.cfg.model_class(self.cfg.model_cfg, self.scene)
        self.robot = self.robots

        self.field.build()
        for robot in self.robots:
            robot.build()
        self.model.build()

    def config(self):
        self.field.config()
        for robot in self.robots:
            robot.config()
        self.model.config()

        self.observation_space = self.model.observation_space
        self.action_space = self.model.action_space
        self.num_agents = len(self.robots)

        self.all_envs_idx = torch.arange(self.num_envs, dtype=torch.long, device=gs.device)
        self.team_reset_pos = [torch.tensor(pos, dtype=torch.float, device=gs.device) for pos in self.cfg.team_reset_pos]
        self.ball_reset_pos = torch.tensor(self.cfg.ball_reset_pos, dtype=torch.float, device=gs.device)

    @torch.compiler.disable
    def __gs_step(self):
        self.scene.step()

    def _team_size(self, robot) -> int:
        if isinstance(robot, TeamedRobot):
            return len(robot.robots)
        return 1

    def _step_robot(self, robot, action: torch.Tensor) -> None:
        if not isinstance(robot, TeamedRobot) and action.ndim == 3 and action.shape[1] == 1:
            action = action[:, 0, :]
        robot.step(action)

    def step(self, action: list[torch.Tensor]): # type: ignore[override]
        action = self.model.preprocess_action(action) # type: ignore[arg-type]

        for _ in range(self.cfg.ctrl_freq_ratio):
            for team_idx, team_action in enumerate(action):
                noise = 1.0 + self.cfg.action_noise * (2.0 * torch.rand_like(team_action) - 1.0)
                self._step_robot(self.robots[team_idx], team_action * noise)
            self.__gs_step()

        kwargs = self.get_state(envs_idx=self.all_envs_idx)
        next_observation = self.model.build_observation(envs_idx=self.all_envs_idx, **kwargs)
        reward = self.model.build_reward(envs_idx=self.all_envs_idx, **kwargs)
        terminated = self.model.build_terminated(envs_idx=self.all_envs_idx, **kwargs)
        truncated = self.model.build_truncated(envs_idx=self.all_envs_idx, **kwargs)
        info = self.model.build_info(envs_idx=self.all_envs_idx, **kwargs)

        need_reset = torch.zeros((self.num_envs,), dtype=torch.bool, device=gs.device)
        for term_i, trunc_i in zip(terminated, truncated):
            need_reset |= torch.logical_or(term_i, trunc_i).squeeze(1)
        if need_reset.any():
            reset_idx = torch.nonzero(need_reset).squeeze(1)
            reset_observation, _ = self.reset(reset_idx)
            for i in range(len(next_observation)):
                next_observation[i][reset_idx] = reset_observation[i]
        return next_observation, reward, terminated, truncated, info

    def reset(self, envs_idx: torch.Tensor | None = None):
        if envs_idx is None:
            envs_idx = self.all_envs_idx
        n_envs = envs_idx.shape[0]

        def sample_pos(center: torch.Tensor, team_size: int) -> torch.Tensor:
            return center.view(1, 1, 2).broadcast_to((n_envs, team_size, 2)) + self.cfg.reset_pos_noise * (
                2.0 * torch.rand((n_envs, team_size, 2), dtype=torch.float, device=gs.device) - 1.0
            )

        def sample_quat(yaw_center: float, team_size: int) -> torch.Tensor:
            yaw = yaw_center + self.cfg.reset_yaw_noise * (
                2.0 * torch.rand((n_envs, team_size), dtype=torch.float, device=gs.device) - 1.0
            )
            quat = torch.zeros((n_envs, team_size, 4), dtype=torch.float, device=gs.device)
            quat[:, :, 0] = torch.cos(0.5 * yaw)
            quat[:, :, 3] = torch.sin(0.5 * yaw)
            return quat

        ball_pos = self.ball_reset_pos + self.cfg.reset_pos_noise * (
            2.0 * torch.rand((n_envs, 2), dtype=torch.float, device=gs.device) - 1.0
        )
        self.field.reset(envs_idx=envs_idx, ball_pos=ball_pos)

        for team_idx, robot in enumerate(self.robots):
            team_size = self._team_size(robot)
            yaw_center = 0.0 if team_idx == 0 else float(np.pi)
            reset_pos = sample_pos(self.team_reset_pos[team_idx], team_size)
            reset_quat = sample_quat(yaw_center, team_size)
            if team_size == 1 and not isinstance(robot, TeamedRobot):
                reset_pos = reset_pos[:, 0, :]
                reset_quat = reset_quat[:, 0, :]
            robot.reset(envs_idx=envs_idx, reset_pos=reset_pos, reset_quat=reset_quat)

        self.model.reset(envs_idx=envs_idx)
        kwargs = self.get_state(envs_idx=envs_idx)
        return self.model.build_observation(envs_idx=envs_idx, **kwargs), self.model.build_info(envs_idx=envs_idx, **kwargs)

    def get_state(self, envs_idx: torch.Tensor) -> dict[str, object]: # type: ignore[override]
        robot_states = [robot.get_state(envs_idx=envs_idx) for robot in self.robots]
        listed_states: dict[str, list[torch.Tensor]] = {
            key: [robot_state[key] for robot_state in robot_states]
            for key in robot_states[0].keys()
        }
        return {**listed_states, **self.field.get_state(envs_idx=envs_idx)}
