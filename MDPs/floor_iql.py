"""Sparse-reward floor kick task and its pretrained teacher actor."""

from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
import torch
import genesis as gs

from .kick_sim2sim import KickSim2SimConfig, KickSim2SimMDP


@dataclass(kw_only=True)
class FloorIQLConfig(KickSim2SimConfig):
    ball_pos: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.07], dtype=np.float32))
    kick_dir_yaw: float = 0.0
    success_delta_x: float = 0.1
    max_steps: int = 350
    step_penalty: float = -0.1


class FloorIQLMDP(KickSim2SimMDP):
    """Teacher-compatible task with an explicit observation state machine.

    The ONNX teacher still receives its original 646-D input.  A learner-facing
    observation appends a three-way phase one-hot: ready, kicking, terminal.
    Histories advance only in ``build_observation`` after physics; teacher
    inference reads the cached current observation and does not advance them.
    """

    cfg: FloorIQLConfig

    def config(self):
        super().config()
        self._episode_ball_x = torch.zeros(self.scene.n_envs, device=gs.device)
        self._step_count = torch.zeros(self.scene.n_envs, dtype=torch.long, device=gs.device)
        self._phase = torch.zeros(self.scene.n_envs, dtype=torch.long, device=gs.device)
        self._floor_observation_cache = torch.zeros((self.scene.n_envs, 649), device=gs.device)

    def reset(self, envs_idx, robot_reset_fn, field_reset_fn):
        super().reset(envs_idx, robot_reset_fn, field_reset_fn)
        self._episode_ball_x[envs_idx] = self.cfg.ball_pos[0]
        self._step_count[envs_idx] = 0
        self._phase[envs_idx] = 0

    @property
    def observation_space(self):
        return gym.spaces.Box(-np.inf, np.inf, shape=(649,), dtype=np.float32)

    def _update_phase(self, envs_idx, ball_pos):
        delta_x = ball_pos[:, 0] - self._episode_ball_x[envs_idx]
        success = delta_x > self.cfg.success_delta_x
        timeout = self._step_count[envs_idx] >= self.cfg.max_steps
        self._phase[envs_idx] = torch.where(success | timeout, torch.full_like(self._phase[envs_idx], 2), torch.where(self._step_count[envs_idx] > 0, torch.ones_like(self._phase[envs_idx]), torch.zeros_like(self._phase[envs_idx])))

    def build_observation(self, envs_idx, ball_pos=None, **kwargs):
        core = super().build_observation(envs_idx, ball_pos=ball_pos, **kwargs)
        self._update_phase(envs_idx, ball_pos)
        phase = torch.nn.functional.one_hot(self._phase[envs_idx], num_classes=3).to(torch.float32)
        observation = torch.cat((core, phase), dim=-1)
        self._floor_observation_cache[envs_idx] = observation
        return observation

    def policy_action(self, **state):
        if not self._observation_valid.all():
            self.build_observation(self._all_idx, **state)
        # The pretrained actor was exported before the floor phase feature.
        output = self._ort.run(None, {self._actor_input: self._observation_cache.detach().cpu().numpy().astype(np.float32)})[0]
        return torch.as_tensor(output, device=gs.device)

    def _success(self, envs_idx, ball_pos):
        return (ball_pos[:, 0] - self._episode_ball_x[envs_idx]) > self.cfg.success_delta_x

    def build_reward(self, envs_idx, ball_pos=None, **kwargs):
        success = self._success(envs_idx, ball_pos)
        self._step_count[envs_idx] += 1
        return torch.where(success, torch.ones_like(success, dtype=torch.float32), torch.full_like(success, self.cfg.step_penalty, dtype=torch.float32)).unsqueeze(1)

    def build_terminated(self, envs_idx, ball_pos=None, **kwargs):
        return self._success(envs_idx, ball_pos).unsqueeze(1)

    def build_truncated(self, envs_idx, ball_pos=None, **kwargs):
        timeout = (self._step_count[envs_idx] >= self.cfg.max_steps) & ~self._success(envs_idx, ball_pos)
        return timeout.unsqueeze(1)

    def build_info(self, envs_idx, ball_pos=None, **kwargs):
        success = self._success(envs_idx, ball_pos)
        return {
            "success": success,
            "delta_x": ball_pos[:, 0] - self._episode_ball_x[envs_idx],
            "phase": self._phase[envs_idx],
        }
