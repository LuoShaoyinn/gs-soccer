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
    # 350 * (-1 / 350) = -1.0 on a timeout episode.
    step_penalty: float = -1.0 / 350.0
    # Match the source task's root-height termination threshold.
    fall_height: float = 0.1
    robot_x_randomization: float = 0.05
    robot_y_randomization: float = 0.05
    robot_yaw_randomization: float = 0.1745329252  # 10 degrees


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
        self._episode_ball_start = torch.zeros((self.scene.n_envs, 3), device=gs.device)
        self._episode_forward = torch.zeros((self.scene.n_envs, 2), device=gs.device)
        self._episode_yaw = torch.zeros(self.scene.n_envs, device=gs.device)
        self._step_count = torch.zeros(self.scene.n_envs, dtype=torch.long, device=gs.device)
        self._phase = torch.zeros(self.scene.n_envs, dtype=torch.long, device=gs.device)
        self._episode_return = torch.zeros(self.scene.n_envs, device=gs.device)
        self._floor_observation_cache = torch.zeros((self.scene.n_envs, 649), device=gs.device)

    def reset(self, envs_idx, robot_reset_fn, field_reset_fn):
        n = envs_idx.shape[0]
        device = self._episode_yaw.device
        x = (2.0 * torch.rand(n, device=device) - 1.0) * self.cfg.robot_x_randomization
        y = (2.0 * torch.rand(n, device=device) - 1.0) * self.cfg.robot_y_randomization
        yaw = (2.0 * torch.rand(n, device=device) - 1.0) * self.cfg.robot_yaw_randomization
        robot_pos = self._base_pos.broadcast_to((n, 3)).clone()
        robot_pos[:, 0] += x
        robot_pos[:, 1] += y
        robot_quat = torch.stack((torch.cos(yaw / 2), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw / 2)), dim=-1)
        ball_distance = float(self.cfg.ball_pos[0])
        ball_pos = robot_pos + ball_distance * torch.stack((torch.cos(yaw), torch.sin(yaw), torch.zeros_like(yaw)), dim=-1)
        ball_pos[:, 2] = float(self.cfg.ball_pos[2])
        robot_reset_fn(joint_pos=self._home_pose.broadcast_to((n, 22)), reset_pos=robot_pos, reset_quat=robot_quat)
        field_reset_fn(ball_pos=ball_pos)
        self._episode_ball_start[envs_idx] = ball_pos
        self._episode_forward[envs_idx] = torch.stack((torch.cos(yaw), torch.sin(yaw)), dim=-1)
        self._episode_yaw[envs_idx] = yaw
        self._episode_ball_x[envs_idx] = ball_pos[:, 0]
        self._step_count[envs_idx] = 0
        self._phase[envs_idx] = 0
        self._episode_return[envs_idx] = 0.0

        self._last_action[envs_idx] = 0
        self._cmd_hist[envs_idx] = 0
        self._ball_hist[envs_idx] = 0
        self._ball_pub_counter[envs_idx] = 0
        self._history_valid[envs_idx] = False
        self._observation_valid[envs_idx] = False

    @property
    def observation_space(self):
        return gym.spaces.Box(-np.inf, np.inf, shape=(649,), dtype=np.float32)

    def _update_phase(self, envs_idx, ball_pos, body_pos=None):
        delta_x = self._delta_x(envs_idx, ball_pos)
        success = delta_x > self.cfg.success_delta_x
        timeout = self._step_count[envs_idx] >= self.cfg.max_steps
        fallen = self._fallen(envs_idx, body_pos)
        self._phase[envs_idx] = torch.where(success | timeout | fallen, torch.full_like(self._phase[envs_idx], 2), torch.where(self._step_count[envs_idx] > 0, torch.ones_like(self._phase[envs_idx]), torch.zeros_like(self._phase[envs_idx])))

    def build_observation(self, envs_idx, ball_pos=None, body_pos=None, **kwargs):
        core = super().build_observation(envs_idx, ball_pos=ball_pos, body_pos=body_pos, **kwargs)
        self._update_phase(envs_idx, ball_pos, body_pos)
        phase = torch.nn.functional.one_hot(self._phase[envs_idx], num_classes=3).to(torch.float32)
        observation = torch.cat((core, phase), dim=-1)
        self._floor_observation_cache[envs_idx] = observation
        return observation

    def policy_action(self, **state):
        if not self._observation_valid.all():
            self.build_observation(self._all_idx, **state)
        # The pretrained actor was exported before the floor phase feature.
        return self._run_actor(self._observation_cache)

    def _success(self, envs_idx, ball_pos):
        return self._delta_x(envs_idx, ball_pos) > self.cfg.success_delta_x

    def _fallen(self, envs_idx, body_pos):
        if body_pos is None:
            return torch.zeros(envs_idx.shape[0], dtype=torch.bool, device=gs.device)
        return body_pos[:, 2] < self.cfg.fall_height

    def _delta_x(self, envs_idx, ball_pos):
        displacement = ball_pos - self._episode_ball_start[envs_idx]
        return (displacement[:, :2] * self._episode_forward[envs_idx]).sum(dim=-1)

    def _command(self, body_pos, body_quat, ball_pos, envs_idx=None):
        """Build the teacher command with forward kick direction per episode."""
        ball_b = super()._command(body_pos, body_quat, ball_pos)
        w, x, y, z = body_quat.unbind(-1)
        yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        if envs_idx is None:
            envs_idx = self._all_idx
        kick_dir = torch.remainder(self._episode_yaw[envs_idx] + self.cfg.kick_dir_yaw - yaw + np.pi, 2 * np.pi) - np.pi
        ball_b[:, 5] = kick_dir
        return ball_b

    def build_reward(self, envs_idx, ball_pos=None, body_pos=None, **kwargs):
        success = self._success(envs_idx, ball_pos)
        fallen = self._fallen(envs_idx, body_pos) & ~success
        self._step_count[envs_idx] += 1
        timeout_reward = torch.full_like(success, self.cfg.step_penalty, dtype=torch.float32)
        # A fall is terminal, but it represents the same failed outcome as a
        # full timeout: charge every not-yet-issued step penalty immediately.
        remaining_penalty = self.cfg.step_penalty * (self.cfg.max_steps - self._step_count[envs_idx] + 1).to(torch.float32)
        reward = torch.where(success, torch.ones_like(timeout_reward), torch.where(fallen, remaining_penalty, timeout_reward))
        self._episode_return[envs_idx] += reward
        self._update_phase(envs_idx, ball_pos, body_pos)
        return reward.unsqueeze(1)

    def build_terminated(self, envs_idx, ball_pos=None, body_pos=None, **kwargs):
        return (self._success(envs_idx, ball_pos) | self._fallen(envs_idx, body_pos)).unsqueeze(1)

    def build_truncated(self, envs_idx, ball_pos=None, body_pos=None, **kwargs):
        timeout = (self._step_count[envs_idx] >= self.cfg.max_steps) & ~self._success(envs_idx, ball_pos) & ~self._fallen(envs_idx, body_pos)
        return timeout.unsqueeze(1)

    def build_info(self, envs_idx, ball_pos=None, body_pos=None, **kwargs):
        success = self._success(envs_idx, ball_pos)
        fallen = self._fallen(envs_idx, body_pos) & ~success
        timeout = (self._step_count[envs_idx] >= self.cfg.max_steps) & ~success & ~fallen
        return {
            "success": success,
            "fallen": fallen,
            # Env.step auto-resets terminal rows, making a fall the explicit
            # stop point and the following reset the recovery transition.
            "recovery": fallen,
            "timeout": timeout,
            "delta_x": self._delta_x(envs_idx, ball_pos),
            "phase": self._phase[envs_idx],
            "step_count": self._step_count[envs_idx],
            "episode_return": self._episode_return[envs_idx],
        }
