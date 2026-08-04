"""Sim2sim adapter for the exported 22-DoF soccer kick actor."""

from dataclasses import dataclass, field
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import genesis as gs

from .MDP import MDP, MDPConfig


@dataclass(kw_only=True)
class KickSim2SimConfig(MDPConfig):
    actor_path: str
    action_dim: int = 22
    ball_pos: np.ndarray = field(default_factory=lambda: np.array([0.8, 0.0, 0.07], dtype=np.float32))
    kick_dir_yaw: float = 0.0
    kick_speed: float = 1.0
    action_scale: np.ndarray = field(default_factory=lambda: np.array([
        0.09614232, 0.09810339, 0.15381525, 0.09810339, 0.15381525, 0.09614232,
        0.09810339, 0.15381525, 0.09810339, 0.15381525, 0.09810339, 0.15381525,
        0.09810339, 0.15381525, 0.09810339, 0.15381525, 0.09810339, 0.15381525,
        0.09810339, 0.09810339, 0.09810339, 0.09810339,
    ], dtype=np.float32))
    home_pose: np.ndarray = field(default_factory=lambda: np.array([
        0.0, -0.25, 0.0, -0.25, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.65, 0.0, 0.65, 0.0,
        -0.4, -0.4, 0.0, 0.0,
    ], dtype=np.float32))
    base_pos: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.351], dtype=np.float32))
    base_quat: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))


def _quat_rotate_inverse(q, v):
    w, x, y, z = q.unbind(-1)
    qv = torch.stack((-x, -y, -z), dim=-1)
    t = 2.0 * torch.cross(qv, v, dim=-1)
    return v + w.unsqueeze(-1) * t + torch.cross(qv, t, dim=-1)


def _yaw_quat(q):
    w, x, y, z = q.unbind(-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return torch.stack((torch.cos(yaw / 2), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw / 2)), dim=-1)


class KickSim2SimMDP(MDP):
    cfg: KickSim2SimConfig

    def build(self):
        pass

    def config(self):
        if self.cfg.action_dim != 22:
            raise ValueError("The supplied kick actor is the 22-action export")
        self._home_pose = torch.as_tensor(self.cfg.home_pose, device=gs.device)
        self._base_pos = torch.as_tensor(self.cfg.base_pos, device=gs.device)
        self._base_quat = torch.as_tensor(self.cfg.base_quat, device=gs.device)
        self._ball_pos = torch.as_tensor(self.cfg.ball_pos, device=gs.device)
        self._scale = torch.as_tensor(self.cfg.action_scale, device=gs.device)
        self._last_action = torch.zeros((self.scene.n_envs, 22), device=gs.device)
        self._cmd_hist = torch.zeros((self.scene.n_envs, 10, 7), device=gs.device)
        self._ball_hist = torch.zeros((self.scene.n_envs, 10, 2), device=gs.device)
        self._history_valid = torch.zeros(self.scene.n_envs, dtype=torch.bool, device=gs.device)
        self._observation_valid = torch.zeros(self.scene.n_envs, dtype=torch.bool, device=gs.device)
        self._observation_cache = torch.zeros((self.scene.n_envs, 646), device=gs.device)
        self._all_idx = torch.arange(self.scene.n_envs, device=gs.device)
        self._load_actor()

    def _load_actor(self):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError("Install onnxruntime to run kick sim2sim") from exc
        path = Path(self.cfg.actor_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        self._ort = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        inp = self._ort.get_inputs()[0]
        if inp.shape[-1] != 646 or self._ort.get_outputs()[0].shape[-1] != 22:
            raise ValueError("Expected the 646-D / 22-action encdec kick actor")
        self._actor_input = inp.name

    def reset(self, envs_idx, robot_reset_fn, field_reset_fn):
        n = envs_idx.shape[0]
        robot_reset_fn(joint_pos=self._home_pose.broadcast_to((n, 22)), reset_pos=self._base_pos.broadcast_to((n, 3)), reset_quat=self._base_quat.broadcast_to((n, 4)))
        field_reset_fn(ball_pos=self._ball_pos.broadcast_to((n, 3)))
        self._last_action[envs_idx] = 0
        self._cmd_hist[envs_idx] = 0
        self._ball_hist[envs_idx] = 0
        self._history_valid[envs_idx] = False
        self._observation_valid[envs_idx] = False

    @property
    def observation_space(self):
        return gym.spaces.Box(-np.inf, np.inf, shape=(646,), dtype=np.float32)

    @property
    def action_space(self):
        return gym.spaces.Box(-1.0, 1.0, shape=(22,), dtype=np.float32)

    def preprocess_action(self, action):
        action = torch.as_tensor(action, device=gs.device, dtype=torch.float32)
        self._last_action = action.detach().clone()
        return action * self._scale + self._home_pose

    def _command(self, body_pos, body_quat, ball_pos, envs_idx=None):
        ball_b = _quat_rotate_inverse(_yaw_quat(body_quat), ball_pos - body_pos)
        w, x, y, z = body_quat.unbind(-1)
        yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        kick_dir = torch.remainder(self.cfg.kick_dir_yaw - yaw + np.pi, 2 * np.pi) - np.pi
        return torch.cat((torch.zeros_like(ball_b), ball_b[:, :2], kick_dir[:, None], torch.full_like(kick_dir[:, None], self.cfg.kick_speed)), dim=-1)

    def build_observation(self, envs_idx, body_ang_vel=None, body_quat=None, dofs_pos=None, dofs_vel=None, ball_pos=None, body_pos=None, **kwargs):
        if body_quat is None:
            return torch.zeros((envs_idx.shape[0], 646), device=gs.device)
        cmd = self._command(body_pos, body_quat, ball_pos, envs_idx=envs_idx)
        fresh = ~self._history_valid[envs_idx]
        if fresh.any():
            fresh_idx = envs_idx[fresh]
            self._cmd_hist[fresh_idx] = cmd[fresh].unsqueeze(1)
            self._ball_hist[fresh_idx] = cmd[fresh, 3:5].unsqueeze(1)
        if (~fresh).any():
            live_idx = envs_idx[~fresh]
            self._cmd_hist[live_idx] = torch.roll(self._cmd_hist[live_idx], -1, 1)
            self._cmd_hist[live_idx, -1] = cmd[~fresh]
            self._ball_hist[live_idx] = torch.roll(self._ball_hist[live_idx], -1, 1)
            self._ball_hist[live_idx, -1] = cmd[~fresh, 3:5]
        self._history_valid[envs_idx] = True
        gravity = _quat_rotate_inverse(body_quat, torch.tensor([0., 0., -1.], device=gs.device).expand_as(body_quat[:, :3]))
        cmd_hist = self._cmd_hist[envs_idx]
        ball_hist = self._ball_hist[envs_idx]
        last_action = self._last_action[envs_idx]
        parts = [(body_ang_vel * 0.25).unsqueeze(1).expand(-1, 8, -1), gravity.unsqueeze(1).expand(-1, 8, -1), cmd_hist[:, :, [0, 1, 2, 5, 6]], (dofs_pos - self._home_pose).unsqueeze(1).expand(-1, 8, -1), (dofs_vel * 0.05).unsqueeze(1).expand(-1, 8, -1), last_action.unsqueeze(1).expand(-1, 8, -1), ball_hist]
        observation = torch.cat([x.reshape(x.shape[0], -1) for x in parts], dim=-1)
        self._observation_cache[envs_idx] = observation
        self._observation_valid[envs_idx] = True
        return observation

    def policy_action(self, **state):
        if not self._observation_valid.all():
            obs = self.build_observation(self._all_idx, **state)
        else:
            obs = self._observation_cache
        return self._run_actor(obs)

    def _run_actor(self, observation):
        values = observation.detach().cpu().numpy().astype(np.float32)
        input_batch = self._ort.get_inputs()[0].shape[0]
        if input_batch in (None, "None") or input_batch == "batch":
            output = self._ort.run(None, {self._actor_input: values})[0]
        elif input_batch == 1 and values.shape[0] != 1:
            output = np.concatenate([
                self._ort.run(None, {self._actor_input: values[i:i + 1]})[0]
                for i in range(values.shape[0])
            ], axis=0)
        else:
            output = self._ort.run(None, {self._actor_input: values})[0]
        return torch.as_tensor(output, device=gs.device)

    def build_reward(self, envs_idx, **kwargs):
        return torch.zeros((envs_idx.shape[0], 1), device=gs.device)

    def build_terminated(self, envs_idx, **kwargs):
        return torch.zeros((envs_idx.shape[0], 1), dtype=torch.bool, device=gs.device)

    def build_truncated(self, envs_idx, **kwargs):
        return torch.zeros((envs_idx.shape[0], 1), dtype=torch.bool, device=gs.device)

    def build_info(self, envs_idx, **kwargs):
        return {}
