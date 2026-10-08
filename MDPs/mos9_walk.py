"""500 control steps at 50 Hz, or terminate on a fall; sparse reward."""
from dataclasses import dataclass
import numpy as np
import torch
import genesis as gs
import gymnasium as gym
from .MDP import MDP, MDPConfig


@dataclass(kw_only=True)
class MOS9WalkConfig(MDPConfig):
    home: np.ndarray
    height: float
    max_steps: int = 500
    minimum_distance: float = 0.3
    reset_xy: float = 0.0
    reset_yaw: float = 0.0


class MOS9WalkMDP(MDP):
    def build(self):
        pass

    def config(self):
        self.home = torch.as_tensor(self.cfg.home, device=gs.device)
        self.steps = torch.zeros(self.scene.n_envs, dtype=torch.long, device=gs.device)
        self.origins = torch.zeros((self.scene.n_envs, 3), device=gs.device)
        self.yaws = torch.zeros(self.scene.n_envs, device=gs.device)
        self.last_action = self.home.repeat(self.scene.n_envs, 1)
        self.completed = []

    @property
    def action_space(self):
        return gym.spaces.Box(-3.0, 3.0, shape=(len(self.cfg.home),), dtype=np.float32)

    @property
    def observation_space(self):
        return gym.spaces.Box(-np.inf, np.inf, shape=(3*len(self.cfg.home)+14,), dtype=np.float32)

    def reset(self, envs_idx, robot_reset_fn, field_reset_fn):
        n = len(envs_idx)
        pos = torch.zeros((n, 3), device=gs.device)
        pos[:, :2] = (torch.rand((n, 2), device=gs.device)*2-1)*self.cfg.reset_xy
        pos[:, 2] = self.cfg.height
        yaw = (torch.rand(n, device=gs.device)*2-1)*self.cfg.reset_yaw
        quat = torch.stack((torch.cos(yaw/2), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw/2)), dim=1)
        self.steps[envs_idx] = 0
        self.origins[envs_idx] = pos
        self.yaws[envs_idx] = yaw
        self.last_action[envs_idx] = self.home
        robot_reset_fn(joint_pos=self.home.repeat(n, 1), reset_pos=pos, reset_quat=quat)
        field_reset_fn()

    def preprocess_action(self, action):
        self.last_action.copy_(action)
        self.steps += 1
        return action

    def build_observation(self, envs_idx, dofs_pos, dofs_vel, body_quat, body_lin_vel, body_ang_vel, body_pos, **kwargs):
        phase = self.steps[envs_idx].float()/self.cfg.max_steps
        return torch.cat((dofs_pos, dofs_vel, self.last_action[envs_idx], body_quat,
                          body_lin_vel, body_ang_vel, body_pos-self.origins[envs_idx], phase[:, None]), dim=1)

    def fallen(self, body_pos, body_quat):
        upright = 1-2*(body_quat[:, 1].square()+body_quat[:, 2].square())
        return (body_pos[:, 2] < self.cfg.height*0.6) | (upright < 0.65)

    def distance(self, envs_idx, body_pos):
        delta = body_pos-self.origins[envs_idx]
        return delta[:, 0]*torch.cos(self.yaws[envs_idx])+delta[:, 1]*torch.sin(self.yaws[envs_idx])

    def build_reward(self, envs_idx, body_pos, body_quat, **kwargs):
        fall = self.fallen(body_pos, body_quat)
        success = (self.steps[envs_idx] >= self.cfg.max_steps) & ~fall & (self.distance(envs_idx, body_pos) >= self.cfg.minimum_distance)
        return (success.float()-fall.float())[:, None]

    def build_terminated(self, envs_idx, body_pos, body_quat, **kwargs):
        return self.fallen(body_pos, body_quat)[:, None]

    def build_truncated(self, envs_idx, body_pos, body_quat, **kwargs):
        return ((self.steps[envs_idx] >= self.cfg.max_steps) & ~self.fallen(body_pos, body_quat))[:, None]

    def build_info(self, envs_idx, body_pos, body_quat, **kwargs):
        # Capture complete outcomes here, before shared Env auto-reset replaces info.
        fall = self.fallen(body_pos, body_quat)
        done = fall | (self.steps[envs_idx] >= self.cfg.max_steps)
        distance = self.distance(envs_idx, body_pos)
        for row in torch.nonzero(done).flatten().tolist():
            self.completed.append({"steps": int(self.steps[envs_idx[row]]), "fallen": bool(fall[row]),
                                   "env_id": int(envs_idx[row]),
                                   "distance": float(distance[row]),
                                   "success": bool(~fall[row] & (distance[row] >= self.cfg.minimum_distance))})
        return {}
