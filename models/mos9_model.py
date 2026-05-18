import torch
import numpy as np
import genesis as gs
import gymnasium as gym
from dataclasses import dataclass

from .model import Model, ModelConfig


@dataclass(kw_only=True)
class MOS9ModelConfig(ModelConfig):
    n_dofs: int
    target_q_offset: np.ndarray
    action_scale: float = 0.25
    step_dt: float = 0.02
    timeout: float = 20.0
    field_range: float = 1.0
    omega: float = 2.0
    obs_clip: tuple = (-1.0, 1.0)
    history_frames: int = 10


class MOS9Model(Model):
    cfg: MOS9ModelConfig

    def config(self):
        self.target_q_offset = torch.from_numpy(self.cfg.target_q_offset).to(gs.device)
        self.ewma_action = torch.zeros((self.scene.n_envs, self.cfg.n_dofs), device=gs.device, dtype=torch.float)
        self.last_action = torch.zeros((self.scene.n_envs, self.cfg.n_dofs), device=gs.device, dtype=torch.float)
        self.episode_time = torch.zeros((self.scene.n_envs, 1), device=gs.device, dtype=torch.float)
        self._log: dict[str, torch.Tensor] = {}
        self.last_obs = torch.zeros(
            (self.scene.n_envs, self.cfg.history_frames, self.dim_observations),
            device=gs.device,
            dtype=torch.float,
        )

        # [qpos, qvel, ang_vel, projected_gravity, prev_action, cmd, sin, cos]
        self.dim_observations = self.cfg.n_dofs + self.cfg.n_dofs + 3 + 3 + self.cfg.n_dofs + 3 + 1 + 1

    @property
    def observation_space(self) -> gym.spaces.Box:
        return gym.spaces.Box(
            low=self.cfg.obs_clip[0],
            high=self.cfg.obs_clip[1],
            shape=(self.dim_observations * self.cfg.history_frames,),
            dtype=np.float32,
        )

    @property
    def action_space(self) -> gym.spaces.Box:
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(self.cfg.n_dofs,), dtype=np.float32)

    def reset(self, envs_idx: torch.Tensor):
        self.ewma_action[envs_idx] = 0.0
        self.last_action[envs_idx] = 0.0
        self.episode_time[envs_idx] = 0.0
        self.last_obs[envs_idx] = 0.0

    def preprocess_action(self, action: torch.Tensor) -> torch.Tensor:
        self.last_action.copy_(action)
        self.ewma_action = action * self.cfg.action_scale
        self.episode_time += self.cfg.step_dt
        return self.target_q_offset + self.ewma_action

    @torch.compile()
    def _project_gravity(self, quat: torch.Tensor) -> torch.Tensor:
        w = quat[:, 0:1]
        q_vec = quat[:, 1:]
        v = torch.tensor([0.0, 0.0, -1.0], device=quat.device)
        a = torch.cross(q_vec, v.expand_as(q_vec), dim=-1) + w * v
        return v + 2.0 * torch.cross(q_vec, a, dim=-1)

    def build_observation(self, 
                          envs_idx, 
                          dofs_pos, 
                          dofs_vel, 
                          body_ang_vel, 
                          body_quat, 
                          cmd_vel, 
                          **kwargs): # type: ignore
        projected_gravity = self._project_gravity(body_quat)
        obs_sin = torch.sin(self.cfg.omega * self.episode_time[envs_idx])
        obs_cos = torch.cos(self.cfg.omega * self.episode_time[envs_idx])
        obs_single = torch.cat(
            (
                dofs_pos,
                dofs_vel,
                body_ang_vel,
                projected_gravity,
                self.last_action[envs_idx],
                cmd_vel,
                obs_sin,
                obs_cos,
            ),
            dim=1,
        )
        obs_single = torch.clamp(obs_single, self.cfg.obs_clip[0], self.cfg.obs_clip[1])
        self.last_obs[envs_idx] = torch.roll(self.last_obs[envs_idx], shifts=-1, dims=1)
        self.last_obs[envs_idx, -1, :] = obs_single
        return self.last_obs[envs_idx].reshape(envs_idx.shape[0], -1)

    def build_reward(
        self,
        envs_idx,
        body_pos: torch.Tensor,
        body_lin_vel: torch.Tensor,
        body_quat: torch.Tensor,
        cmd_vel: torch.Tensor,
        body_ang_vel: torch.Tensor,
        dofs_vel: torch.Tensor,
        net_contact_force: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor: # type: ignore
        lin_vel_error = torch.sum(torch.square(cmd_vel[:, :2] - body_lin_vel[:, :2]), dim=1)
        rew_lin_vel = torch.exp(-lin_vel_error / 0.25)

        yaw_rate_error = torch.square(cmd_vel[:, 2] - body_ang_vel[:, 2])
        rew_yaw = torch.exp(-yaw_rate_error / 0.25)

        tilt = torch.sum(torch.square(body_quat[:, 1:3]), dim=1)
        rew_upright = torch.exp(-tilt / 0.05)

        height_error = torch.square(body_pos[:, 2] - 0.48)
        rew_height = torch.exp(-height_error / 0.02)

        dof_speed = torch.mean(torch.square(dofs_vel), dim=1)
        rew_smooth = torch.exp(-dof_speed / 25.0)

        peak_contact = net_contact_force.max(dim=1).values
        collision_penalty_raw = torch.clamp((peak_contact - 3000.0) / 7000.0, min=0.0, max=1.0)
        upright_gate = (rew_upright > 0.8).float()
        collision_penalty = collision_penalty_raw * upright_gate

        reward = (
            1.20 * rew_lin_vel
            + 0.30 * rew_yaw
            + 0.60 * rew_upright
            + 0.40 * rew_height
            + 0.20 * rew_smooth
            - 0.50 * collision_penalty
            + 0.20
        )
        reward = torch.clamp(reward, min=0.0, max=3.0).unsqueeze(1)

        self._log = {
            "rew/lin_vel": rew_lin_vel.unsqueeze(1),
            "rew/yaw": rew_yaw.unsqueeze(1),
            "rew/upright": rew_upright.unsqueeze(1),
            "rew/height": rew_height.unsqueeze(1),
            "rew/smooth": rew_smooth.unsqueeze(1),
            "rew/contact_peak": peak_contact.unsqueeze(1),
            "rew/collision_penalty_raw": collision_penalty_raw.unsqueeze(1),
            "rew/collision_penalty": collision_penalty.unsqueeze(1),
            "rew/total": reward,
        }
        return reward

    def build_terminated(self, 
                         envs_idx, 
                         body_pos: torch.Tensor, 
                         body_quat: torch.Tensor, 
                         **kwargs) -> torch.Tensor: # type: ignore
        fallen = body_pos[:, 2] < 0.25
        tilted = torch.sum(torch.square(body_quat[:, 1:3]), dim=1) > 0.35
        terminated = (fallen | tilted).unsqueeze(1)
        self._log["terminated/fallen"] = fallen.float().unsqueeze(1)
        self._log["terminated/tilted"] = tilted.float().unsqueeze(1)
        return terminated

    def build_truncated(self, 
                        envs_idx, 
                        body_pos: torch.Tensor, 
                        **kwargs) -> torch.Tensor: # type: ignore
        out_of_field = (torch.abs(body_pos[:, :2]) > self.cfg.field_range).any(dim=1)
        out_of_time = self.episode_time[envs_idx].squeeze(1) > self.cfg.timeout
        return torch.logical_or(out_of_field, out_of_time).unsqueeze(1)

    @torch.compiler.disable
    def build_info(self, envs_idx, **kwargs) -> dict[str, dict[str, torch.Tensor]]:
        extra = {f"Reward / {k}": v.detach().mean().cpu() for k, v in self._log.items()}
        return {"extra": extra}
