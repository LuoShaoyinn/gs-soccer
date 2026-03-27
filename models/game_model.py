#   game_model.py
#       game model for soccer 1v1 env
#
import torch
import numpy as np
import genesis as gs
import gymnasium as gym
from dataclasses import dataclass
from models.model import Model, ModelConfig


@dataclass(kw_only=True)
class GameModelConfig(ModelConfig):
    half_field_size: tuple[float, float] = (4.5, 3.0)
    goal_width: float = 1.9

    timeout_steps_limit: int = 600
    action_scale: float = 2.0
    reward_goal: float = 8.0
    reward_concede: float = -8.0
    reward_ball_progress: float = 1.5
    reward_possession: float = 0.4
    reward_smooth: float = 0.05
    out_of_field_reward_scale: float = 0.4
    fall_height: float = 0.28


class GameModel(Model):
    cfg: GameModelConfig

    def config(self):
        num_envs = self.scene.n_envs
        self.num_agents = 2
        self.time_steps = torch.zeros((num_envs, 1), dtype=torch.float, device=gs.device)
        self.cmd_vel = [torch.zeros((num_envs, 2, 3), dtype=torch.float, device=gs.device) for _ in range(self.num_agents)]

        half = torch.tensor(self.cfg.half_field_size, dtype=torch.float, device=gs.device)
        self.half_field_size = half.view(1, 2).broadcast_to((num_envs, 2))
        self.goal_width = torch.tensor([self.cfg.goal_width], dtype=torch.float, device=gs.device).view(1, 1).broadcast_to((num_envs, 1))
        self.goal_pos = [
            torch.tensor([self.cfg.half_field_size[0], 0.0], dtype=torch.float, device=gs.device).view(1, 2).broadcast_to((num_envs, 2)),
            torch.tensor([-self.cfg.half_field_size[0], 0.0], dtype=torch.float, device=gs.device).view(1, 2).broadcast_to((num_envs, 2)),
        ]

        self.cache_valid = torch.zeros((num_envs,), dtype=torch.bool, device=gs.device)
        self.cache: dict[str, torch.Tensor] = {
            "team0_pos_2d": torch.zeros((num_envs, 2), dtype=torch.float, device=gs.device),
            "team1_pos_2d": torch.zeros((num_envs, 2), dtype=torch.float, device=gs.device),
            "team0_vel_2d": torch.zeros((num_envs, 2), dtype=torch.float, device=gs.device),
            "team1_vel_2d": torch.zeros((num_envs, 2), dtype=torch.float, device=gs.device),
            "ball_pos_2d": torch.zeros((num_envs, 2), dtype=torch.float, device=gs.device),
            "ball_vel_2d": torch.zeros((num_envs, 2), dtype=torch.float, device=gs.device),
            "team0_heading": torch.zeros((num_envs, 2), dtype=torch.float, device=gs.device),
            "team1_heading": torch.zeros((num_envs, 2), dtype=torch.float, device=gs.device),
            "goal_team0": torch.zeros((num_envs, 1), dtype=torch.bool, device=gs.device),
            "goal_team1": torch.zeros((num_envs, 1), dtype=torch.bool, device=gs.device),
            "ball_out": torch.zeros((num_envs, 1), dtype=torch.bool, device=gs.device),
            "terminated": torch.zeros((num_envs, 1), dtype=torch.bool, device=gs.device),
            "out_reward_team0": torch.zeros((num_envs, 1), dtype=torch.float, device=gs.device),
            "out_reward_team1": torch.zeros((num_envs, 1), dtype=torch.float, device=gs.device),
        }
        self.rewards: list[dict[str, torch.Tensor]] = [{}, {}]

    def reset(self, envs_idx: torch.Tensor):
        self.cache_valid[envs_idx] = False
        self.time_steps[envs_idx] = 0.0
        for idx in range(self.num_agents):
            self.cmd_vel[idx][envs_idx] = 0.0

    def preprocess_action(self, action: list[torch.Tensor]
                          ) -> list[torch.Tensor]: # type: ignore[override]
        self.cache_valid[:] = False
        processed: list[torch.Tensor] = []
        for idx in range(self.num_agents):
            self.cmd_vel[idx] = torch.roll(self.cmd_vel[idx], shifts=-1, dims=1)
            self.cmd_vel[idx][:, -1, :] = torch.clamp(action[idx], -1.0, 1.0)
            processed.append(self.cmd_vel[idx].mean(dim=1) * self.cfg.action_scale)
        self.time_steps += 1.0
        return processed

    @property
    def observation_space(self) -> list[gym.spaces.Box]:
        space = gym.spaces.Box(low=-10.0, high=10.0, shape=(18,), dtype=np.float32)
        return [space, space]

    @property
    def action_space(self) -> list[gym.spaces.Box]:
        space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        return [space, space]

    def _heading2d(self, quat: torch.Tensor) -> torch.Tensor:
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        vx = 1.0 - 2.0 * (y * y + z * z)
        vy = 2.0 * (x * y + w * z)
        norm = torch.sqrt(vx * vx + vy * vy + 1e-8)
        return torch.stack([vx / norm, vy / norm], dim=-1)

    def _team_main(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 3:
            return tensor[:, 0, :]
        return tensor

    def _build_obs_from_perspective(
        self,
        self_pos_2d: torch.Tensor,
        self_vel_2d: torch.Tensor,
        self_heading: torch.Tensor,
        self_ang_vel: torch.Tensor,
        opp_pos_2d: torch.Tensor,
        opp_vel_2d: torch.Tensor,
        ball_pos_2d: torch.Tensor,
        ball_vel_2d: torch.Tensor,
        attack_goal_pos: torch.Tensor,
        last_cmd: torch.Tensor,
    ) -> torch.Tensor:
        opp_rel_pos = opp_pos_2d - self_pos_2d
        opp_rel_vel = opp_vel_2d - self_vel_2d
        ball_rel_pos = ball_pos_2d - self_pos_2d
        ball_rel_vel = ball_vel_2d - self_vel_2d
        ball_to_goal = attack_goal_pos - ball_pos_2d
        return torch.cat(
            [
                self_heading,
                self_vel_2d,
                self_ang_vel[:, 2:3],
                opp_rel_pos,
                opp_rel_vel,
                ball_rel_pos,
                ball_rel_vel,
                ball_to_goal,
                last_cmd,
            ],
            dim=1,
        )

    def _build_cache(
        self,
        envs_idx: torch.Tensor,
        body_pos: list[torch.Tensor],
        body_quat: list[torch.Tensor],
        body_lin_vel: list[torch.Tensor],
        body_ang_vel: list[torch.Tensor],
        ball_pos: torch.Tensor,
        ball_vel: torch.Tensor,
        **kwargs,
    ) -> None:
        if self.cache_valid[envs_idx].all():
            return
        self.cache_valid[envs_idx] = True

        team0_pos = self._team_main(body_pos[0])
        team1_pos = self._team_main(body_pos[1])
        team0_quat = self._team_main(body_quat[0])
        team1_quat = self._team_main(body_quat[1])
        team0_lin_vel = self._team_main(body_lin_vel[0])
        team1_lin_vel = self._team_main(body_lin_vel[1])

        ball_pos_2d = ball_pos[:, 0:2]
        ball_vel_2d = ball_vel[:, 0:2]

        self.cache["team0_pos_2d"][envs_idx] = team0_pos[:, 0:2]
        self.cache["team1_pos_2d"][envs_idx] = team1_pos[:, 0:2]
        self.cache["team0_vel_2d"][envs_idx] = team0_lin_vel[:, 0:2]
        self.cache["team1_vel_2d"][envs_idx] = team1_lin_vel[:, 0:2]
        self.cache["ball_pos_2d"][envs_idx] = ball_pos_2d
        self.cache["ball_vel_2d"][envs_idx] = ball_vel_2d
        self.cache["team0_heading"][envs_idx] = self._heading2d(team0_quat)
        self.cache["team1_heading"][envs_idx] = self._heading2d(team1_quat)

        half_x = self.half_field_size[envs_idx, 0:1]
        half_y = self.half_field_size[envs_idx, 1:2]
        goal_half = 0.5 * self.goal_width[envs_idx]

        goal_team0 = (ball_pos_2d[:, 0:1] >= half_x) & (torch.abs(ball_pos_2d[:, 1:2]) <= goal_half)
        goal_team1 = (ball_pos_2d[:, 0:1] <= -half_x) & (torch.abs(ball_pos_2d[:, 1:2]) <= goal_half)
        is_outside = (torch.abs(ball_pos_2d[:, 0:1]) > half_x) | (torch.abs(ball_pos_2d[:, 1:2]) > half_y)

        ball_out = is_outside & ~(goal_team0 | goal_team1)
        self.cache["goal_team0"][envs_idx] = goal_team0
        self.cache["goal_team1"][envs_idx] = goal_team1
        self.cache["ball_out"][envs_idx] = ball_out

        out_point = torch.stack(
            [
                torch.clamp(ball_pos_2d[:, 0], min=-half_x[:, 0], max=half_x[:, 0]),
                torch.clamp(ball_pos_2d[:, 1], min=-half_y[:, 0], max=half_y[:, 0]),
            ],
            dim=1,
        )
        max_dist = torch.norm(2.0 * self.half_field_size[envs_idx], dim=1, keepdim=True).clamp_min(1e-6)
        out_reward_team0 = self.cfg.out_of_field_reward_scale * (
            1.0 - torch.norm(out_point - self.goal_pos[0][envs_idx], dim=1, keepdim=True) / max_dist
        )
        out_reward_team1 = self.cfg.out_of_field_reward_scale * (
            1.0 - torch.norm(out_point - self.goal_pos[1][envs_idx], dim=1, keepdim=True) / max_dist
        )
        self.cache["out_reward_team0"][envs_idx] = torch.where(ball_out, out_reward_team0, torch.zeros_like(out_reward_team0))
        self.cache["out_reward_team1"][envs_idx] = torch.where(ball_out, out_reward_team1, torch.zeros_like(out_reward_team1))

        team0_fall = team0_pos[:, 2:3] < self.cfg.fall_height
        team1_fall = team1_pos[:, 2:3] < self.cfg.fall_height
        timeout = self.time_steps[envs_idx] >= self.cfg.timeout_steps_limit
        self.cache["terminated"][envs_idx] = goal_team0 | goal_team1 | ball_out | team0_fall | team1_fall | timeout

    def build_observation(self, envs_idx: torch.Tensor, **kwargs
                          ) -> list[torch.Tensor]:  # type: ignore[override]
        self._build_cache(envs_idx=envs_idx, **kwargs)
        obs_team0 = self._build_obs_from_perspective(
            self_pos_2d=self.cache["team0_pos_2d"][envs_idx],
            self_vel_2d=self.cache["team0_vel_2d"][envs_idx],
            self_heading=self.cache["team0_heading"][envs_idx],
            self_ang_vel=self._team_main(kwargs["body_ang_vel"][0]),
            opp_pos_2d=self.cache["team1_pos_2d"][envs_idx],
            opp_vel_2d=self.cache["team1_vel_2d"][envs_idx],
            ball_pos_2d=self.cache["ball_pos_2d"][envs_idx],
            ball_vel_2d=self.cache["ball_vel_2d"][envs_idx],
            attack_goal_pos=self.goal_pos[0][envs_idx],
            last_cmd=self.cmd_vel[0][envs_idx, -1, :],
        )
        obs_team1 = self._build_obs_from_perspective(
            self_pos_2d=self.cache["team1_pos_2d"][envs_idx],
            self_vel_2d=self.cache["team1_vel_2d"][envs_idx],
            self_heading=self.cache["team1_heading"][envs_idx],
            self_ang_vel=self._team_main(kwargs["body_ang_vel"][1]),
            opp_pos_2d=self.cache["team0_pos_2d"][envs_idx],
            opp_vel_2d=self.cache["team0_vel_2d"][envs_idx],
            ball_pos_2d=self.cache["ball_pos_2d"][envs_idx],
            ball_vel_2d=self.cache["ball_vel_2d"][envs_idx],
            attack_goal_pos=self.goal_pos[1][envs_idx],
            last_cmd=self.cmd_vel[1][envs_idx, -1, :],
        )
        return [obs_team0, obs_team1]

    def _team_reward(
        self,
        envs_idx: torch.Tensor,
        team_idx: int,
        attack_goal: torch.Tensor,
        own_pos: torch.Tensor,
        opp_pos: torch.Tensor,
        out_reward: torch.Tensor,
        goal_for: torch.Tensor,
        goal_against: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        ball_pos = self.cache["ball_pos_2d"][envs_idx]
        ball_vel = self.cache["ball_vel_2d"][envs_idx]
        to_goal = attack_goal - ball_pos
        to_goal_unit = to_goal / torch.norm(to_goal, dim=1, keepdim=True).clamp_min(1e-6)
        rew_ball_progress = (to_goal_unit * ball_vel).sum(dim=1, keepdim=True)

        own_ball_dis = torch.norm(ball_pos - own_pos, dim=1, keepdim=True)
        opp_ball_dis = torch.norm(ball_pos - opp_pos, dim=1, keepdim=True)
        rew_possession = opp_ball_dis - own_ball_dis

        cmd_now = self.cmd_vel[team_idx][envs_idx, -1, :]
        cmd_prev = self.cmd_vel[team_idx][envs_idx, 0, :]
        rew_smooth = torch.exp(-torch.norm(cmd_now - cmd_prev, dim=1, keepdim=True))

        reward_parts = {
            "rew_goal": goal_for.float() * self.cfg.reward_goal,
            "rew_concede": goal_against.float() * self.cfg.reward_concede,
            "rew_ball_progress": rew_ball_progress * self.cfg.reward_ball_progress,
            "rew_possession": rew_possession * self.cfg.reward_possession,
            "rew_smooth": rew_smooth * self.cfg.reward_smooth,
            "rew_ball_out_distance": out_reward,
        }
        return sum(reward_parts.values()), reward_parts # type: ignore[return-value]

    def build_reward(self, envs_idx: torch.Tensor, **kwargs
                     ) -> list[torch.Tensor]:  # type: ignore[override]
        self._build_cache(envs_idx=envs_idx, **kwargs)
        reward0, rew0 = self._team_reward(
            envs_idx,
            0,
            self.goal_pos[0][envs_idx],
            self.cache["team0_pos_2d"][envs_idx],
            self.cache["team1_pos_2d"][envs_idx],
            self.cache["out_reward_team0"][envs_idx],
            self.cache["goal_team0"][envs_idx],
            self.cache["goal_team1"][envs_idx],
        )
        reward1, rew1 = self._team_reward(
            envs_idx,
            1,
            self.goal_pos[1][envs_idx],
            self.cache["team1_pos_2d"][envs_idx],
            self.cache["team0_pos_2d"][envs_idx],
            self.cache["out_reward_team1"][envs_idx],
            self.cache["goal_team1"][envs_idx],
            self.cache["goal_team0"][envs_idx],
        )
        self.rewards = [rew0, rew1]
        return [reward0, reward1]

    def build_terminated(self, envs_idx: torch.Tensor, **kwargs
                         ) -> list[torch.Tensor]:  # type: ignore[override]
        self._build_cache(envs_idx=envs_idx, **kwargs)
        term = self.cache["terminated"][envs_idx]
        return [term, term.clone()]

    def build_truncated(self, envs_idx: torch.Tensor, **kwargs
                        ) -> list[torch.Tensor]:  # type: ignore[override]
        trunc = torch.zeros((envs_idx.shape[0], 1), dtype=torch.bool, device=gs.device)
        return [trunc, trunc.clone()]

    @torch.compiler.disable
    def build_info(self, envs_idx: torch.Tensor, **kwargs):
        return {
            "extra": [
                {k: v.detach().mean().cpu() for k, v in self.rewards[0].items()},
                {k: v.detach().mean().cpu() for k, v in self.rewards[1].items()},
            ],
            "goal_team0": self.cache["goal_team0"][envs_idx].detach(),
            "goal_team1": self.cache["goal_team1"][envs_idx].detach(),
            "ball_out": self.cache["ball_out"][envs_idx].detach(),
        }
