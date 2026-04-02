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
    num_robots_in_team: int = 1

    timeout_steps_limit: int = 600
    action_scale: tuple[float, float, float] = (1.5, 1.0, 2.0)
    reward_goal: float = 8.0
    reward_concede: float = -8.0
    reward_ball_progress: float = 1.5
    reward_possession: float = 0.4
    reward_close_to_ball: float = 0.2
    reward_facing_ball: float = 0.1
    reward_ball_contact: float = 0.15
    reward_smooth: float = 0.05
    reward_stall: float = 0.03
    reward_timeout: float = 0.5
    stall_ball_speed_threshold: float = 0.08
    out_of_field_reward_scale: float = 0.4
    out_of_field_penalty: float = 0.3
    fall_height: float = 0.28


class GameModel(Model):
    cfg: GameModelConfig

    def config(self):
        num_envs = self.scene.n_envs
        self.team_names = ("red", "blue")
        self.num_agents = 2
        self.action_scale = torch.tensor(
            self.cfg.action_scale, dtype=torch.float, device=gs.device
        ).view(1, 3)
        self.time_steps = torch.zeros(
            (num_envs, 1), dtype=torch.float, device=gs.device
        )
        self.cmd_vel = {
            "red": torch.zeros((num_envs, 2, 3), dtype=torch.float, device=gs.device),
            "blue": torch.zeros((num_envs, 2, 3), dtype=torch.float, device=gs.device),
        }
        if self.cfg.num_robots_in_team < 1:
            raise ValueError(
                f"num_robots_in_team must be >= 1, got {self.cfg.num_robots_in_team}"
            )
        self.obs_blocks = (
            ("self_heading", 2, "vec2"),
            ("self_vel_2d", 2 * self.cfg.num_robots_in_team, "vec2"),
            ("self_ang_z", 1, "scalar"),
            ("opp_rel_pos", 2, "vec2"),
            ("opp_rel_vel", 2, "vec2"),
            ("ball_rel_pos", 2, "vec2"),
            ("ball_rel_vel", 2, "vec2"),
            ("ball_to_goal", 2, "vec2"),
            ("last_cmd", 3, "cmd3"),
        )
        self.obs_dim = int(sum(dim for _, dim, _ in self.obs_blocks))
        self.obs_mirror_sign = self._build_obs_mirror_sign(self.obs_blocks).to(
            gs.device
        )
        self.action_mirror_sign = torch.tensor(
            [-1.0, -1.0, 1.0], dtype=torch.float, device=gs.device
        )

        half = torch.tensor(
            self.cfg.half_field_size, dtype=torch.float, device=gs.device
        )
        self.half_field_size = half.view(1, 2).broadcast_to((num_envs, 2))
        self.goal_width = (
            torch.tensor([self.cfg.goal_width], dtype=torch.float, device=gs.device)
            .view(1, 1)
            .broadcast_to((num_envs, 1))
        )
        self.goal_pos = {
            "red": torch.tensor(
                [self.cfg.half_field_size[0], 0.0], dtype=torch.float, device=gs.device
            )
            .view(1, 2)
            .broadcast_to((num_envs, 2)),
            "blue": torch.tensor(
                [-self.cfg.half_field_size[0], 0.0], dtype=torch.float, device=gs.device
            )
            .view(1, 2)
            .broadcast_to((num_envs, 2)),
        }

        self.cache_valid = torch.zeros((num_envs,), dtype=torch.bool, device=gs.device)
        self.cache: dict[str, torch.Tensor] = {
            "red_pos_2d": torch.zeros(
                (num_envs, 2), dtype=torch.float, device=gs.device
            ),
            "blue_pos_2d": torch.zeros(
                (num_envs, 2), dtype=torch.float, device=gs.device
            ),
            "red_vel_2d": torch.zeros(
                (num_envs, 2 * self.cfg.num_robots_in_team),
                dtype=torch.float,
                device=gs.device,
            ),
            "blue_vel_2d": torch.zeros(
                (num_envs, 2 * self.cfg.num_robots_in_team),
                dtype=torch.float,
                device=gs.device,
            ),
            "red_main_vel_2d": torch.zeros(
                (num_envs, 2), dtype=torch.float, device=gs.device
            ),
            "blue_main_vel_2d": torch.zeros(
                (num_envs, 2), dtype=torch.float, device=gs.device
            ),
            "ball_pos_2d": torch.zeros(
                (num_envs, 2), dtype=torch.float, device=gs.device
            ),
            "ball_vel_2d": torch.zeros(
                (num_envs, 2), dtype=torch.float, device=gs.device
            ),
            "red_heading": torch.zeros(
                (num_envs, 2), dtype=torch.float, device=gs.device
            ),
            "blue_heading": torch.zeros(
                (num_envs, 2), dtype=torch.float, device=gs.device
            ),
            "goal_team_red": torch.zeros(
                (num_envs, 1), dtype=torch.bool, device=gs.device
            ),
            "goal_team_blue": torch.zeros(
                (num_envs, 1), dtype=torch.bool, device=gs.device
            ),
            "ball_out": torch.zeros((num_envs, 1), dtype=torch.bool, device=gs.device),
            "timeout": torch.zeros((num_envs, 1), dtype=torch.bool, device=gs.device),
            "terminated": torch.zeros(
                (num_envs, 1), dtype=torch.bool, device=gs.device
            ),
            "out_reward_red": torch.zeros(
                (num_envs, 1), dtype=torch.float, device=gs.device
            ),
            "out_reward_blue": torch.zeros(
                (num_envs, 1), dtype=torch.float, device=gs.device
            ),
        }
        self.rewards: dict[str, dict[str, torch.Tensor]] = {
            "red": {},
            "blue": {},
        }

    def reset(self, envs_idx: torch.Tensor):
        self.cache_valid[envs_idx] = False
        self.time_steps[envs_idx] = 0.0
        for name in self.team_names:
            self.cmd_vel[name][envs_idx] = 0.0

    def preprocess_action(
        self, action: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:  # type: ignore[override]
        self.cache_valid[:] = False
        processed: dict[str, torch.Tensor] = {}
        for name in self.team_names:
            cmd = torch.clamp(action[name], -1.0, 1.0)
            self.cmd_vel[name][:, -1, :] = cmd
            processed[name] = cmd * self.action_scale
        self.time_steps += 1.0
        return processed

    @property
    def observation_space(self) -> dict[str, gym.spaces.Box]:
        space = gym.spaces.Box(
            low=-10.0, high=10.0, shape=(self.obs_dim,), dtype=np.float32
        )
        return {"red": space, "blue": space}

    @property
    def action_space(self) -> dict[str, gym.spaces.Box]:
        space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        return {"red": space, "blue": space}

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

    @staticmethod
    def _build_obs_mirror_sign(
        obs_blocks: tuple[tuple[str, int, str], ...],
    ) -> torch.Tensor:
        signs: list[float] = []
        for _, dim, mode in obs_blocks:
            if mode == "vec2":
                if dim % 2 != 0:
                    raise ValueError(f"vec2 block dim must be even, got {dim}")
                signs.extend([-1.0] * dim)
            elif mode == "cmd3":
                if dim != 3:
                    raise ValueError("cmd3 block must have dim=3")
                signs.extend([-1.0, -1.0, 1.0])
            else:
                signs.extend([1.0] * dim)
        return torch.tensor(signs, dtype=torch.float)

    def _team_vel2d(self, lin_vel: torch.Tensor) -> torch.Tensor:
        if lin_vel.ndim == 2:
            vel = lin_vel[:, 0:2].unsqueeze(1)
        elif lin_vel.ndim == 3:
            vel = lin_vel[:, :, 0:2]
        else:
            raise ValueError(f"Unsupported body_lin_vel shape: {tuple(lin_vel.shape)}")

        n_envs = vel.shape[0]
        n_src = vel.shape[1]
        n_dst = self.cfg.num_robots_in_team
        out = torch.zeros((n_envs, n_dst, 2), dtype=vel.dtype, device=vel.device)
        n_copy = min(n_src, n_dst)
        out[:, :n_copy, :] = vel[:, :n_copy, :]
        return out.reshape(n_envs, 2 * n_dst)

    def _build_obs_from_perspective(
        self,
        self_pos_2d: torch.Tensor,
        self_vel_2d: torch.Tensor,
        self_main_vel_2d: torch.Tensor,
        self_heading: torch.Tensor,
        self_ang_vel: torch.Tensor,
        opp_pos_2d: torch.Tensor,
        opp_main_vel_2d: torch.Tensor,
        ball_pos_2d: torch.Tensor,
        ball_vel_2d: torch.Tensor,
        attack_goal_pos: torch.Tensor,
        last_cmd: torch.Tensor,
    ) -> torch.Tensor:
        opp_rel_pos = opp_pos_2d - self_pos_2d
        opp_rel_vel = opp_main_vel_2d - self_main_vel_2d
        ball_rel_pos = ball_pos_2d - self_pos_2d
        ball_rel_vel = ball_vel_2d - self_main_vel_2d
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
        red: dict[str, torch.Tensor],
        blue: dict[str, torch.Tensor],
        ball_pos: torch.Tensor,
        ball_vel: torch.Tensor,
        **kwargs,
    ) -> None:
        if self.cache_valid[envs_idx].all():
            return
        self.cache_valid[envs_idx] = True

        red_pos = self._team_main(red["body_pos"])
        blue_pos = self._team_main(blue["body_pos"])
        red_quat = self._team_main(red["body_quat"])
        blue_quat = self._team_main(blue["body_quat"])
        red_lin_vel = self._team_main(red["body_lin_vel"])
        blue_lin_vel = self._team_main(blue["body_lin_vel"])

        ball_pos_2d = ball_pos[:, 0:2]
        ball_vel_2d = ball_vel[:, 0:2]

        self.cache["red_pos_2d"][envs_idx] = red_pos[:, 0:2]
        self.cache["blue_pos_2d"][envs_idx] = blue_pos[:, 0:2]
        self.cache["red_vel_2d"][envs_idx] = self._team_vel2d(red["body_lin_vel"])
        self.cache["blue_vel_2d"][envs_idx] = self._team_vel2d(blue["body_lin_vel"])
        self.cache["red_main_vel_2d"][envs_idx] = red_lin_vel[:, 0:2]
        self.cache["blue_main_vel_2d"][envs_idx] = blue_lin_vel[:, 0:2]
        self.cache["ball_pos_2d"][envs_idx] = ball_pos_2d
        self.cache["ball_vel_2d"][envs_idx] = ball_vel_2d
        self.cache["red_heading"][envs_idx] = self._heading2d(red_quat)
        self.cache["blue_heading"][envs_idx] = self._heading2d(blue_quat)

        half_x = self.half_field_size[envs_idx, 0:1]
        half_y = self.half_field_size[envs_idx, 1:2]
        goal_half = 0.5 * self.goal_width[envs_idx]

        goal_team_red = (ball_pos_2d[:, 0:1] >= half_x) & (
            torch.abs(ball_pos_2d[:, 1:2]) <= goal_half
        )
        goal_team_blue = (ball_pos_2d[:, 0:1] <= -half_x) & (
            torch.abs(ball_pos_2d[:, 1:2]) <= goal_half
        )
        is_outside = (torch.abs(ball_pos_2d[:, 0:1]) > half_x) | (
            torch.abs(ball_pos_2d[:, 1:2]) > half_y
        )

        ball_out = is_outside & ~(goal_team_red | goal_team_blue)
        self.cache["goal_team_red"][envs_idx] = goal_team_red
        self.cache["goal_team_blue"][envs_idx] = goal_team_blue
        self.cache["ball_out"][envs_idx] = ball_out

        out_point = torch.stack(
            [
                torch.clamp(ball_pos_2d[:, 0], min=-half_x[:, 0], max=half_x[:, 0]),
                torch.clamp(ball_pos_2d[:, 1], min=-half_y[:, 0], max=half_y[:, 0]),
            ],
            dim=1,
        )
        max_dist = torch.norm(
            2.0 * self.half_field_size[envs_idx], dim=1, keepdim=True
        ).clamp_min(1e-6)
        out_reward_red = self.cfg.out_of_field_reward_scale * (
            1.0
            - torch.norm(
                out_point - self.goal_pos["red"][envs_idx], dim=1, keepdim=True
            )
            / max_dist
        )
        out_reward_blue = self.cfg.out_of_field_reward_scale * (
            1.0
            - torch.norm(
                out_point - self.goal_pos["blue"][envs_idx], dim=1, keepdim=True
            )
            / max_dist
        )
        self.cache["out_reward_red"][envs_idx] = torch.where(
            ball_out, out_reward_red, torch.zeros_like(out_reward_red)
        )
        self.cache["out_reward_blue"][envs_idx] = torch.where(
            ball_out, out_reward_blue, torch.zeros_like(out_reward_blue)
        )

        red_fall = red_pos[:, 2:3] < self.cfg.fall_height
        blue_fall = blue_pos[:, 2:3] < self.cfg.fall_height
        timeout = self.time_steps[envs_idx] >= self.cfg.timeout_steps_limit
        self.cache["timeout"][envs_idx] = timeout
        self.cache["terminated"][envs_idx] = (
            goal_team_red | goal_team_blue | ball_out | red_fall | blue_fall | timeout
        )

    def build_observation(
        self, envs_idx: torch.Tensor, **kwargs
    ) -> dict[str, torch.Tensor]:  # type: ignore[override]
        self._build_cache(envs_idx=envs_idx, **kwargs)
        obs_red = self._build_obs_from_perspective(
            self_pos_2d=self.cache["red_pos_2d"][envs_idx],
            self_vel_2d=self.cache["red_vel_2d"][envs_idx],
            self_main_vel_2d=self.cache["red_main_vel_2d"][envs_idx],
            self_heading=self.cache["red_heading"][envs_idx],
            self_ang_vel=self._team_main(kwargs["red"]["body_ang_vel"]),
            opp_pos_2d=self.cache["blue_pos_2d"][envs_idx],
            opp_main_vel_2d=self.cache["blue_main_vel_2d"][envs_idx],
            ball_pos_2d=self.cache["ball_pos_2d"][envs_idx],
            ball_vel_2d=self.cache["ball_vel_2d"][envs_idx],
            attack_goal_pos=self.goal_pos["red"][envs_idx],
            last_cmd=self.cmd_vel["red"][envs_idx, -1, :],
        )
        obs_blue = self._build_obs_from_perspective(
            self_pos_2d=self.cache["blue_pos_2d"][envs_idx],
            self_vel_2d=self.cache["blue_vel_2d"][envs_idx],
            self_main_vel_2d=self.cache["blue_main_vel_2d"][envs_idx],
            self_heading=self.cache["blue_heading"][envs_idx],
            self_ang_vel=self._team_main(kwargs["blue"]["body_ang_vel"]),
            opp_pos_2d=self.cache["red_pos_2d"][envs_idx],
            opp_main_vel_2d=self.cache["red_main_vel_2d"][envs_idx],
            ball_pos_2d=self.cache["ball_pos_2d"][envs_idx],
            ball_vel_2d=self.cache["ball_vel_2d"][envs_idx],
            attack_goal_pos=self.goal_pos["blue"][envs_idx],
            last_cmd=self.cmd_vel["blue"][envs_idx, -1, :],
        )
        return {"red": obs_red, "blue": obs_blue}

    def _team_reward(
        self,
        envs_idx: torch.Tensor,
        team_name: str,
        attack_goal: torch.Tensor,
        own_pos: torch.Tensor,
        own_main_vel: torch.Tensor,
        own_heading: torch.Tensor,
        opp_pos: torch.Tensor,
        out_adv_reward: torch.Tensor,
        ball_out: torch.Tensor,
        timeout: torch.Tensor,
        goal_for: torch.Tensor,
        goal_against: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        ball_pos = self.cache["ball_pos_2d"][envs_idx]
        ball_vel = self.cache["ball_vel_2d"][envs_idx]
        to_goal = attack_goal - ball_pos
        to_goal_unit = to_goal / torch.norm(to_goal, dim=1, keepdim=True).clamp_min(
            1e-6
        )
        rew_ball_progress = (to_goal_unit * ball_vel).sum(dim=1, keepdim=True)

        own_ball_dis = torch.norm(ball_pos - own_pos, dim=1, keepdim=True)
        opp_ball_dis = torch.norm(ball_pos - opp_pos, dim=1, keepdim=True)
        rew_possession = opp_ball_dis - own_ball_dis
        ball_rel = ball_pos - own_pos
        rew_close_to_ball = (ball_rel * own_main_vel).sum(dim=1, keepdim=True) / (
            own_ball_dis + 0.1
        )
        rew_facing_ball = (own_heading * ball_rel).sum(dim=1, keepdim=True) / (
            own_ball_dis + 0.1
        )
        rew_ball_contact = torch.exp(-own_ball_dis / 0.35)

        cmd_now = self.cmd_vel[team_name][envs_idx, -1, :]
        cmd_prev = self.cmd_vel[team_name][envs_idx, 0, :]
        rew_smooth = -torch.norm(cmd_now - cmd_prev, dim=1, keepdim=True)
        rew_ball_out_penalty = -ball_out.float() * self.cfg.out_of_field_penalty
        ball_speed = torch.norm(ball_vel, dim=1, keepdim=True)
        rew_stall = -(ball_speed < self.cfg.stall_ball_speed_threshold).float()
        rew_timeout = -timeout.float()

        reward_parts = {
            "rew_goal": goal_for.float() * self.cfg.reward_goal,
            "rew_concede": goal_against.float() * self.cfg.reward_concede,
            "rew_ball_progress": rew_ball_progress * self.cfg.reward_ball_progress,
            "rew_possession": rew_possession * self.cfg.reward_possession,
            "rew_close_to_ball": rew_close_to_ball * self.cfg.reward_close_to_ball,
            "rew_facing_ball": rew_facing_ball * self.cfg.reward_facing_ball,
            "rew_ball_contact": rew_ball_contact * self.cfg.reward_ball_contact,
            "rew_smooth": rew_smooth * self.cfg.reward_smooth,
            "rew_ball_out_adv": out_adv_reward,
            "rew_ball_out_penalty": rew_ball_out_penalty,
            "rew_stall": rew_stall * self.cfg.reward_stall,
            "rew_timeout": rew_timeout * self.cfg.reward_timeout,
        }
        return sum(reward_parts.values()), reward_parts  # type: ignore[return-value]

    def build_reward(self, envs_idx: torch.Tensor, **kwargs) -> dict[str, torch.Tensor]:  # type: ignore[override]
        self._build_cache(envs_idx=envs_idx, **kwargs)
        out_adv_red = (
            self.cache["out_reward_red"][envs_idx]
            - self.cache["out_reward_blue"][envs_idx]
        )
        out_adv_blue = -out_adv_red
        reward_red, rew_red = self._team_reward(
            envs_idx,
            "red",
            self.goal_pos["red"][envs_idx],
            self.cache["red_pos_2d"][envs_idx],
            self.cache["red_main_vel_2d"][envs_idx],
            self.cache["red_heading"][envs_idx],
            self.cache["blue_pos_2d"][envs_idx],
            out_adv_red,
            self.cache["ball_out"][envs_idx],
            self.cache["timeout"][envs_idx],
            self.cache["goal_team_red"][envs_idx],
            self.cache["goal_team_blue"][envs_idx],
        )
        reward_blue, rew_blue = self._team_reward(
            envs_idx,
            "blue",
            self.goal_pos["blue"][envs_idx],
            self.cache["blue_pos_2d"][envs_idx],
            self.cache["blue_main_vel_2d"][envs_idx],
            self.cache["blue_heading"][envs_idx],
            self.cache["red_pos_2d"][envs_idx],
            out_adv_blue,
            self.cache["ball_out"][envs_idx],
            self.cache["timeout"][envs_idx],
            self.cache["goal_team_blue"][envs_idx],
            self.cache["goal_team_red"][envs_idx],
        )
        competitive_dense_keys = (
            "rew_ball_progress",
            "rew_possession",
            "rew_close_to_ball",
            "rew_facing_ball",
            "rew_ball_contact",
            "rew_ball_out_adv",
        )
        for key in competitive_dense_keys:
            adv = rew_red[key] - rew_blue[key]
            rew_red[key] = 0.5 * adv
            rew_blue[key] = -0.5 * adv

        reward_red = sum(rew_red.values())
        reward_blue = sum(rew_blue.values())
        self.rewards = {
            "red": rew_red,
            "blue": rew_blue,
        }
        return {
            "red": reward_red,
            "blue": reward_blue,
        }

    def build_terminated(
        self, envs_idx: torch.Tensor, **kwargs
    ) -> dict[str, torch.Tensor]:  # type: ignore[override]
        self._build_cache(envs_idx=envs_idx, **kwargs)
        term = self.cache["terminated"][envs_idx]
        return {"red": term, "blue": term.clone()}

    def build_truncated(
        self, envs_idx: torch.Tensor, **kwargs
    ) -> dict[str, torch.Tensor]:  # type: ignore[override]
        trunc = torch.zeros((envs_idx.shape[0], 1), dtype=torch.bool, device=gs.device)
        return {"red": trunc, "blue": trunc.clone()}

    @torch.compiler.disable
    def build_info(self, envs_idx: torch.Tensor, **kwargs):
        return {
            "extra": {
                "red": {
                    k: v.detach().mean().cpu() for k, v in self.rewards["red"].items()
                },
                "blue": {
                    k: v.detach().mean().cpu() for k, v in self.rewards["blue"].items()
                },
            },
            "goal_team_red": self.cache["goal_team_red"][envs_idx].detach(),
            "goal_team_blue": self.cache["goal_team_blue"][envs_idx].detach(),
            "ball_out": self.cache["ball_out"][envs_idx].detach(),
        }
