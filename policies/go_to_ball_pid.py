from __future__ import annotations

import torch


class GoToBallPIDPolicy:
    def __init__(
        self,
        *,
        kp_forward: float = 1.2,
        kd_forward: float = 0.15,
        ki_forward: float = 0.02,
        kp_lateral: float = 0.9,
        kd_lateral: float = 0.1,
        ki_lateral: float = 0.01,
        kp_yaw: float = 1.4,
        kd_yaw: float = 0.2,
        ki_yaw: float = 0.02,
        max_cmd: float = 1.0,
        integral_limit: float = 2.0,
        eps: float = 1e-6,
        device: torch.device | str = "cpu",
    ):
        self.kp_forward = kp_forward
        self.kd_forward = kd_forward
        self.ki_forward = ki_forward
        self.kp_lateral = kp_lateral
        self.kd_lateral = kd_lateral
        self.ki_lateral = ki_lateral
        self.kp_yaw = kp_yaw
        self.kd_yaw = kd_yaw
        self.ki_yaw = ki_yaw
        self.max_cmd = max_cmd
        self.integral_limit = integral_limit
        self.eps = eps
        self.device = torch.device(device)

        self._prev_forward: torch.Tensor | None = None
        self._prev_lateral: torch.Tensor | None = None
        self._prev_yaw: torch.Tensor | None = None
        self._int_forward: torch.Tensor | None = None
        self._int_lateral: torch.Tensor | None = None
        self._int_yaw: torch.Tensor | None = None

    def _ensure_state(self, num_envs: int, device: torch.device) -> None:
        needs_reset = (
            self._prev_forward is None or self._prev_forward.shape[0] != num_envs
        )
        if needs_reset:
            self._prev_forward = torch.zeros(
                (num_envs, 1), dtype=torch.float, device=device
            )
            self._prev_lateral = torch.zeros(
                (num_envs, 1), dtype=torch.float, device=device
            )
            self._prev_yaw = torch.zeros(
                (num_envs, 1), dtype=torch.float, device=device
            )
            self._int_forward = torch.zeros(
                (num_envs, 1), dtype=torch.float, device=device
            )
            self._int_lateral = torch.zeros(
                (num_envs, 1), dtype=torch.float, device=device
            )
            self._int_yaw = torch.zeros((num_envs, 1), dtype=torch.float, device=device)

    def _parse_obs(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Observation layout follows GameModel block order:
        # [self_heading(2), self_vel_2d(2*n), self_ang_z(1), opp_rel_pos(2), opp_rel_vel(2),
        #  ball_rel_pos(2), ball_rel_vel(2), ball_to_goal(2), last_cmd(3)]
        vel_dim = obs.shape[1] - 16
        if vel_dim < 2 or vel_dim % 2 != 0:
            raise RuntimeError(
                f"Unsupported observation dim={obs.shape[1]} for GameModel layout"
            )

        idx = 0
        heading = obs[:, idx : idx + 2]
        idx += 2
        idx += vel_dim
        idx += 1
        idx += 2
        idx += 2
        ball_rel = obs[:, idx : idx + 2]
        return heading, ball_rel, obs[:, 2 : 2 + 2]

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        device = obs.device
        self._ensure_state(obs.shape[0], device)
        assert self._prev_forward is not None
        assert self._prev_lateral is not None
        assert self._prev_yaw is not None
        assert self._int_forward is not None
        assert self._int_lateral is not None
        assert self._int_yaw is not None

        heading, ball_rel, self_vel_world = self._parse_obs(obs)
        hx = heading[:, 0:1]
        hy = heading[:, 1:2]
        bx = ball_rel[:, 0:1]
        by = ball_rel[:, 1:2]
        ball_dist = torch.sqrt(bx * bx + by * by + self.eps)

        # Robot-frame ball coordinates.
        rel_forward = bx * hx + by * hy
        rel_lateral = -bx * hy + by * hx
        yaw_err = torch.atan2(rel_lateral, rel_forward.clamp(min=self.eps))

        # World velocity projected to robot frame for damping.
        vel_forward = self_vel_world[:, 0:1] * hx + self_vel_world[:, 1:2] * hy
        vel_lateral = -self_vel_world[:, 0:1] * hy + self_vel_world[:, 1:2] * hx

        # PID integrators (bounded).
        self._int_forward = torch.clamp(
            self._int_forward + rel_forward, -self.integral_limit, self.integral_limit
        )
        self._int_lateral = torch.clamp(
            self._int_lateral + rel_lateral, -self.integral_limit, self.integral_limit
        )
        self._int_yaw = torch.clamp(
            self._int_yaw + yaw_err, -self.integral_limit, self.integral_limit
        )

        d_forward = rel_forward - self._prev_forward
        d_lateral = rel_lateral - self._prev_lateral
        d_yaw = yaw_err - self._prev_yaw

        cmd_forward = (
            self.kp_forward * rel_forward
            + self.ki_forward * self._int_forward
            + self.kd_forward * d_forward
            - 0.25 * vel_forward
        )
        cmd_lateral = (
            self.kp_lateral * rel_lateral
            + self.ki_lateral * self._int_lateral
            + self.kd_lateral * d_lateral
            - 0.25 * vel_lateral
        )
        cmd_yaw = (
            self.kp_yaw * yaw_err + self.ki_yaw * self._int_yaw + self.kd_yaw * d_yaw
        )

        # Near the ball: reduce lateral motion and keep heading stable to avoid orbiting.
        near_mask = (ball_dist < 0.22).float()
        cmd_lateral = cmd_lateral * (1.0 - 0.7 * near_mask)

        self._prev_forward = rel_forward
        self._prev_lateral = rel_lateral
        self._prev_yaw = yaw_err

        action = torch.cat([cmd_forward, cmd_lateral, cmd_yaw], dim=1)
        return torch.clamp(action, -self.max_cmd, self.max_cmd)


Policy = GoToBallPIDPolicy
