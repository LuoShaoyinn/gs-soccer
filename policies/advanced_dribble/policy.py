from __future__ import annotations

import torch


class AdvancedDribblePolicy:
    """Non-RL dribble policy adapted from mos-brain's advanced dribbler.

    This policy expects the single-robot `GameModel` observation layout:
    [heading(2), self_vel(2), self_ang_z(1), opp_rel(2), opp_vel(2),
     ball_rel(2), ball_vel(2), ball_to_goal(2), last_cmd(3)]
    """

    def __init__(
        self,
        *,
        turn_p: float = 1.8,
        side_p: float = 3.0,
        forward_p: float = 1.4,
        setup_dist: float = 0.40,
        dribble_dist: float = 0.20,
        max_linear: float = 1.0,
        max_angular: float = 1.0,
        min_push_aligned: float = 0.75,
        min_creep_factor: float = 0.2,
        align_y_thresh: float = 0.10,
        behind_x_thresh: float = 0.05,
        possession_front_min: float = 0.06,
        possession_front_max: float = 0.55,
        possession_lat_max: float = 0.12,
        recapture_front_thresh: float = 0.28,
        recapture_side_thresh: float = 0.22,
        near_goal_dist: float = 1.7,
        eps: float = 1e-6,
        device: torch.device | str = "cpu",
    ):
        self.turn_p = turn_p
        self.side_p = side_p
        self.forward_p = forward_p
        self.setup_dist = setup_dist
        self.dribble_dist = dribble_dist
        self.max_linear = max_linear
        self.max_angular = max_angular
        self.min_push_aligned = min_push_aligned
        self.min_creep_factor = min_creep_factor
        self.align_y_thresh = align_y_thresh
        self.behind_x_thresh = behind_x_thresh
        self.possession_front_min = possession_front_min
        self.possession_front_max = possession_front_max
        self.possession_lat_max = possession_lat_max
        self.recapture_front_thresh = recapture_front_thresh
        self.recapture_side_thresh = recapture_side_thresh
        self.near_goal_dist = near_goal_dist
        self.eps = eps
        self.device = torch.device(device)

    @staticmethod
    def _to_robot_frame(vec_world: torch.Tensor, heading: torch.Tensor) -> torch.Tensor:
        hx = heading[:, 0:1]
        hy = heading[:, 1:2]
        vx = vec_world[:, 0:1]
        vy = vec_world[:, 1:2]
        forward = vx * hx + vy * hy
        lateral = -vx * hy + vy * hx
        return torch.cat([forward, lateral], dim=1)

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.ndim != 2 or obs.shape[1] < 18:
            raise RuntimeError(
                f"AdvancedDribblePolicy expects obs shape (N, 18+), got {tuple(obs.shape)}"
            )

        heading = obs[:, 0:2]
        self_vel_world = obs[:, 2:4]
        ball_rel_world = obs[:, 9:11]
        ball_to_goal_world = obs[:, 13:15]
        last_cmd = obs[:, 15:18]

        ball_rel = self._to_robot_frame(ball_rel_world, heading)
        goal_vec = self._to_robot_frame(ball_to_goal_world, heading)
        robot_to_goal_world = ball_rel_world + ball_to_goal_world
        robot_goal_vec = self._to_robot_frame(robot_to_goal_world, heading)
        self_vel = self._to_robot_frame(self_vel_world, heading)

        b_x = ball_rel[:, 0:1]
        b_y = ball_rel[:, 1:2]
        g_x = goal_vec[:, 0:1]
        g_y = goal_vec[:, 1:2]
        rg_x = robot_goal_vec[:, 0:1]
        rg_y = robot_goal_vec[:, 1:2]

        ball_angle = torch.atan2(b_y, b_x)
        goal_angle = torch.atan2(rg_y, rg_x)

        possession = (
            (b_x > self.possession_front_min)
            & (b_x < self.possession_front_max)
            & (torch.abs(b_y) < self.possession_lat_max)
        )

        target_angle = torch.where(possession, goal_angle, ball_angle)
        target_angle_deg = torch.abs(target_angle) * (180.0 / torch.pi)

        c_r = torch.cos(-target_angle)
        s_r = torch.sin(-target_angle)
        b_x_virt = b_x * c_r - b_y * s_r
        b_y_virt = b_x * s_r + b_y * c_r

        ball_goal_dist = torch.sqrt(g_x * g_x + g_y * g_y + self.eps)
        near_goal = ball_goal_dist < self.near_goal_dist

        aligned = (
            (torch.abs(b_y_virt) < self.align_y_thresh)
            & (b_x_virt > 0.06)
            & (target_angle_deg < 35.0)
            & possession
        )

        target_dist = torch.full_like(b_x_virt, self.setup_dist)
        interp = ((25.0 - target_angle_deg) / 20.0).clamp(0.0, 1.0)
        interp_dist = self.setup_dist - interp * (self.setup_dist - self.dribble_dist)
        target_dist = torch.where(target_angle_deg < 25.0, interp_dist, target_dist)
        target_dist = torch.where(
            aligned, torch.full_like(target_dist, self.dribble_dist), target_dist
        )

        err_x = b_x_virt - target_dist
        err_y = b_y_virt

        forward_factor = torch.where(
            aligned,
            torch.ones_like(err_x),
            torch.full_like(err_x, self.min_creep_factor),
        )
        side_miss = torch.abs(b_y_virt) > (self.align_y_thresh * 1.8)
        forward_factor = torch.where(
            side_miss, torch.zeros_like(forward_factor), forward_factor
        )
        cmd_x_virt = self.forward_p * err_x * forward_factor
        cmd_y_virt = self.side_p * err_y

        min_push = torch.where(
            near_goal,
            torch.full_like(cmd_x_virt, 0.9),
            torch.full_like(cmd_x_virt, self.min_push_aligned),
        )
        cmd_x_virt = torch.where(
            aligned, torch.maximum(cmd_x_virt, min_push), cmd_x_virt
        )

        c = torch.cos(target_angle)
        s = torch.sin(target_angle)
        cmd_x = cmd_x_virt * c - cmd_y_virt * s
        cmd_y = cmd_x_virt * s + cmd_y_virt * c

        turn_damp = (torch.sqrt(b_x * b_x + b_y * b_y + self.eps) / 0.4).clamp(0.4, 1.0)
        cmd_w = self.turn_p * target_angle * turn_damp - 0.15 * self_vel[:, 1:2]

        behind_mask = b_x < self.behind_x_thresh
        side_front_mask = (b_x < self.recapture_front_thresh) & (
            torch.abs(b_y) > self.recapture_side_thresh
        )
        turn_to_ball = (1.5 * ball_angle).clamp(-self.max_angular, self.max_angular)
        approach = (0.35 * b_x).clamp(-0.3, 0.5)
        side_recap = (-0.8 * b_y).clamp(-0.6, 0.6)
        cmd_x = torch.where(behind_mask, approach, cmd_x)
        cmd_y = torch.where(behind_mask, side_recap, cmd_y)
        cmd_w = torch.where(behind_mask, turn_to_ball, cmd_w)

        recap_x = torch.where(
            b_x < 0.18, torch.full_like(b_x, -0.08), torch.zeros_like(b_x)
        )
        recap_y = (-0.35 * torch.sign(b_y)).clamp(-0.35, 0.35)
        recap_w = (1.2 * ball_angle).clamp(-self.max_angular, self.max_angular)
        cmd_x = torch.where(side_front_mask, recap_x, cmd_x)
        cmd_y = torch.where(side_front_mask, recap_y, cmd_y)
        cmd_w = torch.where(side_front_mask, recap_w, cmd_w)

        lin = torch.cat([cmd_x, cmd_y], dim=1)
        lin_norm = torch.norm(lin, dim=1, keepdim=True).clamp_min(self.eps)
        scale = torch.clamp(self.max_linear / lin_norm, max=1.0)
        lin = lin * scale

        cmd_w = torch.clamp(cmd_w, -self.max_angular, self.max_angular)
        raw_action = torch.cat([lin, cmd_w], dim=1)
        action = 0.8 * raw_action + 0.2 * last_cmd
        return torch.clamp(action, -1.0, 1.0)


Policy = AdvancedDribblePolicy
