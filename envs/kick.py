import math
import torch
import genesis as gs
from dataclasses import dataclass

from .walk import WalkEnv, WalkEnvConfig


@dataclass(kw_only=True)
class KickEnvConfig(WalkEnvConfig):
    ball_radius: float = 0.07
    ball_mass_range: tuple[float, float] = (0.16, 0.16)
    ball_reset_r_min: float = 0.4
    ball_reset_r_max: float = 1.0
    ball_reset_noise: float = 0.2
    mode: str = "walk"
    vel_cmd: tuple[float, float, float] = (0.5, 0.0, 0.0)
    kick_speed: float = 2.0
    kick_dir_deg: float = 0.0
    kick_zone_radius: float = 1.0
    approach_speed_vx: float = 1.2
    approach_kp_wz: float = 2.0
    approach_max_wz: float = 1.5


class KickEnv(WalkEnv):
    cfg: KickEnvConfig

    def build(self):
        self.ball = self.scene.add_entity(
            gs.morphs.Sphere(
                radius=self.cfg.ball_radius,
                pos=(self.cfg.ball_reset_r_max, 0.0, self.cfg.ball_radius),
            ),
        )
        super().build()

    def config(self):
        super().config()
        nominal_mass = sum(self.cfg.ball_mass_range) / 2.0
        self.ball.set_mass(nominal_mass)
        self._ball_z = self.cfg.ball_radius + 0.05
        self.soccer_cmd = torch.zeros(
            (self.num_envs, 7), dtype=torch.float32, device=gs.device,
        )
        self._in_kick_zone = torch.zeros(self.num_envs, dtype=torch.bool, device=gs.device)
        self._target_pos_w = torch.zeros(
            (self.num_envs, 2), dtype=torch.float64, device=gs.device,
        )

    @torch.compiler.disable
    def reset(self, envs_idx: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
        if envs_idx is None:
            envs_idx = self.all_envs_idx
        B = envs_idx.shape[0]

        self._in_kick_zone[envs_idx] = False
        self._target_pos_w[envs_idx] = 0.0

        m_lo, m_hi = self.cfg.ball_mass_range
        if m_lo != m_hi:
            mass_shift = (m_lo - m_hi) / 2.0 + (m_hi - m_lo) * torch.rand(B, device=gs.device)
            self.ball.set_mass_shift(envs_idx=envs_idx, mass_shift=mass_shift)

        r_min, r_max = self.cfg.ball_reset_r_min, self.cfg.ball_reset_r_max
        r = r_min + (r_max - r_min) * torch.rand(B, device=gs.device)
        angle = 2.0 * math.pi * torch.rand(B, device=gs.device)
        noise = self.cfg.ball_reset_noise * (2.0 * torch.rand(B, 2, device=gs.device) - 1.0)
        ball_pos = torch.zeros(B, 3, dtype=torch.float, device=gs.device)
        ball_pos[:, 0] = r * torch.cos(angle) + noise[:, 0]
        ball_pos[:, 1] = r * torch.sin(angle) + noise[:, 1]
        ball_pos[:, 2] = self._ball_z
        self.ball.set_pos(envs_idx=envs_idx, pos=ball_pos)
        self.ball.zero_all_dofs_velocity(envs_idx=envs_idx)
        return super().reset(envs_idx)

    @torch.compiler.disable
    def _compute_soccer_cmd(self, envs_idx, body_pos, body_quat, ball_pos):
        B = envs_idx.shape[0]
        w = body_quat[:, 0]
        x = body_quat[:, 1]
        y = body_quat[:, 2]
        z = body_quat[:, 3]
        robot_yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

        dx = ball_pos[:, 0] - body_pos[:, 0]
        dy = ball_pos[:, 1] - body_pos[:, 1]
        cos_y = torch.cos(robot_yaw)
        sin_y = torch.sin(robot_yaw)
        ball_x_b = cos_y * dx + sin_y * dy
        ball_y_b = -sin_y * dx + cos_y * dy
        dist = torch.sqrt(dx * dx + dy * dy)

        cmd = torch.zeros((B, 7), dtype=torch.float32, device=gs.device)
        cmd[:, 3] = ball_x_b.float()
        cmd[:, 4] = ball_y_b.float()

        mode = self.cfg.mode

        if mode == "walk":
            cmd[:, 0] = self.cfg.vel_cmd[0]
            cmd[:, 1] = self.cfg.vel_cmd[1]
            cmd[:, 2] = self.cfg.vel_cmd[2]

        elif mode == "approach_kick":
            newly_in_zone = (~self._in_kick_zone[envs_idx]) & (dist < self.cfg.kick_zone_radius)
            newly_global = envs_idx[newly_in_zone]

            kick_dir_yaw = math.radians(self.cfg.kick_dir_deg)
            tgt_x = ball_pos[:, 0].double() + self.cfg.kick_speed * math.cos(kick_dir_yaw)
            tgt_y = ball_pos[:, 1].double() + self.cfg.kick_speed * math.sin(kick_dir_yaw)

            if newly_global.numel() > 0:
                self._in_kick_zone[newly_global] = True
                loc = newly_in_zone.nonzero(as_tuple=True)[0]
                self._target_pos_w[newly_global, 0] = tgt_x[loc]
                self._target_pos_w[newly_global, 1] = tgt_y[loc]

            approaching = ~self._in_kick_zone[envs_idx]
            yaw_err = torch.atan2(ball_y_b, ball_x_b)
            wz = torch.clamp(
                self.cfg.approach_kp_wz * yaw_err,
                -self.cfg.approach_max_wz, self.cfg.approach_max_wz,
            )
            vx = torch.maximum(
                torch.cos(yaw_err), torch.zeros(1, device=gs.device),
            ) * self.cfg.approach_speed_vx

            loc = approaching.nonzero(as_tuple=True)[0]
            cmd[loc, 0] = vx[loc].float()
            cmd[loc, 2] = wz[loc].float()

            kicking = self._in_kick_zone[envs_idx]
            loc = kicking.nonzero(as_tuple=True)[0]
            if loc.numel() > 0:
                to_tgt = self._target_pos_w[envs_idx[loc]] - ball_pos[loc, :2].double()
                dir_yaw = torch.atan2(to_tgt[:, 1], to_tgt[:, 0])
                cmd[loc, 5] = ((dir_yaw.float() - robot_yaw[loc].float()) + math.pi) % (2 * math.pi) - math.pi
                cmd[loc, 6] = self.cfg.kick_speed

        elif mode == "kick":
            kick_dir_yaw = math.radians(self.cfg.kick_dir_deg)
            not_frozen = ~self._in_kick_zone[envs_idx]
            freeze_global = envs_idx[not_frozen]
            if freeze_global.numel() > 0:
                loc = not_frozen.nonzero(as_tuple=True)[0]
                self._in_kick_zone[freeze_global] = True
                self._target_pos_w[freeze_global, 0] = (
                    ball_pos[loc, 0].double() + self.cfg.kick_speed * math.cos(kick_dir_yaw)
                )
                self._target_pos_w[freeze_global, 1] = (
                    ball_pos[loc, 1].double() + self.cfg.kick_speed * math.sin(kick_dir_yaw)
                )

            to_tgt = self._target_pos_w[envs_idx] - ball_pos[:, :2].double()
            dir_yaw = torch.atan2(to_tgt[:, 1], to_tgt[:, 0])
            cmd[:, 5] = ((dir_yaw.float() - robot_yaw.float()) + math.pi) % (2 * math.pi) - math.pi
            cmd[:, 6] = self.cfg.kick_speed

        self.soccer_cmd[envs_idx] = cmd

    @torch.compiler.disable
    def get_state(self, envs_idx: torch.Tensor) -> dict[str, torch.Tensor]:
        state = super().get_state(envs_idx)
        ball_pos = self.ball.get_pos(envs_idx=envs_idx)
        state["ball_pos"] = ball_pos
        state["ball_vel"] = self.ball.get_vel(envs_idx=envs_idx)[:, :3]
        self._compute_soccer_cmd(envs_idx, state["body_pos"], state["body_quat"], ball_pos)
        state["soccer_cmd"] = self.soccer_cmd[envs_idx]
        return state
