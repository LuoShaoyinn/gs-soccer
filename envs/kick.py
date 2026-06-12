import math
import torch
import numpy as np
import genesis as gs
from dataclasses import dataclass, field
from torch.nn import functional as F

from .walk import WalkEnv, WalkEnvConfig


@dataclass(kw_only=True)
class KickEnvConfig(WalkEnvConfig):
    ball_radius: float = 0.07
    ball_mass_range: tuple[float, float] = (0.2, 0.28)
    ball_damping_range: tuple[float, float] = (1.0, 4.0)
    ball_reset_r_min: float = 0.4
    ball_reset_r_max: float = 1.0
    ball_reset_noise: float = 0.2


class KickEnv(WalkEnv):
    cfg: KickEnvConfig

    def build(self):
        self.ball = self.scene.add_entity(
            gs.morphs.Sphere(radius=self.cfg.ball_radius),
            pos=(self.cfg.ball_reset_r_max, 0.0, self.cfg.ball_radius),
        )
        super().build()

    def config(self):
        super().config()
        nominal_mass = sum(self.cfg.ball_mass_range) / 2.0
        self.ball.set_mass(nominal_mass)
        self.ball.set_dofs_damping(
            sum(self.cfg.ball_damping_range) / 2.0,
            dofs_idx_local=(3, 4, 5),
        )
        self._ball_z = self.cfg.ball_radius + 0.05

    @torch.compiler.disable
    def reset(self, envs_idx: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
        if envs_idx is None:
            envs_idx = self.all_envs_idx
        B = envs_idx.shape[0]

        # randomize ball mass
        m_lo, m_hi = self.cfg.ball_mass_range
        mass_shift = (m_lo - m_hi) / 2.0 + (m_hi - m_lo) * torch.rand(B, device=gs.device)
        self.ball.set_mass_shift(envs_idx=envs_idx, mass_shift=mass_shift)

        # randomize ball damping (global — set before position reset)
        d_lo, d_hi = self.cfg.ball_damping_range
        damping = d_lo + (d_hi - d_lo) * torch.rand(1, device=gs.device).item()
        self.ball.set_dofs_damping(damping, dofs_idx_local=(3, 4, 5))

        # randomize ball position
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
    def get_state(self, envs_idx: torch.Tensor) -> dict[str, torch.Tensor]:
        state = super().get_state(envs_idx)
        state["ball_pos"] = self.ball.get_pos(envs_idx=envs_idx)
        state["ball_vel"] = self.ball.get_vel(envs_idx=envs_idx)[:, :3]
        return state
