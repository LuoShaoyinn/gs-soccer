import math
import numpy as np
import torch
import genesis as gs
from torch.nn import functional as F
from dataclasses import dataclass, field

from .field import FieldConfig, Field


@dataclass(kw_only=True)
class BallFieldConfig(FieldConfig):
    ball_radius: float = 0.08
    ball_mass: float = 0.200
    ball_damping: float = 2.0
    ball_friction: float = 0.08
    field_friction: float = 0.08
    ball_init_pos: np.ndarray = field(
        default_factory=lambda: np.array([0.5, 0.0, 0.08], dtype=np.float32)
    )
    ball_mass_range: tuple[float, float] | None = None
    ball_reset_radius: tuple[float, float] | None = None
    ball_reset_noise: float = 0.0


class BallField(Field):
    cfg: BallFieldConfig

    @torch.compiler.disable
    def build(self):
        self.plane = self.scene.add_entity(
            morph=gs.morphs.Plane(),
            material=gs.materials.Rigid(friction=self.cfg.field_friction),
        )
        self.ball = self.scene.add_entity(
            morph=gs.morphs.Sphere(radius=self.cfg.ball_radius),
            material=gs.materials.Rigid(friction=self.cfg.ball_friction),
        )

    @torch.compiler.disable
    def config(self):
        self.ball.set_mass(self.cfg.ball_mass)
        self.ball.set_dofs_damping(self.cfg.ball_damping, dofs_idx_local=(3, 4, 5))
        self.ball_init_pos = torch.from_numpy(self.cfg.ball_init_pos).to(gs.device)
        if self.cfg.ball_mass_range is not None:
            lo, hi = self.cfg.ball_mass_range
            n_envs = self.scene.n_envs
            mass = lo + (hi - lo) * torch.rand(n_envs, device=gs.device)
            envs_idx = torch.arange(n_envs, device=gs.device)
            self.ball.set_mass_shift(
                envs_idx=envs_idx, mass_shift=mass - self.cfg.ball_mass
            )

    def _sample_ball_pos(self, n: int) -> torch.Tensor:
        r_min, r_max = self.cfg.ball_reset_radius
        r = r_min + (r_max - r_min) * torch.rand(n, device=gs.device)
        angle = 2.0 * math.pi * torch.rand(n, device=gs.device)
        noise = self.cfg.ball_reset_noise * (
            2.0 * torch.rand(n, 2, device=gs.device) - 1.0
        )
        pos = torch.zeros(n, 3, device=gs.device)
        pos[:, 0] = r * torch.cos(angle) + noise[:, 0]
        pos[:, 1] = r * torch.sin(angle) + noise[:, 1]
        pos[:, 2] = self.cfg.ball_radius
        return pos

    @torch.compiler.disable
    def reset(self, envs_idx, ball_pos=None, ball_mass_shift=None, **kwargs):
        n = envs_idx.shape[0]
        if ball_pos is None:
            if self.cfg.ball_reset_radius is not None:
                ball_pos = self._sample_ball_pos(n)
            else:
                ball_pos = self.ball_init_pos.broadcast_to((n, 3))
        elif ball_pos.shape[1] == 2:
            ball_pos = F.pad(ball_pos, (0, 1), value=self.cfg.ball_radius)
        if ball_mass_shift is not None:
            self.ball.set_mass_shift(envs_idx=envs_idx, mass_shift=ball_mass_shift)
        self.ball.set_pos(envs_idx=envs_idx, pos=ball_pos)
        self.ball.zero_all_dofs_velocity(envs_idx=envs_idx)

    @torch.compiler.disable
    def get_state(self, envs_idx) -> dict[str, torch.Tensor]:
        return {
            "ball_pos": self.ball.get_pos(envs_idx=envs_idx),
            "ball_vel": self.ball.get_vel(envs_idx=envs_idx),
        }
