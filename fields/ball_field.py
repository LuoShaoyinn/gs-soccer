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

    @torch.compiler.disable
    def reset(self, envs_idx, ball_pos=None, ball_mass_shift=None, **kwargs):
        n = envs_idx.shape[0]
        if ball_pos is None:
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
