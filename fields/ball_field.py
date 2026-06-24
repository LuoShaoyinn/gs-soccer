import numpy as np
import torch
import genesis as gs
from dataclasses import dataclass, field

from .field import FieldConfig, Field


@dataclass(kw_only=True)
class BallFieldConfig(FieldConfig):
    ball_radius:    float   = 0.08
    ball_mass:      float   = 0.200
    ball_damping:   float   = 2.0
    ball_friction:  float   = 0.08
    field_friction: float   = 0.08
    ball_init_pos: np.ndarray = field(
        default_factory=lambda: np.array([0.5, 0.0, 0.08], dtype=np.float32)
    )
    ball_mass_shift: float  = 0.0
    ball_COM_shift: float   = 0.0
    # Terrain (disabled by default — uses flat Plane)
    use_terrain:       bool  = False
    terrain_types:     str   = "flat_terrain"
    n_subterrains:     tuple[int, int]    = (1, 1)
    subterrain_size:   tuple[float, float] = (12.0, 12.0)
    horizontal_scale:  float = 0.25
    vertical_scale:    float = 0.005
    terrain_height:    float = 0.0


class BallField(Field):
    cfg: BallFieldConfig

    @torch.compiler.disable
    def build(self):
        self.plane = self.scene.add_entity(
            morph=gs.morphs.Plane(),
            material=gs.materials.Rigid(friction=self.cfg.field_friction),
        )
        if self.cfg.use_terrain:
            tx = self.cfg.n_subterrains[0] * self.cfg.subterrain_size[0]
            ty = self.cfg.n_subterrains[1] * self.cfg.subterrain_size[1]
            self.terrain = self.scene.add_entity(
                morph=gs.morphs.Terrain(
                    n_subterrains=self.cfg.n_subterrains,
                    subterrain_size=self.cfg.subterrain_size,
                    horizontal_scale=self.cfg.horizontal_scale,
                    vertical_scale=self.cfg.vertical_scale,
                    subterrain_types=self.cfg.terrain_types,
                    pos=(-tx / 2, -ty / 2, self.cfg.terrain_height),
                ),
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

        def rnd(scale: float, shape: tuple):
            return scale * (torch.rand(self.scene.n_envs, *shape) - 0.5)
        self.ball.set_mass_shift(mass_shift=rnd(self.cfg.ball_mass_shift, shape=(1,)))
        self.ball.set_COM_shift(com_shift=rnd(self.cfg.ball_COM_shift, shape=(1, 3,)))

    @torch.compiler.disable
    def reset(self, envs_idx, **kwargs):
        self.ball.set_pos(envs_idx=envs_idx, pos=kwargs["ball_pos"])
        self.ball.zero_all_dofs_velocity(envs_idx=envs_idx)

    @torch.compiler.disable
    def get_state(self, envs_idx) -> dict[str, torch.Tensor]:
        return {
            "ball_pos": self.ball.get_pos(envs_idx=envs_idx),
            "ball_vel": self.ball.get_vel(envs_idx=envs_idx),
        }
