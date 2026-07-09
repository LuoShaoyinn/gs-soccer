# terrain_field.py
#   Field with terrain support (inherits Field)
#

import torch
import genesis as gs
from dataclasses import dataclass, field

from .field import FieldConfig, Field


@dataclass(kw_only=True)
class TerrainFieldConfig(FieldConfig):
    field_friction: float = 0.08
    # Terrain (disabled by default — uses flat Plane)
    use_terrain:           bool  = False
    terrain_types:         str   = "flat_terrain"
    n_subterrains:         tuple[int, int]      = (1, 1)
    subterrain_size:       tuple[float, float]  = (12.0, 12.0)
    horizontal_scale:      float = 0.25
    vertical_scale:        float = 0.005
    terrain_pos:           tuple[float, float, float] = (0.0, 0.0, 0.0)
    subterrain_parameters: dict  = field(default_factory=lambda: {})


class TerrainField(Field):
    cfg: TerrainFieldConfig

    @torch.compiler.disable
    def build(self):
        self.plane = self.scene.add_entity(
            morph=gs.morphs.Plane(),
            material=gs.materials.Rigid(friction=self.cfg.field_friction),
        )
        if self.cfg.use_terrain:
            self.terrain = self.scene.add_entity(
                morph=gs.morphs.Terrain(
                    n_subterrains=self.cfg.n_subterrains,
                    subterrain_size=self.cfg.subterrain_size,
                    horizontal_scale=self.cfg.horizontal_scale,
                    vertical_scale=self.cfg.vertical_scale,
                    subterrain_types=self.cfg.terrain_types,
                    subterrain_parameters=self.cfg.subterrain_parameters,
                    pos=self.cfg.terrain_pos,
                ),
                material=gs.materials.Rigid(friction=self.cfg.field_friction),
            )
