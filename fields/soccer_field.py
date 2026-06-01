# single_walker.py
#   Build up a single walker env
#

import random
import torch
import numpy as np
import genesis as gs
import gymnasium as gym
from torch.nn import functional as F
from dataclasses import dataclass, field
from typing import Optional, Callable

from .field import FieldConfig, Field


@dataclass(kw_only=True)
class SoccerFieldConfig(FieldConfig):
    half_field_size: tuple = (4.5, 3.0)
    fence_height: float = 0.5
    goal_width: float = 1.9
    goal_height: float = 0.7
    field_color: tuple = (0.3, 1.0, 0.3)
    fence_color: tuple = (0.9, 0.3, 0.9)
    red_goal_color: tuple = (1.0, 0.3, 0.3)
    blue_goal_color: tuple = (0.3, 0.3, 1.0)
    ball_radius: float = 0.08
    ball_init_pos: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.05], dtype=np.float32)
    )
    field_friction: float = 0.08
    ball_friction: float = 0.08
    ball_mass: float = 0.200
    ball_damping: float = 2.0
    textured: bool = False


class SoccerField(Field):
    cfg: SoccerFieldConfig

    @torch.compiler.disable
    def build(self):
        self._textures = None
        if self.cfg.textured:
            from .textures import ensure_textures
            self._textures = ensure_textures()

        self.field = self.scene.add_entity(
            morph=gs.morphs.Plane(),
            surface=gs.surfaces.Rough(color=self.cfg.field_color),
            material=gs.materials.Rigid(friction=self.cfg.field_friction),
        )

        if self._textures:
            from .meshes import field_obj_path, sphere_obj_path
            half_l, half_w = self.cfg.half_field_size
            self.field_visual = self.scene.add_entity(
                morph=gs.morphs.Mesh(
                    file=field_obj_path(half_l, half_w),
                    fixed=True,
                    collision=False,
                    decimate=False,
                ),
                surface=gs.surfaces.Rough(
                    diffuse_texture=gs.textures.ImageTexture(
                        image_path=self._textures["field"],
                        encoding="srgb",
                    ),
                ),
            )

            self.ball = self.scene.add_entity(
                morph=gs.morphs.Mesh(
                    file=sphere_obj_path(self.cfg.ball_radius, 24, 12),
                    pos=(0, 0, 1.0),
                ),
                surface=gs.surfaces.Rough(
                    diffuse_texture=gs.textures.ImageTexture(
                        image_path=self._textures["ball"],
                        encoding="srgb",
                    ),
                ),
                material=gs.materials.Rigid(friction=self.cfg.ball_friction),
            )
        else:
            self.ball = self.scene.add_entity(
                morph=gs.morphs.Sphere(
                    radius=self.cfg.ball_radius,
                    pos=(0, 0, 1.0),
                ),
                surface=gs.surfaces.Rough(),
                material=gs.materials.Rigid(friction=self.cfg.ball_friction),
            )

        self.__gs_build_virtual()

    @torch.compiler.disable
    def __gs_build_virtual(self):
        half_field_length = self.cfg.half_field_size[0]
        half_field_width = self.cfg.half_field_size[1]
        fence_height = self.cfg.fence_height
        goal_height = self.cfg.goal_height

        def add_fence(x: float, y: float, xx: float, yy: float):
            return self.scene.add_entity(
                morph=gs.morphs.Box(
                    pos=(x, y, fence_height / 2 - 0.05),
                    size=(xx, yy, fence_height),
                    fixed=True,
                    collision=False,
                ),
                surface=gs.surfaces.Rough(color=self.cfg.fence_color),
            )

        self.fences = [
            add_fence(0.0, half_field_width, 2.0 * half_field_length, 0.05),
            add_fence(0.0, -half_field_width, 2.0 * half_field_length, 0.05),
            add_fence(half_field_length, 0.0, 0.05, 2.0 * half_field_width),
            add_fence(-half_field_length, 0.0, 0.05, 2.0 * half_field_width),
        ]

        if not self._textures:
            def add_marker(x: float, y: float, xx: float, yy: float):
                return self.scene.add_entity(
                    morph=gs.morphs.Box(
                        pos=(x, y, 0.001), size=(xx, yy, 0.001), fixed=True, collision=False
                    ),
                    surface=gs.surfaces.Rough(color=(1.0, 1.0, 1.0)),
                )

            self.markers = [add_marker(0.0, 0.0, 0.05, 2.0 * half_field_width)]

        goal_red_surface = gs.surfaces.Rough(color=self.cfg.red_goal_color)
        goal_blue_surface = gs.surfaces.Rough(color=self.cfg.blue_goal_color)

        self.red_goal = self.scene.add_entity(
            morph=gs.morphs.Box(
                pos=(half_field_length - 0.02, 0.0, goal_height / 2 - 0.05),
                quat=(1.0, 0.0, 0.0, 0.0),
                size=(0.2, self.cfg.goal_width, goal_height),
                fixed=True,
                collision=False,
            ),
            surface=goal_red_surface,
        )
        self.blue_goal = self.scene.add_entity(
            morph=gs.morphs.Box(
                pos=(-half_field_length + 0.02, 0.0, goal_height / 2 - 0.05),
                quat=(1.0, 0.0, 0.0, 0.0),
                size=(0.2, self.cfg.goal_width, goal_height),
                fixed=True,
                collision=False,
            ),
            surface=goal_blue_surface,
        )

        if self._textures:
            from .meshes import sphere_obj_path

            post_surface = gs.surfaces.Smooth(
                color=(0.85, 0.85, 0.85),
                metallic=0.8,
                roughness=0.3,
            )
            post_r = 0.025
            gw2 = self.cfg.goal_width / 2
            for sign, base_x in [(1, half_field_length), (-1, -half_field_length)]:
                px = base_x + sign * post_r / 2
                self.scene.add_entity(
                    morph=gs.morphs.Box(
                        pos=(px, -gw2, goal_height / 2 - 0.05),
                        size=(post_r, post_r, goal_height),
                        fixed=True, collision=False,
                    ),
                    surface=post_surface,
                )
                self.scene.add_entity(
                    morph=gs.morphs.Box(
                        pos=(px, gw2, goal_height / 2 - 0.05),
                        size=(post_r, post_r, goal_height),
                        fixed=True, collision=False,
                    ),
                    surface=post_surface,
                )
                self.scene.add_entity(
                    morph=gs.morphs.Box(
                        pos=(px, 0.0, goal_height - 0.05),
                        size=(post_r, self.cfg.goal_width + post_r, post_r),
                        fixed=True, collision=False,
                    ),
                    surface=post_surface,
                )

            self.sky = self.scene.add_entity(
                morph=gs.morphs.Mesh(
                    file=sphere_obj_path(25.0, 32, 16, inverted=True),
                    fixed=True,
                    collision=False,
                    decimate=False,
                ),
                surface=gs.surfaces.Emission(
                    emissive_texture=gs.textures.ImageTexture(
                        image_path=self._textures["sky"],
                        encoding="srgb",
                    ),
                    double_sided=True,
                ),
            )

    @torch.compiler.disable
    def config(self):
        self.ball.set_mass(self.cfg.ball_mass)
        self.ball.set_dofs_damping(self.cfg.ball_damping, dofs_idx_local=(3, 4, 5))
        self.ball_init_pos = torch.from_numpy(self.cfg.ball_init_pos).to(gs.device)

    @torch.compiler.disable
    def reset(
        self,
        envs_idx: torch.Tensor,
        ball_pos: torch.Tensor | None = None,
        ball_mass_shift: torch.Tensor | None = None,
        **kwargs,
    ) -> None:  # type: ignore[override]
        if ball_pos is None:
            ball_pos = self.ball_init_pos.broadcast_to((envs_idx.shape[0], 3))
        elif ball_pos.shape[1] == 2:
            ball_pos = F.pad(ball_pos, (0, 1), value=self.cfg.ball_radius + 0.05)
        if ball_mass_shift is not None:
            self.ball.set_mass_shift(envs_idx=envs_idx, mass_shift=ball_mass_shift)
        self.ball.set_pos(envs_idx=envs_idx, pos=ball_pos)
        self.ball.zero_all_dofs_velocity(envs_idx=envs_idx)

    @torch.compiler.disable
    def get_state(self, envs_idx=torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "ball_pos": self.ball.get_pos(envs_idx=envs_idx),
            "ball_vel": self.ball.get_vel(envs_idx=envs_idx),
        }
