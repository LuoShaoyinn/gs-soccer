# single_walker.py
#   Build up a single walker env
#

import torch
import numpy as np
import genesis as gs
import gymnasium as gym
import random
from dataclasses import dataclass, field
from typing import Optional, Callable

from .robot import RobotConfig, Robot

@dataclass
class FieldConfig():
    half_field_size:    tuple = (5.0, 3.0)
    fence_height:       float = 0.5
    goal_width:         float = 3
    goal_height:        float = 1
    field_color:        tuple = (0.3, 1.0, 0.3)
    fence_color:        tuple = (0.9, 0.3, 0.9)
    red_goal_color:     tuple = (1.0, 0.3, 0.3)
    blue_goal_color:    tuple = (0.3, 0.3, 1.0)
    ball_radius:        float = 0.1
    field_friction:     float = 1.0
    ball_friction:      float = 1.0
    ball_damping:       float = 5e-3


class Field():
    def __init__(self, cfg: FieldConfig, scene: gs.Scene):
        super().__init__()
        self.cfg = cfg
        self.scene = scene
    
    #@torch.no_grad()
    #@torch.compiler.disable # prevent torch from compiling underlying gs
    def gs_build(self):
        # Only field and ball have collision
        self.field = self.scene.add_entity(
            morph=gs.morphs.Plane(),
            surface=gs.surfaces.Rough(color=self.cfg.field_color),
            material=gs.materials.Rigid(friction=self.cfg.field_friction)
        )
        self.ball = self.scene.add_entity(
            morph=gs.morphs.Sphere(
                radius=self.cfg.ball_radius,
                pos=(0, 0, 1.0), 
            ),
            surface=gs.surfaces.Rough(), 
            material=gs.materials.Rigid(friction=self.cfg.ball_friction)
        )
        self.gs_build_virtual()

    def gs_build_virtual(self):
        half_field_length = self.cfg.half_field_size[0]
        half_field_width  = self.cfg.half_field_size[1]
        fence_height      = self.cfg.fence_height
        goal_height       = self.cfg.goal_height
        def add_fence(x: float, y: float, xx: float, yy: float):
            return self.scene.add_entity(
                morph=gs.morphs.Box(
                    pos  = (x, y, fence_height / 2 - 0.05),
                    size = (xx, yy, fence_height),
                    fixed = True,
                    collision = False
                ),
                surface=gs.surfaces.Rough(color=self.cfg.fence_color),
            )
        self.fences = [
            add_fence(0.0,  half_field_width, 2.0 * half_field_length, 0.05), 
            add_fence(0.0, -half_field_width, 2.0 * half_field_length, 0.05), 
            add_fence( half_field_length, 0.0, 0.05, 2.0 * half_field_width), 
            add_fence(-half_field_length, 0.0, 0.05, 2.0 * half_field_width), 
        ]
        def add_marker(x: float, y: float, xx: float, yy: float):
            return self.scene.add_entity(
                morph=gs.morphs.Box(
                    pos  = (x, y, 0.001), 
                    size = (xx, yy, 0.001), 
                    fixed = True,
                    collision = False
                ),
                surface=gs.surfaces.Rough(color=(1.0, 1.0, 1.0))
            )

        self.markers = [
            add_marker(0.0, 0.0, 0.05, 2.0 * half_field_width)
        ]
        self.red_goal = self.scene.add_entity(
            morph=gs.morphs.Box(
                pos  = (half_field_length - 0.02, 0.0, goal_height / 2 - 0.05),
                quat = (1.0, 0.0, 0.0, 0.0),
                size = (0.2, self.cfg.goal_width, goal_height),
                fixed = True,
                collision = False
            ),
            surface=gs.surfaces.Rough(color=self.cfg.red_goal_color),
        )
        self.blue_goal = self.scene.add_entity(
            morph=gs.morphs.Box(
                pos  = (-half_field_length + 0.02, 0.0, goal_height / 2 - 0.05),
                quat = (1.0, 0.0, 0.0, 0.0),
                size = (0.2, self.cfg.goal_width, goal_height),
                fixed = True,
                collision = False
            ),
            surface=gs.surfaces.Rough(color=self.cfg.blue_goal_color),
        )

    
    #@torch.no_grad()
    #@torch.compiler.disable # prevent torch from compiling underlying gs
    def gs_config(self):
        self.ball.set_dofs_damping(self.cfg.ball_damping)
 
    #@torch.no_grad()
    #@torch.compile()
    def step(self, action : torch.Tensor):
        pass
    
    #@torch.no_grad()
    #@torch.compile()
    def reset(self, envs_idx: torch.Tensor | None = None):
        pass

    #@torch.no_grad()
    def render(self):
        pass

    #@torch.no_grad()
    def close(self):
        pass
