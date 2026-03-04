# single_walker.py
#   Build up a single walker env
#

import torch
import numpy as np
import genesis as gs
import gymnasium as gym
from dataclasses import dataclass, field


@dataclass(kw_only = True)
class FieldConfig():
    pass


class Field():
    def __init__(self, cfg: FieldConfig, scene: gs.Scene):
        super().__init__()
        self.cfg = cfg
        self.scene = scene
    
    @torch.no_grad()
    @torch.compiler.disable
    def build(self):
        self.plane = self.scene.add_entity( morph=gs.morphs.Plane() )
    
    def config(self):
        pass

    def get_state(self, envs_idx: torch.Tensor) -> dict[str, torch.Tensor]:
        return {} 
      
    def reset(self, envs_idx: torch.Tensor, **kwargs):
        pass
