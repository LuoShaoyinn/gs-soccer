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


@dataclass
class FieldConfig():
    pass


class Field():
    def __init__(self, cfg: FieldConfig, scene: gs.Scene):
        super().__init__()
        self.cfg = cfg
        self.scene = scene
    
    #@torch.no_grad()
    #@torch.compiler.disable # prevent torch from compiling underlying gs
    def build(self):
        self.plane = self.scene.add_entity( morph=gs.morphs.Plane() )
    
    #@torch.no_grad()
    #@torch.compiler.disable # prevent torch from compiling underlying gs
    def config(self):
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
