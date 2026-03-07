# robot.py
#   Inherit from robot and build up mos9
#

import torch
import numpy as np
import genesis as gs
import gymnasium as gym
from dataclasses import dataclass, field

from .robot import Robot, RobotConfig


@dataclass(kw_only = True)
class MOS9Config(RobotConfig):
    robot_URDF:     str         = 'assets/MOS9/MOS9_walk.urdf'
    base_link_name: str         = "body"
    joint_names:    list[str]   = field(default_factory=\
            lambda: ['b_Lh','Lh_Ll','Ll_Ll1','Ll1_Ll2','Ll2_La','La_Lf', 
                     'b_Rh','Rh_Rl','Rl_Rl1','Rl1_Rl2','Rl2_Ra','Ra_Rf'])
    kp:             np.ndarray  = field(default_factory=\
            lambda: np.array([100.0, 100.0,100.0, 100.0, 50.0, 24.0,
                              100.0, 100.0,100.0, 100.0, 50.0, 24.0], 
                             dtype=np.float32))
    kv:             np.ndarray  = field(default_factory=\
            lambda: np.array([2.0, 2.0, 2.0, 2.0,1.5,0.3,
                              2.0, 2.0, 2.0, 2.0,1.5,0.3], 
                             dtype=np.float32))
    initial_pos:    np.ndarray  = field(default_factory=\
            lambda: np.array([0.0, 0.0, 0.5], dtype=np.float32))
    force_range:    np.ndarray  = field(default_factory=\
            lambda: np.array([[-100] * 12, [100] * 12], dtype=np.float32))
    velocity_range: np.ndarray  = field(default_factory=\
            lambda: np.array([[-100] * 12, [100] * 12], dtype=np.float32)) 


class MOS9(Robot):
    pass

