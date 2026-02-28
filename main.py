import os
import numpy as np
import genesis as gs
import torch

from envs.field import FieldConfig
from envs.robot import RobotConfig
from envs.controlled_robot import ControlledRobotConfig
from envs.single_walker import SingleWalkerEnv, SingleWalkerEnvConfig
from envs.dribble import DribbleEnv, DribbleEnvConfig


MODEL_PATH = f"models/walk_v7_t5.pt"
NUM_ENVS = 1
DEVICE = "cuda"
FIELD_RANGE = 2.0

gs.init(backend=gs.gpu,  # type: ignore[unsolved-attribute]
        performance_mode=True, 
        logging_level='info')

env = DribbleEnv(DribbleEnvConfig(
    robot_cfg = ControlledRobotConfig(
        policy_path     = "a",
        robot_URDF      = 'assets/MOS9/MOS9.urdf',
        kp              = np.array([70.0] * 12, dtype=np.float32),
        kv              = np.array([3.0] * 12, dtype=np.float32), 
        joint_names     = ['b_Rh', 'b_Lh', 'Rh_Rl', 'Lh_Ll', 'Rl_Rl1', 'Ll_Ll1', 
                           'Rl1_Rl2', 'Ll1_Ll2', 'Rl2_Ra', 'Ll2_La', 'Ra_Rf', 'La_Lf'], 
        initial_pos     = np.array([0.0, 0.0, 0.51]),
        force_range     = np.array([[-100] * 12, [100] * 12], dtype=np.float32), 
        velocity_range  = np.array([[-100] * 12, [100] * 12], dtype=np.float32), 
    ),  
    field_cfg = FieldConfig(), 
    field_range     = FIELD_RANGE, 
    num_envs        = NUM_ENVS,
    show_viewer     = True, 
))



env.reset()
actions = torch.ones((NUM_ENVS, 3), dtype=torch.float, device=DEVICE)
while True: 
    env.step(actions) 
