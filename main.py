import os, time
import numpy as np
import genesis as gs
import torch

from envs.field import FieldConfig
from envs.robot import RobotConfig
from envs.controlled_robot import ControlledRobotConfig
from envs.single_walker import SingleWalkerEnv, SingleWalkerEnvConfig
from envs.dribble import DribbleEnv, DribbleEnvConfig


MODEL_PATH = f"models/walk_v5_t5.pt"
NUM_ENVS = 16
DEVICE = "cuda"
FIELD_RANGE = 2.0

gs.init(backend=gs.gpu,  # type: ignore[unsolved-attribute]
        performance_mode=True, 
        logging_level='info')

'''
        init_joint_pos = {
            "b_Lh": 0.3,
            "Ll1_Ll2": -0.6,
            "Ll2_La": 0.3,

            "b_Rh": -0.3,
            "Rl1_Rl2": 0.6,
            "Rl2_Ra": -0.3,
            }
'''

env = SingleWalkerEnv(SingleWalkerEnvConfig(
    robot_cfg = RobotConfig(
        robot_URDF      = 'assets/MOS9/MOS9_walk.urdf',
        kp              = np.array([100.0, 100.0,100.0, 100.0, 50.0, 24.0,
                                    100.0, 100.0,100.0, 100.0, 50.0, 24.0], 
                                   dtype=np.float32),
        kv              = np.array([2.0, 2.0, 2.0, 2.0,1.5,0.3,
                                    2.0, 2.0, 2.0, 2.0,1.5,0.3], 
                                   dtype=np.float32), 
        target_q_offset = np.array([0.3, 0.0, 0.0, -0.6, 0.3, 0.0,
                                    -0.3, 0.0, 0.0, 0.6, -0.3, 0.0],
                                   dtype=np.float32),
        joint_names     = ['b_Lh','Lh_Ll','Ll_Ll1','Ll1_Ll2','Ll2_La','La_Lf', 
                           'b_Rh','Rh_Rl','Rl_Rl1','Rl1_Rl2','Rl2_Ra','Ra_Rf'],
        base_link_name  = "body", 
        initial_pos     = np.array([0.0, 0.0, 0.51]),
        force_range     = np.array([[-100] * 12, [100] * 12], dtype=np.float32), 
        velocity_range  = np.array([[-100] * 12, [100] * 12], dtype=np.float32), 
    ),  
    field_range     = FIELD_RANGE, 
    num_envs        = NUM_ENVS,
    show_viewer     = True, 
))


actor = torch.jit.load(MODEL_PATH).to(gs.device)
obs, info = env.reset()
time.sleep(1)
while True: 
    actions = actor(obs)
    obs, rew, terminated, truncated, info = env.step(actions) 
