import os, time
import numpy as np
import genesis as gs
import torch

from envs.single_walker import SingleWalkerEnv, SingleWalkerEnvConfig
from robots.mos9 import MOS9, MOS9Config
from fields.field import Field, FieldConfig

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
    robot_cfg       = MOS9Config(),  
    robot_class     = MOS9,
    field_cfg       = FieldConfig(),
    field_class     = Field,
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
