import numpy as np
import genesis as gs
import torch

from envs.walker    import WalkEnv,     WalkEnvConfig
from envs.dribble   import DribbleEnv,  DribbleEnvConfig
from robots.mos9    import MOS9,        MOS9Config
from fields.field   import Field,       FieldConfig
from fields.soccer_field        import (SoccerField, 
                                        SoccerFieldConfig)
from robots.controlled_robot    import (ControlledRobotWrapper, 
                                        ControlledRobotWrapperConfig)

MODEL_PATH = f"models_ckpt/walk_v3_t8.pt"
NUM_ENVS = 1
DEVICE = "cuda"
FIELD_RANGE = 2.0

gs.init(backend=gs.gpu,  # type: ignore[unsolved-attribute]
        performance_mode=True, 
        logging_level='info')

env = DribbleEnv(DribbleEnvConfig(
    robot_cfg       = ControlledRobotWrapperConfig(
        robot_cfg      = MOS9Config(),
        robot_class    = MOS9,
        policy_path    = MODEL_PATH,
    ),
    robot_class     = ControlledRobotWrapper,
    field_cfg       = SoccerFieldConfig(),
    field_class     = SoccerField,
    env_spacing     = FIELD_RANGE, 
    num_envs        = NUM_ENVS,
    show_viewer     = True, 
))


obs, info = env.reset()
actions = torch.tensor([0.2, 0, 0], 
                       dtype=torch.float, 
                       device=gs.device) \
                .broadcast_to((NUM_ENVS, 3))
while True: 
    obs, rew, terminated, truncated, info = env.step(actions) 
