import torch
import numpy as np
import genesis as gs

from envs.walker    import WalkEnv,     WalkEnvConfig
from robots.mos9    import MOS9,        MOS9Config
from fields.field   import Field,       FieldConfig
from fields.soccer_field        import (SoccerField, 
                                        SoccerFieldConfig)
from models.mos9_walk_model     import (MOS9WalkModelConfig, 
                                        MOS9WalkModel)

NUM_ENVS = 1
MODEL_PATH = f"models_ckpt/walk_v3_t8.pt"
COMPILE = False

gs.init(backend=gs.gpu,  # type: ignore[unsolved-attribute]
        performance_mode=True, 
        logging_level='info')

env = WalkEnv(WalkEnvConfig(
    robot_cfg       = MOS9Config(),
    robot_class     = MOS9,
    field_cfg       = FieldConfig(),
    field_class     = Field,
    model_cfg       = MOS9WalkModelConfig(
        n_dofs = 12, 
        target_q_offset = np.array([0.0] * 12)
    ), 
    policy_freq     = 50, 
    sim_freq        = 500,
    model_class     = MOS9WalkModel,
    env_spacing     = 0.0, 
    num_envs        = NUM_ENVS,
    show_viewer     = True, 
))


obs, info = env.reset()
actions = torch.zeros((NUM_ENVS, 12), dtype=torch.float, device=gs.device)
actor = torch.jit.load(MODEL_PATH).to(gs.device)
if COMPILE:
    actor = torch.compile(actor)
while True: 
    actions = actor(obs)
    obs, rew, terminated, truncated, info = env.step(actions) 
