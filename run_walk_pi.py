import torch
import numpy as np
import genesis as gs

from envs.walker import WalkEnv, WalkEnvConfig
from robots.pi import PI, PIConfig
from fields.field import Field, FieldConfig
from models.pi_walk_model import PIWalkModelConfig, PIWalkModel


NUM_ENVS = 1
MODEL_PATH = "models_ckpt/pi_policy.pt"
COMPILE = False

gs.init(
    backend=gs.gpu,  # type: ignore[unsolved-attribute]
    performance_mode=True,
    logging_level="info",
)

env = WalkEnv(
    WalkEnvConfig(
        robot_cfg=PIConfig(),
        robot_class=PI,
        field_cfg=FieldConfig(),
        field_class=Field,
        model_cfg=PIWalkModelConfig(
            n_dofs=20,
            target_q_offset=np.array(
                [
                    -0.25,
                    0.0,
                    -0.25,
                    0.0,
                    0.0,
                    0.2,
                    0.0,
                    -0.2,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.65,
                    -1.2,
                    0.65,
                    -1.2,
                    -0.4,
                    -0.4,
                    0.0,
                    0.0,
                ],
                dtype=np.float32,
            ),
        ),
        policy_freq=50,
        sim_freq=500,
        model_class=PIWalkModel,
        env_spacing=0.0,
        num_envs=NUM_ENVS,
        show_viewer=True,
    )
)

obs, info = env.reset()
actions = torch.zeros((NUM_ENVS, 20), dtype=torch.float, device=gs.device)
actor = torch.jit.load(MODEL_PATH).to(gs.device)
if COMPILE:
    actor = torch.compile(actor)
while True:
    actions = actor(obs)
    obs, rew, terminated, truncated, info = env.step(actions)
