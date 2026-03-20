import numpy as np
import genesis as gs

from algorithms.algorithm import Algorithm, RunConfig
from envs.walker import WalkEnv, WalkEnvConfig
from robots.mos9 import MOS9, MOS9Config
from fields.field import Field, FieldConfig
from models.mos9_walk_model import MOS9WalkModelConfig, MOS9WalkModel


gs.init(
    backend=gs.gpu,  # type: ignore[unsolved-attribute]
    performance_mode=True,
    logging_level="info",
)

Algorithm(
    WalkEnv(
        WalkEnvConfig(
            robot_cfg=MOS9Config(),
            robot_class=MOS9,
            field_cfg=FieldConfig(),
            field_class=Field,
            model_cfg=MOS9WalkModelConfig(
                n_dofs=12,
                target_q_offset=np.array([0.0] * 12),
            ),
            policy_freq=50,
            sim_freq=500,
            model_class=MOS9WalkModel,
            env_spacing=0.0,
            num_envs=1,
            show_viewer=True,
        )
    ),
    RunConfig(
        checkpoint_path="models_ckpt/walk_v3_t8.pt",
    ),
).eval()
