import numpy as np
import genesis as gs

from algorithms.algorithm import Algorithm, RunConfig
from envs.walker import WalkEnv, WalkEnvConfig
from robots.pi import PI, PIConfig
from fields.field import Field, FieldConfig
from models.pi_walk_model import PIWalkModelConfig, PIWalkModel


gs.init(
    backend=gs.gpu,  # type: ignore[unsolved-attribute]
    performance_mode=True,
    logging_level="info",
)

Algorithm(
    WalkEnv(
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
            num_envs=1,
            show_viewer=True,
        )
    ),
    RunConfig(
        checkpoint_path="models_ckpt/pi_policy.pt",
    ),
).eval()
