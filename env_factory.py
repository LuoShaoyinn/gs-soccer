import numpy as np

from envs.env import Env, EnvConfig
from fields.terrain_field import TerrainField, TerrainFieldConfig
from MDPs.sim2sim_walk import WalkConfig, WalkMDP
from robots.pi import PI, PIConfig


def make_env(num_envs=1, viewer=False, domain_randomization=False):
    n_joints = 20
    robot_cfg = PIConfig(
        initial_pos=np.array([0.0, 0.0, 0.50], dtype=np.float32),
        kp=np.array(
            [
                50.97, 32.51, 50.97, 32.51,
                50.97, 32.51, 50.97, 32.51,
                50.97, 32.51, 50.97, 32.51,
                50.97, 32.51, 50.97, 32.51,
                50.97, 50.97, 50.97, 50.97,
            ],
            dtype=np.float32,
        ),
        kv=np.array(
            [
                3.24, 2.07, 3.24, 2.07,
                3.24, 2.07, 3.24, 2.07,
                3.24, 2.07, 3.24, 2.07,
                3.24, 2.07, 3.24, 2.07,
                3.24, 3.24, 3.24, 3.24,
            ],
            dtype=np.float32,
        ),
        force_range=np.array(
            [
                np.full(n_joints, -20.0, dtype=np.float32),
                np.full(n_joints, 20.0, dtype=np.float32),
            ],
            dtype=np.float32,
        ),
        armature=np.array(
            [
                0.01291, 0.008234, 0.01291, 0.008234,
                0.01291, 0.008234, 0.01291, 0.008234,
                0.01291, 0.008234, 0.01291, 0.008234,
                0.01291, 0.008234, 0.01291, 0.008234,
                0.01291, 0.01291, 0.01291, 0.01291,
            ],
            dtype=np.float32,
        ),
        damping=np.zeros(n_joints, dtype=np.float32),
    )

    field_cfg = TerrainFieldConfig(
        field_friction=1.0,
        use_terrain=True,
        terrain_types="random_uniform_terrain",
        n_subterrains=(1, 1),
        subterrain_size=(6.0, 6.0),
        horizontal_scale=0.1,
        vertical_scale=0.1,
        terrain_pos=(-3.0, -3.0, 0.01),
        subterrain_parameters={
            "random_uniform_terrain": {
                "min_height": 0.0,
                "max_height": 0.01,
            },
        },
    )

    return Env(
        EnvConfig(
            robot_cfg=robot_cfg,
            robot_class=PI,
            field_cfg=field_cfg,
            field_class=TerrainField,
            MDP_cfg=WalkConfig(
                vel_cmd=(0.5, 0.0, 0.0),
                init_height=float(robot_cfg.initial_pos[2]),
                termination_grace_steps=50,
                max_episode_steps=500,
            ),
            MDP_class=WalkMDP,
            policy_freq=50,
            sim_freq=200,
            show_viewer=viewer,
            num_envs=num_envs,
            env_spacing=3.0,
            domain_randomization=domain_randomization,
        )
    )
