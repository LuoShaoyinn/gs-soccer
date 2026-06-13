import os
from argparse import ArgumentParser

import numpy as np
import torch

import genesis as gs
from envs.env import Env, EnvConfig
from robots.pi import PI, PIConfig
from fields.ball_field import BallField, BallFieldConfig
from models.sim2sim_soccer import Sim2SimSoccerModel, Sim2SimSoccerConfig
from algorithm.actor import Actor, ActorConfig


def parse_args():
    p = ArgumentParser(description="PiPlus soccer sim2sim — Genesis")
    p.add_argument("--mode", type=str, default="walk",
                   choices=["walk", "kick", "approach_kick"])
    p.add_argument("--kick-speed", type=float, default=2.0)
    p.add_argument("--kick-dir-deg", type=float, default=0.0)
    p.add_argument("--vel-x", type=float, default=0.5)
    p.add_argument("--vel-y", type=float, default=0.0)
    p.add_argument("--vel-yaw", type=float, default=0.0)
    p.add_argument("--num-envs", type=int, default=1)
    p.add_argument("--steps", type=int, default=2500)
    p.add_argument("--viewer", action="store_true", default=True)
    p.add_argument("--no-viewer", dest="viewer", action="store_false")
    return p.parse_args()


def main():
    args = parse_args()
    if args.viewer:
        os.environ.pop("PYOPENGL_PLATFORM", None)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

    gs.init(backend=gs.gpu, performance_mode=True, logging_level="warning")

    ball_r_min, ball_r_max = {
        "walk": (3.8, 4.2),
        "approach_kick": (3.0, 4.0),
        "kick": (0.4, 1.0),
    }[args.mode]

    n_joints = 20

    robot_cfg = PIConfig(
        initial_pos=np.array([0.0, 0.0, 0.351], dtype=np.float32),
        kp=np.array([
            50.97, 32.51, 50.97, 32.51,
            50.97, 32.51, 50.97, 32.51,
            50.97, 32.51, 50.97, 32.51,
            50.97, 32.51, 50.97, 32.51,
            50.97, 50.97, 50.97, 50.97,
        ], dtype=np.float32),
        kv=np.array([
            3.24, 2.07, 3.24, 2.07,
            3.24, 2.07, 3.24, 2.07,
            3.24, 2.07, 3.24, 2.07,
            3.24, 2.07, 3.24, 2.07,
            3.24, 3.24, 3.24, 3.24,
        ], dtype=np.float32),
        force_range=np.array([
            np.full(n_joints, -20.0, dtype=np.float32),
            np.full(n_joints, 20.0, dtype=np.float32),
        ]),
        armature=np.array([
            0.01291, 0.008234, 0.01291, 0.008234,
            0.01291, 0.008234, 0.01291, 0.008234,
            0.01291, 0.008234, 0.01291, 0.008234,
            0.01291, 0.008234, 0.01291, 0.008234,
            0.01291, 0.01291, 0.01291, 0.01291,
        ], dtype=np.float32),
        damping=0.0,
    )

    env = Env(EnvConfig(
        robot_cfg=robot_cfg,
        robot_class=PI,
        field_cfg=BallFieldConfig(
            ball_radius=0.07,
            ball_mass=0.16,
            ball_damping=0.0,
            ball_friction=0.6,
            field_friction=1.0,
            ball_reset_radius=(ball_r_min, ball_r_max),
            ball_reset_noise=0.0,
        ),
        field_class=BallField,
        model_cfg=Sim2SimSoccerConfig(
            mode=args.mode,
            vel_cmd=(args.vel_x, args.vel_y, args.vel_yaw),
            kick_speed=args.kick_speed,
            kick_dir_deg=args.kick_dir_deg,
        ),
        model_class=Sim2SimSoccerModel,
        policy_freq=50,
        sim_freq=200,
        show_viewer=args.viewer,
        num_envs=args.num_envs,
        env_spacing=3.0,
    ))

    algo = Actor(env, ActorConfig(max_steps=args.steps))
    print(f"mode={args.mode}, kick_speed={args.kick_speed}")
    algo.eval()

    env.close()


if __name__ == "__main__":
    main()
