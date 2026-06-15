import os
from argparse import ArgumentParser

import numpy as np
import torch

import genesis as gs
from envs.env import Env, EnvConfig
from robots.pi import PI, PIConfig
from fields.ball_field import BallField, BallFieldConfig
from MDPs import DummyMDP, DummyMDPConfig


def parse_args():
    p = ArgumentParser()
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--num-envs", type=int, default=1)
    p.add_argument("--viewer", action="store_true", default=True)
    p.add_argument("--no-viewer", dest="viewer", action="store_false")
    return p.parse_args()


def main():
    args = parse_args()
    if args.viewer:
        os.environ.pop("PYOPENGL_PLATFORM", None)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

    gs.init(backend=gs.gpu, performance_mode=True, logging_level="warning")

    robot_cfg = PIConfig()
    field_cfg = BallFieldConfig()
    cfg = EnvConfig(
        robot_cfg=robot_cfg,
        robot_class=PI,
        field_cfg=field_cfg,
        field_class=BallField,
        MDP_cfg=DummyMDPConfig(
            action_dim=len(robot_cfg.joint_names),
            home_pose=np.zeros(len(robot_cfg.joint_names), dtype=np.float32),
            base_pos=robot_cfg.initial_pos,
            base_quat=robot_cfg.initial_quat,
            ball_pos=field_cfg.ball_init_pos,
        ),
        MDP_class=DummyMDP,
        policy_freq=50,
        sim_freq=500,
        num_envs=args.num_envs,
        show_viewer=args.viewer,
    )
    env = Env(cfg)

    envs_idx = torch.arange(args.num_envs, dtype=torch.long, device=gs.device)
    env.reset(envs_idx)
    action = torch.zeros((args.num_envs, len(robot_cfg.joint_names)), device=gs.device)

    for _ in range(args.steps):
        env.step(action)

    env.close()


if __name__ == "__main__":
    main()
