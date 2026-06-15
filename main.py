import os
from argparse import ArgumentParser
from dataclasses import dataclass

import numpy as np
import torch
import gymnasium as gym

import genesis as gs
from envs.env import Env, EnvConfig
from robots.pi import PI, PIConfig
from fields.ball_field import BallField, BallFieldConfig
from MDPs import MDP, MDPConfig

# 
# Example of model - but please implement it inside the model/ dir
#
@dataclass(kw_only=True)
class StandingMDPConfig(MDPConfig):
    action_dim: int

class StandingMDP(MDP):
    cfg: StandingMDPConfig

    @property
    def observation_space(self) -> gym.spaces.Box:
        dim = self.cfg.action_dim
        return gym.spaces.Box(low=-1e6, high=1e6, shape=(dim * 2,), dtype=np.float32)

    @property
    def action_space(self) -> gym.spaces.Box:
        return gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.cfg.action_dim,), dtype=np.float32
        )

    def build_observation(self, envs_idx, dofs_pos=None, dofs_vel=None, **kwargs):
        if dofs_pos is None or dofs_vel is None:
            n = envs_idx.shape[0]
            return torch.zeros(
                (n, self.cfg.action_dim * 2), dtype=torch.float32, device=gs.device
            )
        return torch.cat([dofs_pos, dofs_vel], dim=-1)


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
    cfg = EnvConfig(
        robot_cfg=robot_cfg,
        robot_class=PI,
        field_cfg=BallFieldConfig(),
        field_class=BallField,
        model_cfg=StandingModelConfig(action_dim=len(robot_cfg.joint_names)),
        model_class=StandingModel,
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
