"""Evaluate the pretrained kick actor on the sparse floor-IQL task."""

import argparse
import os
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("XDG_CACHE_HOME", "/tmp/gs-soccer-cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/gs-soccer-matplotlib")

import genesis as gs

from envs import Env, EnvConfig
from fields import BallField, BallFieldConfig
from robots import KickPI, KickPIConfig
from MDPs import FloorIQLConfig, FloorIQLMDP
from algorithm import TeacherActor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--actor", default="refs/kick_ball_0625/20260624_144537_from20260624_111401/exported/actor.onnx")
    p.add_argument("--steps", type=int, default=350)
    p.add_argument("--num-envs", type=int, default=1)
    p.add_argument("--no-viewer", action="store_true")
    p.add_argument("--diagnostics", action="store_true")
    args = p.parse_args()

    gs.init(backend=gs.gpu, performance_mode=True, logging_level="warning")
    robot_cfg = KickPIConfig()
    field_cfg = BallFieldConfig(ball_radius=0.07, ball_mass=0.25, ball_damping=0.05, ball_friction=0.6, ball_init_pos=np.array([1.0, 0.0, 0.07], dtype=np.float32))
    cfg = EnvConfig(robot_cfg=robot_cfg, robot_class=KickPI, field_cfg=field_cfg, field_class=BallField, MDP_cfg=FloorIQLConfig(actor_path=str(Path(args.actor))), MDP_class=FloorIQLMDP, policy_freq=50, sim_freq=500, num_envs=args.num_envs, show_viewer=not args.no_viewer)
    env = Env(cfg)
    teacher = TeacherActor(env)
    env.reset()
    finished = np.zeros(args.num_envs, dtype=bool)
    for step in range(args.steps):
        state = env.get_state(env.all_envs_idx)
        action = teacher.act(state)
        if args.diagnostics and (step == 0 or (step + 1) % 25 == 0):
            phases = env.MDP._phase.tolist()
            delta_x = env.MDP._delta_x(env.all_envs_idx, state["ball_pos"]).tolist()
            print(f"step={step + 1:4d} phases={phases} delta_x={[round(v, 3) for v in delta_x]} robot_xy={state['body_pos'][:, :2].detach().cpu().numpy().round(3).tolist()} action=[{action.min().item():+.2f},{action.max().item():+.2f}]")
        _, reward, terminated, truncated, info = env.step(action)
        done = (terminated | truncated).squeeze(1).detach().cpu().numpy()
        new_done = done & ~finished
        if new_done.any():
            done_idx = torch.as_tensor(new_done, dtype=torch.bool, device=reward.device)
            print(f"episode_end step={step + 1} envs={np.flatnonzero(new_done).tolist()} success_mean={info['success'][done_idx].detach().float().mean().item():.3f} timeout_mean={info['timeout'][done_idx].detach().float().mean().item():.3f} reward_mean={reward.squeeze(1)[done_idx].detach().mean().item():+.6f} episode_return_mean={info['episode_return'][done_idx].detach().mean().item():+.6f}")
            finished |= done
        if finished.all():
            break
    env.close()


if __name__ == "__main__":
    main()
