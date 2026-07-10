import argparse
import os
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

import genesis as gs
from algorithm.pi_visual_walk.teacher import WalkTeacher, action_clip_bounds, clip_walk_action
from env_factory import make_env


def parse_args():
    p = argparse.ArgumentParser(description="Check the exported sim2sim walk teacher in Genesis")
    p.add_argument("--num-envs", type=int, default=32)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--settle-steps", type=int, default=50)
    p.add_argument("--model-dir", type=str, default="refs/piplus_soccer_sim2sim/models/exported")
    p.add_argument("--model-file", type=str, default=None)
    p.add_argument("--log-dir", type=str, default=None)
    p.add_argument("--viewer", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def main():
    args = parse_args()
    if args.viewer:
        os.environ.pop("PYOPENGL_PLATFORM", None)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    gs.init(backend=gs.gpu, performance_mode=True, logging_level="warning")
    dev = torch.device(gs.device)

    env = make_env(num_envs=args.num_envs, viewer=args.viewer)
    teacher = WalkTeacher(args.model_dir, model_file=args.model_file, device=dev)
    action_low, action_high = action_clip_bounds(dev)

    log_dir = args.log_dir or f"runs/teacher_walk_{datetime.now().strftime('%m%d_%H%M')}"
    writer = SummaryWriter(log_dir)

    obs, info = env.reset()
    teacher.reset(args.num_envs, dev)
    zeros = torch.zeros(args.num_envs, env.action_space.shape[0], device=dev)
    for _ in range(args.settle_steps):
        obs, _, _, _, info = env.step(zeros)

    ep_reward = torch.zeros(args.num_envs, device=dev)
    reset_mask = torch.zeros(args.num_envs, dtype=torch.bool, device=dev)
    for step in range(args.steps):
        with torch.no_grad():
            action = clip_walk_action(teacher.infer(obs, reset_mask=reset_mask), action_low, action_high)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated | truncated
        reset_mask = done.squeeze(-1)
        ep_reward += reward.squeeze(-1)

        if step % 10 == 0:
            writer.add_scalar("reward/total", reward.float().mean().detach().item(), step)
            writer.add_scalar("teacher/action_abs", action.abs().mean().detach().item(), step)
            writer.add_scalar("episode/reward_live", ep_reward.mean().detach().item(), step)
            for key, value in info.items():
                if torch.is_tensor(value):
                    scalar = value.float().mean().detach().item()
                else:
                    scalar = float(value)
                if key.startswith("r_"):
                    writer.add_scalar(f"reward/{key[2:]}", scalar, step)
                else:
                    writer.add_scalar(f"env/{key}", scalar, step)

        if step % 100 == 0:
            n_fall = int(reset_mask.sum().item())
            print(
                f"step={step:4d} z={info['body_pos_z'].float().mean().item():.3f} "
                f"vx={info['walk_vx_body'].float().mean().item():+.3f} "
                f"r={reward.float().mean().item():+.4f} |a|={action.abs().mean().item():.2f} "
                f"falls(total)={n_fall}"
            )

        if done.any():
            ep_reward[done.squeeze(-1)] = 0.0

    writer.close()
    print(f"teacher check logs: {log_dir}")


if __name__ == "__main__":
    main()
