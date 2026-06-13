import os
import math
from argparse import ArgumentParser

import numpy as np
import torch

import genesis as gs
from envs.env import Env, EnvConfig
from robots.pi import PI, PIConfig
from fields.ball_field import BallField, BallFieldConfig
from models.sim2sim_soccer import Sim2SimSoccerModel, Sim2SimSoccerConfig

NUM_GENESIS_JOINTS = 20

_KP = np.array([
    50.97, 32.51, 50.97, 32.51,
    50.97, 32.51, 50.97, 32.51,
    50.97, 32.51, 50.97, 32.51,
    50.97, 32.51, 50.97, 32.51,
    50.97, 50.97, 50.97, 50.97,
], dtype=np.float32)

_KD = np.array([
    3.24, 2.07, 3.24, 2.07,
    3.24, 2.07, 3.24, 2.07,
    3.24, 2.07, 3.24, 2.07,
    3.24, 2.07, 3.24, 2.07,
    3.24, 3.24, 3.24, 3.24,
], dtype=np.float32)

_ARMATURE = np.array([
    0.01291, 0.008234, 0.01291, 0.008234,
    0.01291, 0.008234, 0.01291, 0.008234,
    0.01291, 0.008234, 0.01291, 0.008234,
    0.01291, 0.008234, 0.01291, 0.008234,
    0.01291, 0.01291, 0.01291, 0.01291,
], dtype=np.float32)

_DEFAULT_POS = np.array([
    -0.25, 0.0, -0.25, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.65, 0.0, 0.65, 0.0,
    -0.4, -0.4, 0.0, 0.0,
], dtype=np.float32)


def parse_args():
    p = ArgumentParser(description="PiPlus soccer sim2sim — Genesis")
    p.add_argument("--model-dir", type=str, default="runs")
    p.add_argument("--model-file", type=str, default="pi_plus_actor.pt")
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

    robot_cfg = PIConfig(
        initial_pos=np.array([0.0, 0.0, 0.351], dtype=np.float32),
        kp=_KP,
        kv=_KD,
        force_range=np.array([
            np.full(NUM_GENESIS_JOINTS, -20.0, dtype=np.float32),
            np.full(NUM_GENESIS_JOINTS, 20.0, dtype=np.float32),
        ]),
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
            model_dir=args.model_dir,
            model_file=args.model_file,
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

    # zero URDF joint damping + set armature
    dofs_idx = env.robot.dofs_idx_local
    env.robot.robot.set_dofs_armature(
        torch.from_numpy(_ARMATURE).to(gs.device),
        dofs_idx_local=dofs_idx,
    )
    env.robot.robot.set_dofs_damping(
        torch.zeros(NUM_GENESIS_JOINTS, dtype=torch.float32, device=gs.device),
        dofs_idx_local=dofs_idx,
    )
    env.field.ball.set_dofs_damping(0.0, dofs_idx_local=(3, 4, 5))

    # reset + settle
    obs, info = env.reset()
    default_q = torch.from_numpy(_DEFAULT_POS).to(gs.device).unsqueeze(0).expand(args.num_envs, -1)
    for _ in range(50):
        env.robot.step(default_q)
        env.gs_step()

    zeros_22 = torch.zeros(args.num_envs, 22, dtype=torch.float32, device=gs.device)
    obs, _, _, _, info = env.step(zeros_22)

    print(f"mode={args.mode}, kick_speed={args.kick_speed}")

    for step in range(args.steps):
        action = env.model.act(obs)
        obs, reward, done, trunc, info = env.step(action)

        if step % 50 == 0:
            bp = info["body_pos"][0]
            bl = info["ball_pos"][0]
            bv = info["ball_vel"][0]
            cmd = info["soccer_cmd"][0]
            print(
                f"  t={step:4d}  "
                f"robot=[{bp[0]:.2f},{bp[1]:.2f},{bp[2]:.2f}]  "
                f"ball=[{bl[0]:.2f},{bl[1]:.2f}]  "
                f"|v_ball|={bv.norm():.2f}  "
                f"cmd=[{cmd[0]:.2f},{cmd[1]:.2f},{cmd[2]:.2f},"
                f"{cmd[3]:.2f},{cmd[4]:.2f},{cmd[5]:.2f},{cmd[6]:.2f}]"
            )

    print(f"\ndone. steps={args.steps}")


if __name__ == "__main__":
    main()
