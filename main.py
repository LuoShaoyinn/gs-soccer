import os
from argparse import ArgumentParser

import torch

import genesis as gs

NUM_GENESIS_JOINTS = 20

KP = torch.tensor([
    50.97, 32.51, 50.97, 32.51,
    50.97, 32.51, 50.97, 32.51,
    50.97, 32.51, 50.97, 32.51,
    50.97, 32.51, 50.97, 32.51,
    50.97, 50.97, 50.97, 50.97,
], dtype=torch.float32)

KD = torch.tensor([
    3.24, 2.07, 3.24, 2.07,
    3.24, 2.07, 3.24, 2.07,
    3.24, 2.07, 3.24, 2.07,
    3.24, 2.07, 3.24, 2.07,
    3.24, 3.24, 3.24, 3.24,
], dtype=torch.float32)

ARMATURE = torch.tensor([
    0.01291, 0.008234, 0.01291, 0.008234,
    0.01291, 0.008234, 0.01291, 0.008234,
    0.01291, 0.008234, 0.01291, 0.008234,
    0.01291, 0.008234, 0.01291, 0.008234,
    0.01291, 0.01291, 0.01291, 0.01291,
], dtype=torch.float32)

DEFAULT_POS = torch.tensor([
    -0.25, 0.0, -0.25, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.65, 0.0, 0.65, 0.0,
    -0.4, -0.4, 0.0, 0.0,
], dtype=torch.float32)


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
    p.add_argument("--viewer", action="store_true")
    p.add_argument("--no-render", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.viewer:
        os.environ.pop("PYOPENGL_PLATFORM", None)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

    import numpy as np
    from envs.kick import KickEnv, KickEnvConfig
    from robots.pi import PI, PIConfig
    from fields.field import Field, FieldConfig
    from models.sim2sim_soccer import Sim2SimSoccerModel, Sim2SimSoccerConfig

    gs.init(backend=gs.gpu, performance_mode=True, logging_level="warning")

    ball_r_min, ball_r_max = {
        "walk": (3.8, 4.2),
        "approach_kick": (3.0, 4.0),
        "kick": (0.4, 1.0),
    }[args.mode]

    env = KickEnv(KickEnvConfig(
        robot_cfg=PIConfig(
            initial_pos=np.array([0.0, 0.0, 0.351], dtype=np.float32),
            kp=KP.numpy(),
            kv=KD.numpy(),
            force_range=np.stack([
                np.full(NUM_GENESIS_JOINTS, -20.0, dtype=np.float32),
                np.full(NUM_GENESIS_JOINTS, 20.0, dtype=np.float32),
            ]),
        ),
        robot_class=PI,
        field_cfg=FieldConfig(),
        field_class=Field,
        model_cfg=Sim2SimSoccerConfig(
            model_dir=args.model_dir,
            model_file=args.model_file,
        ),
        model_class=Sim2SimSoccerModel,
        policy_freq=50,
        sim_freq=200,
        show_viewer=args.viewer,
        num_envs=args.num_envs,
        env_spacing=3.0,
        ball_radius=0.07,
        ball_mass_range=(0.16, 0.16),
        ball_reset_r_min=ball_r_min,
        ball_reset_r_max=ball_r_max,
        ball_reset_noise=0.0,
        mode=args.mode,
        vel_cmd=(args.vel_x, args.vel_y, args.vel_yaw),
        kick_speed=args.kick_speed,
        kick_dir_deg=args.kick_dir_deg,
    ))

    dofs_idx = env.robot.dofs_idx_local
    env.robot.robot.set_dofs_armature(ARMATURE.to(gs.device), dofs_idx_local=dofs_idx)
    env.robot.robot.set_dofs_damping(
        torch.zeros(NUM_GENESIS_JOINTS, dtype=torch.float32, device=gs.device),
        dofs_idx_local=dofs_idx,
    )
    env.ball.set_dofs_damping(0.0, dofs_idx_local=(3, 4, 5))

    obs, info = env.reset()
    default_q = DEFAULT_POS.to(gs.device).unsqueeze(0).expand(args.num_envs, -1)
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
