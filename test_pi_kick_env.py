import os
import torch
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--num-envs", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.viewer:
        os.environ.pop("PYOPENGL_PLATFORM", None)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

    import genesis as gs
    from robots.pi import PI, PIConfig
    from fields.soccer_field import SoccerField, SoccerFieldConfig

    gs.init(backend=gs.gpu, performance_mode=True, logging_level="warning")

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.5, -2.5, 1.5),
            camera_lookat=(0.0, 0.0, 0.4),
            camera_fov=35,
            res=(960, 640),
            max_FPS=60,
            enable_help_text=False,
        ),
        sim_options=gs.options.SimOptions(dt=0.01, substeps=2),
        rigid_options=gs.options.RigidOptions(
            enable_neutral_collision=True,
        ),
        show_viewer=args.viewer,
    )

    field = SoccerField(
        SoccerFieldConfig(
            ball_radius=0.07,
            ball_mass=0.25,
            ball_damping=2.0,
            half_field_size=(4.5, 3.0),
            field_color=(0.3, 1.0, 0.3),
        ),
        scene,
    )
    robot = PI(PIConfig(), scene)

    field.build()
    robot.build()
    scene.build(n_envs=args.num_envs, env_spacing=(3.0, 3.0))
    field.config()
    robot.config()

    envs_idx = torch.arange(args.num_envs, dtype=torch.long, device=gs.device)

    # reset
    robot.reset(envs_idx=envs_idx)
    ball_pos = torch.tensor([[0.5, 0.0, 0.12]], dtype=torch.float, device=gs.device)
    ball_pos = ball_pos.broadcast_to((args.num_envs, 3))
    field.reset(envs_idx=envs_idx, ball_pos=ball_pos)

    standing_q = torch.zeros((args.num_envs, len(robot.cfg.joint_names)), device=gs.device)

    for i in range(args.steps):
        robot.step(standing_q)
        scene.step()
        if i % 100 == 0:
            bp = field.ball.get_pos(envs_idx=envs_idx)
            bv = field.ball.get_vel(envs_idx=envs_idx)
            rp = robot.robot.get_pos(envs_idx=envs_idx)
            print(
                f"  t={i:4d}  "
                f"robot=[{rp[0,0]:.2f},{rp[0,1]:.2f},{rp[0,2]:.2f}]  "
                f"ball=[{bp[0,0]:.2f},{bp[0,1]:.2f},{bp[0,2]:.2f}]  "
                f"ball_vel=[{bv[0,0]:.2f},{bv[0,1]:.2f},{bv[0,2]:.2f}]"
            )

    print("done")


if __name__ == "__main__":
    main()
