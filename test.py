import os
import torch
from argparse import ArgumentParser
from PIL import Image


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--head-camera", action="store_true")
    parser.add_argument("--head-camera-gui", action="store_true")
    parser.add_argument("--head-camera-res", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.viewer:
        os.environ.pop("PYOPENGL_PLATFORM", None)
    elif args.head_camera:
        os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

    import genesis as gs

    from fields.field import Field, FieldConfig
    from robots.floating_camera import FloatingCameraConfig, FloatingCameraRobot
    from robots.mos9 import MOS9, MOS9Config

    gs.init(
        backend=gs.gpu,  # type: ignore[attr-defined]
        performance_mode=True,
        logging_level="warning",
    )

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.8, -3.0, 1.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=35,
            res=(960, 640),
            max_FPS=60,
        ),
        sim_options=gs.options.SimOptions(
            dt=0.01,
            substeps=2,
        ),
        rigid_options=gs.options.RigidOptions(
            enable_neutral_collision=True,
        ),
        show_viewer=args.viewer,
    )

    field = Field(FieldConfig(), scene)
    robot = MOS9(MOS9Config(), scene)
    camera_robot = None
    if args.head_camera:
        camera_robot = FloatingCameraRobot(
            FloatingCameraConfig(
                res=tuple(args.head_camera_res) if args.head_camera_res else (320, 240),
                GUI=args.head_camera_gui,
            ),
            scene,
        )

    field.build()
    robot.build()
    if camera_robot is not None:
        camera_robot.look_at(robot.robot)
        camera_robot.build()
    scene.build(n_envs=1, env_spacing=(0.0, 0.0))
    field.config()
    robot.config()
    if camera_robot is not None:
        camera_robot.config()

    envs_idx = torch.tensor([0], dtype=torch.long, device=gs.device)
    robot.reset(envs_idx=envs_idx)
    standing_q = torch.zeros((1, len(robot.cfg.joint_names)), device=gs.device)

    for _ in range(args.steps):
        robot.step(standing_q)
        scene.step()
        if camera_robot is not None:
            camera_robot.step()

    if camera_robot is not None:
        frame = camera_robot.render()
        rgb = frame[0] if isinstance(frame, tuple) else frame
        if isinstance(rgb, torch.Tensor):
            rgb = rgb.detach().cpu().numpy()
        Image.fromarray(rgb).save("camera.bmp", format="BMP")
        print(f"floating_camera_rgb_shape={rgb.shape}")


if __name__ == "__main__":
    main()
