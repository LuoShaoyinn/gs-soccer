import os
from argparse import ArgumentParser

import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.animation import FFMpegWriter, PillowWriter

from camera_location_experiment import (
    CameraTrajectoryConfig,
    camera_horizon_quat,
    build_scene,
    make_torch_generator,
    sample_smooth_trajectory,
)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--res", type=int, nargs=2, default=(640, 480), metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--fov", type=float, default=95.0)
    parser.add_argument("--horizon-distance", type=float, default=100.0)
    parser.add_argument("--viewer", dest="viewer", action="store_true")
    parser.add_argument("--no-viewer", dest="viewer", action="store_false")
    parser.add_argument("--textured", dest="textured", action="store_true")
    parser.add_argument("--no-textured", dest="textured", action="store_false")
    parser.add_argument("--madrona-camera", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--video", default="camera_location_trajectory.mp4")
    parser.set_defaults(viewer=True, textured=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.viewer:
        os.environ.pop("PYOPENGL_PLATFORM", None)
    else:
        os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

    import genesis as gs

    gs.init(
        backend=gs.cuda if args.madrona_camera else gs.gpu,  # type: ignore[attr-defined]
        performance_mode=True,
        logging_level="warning",
    )
    scene, field_cfg, camera = build_scene(args)
    traj_cfg = CameraTrajectoryConfig(half_field_size=field_cfg.half_field_size)
    generator = make_torch_generator(args.seed, gs.device)
    pos_seq, yaw_seq, _ = sample_smooth_trajectory(args.steps, generator, traj_cfg, gs.device)
    pos_np = pos_seq.detach().cpu().numpy()
    yaw_np = yaw_seq.detach().cpu().numpy()

    envs_idx = torch.tensor([0], dtype=torch.long, device=gs.device)
    rgb_frames = []
    for step in range(args.steps):
        pos = pos_seq[step]
        yaw = yaw_seq[step]
        quat = camera_horizon_quat(camera, pos, yaw, args.horizon_distance)
        camera.step(torch.cat((pos, quat)))
        scene.step()
        state = camera.get_state(envs_idx)
        rgb = state["image"][0].detach().cpu().numpy()
        rgb_frames.append(rgb)

    matplotlib.use("Agg")
    fig, (ax_img, ax_plot) = plt.subplots(1, 2, figsize=(9, 4), dpi=120)
    half_x, half_y = field_cfg.half_field_size
    writer = FFMpegWriter(fps=30) if matplotlib.animation.writers.is_available("ffmpeg") else PillowWriter(fps=30)
    video_path = args.video if isinstance(writer, FFMpegWriter) else args.video.rsplit(".", 1)[0] + ".gif"
    with writer.saving(fig, video_path, dpi=120):
        for step, rgb in enumerate(rgb_frames):
            ax_img.clear()
            ax_img.imshow(rgb)
            ax_img.set_axis_off()
            ax_img.set_title(f"camera image, fov={args.fov:g}")

            ax_plot.clear()
            ax_plot.set_xlim(-half_x, half_x)
            ax_plot.set_ylim(-half_y, half_y)
            ax_plot.set_aspect("equal", adjustable="box")
            ax_plot.plot(pos_np[:, 0], pos_np[:, 1], color="0.75", linewidth=1.5)
            ax_plot.plot(pos_np[: step + 1, 0], pos_np[: step + 1, 1], color="tab:blue", linewidth=2.0)
            ax_plot.scatter(pos_np[step, 0], pos_np[step, 1], color="tab:red", s=35)
            ax_plot.arrow(
                pos_np[step, 0],
                pos_np[step, 1],
                0.25 * np.cos(yaw_np[step]),
                0.25 * np.sin(yaw_np[step]),
                color="tab:red",
                head_width=0.08,
                length_includes_head=True,
            )
            ax_plot.set_title("camera xy trajectory")
            ax_plot.grid(True, linewidth=0.5, alpha=0.4)
            writer.grab_frame()
    plt.close(fig)
    print(f"saved_video={video_path}")


if __name__ == "__main__":
    main()
