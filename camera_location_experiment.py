import os
from argparse import ArgumentParser
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from models.camera_location_rnn import (
    CameraLocationRNN,
    CameraLocationRNNConfig,
    camera_observation_to_tensors,
)


@dataclass(kw_only=True)
class CameraTrajectoryConfig:
    half_field_size: tuple[float, float] = (4.5, 3.0)
    height: float = 1.0
    margin: float = 0.35
    control_points: int = 6
    slow_factor: int = 10
    pitch_mean: float = 0.0
    pitch_std: float = 0.0
    yaw_smooth_noise: float = 0.025


def yaw_to_quat(yaw: torch.Tensor, pitch: torch.Tensor | None = None) -> torch.Tensor:
    if pitch is None:
        pitch = torch.zeros_like(yaw)
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    # Genesis uses wxyz. This composes yaw around z and a small pitch around y.
    return torch.stack((cy * cp, -sy * sp, cy * sp, sy * cp), dim=-1)


def camera_horizon_quat(
    camera,
    pos: torch.Tensor,
    yaw: torch.Tensor,
    horizon_distance: float,
) -> torch.Tensor:
    import genesis as gs

    # Genesis uses z as height in this repo. Keep the optical axis horizontal:
    # same z as the camera, far ahead in the xy yaw direction.
    distance = torch.as_tensor(horizon_distance, dtype=pos.dtype, device=pos.device)
    lookat = torch.stack(
        (
            pos[0] + torch.cos(yaw) * distance,
            pos[1] + torch.sin(yaw) * distance,
            pos[2],
        )
    )
    up = torch.tensor([0.0, 0.0, 1.0], dtype=pos.dtype, device=pos.device)
    camera_module = type(camera.camera).__module__
    module = __import__(camera_module, fromlist=["pos_lookat_up_to_T"])
    camera_T = module.pos_lookat_up_to_T(pos, lookat, up)
    return gs.T_to_quat(camera_T)


def quat_to_projected_gravity(quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat.unbind(-1)
    gravity = torch.tensor([0.0, 0.0, -1.0], dtype=quat.dtype, device=quat.device)
    r20 = 2.0 * (x * z - w * y)
    r21 = 2.0 * (y * z + w * x)
    r22 = 1.0 - 2.0 * (x * x + y * y)
    r02 = 2.0 * (x * z + w * y)
    r12 = 2.0 * (y * z - w * x)
    # R^T * world_gravity; only z world is nonzero.
    return torch.stack((r20 * gravity[2], r21 * gravity[2], r22 * gravity[2]), dim=-1)


def make_torch_generator(seed: int | None, device: torch.device) -> torch.Generator:
    generator = torch.Generator(device=device)
    if seed is None:
        generator.seed()
    else:
        generator.manual_seed(seed)
    return generator


def sample_smooth_trajectory(
    steps: int,
    generator: torch.Generator,
    cfg: CameraTrajectoryConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if cfg.control_points < 4:
        raise ValueError("CameraTrajectoryConfig.control_points must be >= 4 for cubic B-spline sampling.")

    half_x, half_y = cfg.half_field_size
    low = torch.tensor([-half_x + cfg.margin, -half_y + cfg.margin], dtype=torch.float32, device=device)
    high = torch.tensor([half_x - cfg.margin, half_y - cfg.margin], dtype=torch.float32, device=device)
    points = low + (high - low) * torch.rand((cfg.control_points, 2), generator=generator, device=device)

    spans = cfg.control_points - 3
    visible_span = spans / max(cfg.slow_factor, 1)
    max_start = max(spans - visible_span, 0.0)
    start_t = torch.rand((), generator=generator, device=device) * max_start
    sample_t = torch.linspace(0.0, visible_span, steps, dtype=torch.float32, device=device) + start_t
    segment = torch.floor(sample_t).to(torch.long).clamp(0, spans - 1)
    u = sample_t - segment.to(torch.float32)

    p0 = points[segment]
    p1 = points[segment + 1]
    p2 = points[segment + 2]
    p3 = points[segment + 3]
    u2 = u * u
    u3 = u2 * u
    b0 = ((1.0 - u) ** 3) / 6.0
    b1 = (3.0 * u3 - 6.0 * u2 + 4.0) / 6.0
    b2 = (-3.0 * u3 + 3.0 * u2 + 3.0 * u + 1.0) / 6.0
    b3 = u3 / 6.0
    xy = b0[:, None] * p0 + b1[:, None] * p1 + b2[:, None] * p2 + b3[:, None] * p3
    xy = torch.clamp(xy, min=low, max=high)

    dxy = torch.empty_like(xy)
    if steps == 1:
        dxy[0] = p2[0] - p1[0]
    else:
        dxy[0] = xy[1] - xy[0]
        dxy[-1] = xy[-1] - xy[-2]
        if steps > 2:
            dxy[1:-1] = 0.5 * (xy[2:] - xy[:-2])
    yaw = torch.atan2(dxy[:, 1], dxy[:, 0])
    yaw_noise_count = max(3, steps // 8)
    yaw_noise_knots = torch.randn((yaw_noise_count,), generator=generator, device=device) * cfg.yaw_smooth_noise
    yaw_noise = F.interpolate(
        yaw_noise_knots.view(1, 1, -1),
        size=steps,
        mode="linear",
        align_corners=True,
    ).view(-1)
    yaw = yaw + yaw_noise
    pitch = torch.randn((steps,), generator=generator, device=device) * cfg.pitch_std + cfg.pitch_mean
    pos = torch.cat((xy, torch.full((steps, 1), cfg.height, dtype=torch.float32, device=device)), dim=-1)
    return pos, yaw, pitch


def build_scene(args):
    import genesis as gs

    from fields.soccer_field import SoccerField, SoccerFieldConfig
    from robots.floating_camera import FloatingCameraConfig, FloatingCameraRobot

    scene_kwargs = {}
    if args.madrona_camera:
        scene_kwargs["renderer"] = gs.renderers.BatchRenderer(use_rasterizer=True)
    scene = gs.Scene(
        **scene_kwargs,
        vis_options=gs.options.VisOptions(
            show_world_frame=False,
            show_link_frame=False,
            show_cameras=False,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.0, -6.5, 4.0),
            camera_lookat=(0.0, 0.0, 0.4),
            camera_fov=40,
            res=tuple(args.res),
            max_FPS=60,
            enable_help_text=False,
        ),
        sim_options=gs.options.SimOptions(dt=0.02, substeps=1),
        show_viewer=args.viewer,
    )
    field_cfg = SoccerFieldConfig(textured=args.textured)
    field = SoccerField(field_cfg, scene)
    camera = FloatingCameraRobot(
        FloatingCameraConfig(
            res=tuple(args.res),
            pos=np.array([0.0, 0.0, 1.0], dtype=np.float32),
            fov=args.fov,
            use_madrona=args.madrona_camera,
        ),
        scene,
    )
    field.build()
    camera.build()
    scene.build(n_envs=1, env_spacing=(0.0, 0.0))
    field.config()
    camera.config()
    return scene, field_cfg, camera


def collect_episode(scene, camera, traj_cfg: CameraTrajectoryConfig, args, generator: torch.Generator):
    import genesis as gs

    pos_seq, yaw_seq, _ = sample_smooth_trajectory(args.seq_len, generator, traj_cfg, gs.device)
    visual_steps = []
    proprio_steps = []
    pos_targets = []
    envs_idx = torch.tensor([0], dtype=torch.long, device=gs.device)
    for step in range(args.seq_len):
        pos = pos_seq[step]
        yaw = yaw_seq[step]
        quat = camera_horizon_quat(camera, pos, yaw, args.horizon_distance)
        camera.step(torch.cat((pos, quat)))
        scene.step()
        state = camera.get_state(envs_idx)
        projected_gravity = quat_to_projected_gravity(quat).unsqueeze(0)
        visual, proprio = camera_observation_to_tensors(
            state["image"],
            state["depth_image"],
            projected_gravity,
            yaw.unsqueeze(0),
            tuple(args.res),
        )
        visual_steps.append(visual[0])
        proprio_steps.append(proprio[0])
        pos_targets.append(pos[:2])
    return (
        torch.stack(visual_steps, dim=0),
        torch.stack(proprio_steps, dim=0),
        torch.stack(pos_targets, dim=0),
        pos_seq,
    )


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--trajectories", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=24)
    parser.add_argument("--res", type=int, nargs=2, default=(640, 480), metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--fov", type=float, default=95.0)
    parser.add_argument("--horizon-distance", type=float, default=100.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--textured", dest="textured", action="store_true")
    parser.add_argument("--no-textured", dest="textured", action="store_false")
    parser.add_argument("--madrona-camera", action="store_true")
    parser.add_argument("--save", default="camera_location_rnn.pt")
    parser.set_defaults(textured=True)
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
    episodes = args.episodes if args.trajectories is None else args.trajectories
    generator = make_torch_generator(args.seed, gs.device)
    model = CameraLocationRNN(CameraLocationRNNConfig(image_size=tuple(args.res))).to(gs.device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(args.epochs):
        losses = []
        for episode in range(episodes):
            visual, proprio, target, _ = collect_episode(scene, camera, traj_cfg, args, generator)
            pred = model(visual.unsqueeze(0), proprio.unsqueeze(0))
            loss = F.mse_loss(pred, target.unsqueeze(0))
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            losses.append(float(loss.detach().cpu()))
            del visual, proprio, target, pred, loss
        print(f"epoch={epoch + 1} loss={sum(losses) / len(losses):.6f}")

    torch.save(
        {
            "model": model.state_dict(),
            "model_cfg": CameraLocationRNNConfig(image_size=tuple(args.res)),
            "traj_cfg": traj_cfg,
        },
        args.save,
    )
    print(f"saved={args.save}")


if __name__ == "__main__":
    main()
