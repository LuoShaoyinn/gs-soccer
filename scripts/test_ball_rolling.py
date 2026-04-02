from __future__ import annotations

import argparse

import torch
import genesis as gs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test ball rollout stop time")
    parser.add_argument(
        "--sim-freq", type=int, default=500, help="Physics frequency (Hz)"
    )
    parser.add_argument(
        "--seconds", type=float, default=10.0, help="Max simulation duration"
    )
    parser.add_argument(
        "--sample-freq", type=int, default=10, help="Print frequency (Hz)"
    )
    parser.add_argument(
        "--ball-radius", type=float, default=0.08, help="Ball radius (m)"
    )
    parser.add_argument("--ball-mass", type=float, default=0.2, help="Ball mass (kg)")
    parser.add_argument(
        "--plane-friction", type=float, default=0.08, help="Plane friction"
    )
    parser.add_argument(
        "--ball-friction", type=float, default=0.08, help="Ball friction"
    )
    parser.add_argument(
        "--ang-damping",
        type=float,
        default=2.0,
        help="Angular damping on dofs (3,4,5)",
    )
    parser.add_argument(
        "--init-speed", type=float, default=1.0, help="Initial x speed (m/s)"
    )
    parser.add_argument(
        "--stop-eps",
        type=float,
        default=0.03,
        help="Speed threshold for stop detection (m/s)",
    )
    parser.add_argument("--headless", action="store_true", help="Disable viewer")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dt = 1.0 / args.sim_freq
    n_steps = int(round(args.seconds * args.sim_freq))
    sample_every = max(1, int(round(args.sim_freq / args.sample_freq)))

    gs.init(backend=gs.gpu, logging_level="warning")
    scene = gs.Scene(
        show_viewer=not args.headless,
        rigid_options=gs.options.RigidOptions(dt=dt),
    )

    scene.add_entity(
        morph=gs.morphs.Plane(),
        material=gs.materials.Rigid(friction=args.plane_friction),
    )
    ball = scene.add_entity(
        morph=gs.morphs.Sphere(
            radius=args.ball_radius, pos=(0.0, 0.0, args.ball_radius + 0.002)
        ),
        material=gs.materials.Rigid(friction=args.ball_friction),
    )

    scene.build(n_envs=1)

    ball.set_mass(args.ball_mass)
    ball.set_dofs_damping(args.ang_damping, dofs_idx_local=(3, 4, 5))

    init_vel = torch.zeros((1, 6), dtype=torch.float, device=gs.device)
    init_vel[:, 0] = args.init_speed
    ball.set_dofs_velocity(init_vel)

    stop_step: int | None = None

    print("t,x,speed")
    for step in range(n_steps + 1):
        if step > 0:
            scene.step()

        pos = ball.get_pos()[0, 0:2]
        vel = ball.get_vel()[0, 0:2]
        speed = float(torch.linalg.norm(vel))
        t = step * dt

        if stop_step is None and speed < args.stop_eps:
            stop_step = step

        if step % sample_every == 0 or step == n_steps:
            print(f"{t:.3f},{float(pos[0]):.6f},{speed:.6f}")

        if stop_step is not None and step > stop_step + sample_every:
            break

    final_pos = ball.get_pos()[0, 0:2]
    final_speed = float(torch.linalg.norm(ball.get_vel()[0, 0:2]))
    stop_text = "not reached"
    if stop_step is not None:
        stop_text = f"{stop_step * dt:.4f}s"

    print("\nSummary")
    print(f"- stop threshold: {args.stop_eps:.4f} m/s")
    print(f"- stop time: {stop_text}")
    print(f"- final x: {float(final_pos[0]):.4f} m")
    print(f"- final speed: {final_speed:.6f} m/s")


if __name__ == "__main__":
    main()
