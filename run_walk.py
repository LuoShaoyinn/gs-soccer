import argparse
import atexit
import os


MOS9_TARGET_Q_OFFSET = [
    0.00, 0.00, -0.35, 0.70, -0.35, 0.00,
    0.00, 0.00, -0.35, 0.70, -0.35, 0.00,
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true", help="Evaluation mode")
    parser.add_argument("--resume", action="store_true", help="Load checkpoint")
    parser.add_argument("--viewer", action="store_true", help="Enable on-screen viewer (requires a valid GL context)")
    parser.add_argument("--experiment-name", default="mos9_gait_sac")
    parser.add_argument("--timesteps", type=int, default=30000)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--memory-size",
        type=int,
        default=256,
        help="Replay memory size per environment (skrl uses shape [memory_size, num_envs, ...])",
    )
    parser.add_argument("--learning-starts", type=int, default=1024)
    parser.add_argument("--random-timesteps", type=int, default=256)
    parser.add_argument("--gradient-steps", type=int, default=2)
    return parser.parse_args()


def configure_render_env(viewer: bool) -> None:
    if viewer:
        # Viewer mode: don't force headless backend.
        os.environ.pop("PYGLET_HEADLESS", None)
    else:
        # Headless-safe defaults for training / non-visual eval.
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        os.environ.setdefault("PYGLET_HEADLESS", "true")


def main():
    args = parse_args()
    eval_mode = args.eval
    configure_render_env(args.viewer)

    import numpy as np
    import genesis as gs

    from algorithms.sac import SACAlgorithm, SACAlgorithmConfig
    from envs.walker import WalkEnv, WalkEnvConfig
    from robots.mos9 import MOS9, MOS9Config
    from fields.field import Field, FieldConfig
    from models.mos9_walk_model import MOS9WalkModelConfig, MOS9WalkModel

    gs.init(
        backend=gs.gpu,  # type: ignore[unsolved-attribute]
        performance_mode=True,
        logging_level="warning",
    )
    # Avoid duplicate global teardown path at interpreter exit.
    atexit.unregister(gs.destroy)

    algorithm = SACAlgorithm(
        WalkEnv(
            WalkEnvConfig(
                robot_cfg=MOS9Config(),
                robot_class=MOS9,
                field_cfg=FieldConfig(),
                field_class=Field,
                model_cfg=MOS9WalkModelConfig(
                    n_dofs=12,
                    target_q_offset=np.array(MOS9_TARGET_Q_OFFSET, dtype=np.float32),
                ),
                policy_freq=50,
                sim_freq=500,
                model_class=MOS9WalkModel,
                env_spacing=2.0,
                num_envs=1 if eval_mode else args.num_envs,
                show_viewer=args.viewer,
            )
        ),
        SACAlgorithmConfig(
            experiment_name=args.experiment_name,
            timesteps=args.timesteps,
            checkpoint_path=f"runs/{args.experiment_name}/checkpoints/best_agent.pt",
            resume=args.resume or eval_mode,
            memory_size=args.memory_size,
            batch_size=args.batch_size,
            random_timesteps=args.random_timesteps,
            learning_starts=args.learning_starts,
            gradient_steps=args.gradient_steps,
        ),
    )

    try:
        if eval_mode:
            algorithm.eval()
        else:
            algorithm.train()
    finally:
        try:
            algorithm.env.close()
        except Exception:
            pass
        try:
            gs.destroy()
        except Exception:
            pass


if __name__ == "__main__":
    main()
