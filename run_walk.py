import argparse
import os
import numpy as np
import genesis as gs

from algorithms.ppo import PPOAlgorithm, PPOAlgorithmConfig
from envs.walker import WalkEnv, WalkEnvConfig
from fields.field import Field, FieldConfig
from models.mos9_model import MOS9Model, MOS9ModelConfig
from robots.mos9 import MOS9, MOS9Config


MOS9_TARGET_Q_OFFSET = [
    0.00, 0.00, -0.35, 0.70, -0.35, 0.00,
    0.00, 0.00, -0.35, 0.70, -0.35, 0.00,
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--eval", action="store_true")
    p.add_argument("--viewer", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--experiment-name", default="mos9_core_ppo")
    p.add_argument("--timesteps", type=int, default=1_000_000)
    p.add_argument("--num-envs", type=int, default=4096)
    return p.parse_args()


def configure_render_env(viewer: bool):
    if viewer:
        os.environ.pop("PYGLET_HEADLESS", None)
    else:
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        os.environ.setdefault("PYGLET_HEADLESS", "true")


def main():
    args = parse_args()
    configure_render_env(args.viewer)
    gs.init(backend=gs.gpu, performance_mode=True, logging_level="warning") # type: ignore

    env = WalkEnv(
        WalkEnvConfig(
            robot_cfg=MOS9Config(),
            robot_class=MOS9,
            field_cfg=FieldConfig(),
            field_class=Field,
            model_cfg=MOS9ModelConfig(
                n_dofs=12,
                target_q_offset=np.array(MOS9_TARGET_Q_OFFSET, dtype=np.float32),
                step_dt=0.02,
                field_range=1.0,
                timeout=20.0,
                omega=2.0,
                collision_penalty_weight=0.5,
            ),
            policy_freq=50,
            sim_freq=500,
            model_class=MOS9Model,
            env_spacing=2.0,
            num_envs=1 if args.eval else args.num_envs,
            show_viewer=args.viewer,
        )
    )

    algo = PPOAlgorithm(
        env,
        PPOAlgorithmConfig(
            experiment_name=args.experiment_name,
            timesteps=args.timesteps,
            checkpoint_path=f"runs/{args.experiment_name}/checkpoints/best_agent.pt",
            resume=args.resume or args.eval,
            environment_info="extra",
        ),
    )
    if args.eval:
        algo.eval()
    else:
        algo.train()


if __name__ == "__main__":
    main()
