import argparse
import os


MOS9_TARGET_Q_OFFSET = [
    0.00, 0.00, -0.35, 0.70, -0.35, 0.00,
    0.00, 0.00, -0.35, 0.70, -0.35, 0.00,
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true", help="Evaluation mode")
    parser.add_argument("--resume", action="store_true", help="Load checkpoint")
    parser.add_argument("--viewer", action="store_true", help="Enable on-screen viewer")
    parser.add_argument("--experiment-name", default="mos9_gait_ppo_unitree")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--num-envs", type=int, default=8192)
    parser.add_argument("--rollouts", type=int, default=48)
    parser.add_argument("--learning-epochs", type=int, default=8)
    parser.add_argument("--mini-batches", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    return parser.parse_args()


def configure_render_env(viewer: bool) -> None:
    if viewer:
        os.environ.pop("PYGLET_HEADLESS", None)
    else:
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        os.environ.setdefault("PYGLET_HEADLESS", "true")


def main():
    args = parse_args()
    eval_mode = args.eval
    configure_render_env(args.viewer)

    import numpy as np
    import genesis as gs

    from algorithms.ppo import PPOAlgorithm, PPOAlgorithmConfig
    from envs.walker import WalkEnv, WalkEnvConfig
    from envs.env import Env
    from robots.mos9 import MOS9, MOS9Config
    from fields.field import Field, FieldConfig
    from models.mos9_walk_model import MOS9WalkModelConfig, MOS9WalkModel

    class SkrlWalkEnv(WalkEnv):
        # skrl trainer calls env.state()
        def state(self):
            return None

        # main walker has a reset-index mismatch in target_q broadcasting.
        # keep infra file untouched and patch behavior at entry layer.
        def get_state(self, envs_idx):
            state = {"cmd_vel": self.cmd_vel[envs_idx], **Env.get_state(self, envs_idx)}
            state["non_foot_heights"] = self._get_non_foot_heights(envs_idx)
            target_q = self.model.target_q[envs_idx]
            state["dofs_torque"] = self._kp * (target_q - state["dofs_pos"]) - self._kv * state["dofs_vel"]
            return state

    gs.init(
        backend=gs.gpu,  # type: ignore[unsolved-attribute]
        performance_mode=True,
        logging_level="warning",
    )

    algorithm = PPOAlgorithm(
        SkrlWalkEnv(
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
        PPOAlgorithmConfig(
            experiment_name=args.experiment_name,
            timesteps=args.timesteps,
            checkpoint_path=f"runs/{args.experiment_name}/checkpoints/best_agent.pt",
            resume=args.resume or eval_mode,
            rollouts=args.rollouts,
            learning_epochs=args.learning_epochs,
            mini_batches=args.mini_batches,
            learning_rate=args.learning_rate,
        ),
    )

    try:
        if eval_mode:
            algorithm.eval()
        else:
            algorithm.train()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            algorithm.env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()

