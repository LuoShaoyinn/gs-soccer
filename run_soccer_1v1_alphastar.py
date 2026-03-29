import argparse
import numpy as np
import genesis as gs

from algorithms.alphastar_evolution import AlphaStarEvolutionAlgorithm, AlphaStarEvolutionConfig
from envs.game import GameEnv, GameEnvConfig
from fields.soccer_field import SoccerField, SoccerFieldConfig
from models.pi_walk_model import PIWalkModel, PIWalkModelConfig
from models.game_model import GameModel, GameModelConfig
from robots.controlled_robot import ControlledRobotWrapper, ControlledRobotWrapperConfig
from robots.pi import PI, PIConfig


PI_TARGET_Q_OFFSET = np.array(
    [
        -0.25,
        0.0,
        -0.25,
        0.0,
        0.0,
        0.2,
        0.0,
        -0.2,
        0.0,
        0.0,
        0.0,
        0.0,
        0.65,
        -1.2,
        0.65,
        -1.2,
        -0.4,
        -0.4,
        0.0,
        0.0,
    ],
    dtype=np.float32,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true", help="Run evaluation mode")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--experiment-name", default="soccer_1v1_alphastar", help="Experiment name under runs/")
    parser.add_argument("--num-envs", type=int, default=8192, help="Number of vectorized envs")
    parser.add_argument("--generations", type=int, default=8, help="League training generations")
    parser.add_argument("--timesteps-per-gen", type=int, default=131072, help="Training timesteps each generation")
    parser.add_argument("--eval-episodes", type=int, default=32, help="Evaluation episodes per matchup")
    parser.add_argument("--rollout-steps", type=int, default=16, help="PPO rollout horizon")
    parser.add_argument("--learning-epochs", type=int, default=4, help="PPO learning epochs")
    parser.add_argument("--mini-batches", type=int, default=16, help="PPO mini-batches")
    parser.add_argument(
        "--scripted-opponent",
        choices=("pid", "legacy"),
        default="pid",
        help="Scripted opponent used when league member is scripted_bot",
    )
    parser.add_argument("--headless", action="store_true", help="Force headless mode")
    return parser.parse_args()


def make_robot_wrapper(initial_pos: np.ndarray) -> ControlledRobotWrapperConfig:
    return ControlledRobotWrapperConfig(
        robot_cfg=PIConfig(initial_pos=initial_pos),
        robot_class=PI,
        ctrl_model_cfg=PIWalkModelConfig(n_dofs=20, target_q_offset=PI_TARGET_Q_OFFSET),
        ctrl_model_class=PIWalkModel,
        ctrl_policy_path="models_ckpt/pi_policy.pt",
    )


def main() -> None:
    args = parse_args()
    eval_mode = args.eval
    show_viewer = eval_mode and not args.headless

    gs.init(
        backend=gs.gpu,  # type: ignore[unsolved-attribute]
        performance_mode=True,
        logging_level="warning",
    )

    model_cfg = GameModelConfig(
        half_field_size=SoccerFieldConfig().half_field_size,
        goal_width=SoccerFieldConfig().goal_width,
    )

    env = GameEnv(
        GameEnvConfig(
            robot_cfg={
                "red": make_robot_wrapper(np.array([-1.5, 0.0, 0.5], dtype=np.float32)),
                "blue": make_robot_wrapper(np.array([1.5, 0.0, 0.5], dtype=np.float32)),
            },
            robot_class={
                "red": ControlledRobotWrapper,
                "blue": ControlledRobotWrapper,
            },
            field_cfg=SoccerFieldConfig(),
            field_class=SoccerField,
            model_cfg=model_cfg,
            model_class=GameModel,
            num_envs=1 if eval_mode else args.num_envs,
            show_viewer=show_viewer,
            env_spacing=0.0,
            policy_freq=50,
            sim_freq=500,
            ctrl_freq_ratio=10,
            max_collision_pairs=400,
            multiplier_collision_broad_phase=16,
            robot_reset_pos={
                "red": np.array([-1.5, 0.0, 0.0], dtype=np.float32),
                "blue": np.array([1.5, 0.0, np.pi], dtype=np.float32),
            },
        )
    )

    algorithm = AlphaStarEvolutionAlgorithm(
        env,
        AlphaStarEvolutionConfig(
            experiment_name=args.experiment_name,
            generations=args.generations,
            train_timesteps_per_generation=args.timesteps_per_gen,
            eval_episodes=args.eval_episodes,
            rollout_steps=args.rollout_steps,
            learning_epochs=args.learning_epochs,
            mini_batches=args.mini_batches,
            scripted_opponent=args.scripted_opponent,
            checkpoint_path=f"runs/{args.experiment_name}/checkpoints/best_agent.pt",
            resume=args.resume or eval_mode,
            compile_policy=False,
            headless=not show_viewer,
            learner_idx=0,
        ),
    )

    if eval_mode:
        algorithm.eval()
    else:
        algorithm.train()


if __name__ == "__main__":
    main()
