import argparse
import numpy as np
import genesis as gs

from algorithms.ppo import PPOAlgorithm, PPOAlgorithmConfig
from robots.pi import PI, PIConfig
from robots.controlled_robot import ControlledRobotWrapper, ControlledRobotWrapperConfig
from fields.soccer_field import SoccerField, SoccerFieldConfig
from models.pi_walk_model import PIWalkModelConfig, PIWalkModel
from models.dribble_model import DribbleModelConfig, DribbleModel
from envs.dribble import DribbleEnv, DribbleEnvConfig

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
    parser.add_argument("--resume", action="store_true", help="Load checkpoint before train/eval")
    parser.add_argument(
        "--experiment-name",
        default="dribble_PI_v4",
        help="Experiment directory name under runs/",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    eval_mode = args.eval
    resume = args.resume
    experiment_name = args.experiment_name

    gs.init(
        backend=gs.gpu,  # type: ignore[unsolved-attribute]
        performance_mode=True,
        logging_level="warning",
    )

    algorithm = PPOAlgorithm(
        DribbleEnv(
            DribbleEnvConfig(
                robot_cfg=ControlledRobotWrapperConfig(
                    robot_cfg=PIConfig(
                        initial_pos=np.array([-1.0, 0.0, 0.5], dtype=np.float32),
                    ),
                    robot_class=PI,
                    ctrl_model_cfg=PIWalkModelConfig(
                        n_dofs=20,
                        target_q_offset=PI_TARGET_Q_OFFSET,
                    ),
                    ctrl_model_class=PIWalkModel,
                    ctrl_policy_path="models_ckpt/pi_policy.pt",
                ),
                robot_class=ControlledRobotWrapper,
                field_cfg=SoccerFieldConfig(),
                field_class=SoccerField,
                model_cfg=DribbleModelConfig(),
                model_class=DribbleModel,
                num_envs=1 if eval_mode else 8192,
                show_viewer=eval_mode,
                ctrl_freq_ratio=10,
                policy_freq=50,
                sim_freq=500,
            )
        ),
        PPOAlgorithmConfig(
            experiment_name=experiment_name,
            timesteps = 30000, 
            checkpoint_path=f"runs/{experiment_name}/checkpoints/best_agent.pt",
            resume=resume or eval_mode,
        ),
    )
    if eval_mode:
        algorithm.eval()
    else:
        algorithm.train()


if __name__ == "__main__":
    main()
