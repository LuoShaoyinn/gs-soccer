import argparse
import numpy as np
import torch
import genesis as gs

from envs.game import GameEnv, GameEnvConfig
from fields.soccer_field import SoccerField, SoccerFieldConfig
from models.pi_walk_model import PIWalkModel, PIWalkModelConfig
from models.game_model import GameModel, GameModelConfig
from robots.controlled_robot import ControlledRobotWrapper, ControlledRobotWrapperConfig
from robots.pi import PI, PIConfig
from policies import available_policy_modules, build_policy


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
    policy_modules = available_policy_modules()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-envs", type=int, default=1, help="Number of vectorized envs"
    )
    parser.add_argument("--steps", type=int, default=2000, help="Simulation steps")
    parser.add_argument("--headless", action="store_true", help="Disable viewer")
    parser.add_argument(
        "--sim-freq", type=int, default=500, help="Physics simulation frequency"
    )
    parser.add_argument(
        "--policy-freq", type=int, default=50, help="High-level policy frequency"
    )
    parser.add_argument(
        "--ctrl-freq-ratio",
        type=int,
        default=5,
        help="Low-level controller updates every N sim steps",
    )
    parser.add_argument(
        "--red-policy",
        choices=policy_modules,
        default="go_to_ball_pid",
        help="Policy module in policies/ for red team",
    )
    parser.add_argument(
        "--blue-policy",
        choices=policy_modules,
        default="advanced_dribble",
        help="Policy module in policies/ for blue team",
    )
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
    gs.init(
        backend=gs.gpu,  # type: ignore[unsolved-attribute]
        performance_mode=True,
        logging_level="warning",
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
            model_cfg=GameModelConfig(
                half_field_size=SoccerFieldConfig().half_field_size,
                goal_width=SoccerFieldConfig().goal_width,
            ),
            model_class=GameModel,
            num_envs=args.num_envs,
            show_viewer=not args.headless,
            env_spacing=0.0,
            policy_freq=args.policy_freq,
            sim_freq=args.sim_freq,
            ctrl_freq_ratio=args.ctrl_freq_ratio,
            max_collision_pairs=400,
            multiplier_collision_broad_phase=16,
            robot_reset_pos={
                "red": np.array([-1.5, 0.0, 0.0], dtype=np.float32),
                "blue": np.array([1.5, 0.0, np.pi], dtype=np.float32),
            },
        )
    )

    red_policy = build_policy(args.red_policy, device=gs.device)
    blue_policy = build_policy(args.blue_policy, device=gs.device)

    obs, _ = env.reset()
    for _ in range(args.steps):
        with torch.no_grad():
            actions = {
                "red": red_policy.act(obs["red"]),
                "blue": blue_policy.act(obs["blue"]),
            }
        obs, _, _, _, _ = env.step(actions)


if __name__ == "__main__":
    main()
