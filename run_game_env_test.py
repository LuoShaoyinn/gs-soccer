import torch
import numpy as np
import genesis as gs

from envs.game import GameEnv, GameEnvConfig
from fields.soccer_field import SoccerField, SoccerFieldConfig
from models.pi_walk_model import PIWalkModel, PIWalkModelConfig
from models.game_model import GameModel, GameModelConfig
from robots.pi import PI, PIConfig
from robots.controlled_robot import ControlledRobotWrapper, ControlledRobotWrapperConfig


HALF_FIELD_SIZE = (4.5, 3.0)
GOAL_WIDTH = 1.9
PI_TARGET_Q_OFFSET = np.array([ -0.25,   0.0, -0.25,   0.0, 
                                  0.0,   0.2,   0.0,  -0.2, 
                                  0.0,   0.0,   0.0,   0.0, 
                                 0.65,  -1.2,  0.65,  -1.2, 
                                 -0.4,  -0.4,   0.0,   0.0, ], 
                              dtype=np.float32)

gs.init(
    backend=gs.gpu,  # type: ignore[unsolved-attribute]
    performance_mode=True,
    logging_level="info",
)

env = GameEnv(
    GameEnvConfig(
        robot_cfg = {
            "red":  ControlledRobotWrapperConfig(
                robot_cfg=PIConfig(initial_pos=np.array([-2.0, 0.0, 0.5], dtype=np.float32)),
                robot_class=PI,
                ctrl_model_cfg=PIWalkModelConfig(n_dofs=20, target_q_offset=PI_TARGET_Q_OFFSET),
                ctrl_model_class=PIWalkModel,
                ctrl_policy_path="models_ckpt/pi_policy.pt"
            ),
            "blue": ControlledRobotWrapperConfig(
                robot_cfg=PIConfig(initial_pos=np.array([2.0, 0.0, 0.5], dtype=np.float32)),
                robot_class=PI,
                ctrl_model_cfg=PIWalkModelConfig(n_dofs=20, target_q_offset=PI_TARGET_Q_OFFSET),
                ctrl_model_class=PIWalkModel,
                ctrl_policy_path="models_ckpt/pi_policy.pt"
            ),
        },
        robot_class     = {
            "red":  ControlledRobotWrapper,
            "blue": ControlledRobotWrapper,
        },
        robot_reset_pos = {
            "red": np.array([-2.0, 0.0, 0.0], dtype=np.float32),
            "blue": np.array([2.0, 0.0, 0.0], dtype=np.float32),
        },
        field_cfg       = SoccerFieldConfig(
            half_field_size = HALF_FIELD_SIZE,
            goal_width      = GOAL_WIDTH,
        ),
        field_class     = SoccerField,
        model_cfg       = GameModelConfig(
            half_field_size = HALF_FIELD_SIZE,
            goal_width      = GOAL_WIDTH,
        ),
        model_class     = GameModel,
        num_envs        = 1,
        show_viewer     = True,
        policy_freq     = 50,
        sim_freq        = 500,
        ctrl_freq_ratio = 10,
    )
)

env.reset()
action = torch.zeros((1, 3), dtype=torch.float, device=gs.device)
while True:
    _ = env.step({"red": action, "blue": action})
