# robot.py
#   Inherit from robot and build up pi
#

import numpy as np
from dataclasses import dataclass, field

from .robot import Robot, RobotConfig


@dataclass(kw_only=True)
class PIConfig(RobotConfig):
    robot_URDF: str = "assets/PI/pi_plus.urdf"
    base_link_name: str = "base_link"
    joint_names: list[str] = field(
        default_factory=lambda: [
            "l_hip_pitch_joint",
            "l_shoulder_pitch_joint",
            "r_hip_pitch_joint",
            "r_shoulder_pitch_joint",
            "l_hip_roll_joint",
            "l_shoulder_roll_joint",
            "r_hip_roll_joint",
            "r_shoulder_roll_joint",
            "l_thigh_joint",
            "l_upper_arm_joint",
            "r_thigh_joint",
            "r_upper_arm_joint",
            "l_calf_joint",
            "l_elbow_joint",
            "r_calf_joint",
            "r_elbow_joint",
            "l_ankle_pitch_joint",
            "r_ankle_pitch_joint",
            "l_ankle_roll_joint",
            "r_ankle_roll_joint",
        ]
    )
    kp: np.ndarray = field(
        default_factory=lambda: np.array(
            [80, 30, 80, 30, 80, 30, 80, 30, 80, 30, 80, 30, 80, 30, 80, 30, 60, 60, 60, 60],
            dtype=np.float32,
        )
    )
    kv: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                1.1,
                0.6,
                1.1,
                0.6,
                1.1,
                0.6,
                1.1,
                0.6,
                1.1,
                0.6,
                1.1,
                0.6,
                1.1,
                0.6,
                1.1,
                0.6,
                1.2,
                1.2,
                1.2,
                1.2,
            ],
            dtype=np.float32,
        )
    )
    initial_pos: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.5], dtype=np.float32))
    force_range: np.ndarray = field(
        default_factory=lambda: np.array([[-100] * 20, [100] * 20], dtype=np.float32)
    )
    velocity_range: np.ndarray = field(
        default_factory=lambda: np.array([[-100] * 20, [100] * 20], dtype=np.float32)
    )


class PI(Robot):
    pass
