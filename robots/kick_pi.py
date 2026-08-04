"""PiPlus configuration matching the 22-DoF kick policy export."""

import numpy as np
from dataclasses import dataclass, field

from .pi import PI
from .robot import RobotConfig


@dataclass(kw_only=True)
class KickPIConfig(RobotConfig):
    robot_URDF: str = "assets/PI/pi_plus.urdf"
    base_link_name: str = "base_link"
    joint_names: list[str] = field(default_factory=lambda: [
        # This is the IsaacSim/training order from robot_cfgs.py, not URDF order.
        "head_yaw_joint", "l_hip_pitch_joint", "l_shoulder_pitch_joint",
        "r_hip_pitch_joint", "r_shoulder_pitch_joint", "head_pitch_joint",
        "l_hip_roll_joint", "l_shoulder_roll_joint", "r_hip_roll_joint",
        "r_shoulder_roll_joint", "l_thigh_joint", "l_upper_arm_joint",
        "r_thigh_joint", "r_upper_arm_joint", "l_calf_joint", "l_elbow_joint",
        "r_calf_joint", "r_elbow_joint", "l_ankle_pitch_joint",
        "r_ankle_pitch_joint", "l_ankle_roll_joint", "r_ankle_roll_joint",
    ])
    foot_link_names: list[str] = field(default_factory=lambda: ["l_ankle_roll_link", "r_ankle_roll_link"])
    kp: np.ndarray = field(default_factory=lambda: np.array([
        7.80094, 50.9666, 32.5065, 50.9666, 32.5065, 7.80094,
        50.9666, 32.5065, 50.9666, 32.5065, 50.9666, 32.5065,
        50.9666, 32.5065, 50.9666, 32.5065, 50.9666, 32.5065,
        50.9666, 50.9666, 50.9666, 50.9666,
    ], dtype=np.float32))
    kv: np.ndarray = field(default_factory=lambda: np.array([
        0.496623, 3.24464, 2.06943, 3.24464, 2.06943, 0.496623,
        3.24464, 2.06943, 3.24464, 2.06943, 3.24464, 2.06943,
        3.24464, 2.06943, 3.24464, 2.06943, 3.24464, 2.06943,
        3.24464, 3.24464, 3.24464, 3.24464,
    ], dtype=np.float32))
    armature: np.ndarray = field(default_factory=lambda: np.array([
        0.001976, 0.01291, 0.008234, 0.01291, 0.008234, 0.001976,
        0.01291, 0.008234, 0.01291, 0.008234, 0.01291, 0.008234,
        0.01291, 0.008234, 0.01291, 0.008234, 0.01291, 0.008234,
        0.01291, 0.01291, 0.01291, 0.01291,
    ], dtype=np.float32))
    damping: np.ndarray = field(default_factory=lambda: np.zeros(22, dtype=np.float32))
    velocity_range: np.ndarray = field(default_factory=lambda: np.array([[-60.0] * 22, [60.0] * 22], dtype=np.float32))
    force_range: np.ndarray = field(default_factory=lambda: np.array([
        [-10.0, -20.0, -10.0, -20.0, -10.0, -10.0,
         -20.0, -10.0, -20.0, -10.0, -20.0, -10.0,
         -20.0, -10.0, -20.0, -10.0, -20.0, -10.0,
         -20.0, -20.0, -20.0, -20.0],
        [10.0, 20.0, 10.0, 20.0, 10.0, 10.0,
         20.0, 10.0, 20.0, 10.0, 20.0, 10.0,
         20.0, 10.0, 20.0, 10.0, 20.0, 10.0,
         20.0, 20.0, 20.0, 20.0],
    ], dtype=np.float32))
    initial_pos: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.351], dtype=np.float32))


class KickPI(PI):
    """Robot marker class for the kick sim2sim experiment."""
