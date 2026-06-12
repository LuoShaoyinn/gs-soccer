import numpy as np

POLICY_JOINT_NAMES = (
    "head_yaw_joint",
    "l_hip_pitch_joint",
    "l_shoulder_pitch_joint",
    "r_hip_pitch_joint",
    "r_shoulder_pitch_joint",
    "head_pitch_joint",
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
)

NUM_POLICY_JOINTS = len(POLICY_JOINT_NAMES)

POLICY_DEFAULT_POS = np.array([
    0.0, -0.25, 0.0, -0.25, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.65, 0.0, 0.65, 0.0,
    -0.4, -0.4, 0.0, 0.0,
], dtype=np.float32)

POLICY_ACTION_SCALE = np.array([
    0.096, 0.098, 0.154, 0.098, 0.154, 0.096,
    0.098, 0.154, 0.098, 0.154,
    0.098, 0.154, 0.098, 0.154,
    0.098, 0.154, 0.098, 0.154,
    0.098, 0.098, 0.098, 0.098,
], dtype=np.float32)

POLICY_KP = np.array([
    7.80, 50.97, 32.51, 50.97, 32.51, 7.80,
    50.97, 32.51, 50.97, 32.51,
    50.97, 32.51, 50.97, 32.51,
    50.97, 32.51, 50.97, 32.51,
    50.97, 50.97, 50.97, 50.97,
], dtype=np.float32)

POLICY_KD = np.array([
    0.50, 3.24, 2.07, 3.24, 2.07, 0.50,
    3.24, 2.07, 3.24, 2.07,
    3.24, 2.07, 3.24, 2.07,
    3.24, 2.07, 3.24, 2.07,
    3.24, 3.24, 3.24, 3.24,
], dtype=np.float32)

GENESIS_JOINT_NAMES = (
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
)

NUM_GENESIS_JOINTS = len(GENESIS_JOINT_NAMES)

POLICY_TO_GENESIS = np.array([1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21], dtype=np.int32)

GENESIS_DEFAULT_POS = POLICY_DEFAULT_POS[POLICY_TO_GENESIS]
GENESIS_ACTION_SCALE = POLICY_ACTION_SCALE[POLICY_TO_GENESIS]
GENESIS_KP = POLICY_KP[POLICY_TO_GENESIS]
GENESIS_KD = POLICY_KD[POLICY_TO_GENESIS]
