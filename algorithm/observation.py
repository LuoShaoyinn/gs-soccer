import math
import numpy as np
from .joints import NUM_POLICY_JOINTS


class ObservationBuilder:
    def __init__(self, num_envs: int, num_history: int = 8):
        self.num_envs = num_envs
        self.num_history = num_history
        self.num_joints = NUM_POLICY_JOINTS
        self.num_commands = 7
        self.frame_dim = 6 + self.num_commands + 3 * self.num_joints  # 79
        self.obs_dim = self.num_history * self.frame_dim  # 632
        self.ang_vel_scale = 0.25
        self.joint_vel_scale = 0.05
        self.history = np.zeros((num_envs, num_history, self.frame_dim), dtype=np.float32)
        self._default_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self.history[:, :, 3:6] = self._default_gravity

    def reset(self, envs_idx: np.ndarray, command: np.ndarray | None = None):
        self.history[envs_idx] = 0.0
        self.history[envs_idx, :, 3:6] = self._default_gravity
        if command is not None:
            for h in range(self.num_history):
                self.history[envs_idx, h, 6:13] = command

    def push(self, ang_vel: np.ndarray, proj_gravity: np.ndarray,
             command: np.ndarray, joint_pos_rel: np.ndarray,
             joint_vel: np.ndarray, actions: np.ndarray):
        frame = np.concatenate([
            ang_vel * self.ang_vel_scale,
            proj_gravity,
            command,
            joint_pos_rel,
            joint_vel * self.joint_vel_scale,
            actions,
        ], axis=-1)
        self.history[:, :-1, :] = self.history[:, 1:, :]
        self.history[:, -1, :] = frame

    def get(self) -> np.ndarray:
        return self.history.reshape(self.num_envs, -1).copy()

    @staticmethod
    def quat_to_projected_gravity(quat_wxyz: np.ndarray) -> np.ndarray:
        w = quat_wxyz[..., 0]
        x = quat_wxyz[..., 1]
        y = quat_wxyz[..., 2]
        z = quat_wxyz[..., 3]
        g_x = -2.0 * (x * z + w * y)
        g_y = -2.0 * (y * z - w * x)
        g_z = -(1.0 - 2.0 * (x * x + y * y))
        return np.stack([g_x, g_y, g_z], axis=-1).astype(np.float32)
