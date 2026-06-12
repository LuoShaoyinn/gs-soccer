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
        self._default_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self.ang_vel_buf = np.zeros((num_envs, num_history, 3), dtype=np.float32)
        self.grav_buf = np.zeros((num_envs, num_history, 3), dtype=np.float32)
        self.grav_buf[:] = self._default_gravity
        self.cmd_buf = np.zeros((num_envs, num_history, self.num_commands), dtype=np.float32)
        self.jpos_buf = np.zeros((num_envs, num_history, self.num_joints), dtype=np.float32)
        self.jvel_buf = np.zeros((num_envs, num_history, self.num_joints), dtype=np.float32)
        self.act_buf = np.zeros((num_envs, num_history, self.num_joints), dtype=np.float32)

    def reset(self, envs_idx: np.ndarray, command: np.ndarray | None = None):
        self.ang_vel_buf[envs_idx] = 0.0
        self.grav_buf[envs_idx] = self._default_gravity
        self.cmd_buf[envs_idx] = 0.0
        self.jpos_buf[envs_idx] = 0.0
        self.jvel_buf[envs_idx] = 0.0
        self.act_buf[envs_idx] = 0.0
        if command is not None:
            for h in range(self.num_history):
                self.cmd_buf[envs_idx, h] = command

    def push(self, ang_vel: np.ndarray, proj_gravity: np.ndarray,
             command: np.ndarray, joint_pos_rel: np.ndarray,
             joint_vel: np.ndarray, actions: np.ndarray):
        for buf in (self.ang_vel_buf, self.grav_buf, self.cmd_buf,
                    self.jpos_buf, self.jvel_buf, self.act_buf):
            buf[:, :-1] = buf[:, 1:]
        self.ang_vel_buf[:, -1] = ang_vel * self.ang_vel_scale
        self.grav_buf[:, -1] = proj_gravity
        self.cmd_buf[:, -1] = command
        self.jpos_buf[:, -1] = joint_pos_rel
        self.jvel_buf[:, -1] = joint_vel * self.joint_vel_scale
        self.act_buf[:, -1] = actions

    def get(self) -> np.ndarray:
        return np.concatenate([
            self.ang_vel_buf.reshape(self.num_envs, -1),
            self.grav_buf.reshape(self.num_envs, -1),
            self.cmd_buf.reshape(self.num_envs, -1),
            self.jpos_buf.reshape(self.num_envs, -1),
            self.jvel_buf.reshape(self.num_envs, -1),
            self.act_buf.reshape(self.num_envs, -1),
        ], axis=-1).copy()

    @staticmethod
    def quat_to_projected_gravity(quat_wxyz: np.ndarray) -> np.ndarray:
        w = quat_wxyz[..., 0]
        x = quat_wxyz[..., 1]
        y = quat_wxyz[..., 2]
        z = quat_wxyz[..., 3]
        g_x = -2.0 * (x * z - w * y)
        g_y = -2.0 * (y * z + w * x)
        g_z = -(1.0 - 2.0 * (x * x + y * y))
        return np.stack([g_x, g_y, g_z], axis=-1).astype(np.float32)
