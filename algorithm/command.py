import math
import numpy as np


class SoccerCommandBuilder:
    NUM_COMMANDS = 7

    def __init__(self, mode: str = "approach_kick",
                 vel_cmd: tuple = (0.0, 0.0, 0.0),
                 kick_speed: float = 2.0,
                 kick_dir_deg: float = 0.0,
                 kick_zone_radius: float = 1.0,
                 approach_speed_vx: float = 1.2,
                 approach_kp_wz: float = 2.0,
                 approach_max_wz: float = 1.5):
        self.mode = mode
        self.vel_cmd = np.asarray(vel_cmd, dtype=np.float32).reshape(3)
        self.kick_speed = float(kick_speed)
        self.kick_dir_yaw_w = math.radians(kick_dir_deg)
        self.kick_zone_radius = float(kick_zone_radius)
        self.approach_speed_vx = float(approach_speed_vx)
        self.approach_kp_wz = float(approach_kp_wz)
        self.approach_max_wz = float(approach_max_wz)
        self._in_kick_zone = False
        self._target_pos_w = np.zeros(2, dtype=np.float64)

    def reset(self):
        self._in_kick_zone = False

    def set_velocity(self, vel_cmd: np.ndarray):
        self.vel_cmd = np.asarray(vel_cmd, dtype=np.float32).reshape(3)

    def _freeze_target(self, ball_xy_w: np.ndarray):
        self._target_pos_w[0] = ball_xy_w[0] + self.kick_speed * math.cos(self.kick_dir_yaw_w)
        self._target_pos_w[1] = ball_xy_w[1] + self.kick_speed * math.sin(self.kick_dir_yaw_w)

    def compute(self, robot_pos_w: np.ndarray, robot_yaw_w: float,
                ball_pos_w: np.ndarray) -> np.ndarray:
        robot_xy = np.asarray(robot_pos_w, dtype=np.float64)[:2]
        ball_xy = np.asarray(ball_pos_w, dtype=np.float64)[:2]
        cos_y, sin_y = math.cos(robot_yaw_w), math.sin(robot_yaw_w)
        dx = ball_xy[0] - robot_xy[0]
        dy = ball_xy[1] - robot_xy[1]
        ball_x_b = cos_y * dx + sin_y * dy
        ball_y_b = -sin_y * dx + cos_y * dy
        dist = math.hypot(dx, dy)
        cmd = np.zeros(self.NUM_COMMANDS, dtype=np.float32)
        cmd[3] = ball_x_b
        cmd[4] = ball_y_b

        if self.mode == "walk":
            cmd[0:3] = self.vel_cmd
            return cmd

        if self.mode == "approach_kick":
            if not self._in_kick_zone and dist < self.kick_zone_radius:
                self._in_kick_zone = True
                self._freeze_target(ball_xy)
            if not self._in_kick_zone:
                yaw_err = math.atan2(ball_y_b, ball_x_b)
                wz = float(np.clip(self.approach_kp_wz * yaw_err,
                                   -self.approach_max_wz, self.approach_max_wz))
                vx = max(math.cos(yaw_err), 0.0) * self.approach_speed_vx
                cmd[0], cmd[1], cmd[2] = vx, 0.0, wz
                return cmd

        if not self._in_kick_zone:
            self._freeze_target(ball_xy)
        cmd[0:3] = 0.0
        to_target = self._target_pos_w - ball_xy
        dir_yaw_w = math.atan2(to_target[1], to_target[0])
        cmd[5] = (dir_yaw_w - robot_yaw_w + math.pi) % (2.0 * math.pi) - math.pi
        cmd[6] = self.kick_speed
        return cmd
