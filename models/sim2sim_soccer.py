import math
import torch
import gymnasium as gym
import numpy as np
from dataclasses import dataclass

import genesis as gs

from models.model import ModelConfig, Model
from algorithm.actor import NUM_POLICY_JOINTS, POLICY_TO_GENESIS, POLICY_DEFAULT_POS

HISTORY_LEN = 8
NUM_ANG_VEL = 3
NUM_GRAVITY = 3
NUM_CMD = 7
OBS_DIM = HISTORY_LEN * (NUM_ANG_VEL + NUM_GRAVITY + NUM_CMD + 3 * NUM_POLICY_JOINTS)

ANG_VEL_SCALE = 0.25
JOINT_VEL_SCALE = 0.05


@dataclass(kw_only=True)
class Sim2SimSoccerConfig(ModelConfig):
    mode: str = "walk"
    vel_cmd: tuple[float, float, float] = (0.5, 0.0, 0.0)
    kick_speed: float = 2.0
    kick_dir_deg: float = 0.0
    kick_zone_radius: float = 1.0
    approach_speed_vx: float = 1.2
    approach_kp_wz: float = 2.0
    approach_max_wz: float = 1.5


class Sim2SimSoccerModel(Model):
    cfg: Sim2SimSoccerConfig

    def config(self):
        dev = gs.device
        self.idx = torch.tensor(POLICY_TO_GENESIS, dtype=torch.long, device=dev)
        self.policy_default_pos = torch.tensor(POLICY_DEFAULT_POS, dtype=torch.float32, device=dev)

        B = self.scene.n_envs
        H = HISTORY_LEN
        self._buf_ang_vel = torch.zeros((B, H, NUM_ANG_VEL), dtype=torch.float32, device=dev)
        self._buf_grav = torch.zeros((B, H, NUM_GRAVITY), dtype=torch.float32, device=dev)
        self._buf_cmd = torch.zeros((B, H, NUM_CMD), dtype=torch.float32, device=dev)
        self._buf_jpos = torch.zeros((B, H, NUM_POLICY_JOINTS), dtype=torch.float32, device=dev)
        self._buf_jvel = torch.zeros((B, H, NUM_POLICY_JOINTS), dtype=torch.float32, device=dev)
        self._buf_act = torch.zeros((B, H, NUM_POLICY_JOINTS), dtype=torch.float32, device=dev)
        self._default_gravity = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=dev)
        self._last_action = torch.zeros((B, NUM_POLICY_JOINTS), dtype=torch.float32, device=dev)
        self._pending_action = None

        self._in_kick_zone = torch.zeros(B, dtype=torch.bool, device=dev)
        self._target_pos_w = torch.zeros((B, 2), dtype=torch.float64, device=dev)
        self._kick_dir_yaw = math.radians(self.cfg.kick_dir_deg)
        self._last_cmd = torch.zeros((B, NUM_CMD), dtype=torch.float32, device=dev)

    def reset(self, envs_idx: torch.Tensor):
        for buf in (self._buf_ang_vel, self._buf_grav, self._buf_cmd,
                    self._buf_jpos, self._buf_jvel, self._buf_act):
            buf[envs_idx] = 0.0
        self._buf_grav[envs_idx] = self._default_gravity
        self._last_action[envs_idx] = 0.0
        self._pending_action = None
        self._in_kick_zone[envs_idx] = False
        self._target_pos_w[envs_idx] = 0.0

    def _compute_soccer_cmd(self, body_pos, body_quat, ball_pos):
        B = body_pos.shape[0]
        w = body_quat[:, 0]
        x = body_quat[:, 1]
        y = body_quat[:, 2]
        z = body_quat[:, 3]
        robot_yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

        dx = ball_pos[:, 0] - body_pos[:, 0]
        dy = ball_pos[:, 1] - body_pos[:, 1]
        cos_y = torch.cos(robot_yaw)
        sin_y = torch.sin(robot_yaw)
        ball_x_b = cos_y * dx + sin_y * dy
        ball_y_b = -sin_y * dx + cos_y * dy
        dist = torch.sqrt(dx * dx + dy * dy)

        cmd = torch.zeros((B, NUM_CMD), dtype=torch.float32, device=gs.device)
        cmd[:, 3] = ball_x_b.float()
        cmd[:, 4] = ball_y_b.float()

        mode = self.cfg.mode
        ks = self.cfg.kick_speed

        if mode == "walk":
            cmd[:, 0] = self.cfg.vel_cmd[0]
            cmd[:, 1] = self.cfg.vel_cmd[1]
            cmd[:, 2] = self.cfg.vel_cmd[2]

        elif mode == "approach_kick":
            newly = (~self._in_kick_zone) & (dist < self.cfg.kick_zone_radius)
            newly_idx = newly.nonzero(as_tuple=True)[0]
            if newly_idx.numel() > 0:
                self._in_kick_zone[newly_idx] = True
                self._target_pos_w[newly_idx, 0] = (
                    ball_pos[newly_idx, 0].double() + ks * math.cos(self._kick_dir_yaw)
                )
                self._target_pos_w[newly_idx, 1] = (
                    ball_pos[newly_idx, 1].double() + ks * math.sin(self._kick_dir_yaw)
                )

            app = (~self._in_kick_zone).nonzero(as_tuple=True)[0]
            if app.numel() > 0:
                yaw_err = torch.atan2(ball_y_b[app], ball_x_b[app])
                wz = torch.clamp(
                    self.cfg.approach_kp_wz * yaw_err,
                    -self.cfg.approach_max_wz, self.cfg.approach_max_wz,
                )
                vx = torch.maximum(
                    torch.cos(yaw_err), torch.zeros(1, device=gs.device),
                ) * self.cfg.approach_speed_vx
                cmd[app, 0] = vx.float()
                cmd[app, 2] = wz.float()

            kick = self._in_kick_zone.nonzero(as_tuple=True)[0]
            if kick.numel() > 0:
                to_tgt = self._target_pos_w[kick] - ball_pos[kick, :2].double()
                dir_yaw = torch.atan2(to_tgt[:, 1], to_tgt[:, 0])
                cmd[kick, 5] = ((dir_yaw.float() - robot_yaw[kick].float()) + math.pi) % (2 * math.pi) - math.pi
                cmd[kick, 6] = ks

        elif mode == "kick":
            nf = (~self._in_kick_zone).nonzero(as_tuple=True)[0]
            if nf.numel() > 0:
                self._in_kick_zone[nf] = True
                self._target_pos_w[nf, 0] = (
                    ball_pos[nf, 0].double() + ks * math.cos(self._kick_dir_yaw)
                )
                self._target_pos_w[nf, 1] = (
                    ball_pos[nf, 1].double() + ks * math.sin(self._kick_dir_yaw)
                )

            to_tgt = self._target_pos_w - ball_pos[:, :2].double()
            dir_yaw = torch.atan2(to_tgt[:, 1], to_tgt[:, 0])
            cmd[:, 5] = ((dir_yaw.float() - robot_yaw.float()) + math.pi) % (2 * math.pi) - math.pi
            cmd[:, 6] = ks

        return cmd

    def build_observation(self, envs_idx, **kwargs) -> torch.Tensor:
        ang_vel = kwargs["body_ang_vel"]
        body_quat = kwargs["body_quat"]
        body_pos = kwargs["body_pos"]
        dofs_pos = kwargs["dofs_pos"]
        dofs_vel = kwargs["dofs_vel"]
        ball_pos = kwargs["ball_pos"]
        B = ang_vel.shape[0]

        soccer_cmd = self._compute_soccer_cmd(body_pos, body_quat, ball_pos)
        self._last_cmd = soccer_cmd

        jpos_policy = torch.zeros((B, NUM_POLICY_JOINTS), dtype=torch.float32, device=gs.device)
        jvel_policy = torch.zeros((B, NUM_POLICY_JOINTS), dtype=torch.float32, device=gs.device)
        jpos_policy[:, self.idx] = dofs_pos
        jvel_policy[:, self.idx] = dofs_vel
        jpos_rel = jpos_policy - self.policy_default_pos

        proj_grav = self._quat_to_projected_gravity(body_quat)

        for buf in (self._buf_ang_vel, self._buf_grav, self._buf_cmd,
                    self._buf_jpos, self._buf_jvel, self._buf_act):
            buf[:, :-1] = buf[:, 1:].clone()

        self._buf_ang_vel[:, -1] = ang_vel * ANG_VEL_SCALE
        self._buf_grav[:, -1] = proj_grav
        self._buf_cmd[:, -1] = soccer_cmd
        self._buf_jpos[:, -1] = jpos_rel
        self._buf_jvel[:, -1] = jvel_policy * JOINT_VEL_SCALE
        self._buf_act[:, -1] = self._last_action

        obs = torch.cat([
            self._buf_ang_vel.reshape(B, -1),
            self._buf_grav.reshape(B, -1),
            self._buf_cmd.reshape(B, -1),
            self._buf_jpos.reshape(B, -1),
            self._buf_jvel.reshape(B, -1),
            self._buf_act.reshape(B, -1),
        ], dim=-1)

        if self._pending_action is not None:
            self._last_action = self._pending_action
            self._pending_action = None

        return obs

    def preprocess_action(self, action: torch.Tensor) -> torch.Tensor:
        self._pending_action = action.clone()
        idx = self.idx
        default_pos = self.policy_default_pos[idx]
        from algorithm.actor import POLICY_ACTION_SCALE
        scale = torch.tensor(POLICY_ACTION_SCALE, dtype=torch.float32, device=gs.device)[idx]
        return default_pos + action[:, idx] * scale

    def build_info(self, envs_idx, **kwargs) -> dict[str, torch.Tensor]:
        return {
            "body_pos": kwargs["body_pos"],
            "ball_pos": kwargs["ball_pos"],
            "ball_vel": kwargs["ball_vel"],
            "soccer_cmd": self._last_cmd,
        }

    @staticmethod
    def _quat_to_projected_gravity(quat_wxyz: torch.Tensor) -> torch.Tensor:
        w, x, y, z = quat_wxyz[:, 0], quat_wxyz[:, 1], quat_wxyz[:, 2], quat_wxyz[:, 3]
        g_x = -2.0 * (x * z - w * y)
        g_y = -2.0 * (y * z + w * x)
        g_z = -(1.0 - 2.0 * (x * x + y * y))
        return torch.stack([g_x, g_y, g_z], dim=-1)

    @property
    def observation_space(self) -> gym.spaces.Box:
        return gym.spaces.Box(-float("inf"), float("inf"), (OBS_DIM,), dtype=np.float32)

    @property
    def action_space(self) -> gym.spaces.Box:
        return gym.spaces.Box(-float("inf"), float("inf"), (NUM_POLICY_JOINTS,), dtype=np.float32)
