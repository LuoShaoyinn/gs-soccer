import os
import math
import torch
import gymnasium as gym
import numpy as np
from dataclasses import dataclass

import genesis as gs

from models.model import ModelConfig, Model

NUM_POLICY_JOINTS = 22
NUM_GENESIS_JOINTS = 20

_POLICY_TO_GENESIS = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]

_POLICY_DEFAULT_POS = [
    0.0, -0.25, 0.0, -0.25, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.65, 0.0, 0.65, 0.0,
    -0.4, -0.4, 0.0, 0.0,
]

_POLICY_ACTION_SCALE = [
    0.096, 0.098, 0.154, 0.098, 0.154, 0.096,
    0.098, 0.154, 0.098, 0.154,
    0.098, 0.154, 0.098, 0.154,
    0.098, 0.154, 0.098, 0.154,
    0.098, 0.098, 0.098, 0.098,
]

HISTORY_LEN = 8
NUM_ANG_VEL = 3
NUM_GRAVITY = 3
NUM_CMD = 7
OBS_DIM = HISTORY_LEN * (NUM_ANG_VEL + NUM_GRAVITY + NUM_CMD + 3 * NUM_POLICY_JOINTS)

ANG_VEL_SCALE = 0.25
JOINT_VEL_SCALE = 0.05


class ActorMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc0 = torch.nn.Linear(632, 512)
        self.fc1 = torch.nn.Linear(512, 256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc3 = torch.nn.Linear(128, 22)
        self.act = torch.nn.ELU()

    def forward(self, x):
        x = self.act(self.fc0(x))
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return self.fc3(x)


@dataclass(kw_only=True)
class Sim2SimSoccerConfig(ModelConfig):
    model_dir: str = "runs"
    model_file: str = "pi_plus_actor.pt"


class Sim2SimSoccerModel(Model):
    cfg: Sim2SimSoccerConfig

    def build(self):
        self.actor = ActorMLP().to(gs.device)
        ckpt = os.path.join(self.cfg.model_dir, self.cfg.model_file)
        self.actor.load_state_dict(torch.load(ckpt, map_location=gs.device))
        self.actor.eval()

    def config(self):
        dev = gs.device
        self.idx = torch.tensor(_POLICY_TO_GENESIS, dtype=torch.long, device=dev)
        self.policy_default_pos = torch.tensor(_POLICY_DEFAULT_POS, dtype=torch.float32, device=dev)
        self.default_pos = self.policy_default_pos[self.idx]

        policy_action_scale = torch.tensor(_POLICY_ACTION_SCALE, dtype=torch.float32, device=dev)
        self.action_scale = policy_action_scale[self.idx]

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

    def reset(self, envs_idx: torch.Tensor):
        for buf in (self._buf_ang_vel, self._buf_grav, self._buf_cmd,
                    self._buf_jpos, self._buf_jvel, self._buf_act):
            buf[envs_idx] = 0.0
        self._buf_grav[envs_idx] = self._default_gravity
        self._last_action[envs_idx] = 0.0
        self._pending_action = None

    def build_observation(self, envs_idx, **kwargs) -> torch.Tensor:
        ang_vel = kwargs["body_ang_vel"]
        body_quat = kwargs["body_quat"]
        dofs_pos = kwargs["dofs_pos"]
        dofs_vel = kwargs["dofs_vel"]
        soccer_cmd = kwargs["soccer_cmd"]
        B = ang_vel.shape[0]

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
        return self.default_pos + action[:, self.idx] * self.action_scale

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.actor(obs)

    def build_info(self, envs_idx, **kwargs) -> dict[str, torch.Tensor]:
        return {
            "body_pos": kwargs["body_pos"],
            "ball_pos": kwargs["ball_pos"],
            "ball_vel": kwargs["ball_vel"],
            "soccer_cmd": kwargs["soccer_cmd"],
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
