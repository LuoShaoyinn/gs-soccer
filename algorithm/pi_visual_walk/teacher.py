from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HISTORY_LEN = 8
NUM_GENESIS_JOINTS = 20
NUM_POLICY_JOINTS = 22
NUM_WALK_COMMANDS = 3
NUM_SOCCER_COMMANDS = 7

POLICY_TO_GENESIS = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
HEAD_POLICY_IDX = (0, 5)

POLICY_DEFAULT_POS = [
    0.0, -0.25, 0.0, -0.25, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.65, 0.0, 0.65, 0.0,
    -0.4, -0.4, 0.0, 0.0,
]

POLICY_ACTION_SCALE = [
    0.096, 0.098, 0.154, 0.098, 0.154, 0.096,
    0.098, 0.154, 0.098, 0.154,
    0.098, 0.154, 0.098, 0.154,
    0.098, 0.154, 0.098, 0.154,
    0.098, 0.098, 0.098, 0.098,
]

WALK_OBS_FRAME_DIM = 3 + 3 + NUM_WALK_COMMANDS + 3 * NUM_GENESIS_JOINTS
WALK_OBS_DIM = HISTORY_LEN * WALK_OBS_FRAME_DIM
TEACHER_OBS_FRAME_DIM = 3 + 3 + NUM_SOCCER_COMMANDS + 3 * NUM_POLICY_JOINTS
TEACHER_OBS_DIM = HISTORY_LEN * TEACHER_OBS_FRAME_DIM


class ActorMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc0 = nn.Linear(TEACHER_OBS_DIM, 512)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, NUM_POLICY_JOINTS)
        self.act = nn.ELU()

    def forward(self, x):
        x = self.act(self.fc0(x))
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return self.fc3(x)


def _sim2sim_root() -> Path:
    return Path(__file__).resolve().parents[2] / "refs" / "piplus_soccer_sim2sim"


class WalkTeacher:
    """Adapter from 20-DOF walk observations to the 22-DOF soccer teacher.

    The student/env observation is term-major:
    [ang_hist][grav_hist][cmd_hist][jpos_hist][jvel_hist][act_hist].
    The teacher expects the same term-major convention, with fixed head
    qpos/qvel slots and the real 22-D previous action history.
    """

    def __init__(
        self,
        model_dir: str = "refs/piplus_soccer_sim2sim/models/exported",
        model_file: str | None = None,
        device=None,
    ):
        root = _sim2sim_root()
        sim2sim_dir = root / "sim2sim"
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        if str(sim2sim_dir) not in sys.path:
            sys.path.insert(0, str(sim2sim_dir))

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._action_history = None
        self._last_action = None
        self.policy = None
        self.net = None
        if model_file is not None:
            self.net = ActorMLP().to(self.device)
            self.net.load_state_dict(torch.load(model_file, map_location=self.device))
            self.net.eval()
        else:
            try:
                from sim2sim.onnx_policy_soccer import SoccerOnnxPolicy
            except ModuleNotFoundError as exc:
                if exc.name == "onnxruntime":
                    raise ModuleNotFoundError(
                        "onnxruntime is required for the exported sim2sim teacher. "
                        "Run `uv sync --extra rocm` after updating pyproject.toml."
                    ) from exc
                raise
            self.policy = SoccerOnnxPolicy(model_dir)
            if self.policy.proprio_dim != TEACHER_OBS_DIM or self.policy.action_dim != NUM_POLICY_JOINTS:
                raise ValueError(
                    f"Unexpected teacher shape: obs={self.policy.proprio_dim}, action={self.policy.action_dim}"
                )

    @torch.no_grad()
    def infer(self, walk_obs: torch.Tensor, reset_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch = walk_obs.shape[0]
        dev = walk_obs.device
        self._ensure_buffers(batch, dev)

        if reset_mask is not None and reset_mask.any():
            idx = torch.nonzero(reset_mask, as_tuple=False).squeeze(-1)
            self._action_history[idx] = 0.0
            self._last_action[idx] = 0.0

        self._action_history[:, :-1] = self._action_history[:, 1:].clone()
        self._action_history[:, -1] = self._last_action

        teacher_obs_t = expand_walk_obs_to_teacher(walk_obs, action_history22=self._action_history)
        if self.net is not None:
            action22 = self.net(teacher_obs_t.to(self.device)).detach().cpu().numpy()
        else:
            teacher_obs = teacher_obs_t.detach().cpu().numpy()
            if self.policy.proprio_dim == TEACHER_OBS_DIM and teacher_obs.shape[0] != 1:
                action22 = np.concatenate([self.policy.infer(row[None, :]) for row in teacher_obs], axis=0)
            else:
                action22 = self.policy.infer(teacher_obs)
        self._last_action = torch.as_tensor(action22, dtype=torch.float32, device=dev)
        action20 = action22[:, POLICY_TO_GENESIS]
        return torch.as_tensor(action20, dtype=torch.float32, device=dev)

    def reset(self, num_envs: int | None = None, device=None):
        if num_envs is None:
            self._action_history = None
            self._last_action = None
            return
        dev = torch.device(device or self.device)
        self._action_history = torch.zeros(num_envs, HISTORY_LEN, NUM_POLICY_JOINTS, device=dev)
        self._last_action = torch.zeros(num_envs, NUM_POLICY_JOINTS, device=dev)

    def _ensure_buffers(self, batch: int, device):
        if (
            self._action_history is None
            or self._action_history.shape[0] != batch
            or self._action_history.device != device
        ):
            self.reset(batch, device)


def expand_walk_obs_to_teacher(
    walk_obs: torch.Tensor,
    action_history22: torch.Tensor | None = None,
) -> torch.Tensor:
    """Expand 20-DOF walk obs into the teacher's 22-DOF term-major obs."""
    if walk_obs.shape[-1] != WALK_OBS_DIM:
        raise ValueError(f"Expected walk obs dim {WALK_OBS_DIM}, got {walk_obs.shape[-1]}")

    leading = walk_obs.shape[:-1]
    h = HISTORY_LEN

    def term(start: int, dim: int) -> torch.Tensor:
        end = start + h * dim
        return walk_obs[..., start:end].reshape(*leading, h, dim)

    offset = 0
    ang_vel = term(offset, 3); offset += h * 3
    gravity = term(offset, 3); offset += h * 3
    walk_cmd = term(offset, NUM_WALK_COMMANDS); offset += h * NUM_WALK_COMMANDS
    jpos20 = term(offset, NUM_GENESIS_JOINTS); offset += h * NUM_GENESIS_JOINTS
    jvel20 = term(offset, NUM_GENESIS_JOINTS); offset += h * NUM_GENESIS_JOINTS

    cmd7 = torch.zeros(*leading, h, NUM_SOCCER_COMMANDS, dtype=walk_obs.dtype, device=walk_obs.device)
    cmd7[..., 0:3] = walk_cmd

    jpos22 = _insert_fake_head_slots(jpos20)
    jvel22 = _insert_fake_head_slots(jvel20)
    if action_history22 is not None:
        act22 = action_history22.to(dtype=walk_obs.dtype, device=walk_obs.device)
    else:
        act20 = term(offset, NUM_GENESIS_JOINTS)
        act22 = _insert_fake_head_slots(act20)

    return torch.cat(
        [
            ang_vel.reshape(*leading, h * 3),
            gravity.reshape(*leading, h * 3),
            cmd7.reshape(*leading, h * NUM_SOCCER_COMMANDS),
            jpos22.reshape(*leading, h * NUM_POLICY_JOINTS),
            jvel22.reshape(*leading, h * NUM_POLICY_JOINTS),
            act22.reshape(*leading, h * NUM_POLICY_JOINTS),
        ],
        dim=-1,
    )


def _insert_fake_head_slots(x20: torch.Tensor) -> torch.Tensor:
    x22 = torch.zeros(*x20.shape[:-1], NUM_POLICY_JOINTS, dtype=x20.dtype, device=x20.device)
    x22[..., POLICY_TO_GENESIS] = x20
    return x22


def action_clip_bounds(device):
    default = torch.tensor(POLICY_DEFAULT_POS, dtype=torch.float32, device=device)[POLICY_TO_GENESIS]
    scale = torch.tensor(POLICY_ACTION_SCALE, dtype=torch.float32, device=device)[POLICY_TO_GENESIS]
    low = torch.ceil((-1.57 - default) / scale)
    high = torch.floor((1.57 - default) / scale)
    return low, high


def clip_walk_action(action: torch.Tensor, action_low: torch.Tensor, action_high: torch.Tensor) -> torch.Tensor:
    return action.clamp(action_low.unsqueeze(0), action_high.unsqueeze(0))
