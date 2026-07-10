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
    """Adapter from 20-DOF walk observations to the exported 22-DOF soccer ONNX actor.

    The Genesis robot only actuates 20 joints: the head_yaw / head_pitch motors
    (policy indices 0 and 5) are fixed. Their joint-pos / joint-vel observation
    slots therefore stay at zero, but the *action* history slot is filled with the
    real last_action the policy actually produced (all 22 dims), tracked here so
    that every history frame is aligned with the env's proprioceptive state.
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

        # Push the real previous output as the most recent action frame so the
        # action history is temporally aligned with the env's state history.
        self._action_history[:, :-1] = self._action_history[:, 1:].clone()
        self._action_history[:, -1] = self._last_action

        teacher_obs_t = expand_walk_obs_to_teacher(walk_obs, action_history22=self._action_history)
        if self.net is not None:
            action22 = self.net(teacher_obs_t.to(self.device)).detach().cpu().numpy()
        else:
            teacher_obs = teacher_obs_t.detach().cpu().numpy()
            if teacher_obs.shape[0] != 1:
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
    """Expand the 20-DOF walk obs into the 22-DOF teacher obs.

    The walk obs is laid out *term-major*: ``[ang(8*3)][grav(8*3)][cmd(8*3)]
    [jpos(8*20)][jvel(8*20)][act(8*20)]``. The exported policy expects the same
    term-major order over its 79-wide frame (``[ang][grav][cmd7][jpos][jvel][act]``).
    """
    if walk_obs.shape[-1] != WALK_OBS_DIM:
        raise ValueError(f"Expected walk obs dim {WALK_OBS_DIM}, got {walk_obs.shape[-1]}")

    leading = walk_obs.shape[:-1]
    H = HISTORY_LEN

    def term(lo: int, hi: int, d: int) -> torch.Tensor:
        return walk_obs[..., lo:hi].reshape(*leading, H, d)

    ang_vel = term(0, H * 3, 3)
    gravity = term(H * 3, H * 6, 3)
    walk_cmd = term(H * 6, H * 9, 3)
    jpos20 = term(H * 9, H * 29, NUM_GENESIS_JOINTS)
    jvel20 = term(H * 29, H * 49, NUM_GENESIS_JOINTS)

    cmd7 = torch.zeros(*leading, H, NUM_SOCCER_COMMANDS, dtype=walk_obs.dtype, device=walk_obs.device)
    cmd7[..., 0:3] = walk_cmd

    # Head/neck motors are fixed: their pos/vel stay at default (zero relative).
    jpos22 = _insert_fake_head_slots(jpos20)
    jvel22 = _insert_fake_head_slots(jvel20)
    # The action slot carries the *real* last_action for all 22 joints.
    if action_history22 is not None:
        act22 = action_history22.to(dtype=walk_obs.dtype, device=walk_obs.device)
    else:
        act20 = term(H * 49, H * 69, NUM_GENESIS_JOINTS)
        act22 = _insert_fake_head_slots(act20)

    # term-major flatten: each term's full history as one contiguous block.
    return torch.cat(
        [
            ang_vel.reshape(*leading, H * 3),
            gravity.reshape(*leading, H * 3),
            cmd7.reshape(*leading, H * NUM_SOCCER_COMMANDS),
            jpos22.reshape(*leading, H * NUM_POLICY_JOINTS),
            jvel22.reshape(*leading, H * NUM_POLICY_JOINTS),
            act22.reshape(*leading, H * NUM_POLICY_JOINTS),
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
