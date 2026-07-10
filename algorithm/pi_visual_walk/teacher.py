from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

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


def _sim2sim_root() -> Path:
    return Path(__file__).resolve().parents[2] / "refs" / "piplus_soccer_sim2sim"


class WalkTeacher:
    """Adapter from 20-DOF walk observations to the exported 22-DOF soccer ONNX actor."""

    def __init__(self, model_dir: str = "refs/piplus_soccer_sim2sim/models/exported", device=None):
        root = _sim2sim_root()
        sim2sim_dir = root / "sim2sim"
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        if str(sim2sim_dir) not in sys.path:
            sys.path.insert(0, str(sim2sim_dir))

        try:
            from sim2sim.onnx_policy_soccer import SoccerOnnxPolicy
        except ModuleNotFoundError as exc:
            if exc.name == "onnxruntime":
                raise ModuleNotFoundError(
                    "onnxruntime is required for the exported sim2sim teacher. "
                    "Run `uv sync --extra rocm` after updating pyproject.toml."
                ) from exc
            raise

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.policy = SoccerOnnxPolicy(model_dir)
        if self.policy.proprio_dim != TEACHER_OBS_DIM or self.policy.action_dim != NUM_POLICY_JOINTS:
            raise ValueError(
                f"Unexpected teacher shape: obs={self.policy.proprio_dim}, action={self.policy.action_dim}"
            )

    @torch.no_grad()
    def infer(self, walk_obs: torch.Tensor) -> torch.Tensor:
        teacher_obs = expand_walk_obs_to_teacher(walk_obs).detach().cpu().numpy()
        if self.policy.proprio_dim == TEACHER_OBS_DIM and teacher_obs.shape[0] != 1:
            action22 = np.concatenate([self.policy.infer(row[None, :]) for row in teacher_obs], axis=0)
        else:
            action22 = self.policy.infer(teacher_obs)
        action20 = action22[:, POLICY_TO_GENESIS]
        return torch.as_tensor(action20, dtype=torch.float32, device=walk_obs.device)


def expand_walk_obs_to_teacher(walk_obs: torch.Tensor) -> torch.Tensor:
    if walk_obs.shape[-1] != WALK_OBS_DIM:
        raise ValueError(f"Expected walk obs dim {WALK_OBS_DIM}, got {walk_obs.shape[-1]}")

    leading_shape = walk_obs.shape[:-1]
    frames = walk_obs.reshape(*leading_shape, HISTORY_LEN, WALK_OBS_FRAME_DIM)

    ang_vel = frames[..., 0:3]
    gravity = frames[..., 3:6]
    walk_cmd = frames[..., 6:9]
    jpos20 = frames[..., 9:29]
    jvel20 = frames[..., 29:49]
    act20 = frames[..., 49:69]

    cmd7 = torch.zeros(*leading_shape, HISTORY_LEN, NUM_SOCCER_COMMANDS, dtype=walk_obs.dtype, device=walk_obs.device)
    cmd7[..., 0:3] = walk_cmd

    jpos22 = _insert_fake_head_slots(jpos20)
    jvel22 = _insert_fake_head_slots(jvel20)
    act22 = _insert_fake_head_slots(act20)

    teacher_frames = torch.cat([ang_vel, gravity, cmd7, jpos22, jvel22, act22], dim=-1)
    return teacher_frames.reshape(*leading_shape, TEACHER_OBS_DIM)


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
