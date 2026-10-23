"""Validated MOS9 reference library. Human/G1 trajectories cannot be loaded directly."""

import hashlib
from pathlib import Path

import numpy as np
import torch

from .config import JOINT_NAMES, URDF
from .geometry import conjugate, multiply, normalize, rotation_vector


def derivative(x, dt):
    return torch.from_numpy(np.gradient(x.cpu().numpy(), dt, axis=0).astype(np.float32))


class MotionLibrary:
    def __init__(self, path, task, device="cpu", allow_smoke=False):
        path = Path(path)
        self.path = path.resolve()
        files = sorted(path.glob("*.npz")) if path.is_dir() else [path]
        if not files or any(not f.is_file() for f in files):
            raise ValueError(
                f"no MOS9 reference .npz files at {path}; see docs/supermimic.md"
            )
        self.names, clips, hashes = [], [], {}
        self.smoke_only = False
        for f in files:
            with np.load(f, allow_pickle=False) as data:
                required = {
                    "fps",
                    "joint_names",
                    "joint_pos",
                    "root_pos",
                    "root_quat",
                    "robot",
                }
                if required - set(data.files):
                    raise ValueError(
                        f"{f}: missing {sorted(required - set(data.files))}"
                    )
                if str(data["robot"].item()) != "MOS9":
                    raise ValueError(f"{f}: retarget to MOS9 first")
                smoke = "smoke_only" in data and bool(data["smoke_only"].item())
                if smoke and not allow_smoke:
                    raise ValueError(
                        "smoke-only reference cannot be used for walking training"
                    )
                self.smoke_only |= smoke
                names = tuple(str(n) for n in data["joint_names"])
                if len(names) != len(set(names)) or set(names) != set(JOINT_NAMES):
                    raise ValueError(
                        f"{f}: joint_names must name all {len(JOINT_NAMES)} MOS9 joints exactly once"
                    )
                order = [names.index(n) for n in JOINT_NAMES]
                fps = float(data["fps"].item())
                if not np.isfinite(fps) or not np.isclose(fps, 1 / task.cfg.dt):
                    raise ValueError(
                        f"{f}: resample to {1 / task.cfg.dt:g} Hz before loading"
                    )
                q = torch.tensor(data["joint_pos"], dtype=torch.float32)
                if q.ndim != 2 or q.shape[1] != task.n:
                    raise ValueError(f"{f}: joint_pos must be [T, {task.n}]")
                q = q[:, order]
                p = torch.tensor(data["root_pos"], dtype=torch.float32)
                r = torch.tensor(data["root_quat"], dtype=torch.float32)
                if (
                    len(q)
                    < task.cfg.horizon + task.cfg.future + task.cfg.model_window + 1
                ):
                    raise ValueError(
                        f"{f}: clip is too short for the configured horizons"
                    )
                if p.shape != (len(q), 3) or r.shape != (len(q), 4):
                    raise ValueError(f"{f}: root_pos/root_quat shape mismatch")
                if not all(torch.isfinite(t).all() for t in (q, p, r)):
                    raise ValueError(f"{f}: nonfinite motion values")
                if not torch.allclose(r.norm(dim=-1), torch.ones(len(r)), atol=0.01):
                    raise ValueError(f"{f}: root_quat must be unit wxyz quaternions")
                if (
                    (q < task.kin.lower.cpu() - 0.05)
                    | (q > task.kin.upper.cpu() + 0.05)
                ).any():
                    raise ValueError(
                        f"{f}: joint angles exceed URDF limits (check order, units, retargeting)"
                    )
                # Origin xy is arbitrary; floor z must be calibrated by retargeting.
                p[:, :2] -= p[0, :2].clone()
                r = normalize(r)
                if "joint_vel" in data:
                    dq = torch.tensor(data["joint_vel"], dtype=torch.float32)
                    if dq.shape != q.shape:
                        raise ValueError(f"{f}: joint_vel shape mismatch")
                    dq = dq[:, order]
                else:
                    dq = derivative(q, task.cfg.dt)
                v = (
                    torch.tensor(data["root_vel"], dtype=torch.float32)
                    if "root_vel" in data
                    else derivative(p, task.cfg.dt)
                )
                if "root_ang_vel" in data:
                    w = torch.tensor(data["root_ang_vel"], dtype=torch.float32)
                else:
                    # World angular velocity from consecutive authored-root quaternions.
                    delta = (
                        rotation_vector(multiply(r[1:], conjugate(r[:-1])))
                        / task.cfg.dt
                    )
                    w = torch.cat(
                        (delta[:1], (delta[:-1] + delta[1:]) / 2, delta[-1:]), 0
                    )
                if dq.shape != q.shape or v.shape != p.shape or w.shape != p.shape:
                    raise ValueError(f"{f}: velocity shape mismatch")
                if not all(torch.isfinite(t).all() for t in (dq, v, w)):
                    raise ValueError(f"{f}: nonfinite velocities")
                clips.append(task.pack(q, dq, p, r, v, w))
                self.names.append(f.name)
                hashes[f.name] = hashlib.sha256(f.read_bytes()).hexdigest()
        self.lengths = torch.tensor([len(c) for c in clips], device=device)
        self.states = torch.zeros(
            len(clips), max(map(len, clips)), task.state_dim, device=device
        )
        for i, c in enumerate(clips):
            self.states[i, : len(c)] = c.to(device)
        self.fingerprint = {
            "clips": hashes,
            "urdf": hashlib.sha256(URDF.read_bytes()).hexdigest(),
        }
        self.device, self.task = device, task

    def reference(self, clips, frames, count=1):
        idx = frames[..., None] + torch.arange(count, device=self.states.device)
        idx = torch.minimum(idx, self.lengths[clips][..., None] - 1)
        return self.states[clips[..., None], idx]


class ResetSampler:
    """Paper D.1: one-second bins, 50% uniform / 40% hard / 10% historical."""

    def __init__(self, library, gamma):
        self.library, self.gamma = library, gamma
        self.bin_frames = round(1 / library.task.cfg.dt)
        clips, frames = [], []
        for i, length in enumerate(library.lengths.tolist()):
            for f in range(0, length - 1, self.bin_frames):
                clips.append(i)
                frames.append(f)
        self.clips, self.frames = torch.tensor(clips), torch.tensor(frames)
        self.bin_index = {(c, f): i for i, (c, f) in enumerate(zip(clips, frames))}
        self.returns = torch.zeros(len(clips))
        self.seen = torch.zeros(len(clips), dtype=torch.bool)
        self.past = torch.ones(len(clips)) / len(clips)
        self.calls = 0

    def sample(self, n):
        gmax = 5 * self.library.task.cfg.dt / (1 - self.gamma)
        hard = torch.softmax(-25 * self.returns / gmax, 0)
        weights = 0.5 / len(hard) + 0.4 * hard + 0.1 * self.past / self.past.sum()
        self.calls += 1
        self.past += (hard - self.past) / self.calls
        bins = torch.multinomial(weights, n, replacement=True)
        clips = self.clips[bins]
        maxoffset = torch.minimum(
            torch.full((n,), self.bin_frames),
            self.library.lengths.cpu()[clips] - 1 - self.frames[bins],
        )
        frames = self.frames[bins] + (torch.rand(n) * maxoffset).long()
        return clips.to(self.library.states.device), frames.to(
            self.library.states.device
        )

    def update(self, trajectory):
        value = 0.0
        by_bin = {}
        for clip, frame, reward in reversed(trajectory):
            value = reward + self.gamma * value
            idx = self.bin_index[clip, frame // self.bin_frames * self.bin_frames]
            by_bin.setdefault(idx, []).append(value)
        for idx, values in by_bin.items():
            target = sum(values) / len(values)
            self.returns[idx] = (
                0.9 * self.returns[idx] + 0.1 * target if self.seen[idx] else target
            )
            self.seen[idx] = True

    def state_dict(self):
        return {k: getattr(self, k) for k in ("returns", "seen", "past", "calls")}

    def load_state_dict(self, data):
        for k, v in data.items():
            setattr(self, k, v)
