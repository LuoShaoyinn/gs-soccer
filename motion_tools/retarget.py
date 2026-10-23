"""Contact-aware G1 -> MOS9 walking retargeting using the exact target URDF.

This solves a bounded kinematic trajectory fit, not inverse dynamics or RL.
No source joint angles are copied into the target robot.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import CubicSpline
from scipy.ndimage import binary_closing, binary_opening
from scipy.spatial.transform import Rotation, Slerp

from .quality import RobotModel, inspect, rotate

ROOT = Path(__file__).resolve().parents[1]


def sole_offset(model, link):
    poses = model.fk(
        np.zeros((1, len(model.names))), np.zeros((1, 3)), np.array([[1.0, 0, 0, 0]])
    )
    points, _ = model.collision_points(poses, {link})
    points = points[0]
    # In the flat-foot degeneracy, average the equally low primitive supports.
    point = points[points[:, 2] < points[:, 2].min() + 1e-6].mean(axis=0)
    p, r = poses[link]
    return r[0].T @ (point - p[0]), r[0], point


class TorchModel:
    def __init__(self, model, device="cpu"):
        self.model, self.device = model, device
        self.tensor = lambda x: torch.as_tensor(x, dtype=torch.float64, device=device)
        self.joints = []
        for parent, child, name, offset, r0, axis in model.joints:
            x, y, z = axis
            k = self.tensor([[0, -z, y], [z, 0, -x], [-y, x, 0]])
            self.joints.append(
                (
                    parent,
                    child,
                    model.names.index(name) if name else None,
                    self.tensor(offset),
                    self.tensor(r0),
                    k,
                    k @ k,
                )
            )

    def fk(self, q, p, r):
        poses = {self.model.root: (p, r)}
        for parent, child, idx, offset, r0, k, kk in self.joints:
            pp, pr = poses[parent]
            cr = pr @ r0
            if idx is not None:
                angle = q[:, idx, None, None]
                cr = cr @ (
                    torch.eye(3, device=q.device)
                    + angle.sin() * k
                    + (1 - angle.cos()) * kk
                )
            poses[child] = pp + (pr @ offset[..., None]).squeeze(-1), cr
        return poses

    def heights(self, poses, link):
        heights = []
        for name, offset, r0, kind, dimensions in self.model.collisions:
            if name != link:
                continue
            p, r = poses[name]
            center = p + (r @ self.tensor(offset)[:, None]).squeeze(-1)
            rotation = r @ self.tensor(r0)
            if kind == "cylinder":
                c = rotation[:, 2, 2]
                support = (
                    c.abs() * dimensions[1] / 2
                    + dimensions[0] * (1 - c * c + 1e-12).sqrt()
                )
            elif kind == "sphere":
                support = dimensions[0]
            elif kind == "box":
                support = (rotation[:, 2].abs() * self.tensor(dimensions) / 2).sum(-1)
            else:
                raise ValueError(kind)
            heights.append(center[:, 2] - support)
        return torch.stack(heights, -1).min(-1).values


def segments(mask):
    boundaries = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(boundaries == 1), np.flatnonzero(boundaries == -1)))


def detect_contacts(sole_positions, lowest_heights, dt):
    speed = np.linalg.norm(np.gradient(sole_positions, dt, axis=0), axis=-1)
    # Small morphology-independent stance threshold in source coordinates.
    mask = (lowest_heights < 0.025) & (speed < 0.3)
    for side in range(2):
        mask[:, side] = binary_closing(mask[:, side], structure=np.ones(3))
        mask[:, side] = binary_opening(mask[:, side], structure=np.ones(4))
    # Real walking should have a support foot; do not fabricate one for a flight phase.
    return mask, speed


def prepare_source(csv, source, target, seconds, start_seconds=None):
    data = np.loadtxt(csv, delimiter=",")
    if data.ndim != 2 or data.shape[1] != 36 or not np.isfinite(data).all():
        raise ValueError("expected finite Unitree G1 CSV: xyz + xyzw + 29 joints")
    src_times = np.arange(len(data)) / 30
    r = Rotation.from_quat(data[:, 3:7])
    pose = source.fk(data[:, 7:], data[:, :3], data[:, 3:7][:, [3, 0, 1, 2]])
    source_soles, source_heights, neutral_source = [], [], []
    target_offsets, neutral_target, neutral_rot = [], [], []
    for side, mos in (("right", "R"), ("left", "L")):
        link = side + "_ankle_roll_link"
        offset, _, neutral = sole_offset(source, link)
        pp, rr = pose[link]
        source_soles.append(pp + rotate(rr, offset))
        lowest, _ = source.collision_points(pose, {link})
        source_heights.append(lowest[..., 2].min(-1))
        neutral_source.append(neutral)
        off, rot, neu = sole_offset(target, mos + "foot")
        target_offsets.append(off)
        neutral_target.append(neu)
        neutral_rot.append(rot)
    source_soles = np.stack(source_soles, 1)
    source_heights = np.stack(source_heights, 1)
    scale = -np.mean(np.array(neutral_target)[:, 2]) / -np.mean(
        np.array(neutral_source)[:, 2]
    )
    if start_seconds is None:
        # Select before retargeting: score source-only windows for upright continuous walking.
        velocity = (
            np.linalg.norm(np.gradient(data[:, :2], 1 / 30, axis=0), axis=-1) * scale
        )
        root_up = r.as_matrix()[:, 2, 2]
        contacts, _ = detect_contacts(source_soles, source_heights, 1 / 30)
        valid = (
            (velocity > 0.12) & (velocity < 0.8) & (root_up > 0.9) & contacts.any(-1)
        )
        width = round(seconds * 30) + 1
        scores = np.convolve(valid.astype(float), np.ones(width) / width, mode="valid")
        start = int(np.argmax(scores))
    else:
        start = round(start_seconds * 30)
    stop = start + round(seconds * 30)
    if stop >= len(data):
        raise ValueError("requested segment exceeds source clip")
    times = src_times[start] + np.arange(round(seconds * 50) + 1) / 50
    q = CubicSpline(src_times, data[:, 7:])(times)
    root_pos = CubicSpline(src_times, data[:, :3])(times)
    root_rot = Slerp(src_times, r)(times)
    poses = source.fk(q, root_pos, root_rot.as_quat()[:, [3, 0, 1, 2]])
    soles, heights, foot_rotations = [], [], []
    for side in ("right", "left"):
        link = side + "_ankle_roll_link"
        off, _, _ = sole_offset(source, link)
        p, rr = poses[link]
        soles.append(p + rotate(rr, off))
        pts, _ = source.collision_points(poses, {link})
        heights.append(pts[..., 2].min(-1))
        foot_rotations.append(rr)
    soles, heights = np.stack(soles, 1), np.stack(heights, 1)
    contacts, speeds = detect_contacts(soles, heights, 0.02)
    # Express the entire segment with its initial heading and xy removed.
    initial_yaw = root_rot.as_euler("xyz")[0, 2]
    change = Rotation.from_euler("z", -initial_yaw).as_matrix()
    origin = root_pos[0].copy()
    origin[2] = 0
    root_pos = rotate(change, root_pos - origin) * scale
    root_matrix = change @ root_rot.as_matrix()
    desired_feet, desired_rotations = [], []
    for side in range(2):
        # Match target hip/stance width separately from leg-length scaling.
        offset = np.array(neutral_target)[side] - scale * np.array(neutral_source)[side]
        heading = Rotation.from_euler(
            "z", Rotation.from_matrix(root_matrix).as_euler("xyz")[:, 2:3]
        ).as_matrix()
        feet = rotate(change, soles[:, side] - origin) * scale + rotate(heading, offset)
        foot_rotation = change @ foot_rotations[side]
        yaw = Rotation.from_matrix(foot_rotation).as_euler("xyz")[:, 2]
        for a, b in segments(contacts[:, side]):
            feet[a:b, :2] = np.median(feet[a:b, :2], axis=0)
            feet[a:b, 2] = 0.001
            yaw[a:b] = np.median(np.unwrap(yaw[a:b]))
        for a, b in segments(~contacts[:, side]):
            if b - a < 4:
                continue
            # Transfer steps, not another robot's millimeter-scale foot scuffing.
            phase = np.linspace(0, 1, b - a)
            blend = 3 * phase**2 - 2 * phase**3
            begin = feet[max(a - 1, 0)].copy()
            end = feet[min(b, len(feet) - 1)].copy()
            feet[a:b, :2] = begin[:2] + blend[:, None] * (end[:2] - begin[:2])
            feet[a:b, 2] = np.maximum(
                feet[a:b, 2], 0.05 * np.sin(np.pi * phase) ** 2 + 0.001
            )
        desired_feet.append(feet)
        # Flat-foot walking targets, retaining source heading.
        flat = Rotation.from_euler("z", yaw[:, None]).as_matrix()
        foot_rotation = flat
        desired_rotations.append(foot_rotation @ neutral_rot[side])
    return {
        "root_pos": root_pos,
        "root_rotation": root_matrix,
        "feet": np.stack(desired_feet, 1),
        "foot_rotation": np.stack(desired_rotations, 1),
        "contacts": contacts,
        "source_contact_speed": speeds,
        "sole_offsets": target_offsets,
        "scale": scale,
        "source_start_frame": start,
        "source_stop_frame": stop,
        "unsupported_fraction": float((~contacts.any(-1)).mean()),
    }


def fit(target, reference, iterations=1000, device="cpu"):
    torch.set_num_threads(1)
    model = TorchModel(target, device)
    tensor = model.tensor
    nframes = len(reference["root_pos"])
    low, high = tensor(target.limits.T)
    mid, half = (high + low) / 2, (high - low) / 2
    home = np.zeros(len(target.names))
    home[target.names.index("right_shoulder_roll")] = -1.4
    home[target.names.index("left_shoulder_roll")] = 1.4
    for name, value in (
        ("right_knee", 0.4),
        ("left_knee", -0.4),
        ("right_hip_pitch", -0.2),
        ("left_hip_pitch", 0.2),
    ):
        home[target.names.index(name)] = value
    initial = tensor(np.repeat(home[None], nframes, axis=0))
    latent = torch.atanh(((initial - mid) / half).clamp(-0.99, 0.99)).requires_grad_()
    root_prior = tensor(reference["root_pos"])
    position = root_prior.clone().requires_grad_()
    rotations = tensor(reference["root_rotation"])
    feet = tensor(reference["feet"])
    foot_rotations = tensor(reference["foot_rotation"])
    contact = tensor(reference["contacts"])
    offsets = tensor(np.array(reference["sole_offsets"]))
    optimizer = torch.optim.LBFGS(
        [latent, position],
        lr=0.7,
        max_iter=iterations,
        tolerance_grad=1e-8,
        tolerance_change=1e-11,
        line_search_fn="strong_wolfe",
    )
    calls, last = 0, {}
    speed_limits = tensor(
        [
            9.32 if ("shoulder" in n or "ankle_roll" in n) else 15.60
            for n in target.names
        ]
    )

    def objective():
        nonlocal calls, last
        optimizer.zero_grad()
        q = mid + half * latent.tanh()
        poses = model.fk(q, position, rotations)
        sole, rr, heights = [], [], []
        for i, link in enumerate(("Rfoot", "Lfoot")):
            p, r = poses[link]
            sole.append(p + (r @ offsets[i, :, None]).squeeze(-1))
            rr.append(r)
            heights.append(model.heights(poses, link))
        sole, rr = torch.stack(sole, 1), torch.stack(rr, 1)
        h = torch.stack(heights, 1)
        # Weight stance constraints strongly; the root can adjust within its prior.
        weights = 1 + 9 * contact
        losses = {
            "foot_position": 200 * ((sole - feet).square().sum(-1) * weights).mean(),
            "foot_rotation": 2
            * ((rr - foot_rotations).square().sum((-1, -2)) * (1 + contact)).mean(),
            "root_prior": 100 * (position - root_prior).square().mean(),
            "root_velocity": 5
            * ((position[1:] - position[:-1] - root_prior[1:] + root_prior[:-1]) / 0.02)
            .square()
            .mean(),
            "joint_prior": 0.0002 * (q - tensor(home)).square().mean(),
            "joint_acceleration": 100 * (q[2:] - 2 * q[1:-1] + q[:-2]).square().mean(),
            "joint_velocity": 0.0005 * ((q[1:] - q[:-1]) / 0.02).square().mean(),
            "motor_speed_limit": ((q[1:] - q[:-1]).abs() / 0.02 - speed_limits)
            .relu()
            .square()
            .mean(),
            "root_acceleration": 1000
            * (position[2:] - 2 * position[1:-1] + position[:-2]).square().mean(),
            "floor": 5000 * (0.001 - h).relu().square().mean(),
            "stance_height": 2000 * ((h - 0.001).square() * contact).mean(),
        }
        # Contact transition speed is directly constrained, not repaired by framewise root shifts.
        if nframes > 1:
            consecutive = contact[1:] * contact[:-1]
            losses["stance_velocity"] = (
                5
                * ((sole[1:] - sole[:-1]).square().sum(-1) * consecutive).mean()
                / 0.02**2
            )
        loss = sum(losses.values())
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite IK objective")
        loss.backward()
        calls += 1
        last = {k: float(v.detach()) for k, v in losses.items()}
        if calls % 50 == 0:
            print(
                json.dumps(
                    {"objective_calls": calls, "loss": float(loss.detach()), **last}
                ),
                flush=True,
            )
        return loss

    optimizer.step(objective)
    q = (mid + half * latent.tanh()).detach().cpu().numpy()
    p = position.detach().cpu().numpy()
    poses = target.fk(
        q,
        p,
        Rotation.from_matrix(reference["root_rotation"]).as_quat()[:, [3, 0, 1, 2]],
    )
    errors = []
    for i, link in enumerate(("Rfoot", "Lfoot")):
        pp, rr = poses[link]
        errors.append(
            np.linalg.norm(
                pp
                + rotate(rr, np.array(reference["sole_offsets"])[i])
                - reference["feet"][:, i],
                axis=-1,
            )
        )
    return (
        q,
        p,
        {
            "objective_calls": calls,
            "losses": last,
            "foot_position_error_p95_m": float(np.percentile(errors, 95)),
        },
    )


def save(path, model, q, p, r, metadata):
    poses = model.fk(q, p, r)
    bp = np.stack([poses[n][0] for n in model.links], 1)
    br = np.stack(
        [
            Rotation.from_matrix(poses[n][1]).as_quat()[:, [3, 0, 1, 2]]
            for n in model.links
        ],
        1,
    )
    angular = []
    for n in model.links:
        matrices = poses[n][1]
        w = (
            Rotation.from_matrix(
                matrices[1:] @ np.swapaxes(matrices[:-1], -1, -2)
            ).as_rotvec()
            / 0.02
        )
        angular.append(np.concatenate((w[:1], (w[1:] + w[:-1]) / 2, w[-1:])))
    angular = np.stack(angular, 1)
    root_idx = model.links.index(model.root)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        robot="MOS9",
        fps=50.0,
        joint_names=np.array(model.names),
        joint_pos=q.astype(np.float32),
        joint_vel=np.gradient(q, 0.02, axis=0).astype(np.float32),
        root_pos=p.astype(np.float32),
        root_quat=r.astype(np.float32),
        root_vel=np.gradient(p, 0.02, axis=0).astype(np.float32),
        root_ang_vel=angular[:, root_idx].astype(np.float32),
        body_names=np.array(model.links),
        body_pos_w=bp.astype(np.float32),
        body_quat_w=br.astype(np.float32),
        body_lin_vel_w=np.gradient(bp, 0.02, axis=0).astype(np.float32),
        body_ang_vel_w=angular.astype(np.float32),
        provenance=json.dumps(metadata),
        quality_status="kinematic_candidate_not_dynamic_validation",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--start", type=float)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.seconds < 1 or args.iterations < 1:
        parser.error("seconds must be at least one, and iterations positive")
    source = RobotModel(ROOT / "assets/g1/g1_29dof.urdf", skip_meshes=True)
    target = RobotModel(ROOT / "assets/MOS9/MOS9_walk.urdf")
    reference = prepare_source(args.input, source, target, args.seconds, args.start)
    start = time.monotonic()
    q, p, fit_metrics = fit(target, reference, args.iterations, args.device)
    r = Rotation.from_matrix(reference["root_rotation"]).as_quat()[:, [3, 0, 1, 2]]
    metadata = {
        "retarget_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "source_file": str(args.input),
        "source_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "source_urdf_sha256": hashlib.sha256(source.path.read_bytes()).hexdigest(),
        "target_urdf_sha256": hashlib.sha256(target.path.read_bytes()).hexdigest(),
        **{
            k: reference[k]
            for k in (
                "scale",
                "source_start_frame",
                "source_stop_frame",
                "unsupported_fraction",
            )
        },
        **fit_metrics,
        "seconds_to_fit": time.monotonic() - start,
        "dynamic_validation": False,
    }
    save(args.output, target, q, p, r, metadata)
    quality, _ = inspect(args.output, target)
    report = {"provenance": metadata, "quality": quality}
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    diagnostics = args.output.parent / "diagnostics"
    diagnostics.mkdir(exist_ok=True)
    np.savez_compressed(
        diagnostics / (args.output.stem + ".targets.npz"),
        feet=reference["feet"],
        contacts=reference["contacts"],
        root_prior=reference["root_pos"],
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
