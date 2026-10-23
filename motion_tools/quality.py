"""Audit named FK motion files against exact authored URDF frames/collision primitives.

Contact gates are necessary kinematic checks for flat-ground walking. They do
not certify dynamic feasibility and do not use foot-link origins as sole heights.
"""

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def xyz(element, attribute, default="0 0 0"):
    return np.array([float(v) for v in element.get(attribute, default).split()])


def origin(element):
    node = element.find("origin")
    if node is None:
        return np.zeros(3), np.eye(3)
    return xyz(node, "xyz"), Rotation.from_euler("xyz", xyz(node, "rpy")).as_matrix()


def rotate(matrix, vector):
    return np.einsum("...ij,...j->...i", matrix, vector)


def lowest_point(position, rotation, kind, dimensions):
    """World point achieving the primitive's minimum z (exact support function)."""
    down = np.zeros_like(position)
    down[..., 2] = -1
    local_down = rotate(np.swapaxes(rotation, -1, -2), down)
    if kind == "sphere":
        local = local_down * dimensions[0]
    elif kind == "box":
        local = np.sign(local_down) * np.array(dimensions) / 2
    elif kind == "cylinder":
        local = np.zeros_like(local_down)
        radial_norm = np.linalg.norm(local_down[..., :2], axis=-1, keepdims=True)
        local[..., :2] = (
            dimensions[0] * local_down[..., :2] / np.maximum(radial_norm, 1e-12)
        )
        axial = local_down[..., 2]
        local[..., 2] = (
            np.where(np.abs(axial) < 1e-10, 0, np.sign(axial)) * dimensions[1] / 2
        )
    else:
        raise ValueError(f"unsupported collision primitive {kind}")
    return position + rotate(rotation, local)


class RobotModel:
    def __init__(self, urdf, skip_meshes=False):
        self.path = Path(urdf)
        root = ET.parse(urdf).getroot()
        self.links = [n.get("name") for n in root.findall("link")]
        self.names = [
            j.get("name") for j in root.findall("joint") if j.get("type") == "revolute"
        ]
        pending = list(root.findall("joint"))
        children = {j.find("child").get("link") for j in pending}
        roots = set(self.links) - children
        if len(roots) != 1:
            raise ValueError(f"expected one URDF root, got {roots}")
        self.root = roots.pop()
        self.joints, known, limits = [], {self.root}, {}
        while pending:
            ready = [j for j in pending if j.find("parent").get("link") in known]
            if not ready:
                raise ValueError("disconnected URDF")
            for j in ready:
                parent, child = (
                    j.find("parent").get("link"),
                    j.find("child").get("link"),
                )
                name = j.get("name")
                if j.get("type") not in ("fixed", "revolute"):
                    raise ValueError(f"unsupported joint type {j.get('type')}")
                axis = (
                    np.zeros(3)
                    if j.get("type") == "fixed"
                    else xyz(j.find("axis"), "xyz")
                )
                axis /= max(np.linalg.norm(axis), 1e-12)
                self.joints.append(
                    (
                        parent,
                        child,
                        name if name in self.names else None,
                        *origin(j),
                        axis,
                    )
                )
                if name in self.names:
                    limit = j.find("limit")
                    limits[name] = float(limit.get("lower")), float(limit.get("upper"))
                known.add(child)
                pending.remove(j)
        self.limits = np.array([limits[n] for n in self.names])
        self.collisions, self.inertias = [], []
        for link in root.findall("link"):
            name = link.get("name")
            for c in link.findall("collision"):
                g = next(iter(c.find("geometry")))
                if g.tag == "cylinder":
                    dimensions = [float(g.get("radius")), float(g.get("length"))]
                elif g.tag == "sphere":
                    dimensions = [float(g.get("radius"))]
                elif g.tag == "box":
                    dimensions = xyz(g, "size")
                else:
                    if skip_meshes and g.tag == "mesh":
                        continue
                    raise ValueError(
                        f"collision audit requires primitive geometry, found {g.tag}"
                    )
                self.collisions.append((name, *origin(c), g.tag, dimensions))
            inertial = link.find("inertial")
            if inertial is not None:
                self.inertias.append(
                    (
                        name,
                        float(inertial.find("mass").get("value")),
                        origin(inertial)[0],
                    )
                )

    def fk(self, q, position, quaternion):
        """Quaternions use wxyz; returns poses of authored link origins."""
        q = np.asarray(q)
        rot = Rotation.from_quat(np.asarray(quaternion)[..., [1, 2, 3, 0]]).as_matrix()
        poses = {self.root: (np.asarray(position), rot)}
        for parent, child, name, offset, r0, axis in self.joints:
            p, r = poses[parent]
            rc = r @ r0
            if name is not None:
                rc = (
                    rc
                    @ Rotation.from_rotvec(
                        q[..., self.names.index(name), None] * axis
                    ).as_matrix()
                )
            poses[child] = p + rotate(r, offset), rc
        return poses

    def collision_points(self, poses, links=None):
        points, names = [], []
        for name, offset, r0, kind, dimensions in self.collisions:
            if links is not None and name not in links:
                continue
            p, r = poses[name]
            points.append(lowest_point(p + rotate(r, offset), r @ r0, kind, dimensions))
            names.append(name)
        return np.stack(points, axis=-2), names


def percentiles(x):
    return dict(
        zip(
            ("min", "p05", "median", "p95", "max"),
            np.percentile(x, [0, 5, 50, 95, 100]).tolist(),
        )
    )


def longest(mask, dt):
    best = current = 0
    for value in mask:
        current = current + 1 if value else 0
        best = max(best, current)
    return best * dt


def inspect(path, model):
    with np.load(path, allow_pickle=False) as d:
        names = d["joint_names"].tolist()
        bodies = d["body_names"].tolist()
        q = d["joint_pos"][:, [names.index(n) for n in model.names]].astype(np.float64)
        dt = 1 / float(d["fps"].item())
        p = d["body_pos_w"][:, bodies.index(model.root)].astype(np.float64)
        quat = d["body_quat_w"][:, bodies.index(model.root)].astype(np.float64)
        if not all(np.isfinite(x).all() for x in (q, p, quat)):
            raise ValueError("nonfinite input")
        poses = model.fk(q, p, quat)
        position_error, rotation_error = [], []
        for name in model.links:
            if name not in bodies:
                continue
            i = bodies.index(name)
            position_error.append(
                np.linalg.norm(poses[name][0] - d["body_pos_w"][:, i], axis=-1)
            )
            stored = Rotation.from_quat(d["body_quat_w"][:, i][:, [1, 2, 3, 0]])
            predicted = Rotation.from_matrix(poses[name][1])
            rotation_error.append((predicted.inv() * stored).magnitude())
        points, point_names = model.collision_points(poses, {"Rfoot", "Lfoot"})
        feet = []
        speeds = []
        for name in ("Rfoot", "Lfoot"):
            idx = np.array([i for i, n in enumerate(point_names) if n == name])
            local_points = points[:, idx]
            lowest = local_points[
                np.arange(len(q)), np.argmin(local_points[..., 2], axis=1)
            ]
            feet.append(lowest[:, 2])
            lp, lr = poses[name]
            velocity = np.gradient(lp, dt, axis=0)
            delta = (
                Rotation.from_matrix(lr[1:] @ np.swapaxes(lr[:-1], -1, -2)).as_rotvec()
                / dt
            )
            angular = np.concatenate(
                (delta[:1], (delta[:-1] + delta[1:]) / 2, delta[-1:])
            )
            contact_velocity = velocity + np.cross(angular, lowest - lp)
            speeds.append(np.linalg.norm(contact_velocity[:, :2], axis=-1))
        heights = np.stack(feet, axis=-1)
        speeds = np.stack(speeds, axis=-1)
        minimum = heights.min(axis=1)
        offset = -float(np.median(minimum))
        calibrated = heights + offset
        contact = np.abs(calibrated) < 0.015
        support_contact = contact & (
            heights <= heights.min(axis=1, keepdims=True) + 0.003
        )
        slip = speeds[contact]
        support_slip = speeds[support_contact]
        finite_qv = np.gradient(q, dt, axis=0)
        step_velocity = np.abs(np.diff(q, axis=0)) / dt
        motor_limits = np.array(
            [
                9.32 if ("shoulder" in n or "ankle_roll" in n) else 15.60
                for n in model.names
            ]
        )
        speed_violation = np.maximum(step_velocity - motor_limits, 0)
        logged_qv = d["joint_vel"][:, [names.index(n) for n in model.names]]
        root_v = np.gradient(p, dt, axis=0)
        lower, upper = model.limits.T
        violation = np.maximum(lower - q, 0) + np.maximum(q - upper, 0)
        nonfeet_points, _ = model.collision_points(
            poses, set(model.links) - {"Rfoot", "Lfoot"}
        )
        payload = b"".join(np.ascontiguousarray(x).tobytes() for x in (q, p, quat))
        metrics = {
            "file": str(path),
            "frames": len(q),
            "seconds": (len(q) - 1) * dt,
            "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            "trajectory_sha256": hashlib.sha256(payload).hexdigest(),
            "root_displacement_xy_m": (p[-1, :2] - p[0, :2]).tolist(),
            "root_speed_m_s": percentiles(np.linalg.norm(root_v[:, :2], axis=-1)),
            "joint_limit_violation_max_rad": float(violation.max()),
            "joint_limit_violation_frame_fraction": float(
                (violation.max(axis=1) > 0.05).mean()
            ),
            "joint_step_max_rad": float(np.abs(np.diff(q, axis=0)).max()),
            "motor_speed_violation_max_rad_s": float(speed_violation.max()),
            "motor_speed_violation_frame_fraction": float(
                (speed_violation.max(axis=1) > 0.05).mean()
            ),
            "joint_speed_abs_rad_s": percentiles(np.abs(finite_qv)),
            "logged_joint_velocity_rmse_rad_s": float(
                np.sqrt(np.mean((logged_qv - finite_qv) ** 2))
            ),
            "root_quaternion_norm_error_max": float(
                np.abs(np.linalg.norm(quat, axis=-1) - 1).max()
            ),
            "stored_fk_position_error_m": percentiles(np.array(position_error)),
            "stored_fk_orientation_error_rad": percentiles(np.array(rotation_error)),
            "foot_lowest_height_m": percentiles(minimum),
            "raw_penetration_over_2cm_frame_fraction": float((minimum < -0.02).mean()),
            "raw_both_feet_over_2cm_frame_fraction": float((minimum > 0.02).mean()),
            "best_constant_ground_offset_m": offset,
            "calibrated_penetration_over_2cm_frame_fraction": float(
                (minimum + offset < -0.02).mean()
            ),
            "calibrated_both_feet_over_2cm_frame_fraction": float(
                (minimum + offset > 0.02).mean()
            ),
            "calibrated_airborne_longest_seconds": longest(minimum + offset > 0.02, dt),
            "contact_material_speed_xy_m_s": percentiles(slip) if len(slip) else None,
            "contact_speed_over_02m_s_fraction": float((slip > 0.2).mean())
            if len(slip)
            else None,
            "lowest_support_material_speed_xy_m_s": percentiles(support_slip)
            if len(support_slip)
            else None,
            "lowest_support_speed_over_02m_s_fraction": float(
                (support_slip > 0.2).mean()
            )
            if len(support_slip)
            else None,
            "nonfoot_minimum_height_m": float(nonfeet_points[..., 2].min()),
        }
        return metrics, {
            "q": q,
            "root_pos": p,
            "root_quat": quat,
            "foot_heights": heights,
            "foot_contact_speed": speeds,
        }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--urdf", type=Path, default=Path("assets/MOS9/MOS9_walk.urdf"))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    model = RobotModel(args.urdf)
    rows = []
    for path in sorted(args.input.rglob("*.npz")):
        if path.name.endswith(".targets.npz"):
            continue
        if "mos9_fk_" not in str(path):
            continue
        row, _ = inspect(path, model)
        rows.append(row)
    duplicates = {}
    for row in rows:
        duplicates.setdefault(row["trajectory_sha256"], []).append(row["file"])
    result = {
        "urdf": str(args.urdf),
        "urdf_sha256": hashlib.sha256(args.urdf.read_bytes()).hexdigest(),
        "clips": rows,
        "duplicate_groups": [g for g in duplicates.values() if len(g) > 1],
        "gates": {
            "ground_tolerance_m": 0.02,
            "contact_height_tolerance_m": 0.015,
            "contact_speed_flag_m_s": 0.2,
            "joint_limit_tolerance_rad": 0.05,
        },
        "limitations": "Kinematic screening, not dynamic feasibility; static ground offset is reported but never applied to source files.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "clips": len(rows),
                "duplicate_groups": len(result["duplicate_groups"]),
                "report": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
