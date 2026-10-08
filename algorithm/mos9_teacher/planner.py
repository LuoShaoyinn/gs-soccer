"""Offline ZMP/LIPM pattern generation with URDF-based leg inverse kinematics.

No learned policy, motion clip, or privileged simulator pose control is used.
The resulting joint targets are executed by the ordinary PD actuators.
"""
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.linalg import solve_banded
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


@dataclass
class GaitConfig:
    step_length: float = 0.035
    step_time: float = 0.6
    swing_height: float = 0.012
    double_support: float = 0.6
    crouch: float = 0.045
    start_time: float = 1.0
    zmp_x_offset: float = -0.012
    zmp_y_scale: float = 0.95
    com_x_offset: float = 0.0
    stance_width: float = 0.12
    lean_gain: float = 1.7


class Kinematics:
    def __init__(self, path="assets/MOS9/MOS9_walk.urdf"):
        root = ET.parse(path).getroot()
        self.links = {link.get("name"): link for link in root.findall("link")}
        self.joints = list(root.findall("joint"))
        self.names = [j.get("name") for j in self.joints if j.get("type") == "revolute"]
        self.index = {name: i for i, name in enumerate(self.names)}
        self.origins = {}
        self.axes = {}
        self.bounds = {}
        for j in self.joints:
            origin = j.find("origin")
            transform = np.eye(4)
            transform[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
            transform[:3, :3] = Rotation.from_euler("xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")).as_matrix()
            self.origins[j.get("name")] = transform
            if j.get("type") == "revolute":
                self.axes[j.get("name")] = np.fromstring(j.find("axis").get("xyz"), sep=" ")
                limit = j.find("limit")
                self.bounds[j.get("name")] = (float(limit.get("lower")), float(limit.get("upper")))
        self.leg_indices = [[self.index[f"{side}_{name}"] for name in
                             ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]
                            for side in ("right", "left")]

    def forward(self, q):
        transforms = {"base_link": np.eye(4)}
        for j in self.joints:
            name = j.get("name")
            joint_rotation = np.eye(4)
            if name in self.index:
                joint_rotation[:3, :3] = Rotation.from_rotvec(self.axes[name] * q[self.index[name]]).as_matrix()
            transforms[j.find("child").get("link")] = transforms[j.find("parent").get("link")] @ self.origins[name] @ joint_rotation
        return transforms

    def com(self, q):
        transforms = self.forward(q)
        points, masses = [], []
        for name, link in self.links.items():
            inertial = link.find("inertial")
            mass = float(inertial.find("mass").get("value"))
            xyz = np.fromstring(inertial.find("origin").get("xyz"), sep=" ")
            points.append((transforms[name] @ np.r_[xyz, 1])[:3])
            masses.append(mass)
        return np.average(points, axis=0, weights=masses)

    def collision_bottom(self, q):
        transforms = self.forward(q)
        heights = []
        for name in self.links:
            for collision in self.links[name].findall("collision"):
                origin = collision.find("origin")
                xyz = np.fromstring(origin.get("xyz"), sep=" ")
                rotation = Rotation.from_euler("xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")).as_matrix()
                cylinder = collision.find("geometry/cylinder")
                world_rotation = transforms[name][:3, :3] @ rotation
                center = (transforms[name] @ np.r_[xyz, 1])[:3]
                box = collision.find("geometry/box")
                sphere = collision.find("geometry/sphere")
                if cylinder is not None:
                    extent = float(cylinder.get("radius")) * np.linalg.norm(world_rotation[2, :2]) + float(cylinder.get("length"))/2 * abs(world_rotation[2, 2])
                elif box is not None:
                    size = np.fromstring(box.get("size"), sep=" ")
                    extent = np.sum(abs(world_rotation[2]) * size/2)
                elif sphere is not None:
                    extent = float(sphere.get("radius"))
                else:
                    import trimesh
                    mesh = collision.find("geometry/mesh")
                    vertices = trimesh.load(Path("assets/MOS9") / mesh.get("filename"), force="mesh").vertices
                    scale = np.fromstring(mesh.get("scale", "1 1 1"), sep=" ")
                    heights.append(np.min((vertices*scale) @ world_rotation[2] + center[2]))
                    continue
                heights.append(center[2] - extent)
        return min(heights)

    def sole_height(self, transforms):
        heights = []
        for name in ("Rfoot", "Lfoot"):
            for collision in self.links[name].findall("collision"):
                xyz = np.fromstring(collision.find("origin").get("xyz"), sep=" ")
                cylinder = collision.find("geometry/cylinder")
                rotation = transforms[name][:3, :3]
                center = (transforms[name] @ np.r_[xyz, 1])[:3]
                extent = float(cylinder.get("radius"))*np.linalg.norm(rotation[2, :2]) + float(cylinder.get("length"))/2*abs(rotation[2, 2])
                heights.append(center[2]-extent)
        return min(heights)

    def ik(self, q, feet, rotations):
        q = q.copy()
        worst = 0.0
        for leg, name, target, rotation in zip(self.leg_indices, ("Rfoot", "Lfoot"), feet, rotations):
            def residual(values):
                candidate = q.copy()
                candidate[leg] = values
                transform = self.forward(candidate)[name]
                return np.r_[transform[:3, 3] - target,
                             0.12 * Rotation.from_matrix(rotation.T @ transform[:3, :3]).as_rotvec()]
            lower, upper = np.array([self.bounds[self.names[i]] for i in leg]).T
            result = least_squares(residual, np.clip(q[leg], lower + 1e-6, upper - 1e-6),
                                   bounds=(lower, upper), max_nfev=35, ftol=1e-8, xtol=1e-8, gtol=1e-8)
            q[leg] = result.x
            worst = max(worst, np.max(np.abs(result.fun)))
        return q, worst


def make_reference(cfg: GaitConfig, dt=0.02, duration=12.0, model_path="assets/MOS9/MOS9_walk.urdf"):
    kin = Kinematics(model_path)
    zero = np.zeros(len(kin.names))
    # MOS9's zero shoulder rolls form a T pose. Use the walking model's
    # arms-down posture before calculating COM and solving leg IK.
    zero[kin.index["right_shoulder_roll"]] = -1.4
    zero[kin.index["left_shoulder_roll"]] = 1.4
    transforms = kin.forward(zero)
    foot0 = np.array([transforms[name][:3, 3] for name in ("Rfoot", "Lfoot")])
    rotations = [transforms[name][:3, :3] for name in ("Rfoot", "Lfoot")]
    base_height = -kin.collision_bottom(zero) - cfg.crouch + 0.002
    feet = foot0.copy()
    feet[:, 2] += cfg.crouch
    feet[:, 1] = [-cfg.stance_width/2, cfg.stance_width/2]
    # Seed the physically valid forward-bending knee branches.
    zero[kin.index["right_knee"]] = 0.5
    zero[kin.index["left_knee"]] = -0.5
    home, error = kin.ik(zero, feet, rotations)
    com_offset = kin.com(home)
    n = round(duration / dt) + 1
    times = np.arange(n) * dt
    foot_path = np.repeat(feet[None], n, axis=0)
    zmp = np.zeros((n, 2))
    foot_center = feet[:, :2].copy()
    # Foot link origins are near the rear of the sole; account for sole COM.
    sole_x = np.mean(foot_center[:, 0]) + 0.015 + cfg.zmp_x_offset
    zmp[:, 0] = sole_x
    for k, t in enumerate(times):
        if t < cfg.start_time:
            continue
        step = int((t - cfg.start_time) / cfg.step_time)
        phase = ((t - cfg.start_time) / cfg.step_time) % 1
        swing = step % 2
        stance = 1 - swing
        # One foot advances per step; initial step is half-length.
        start_x = foot_center[swing, 0] + max(0, step - 1) * cfg.step_length
        end_x = foot_center[swing, 0] + (step + 1) * cfg.step_length
        if step == 0:
            start_x = foot_center[swing, 0]
        foot_path[k, stance, 0] = foot_center[stance, 0] + step * cfg.step_length
        if step == 0:
            foot_path[k, stance, 0] = foot_center[stance, 0]
        ds = cfg.double_support / 2
        u = np.clip((phase - ds) / (1 - cfg.double_support), 0, 1)
        blend = u**3 * (10 - 15*u + 6*u*u)
        foot_path[k, swing, 0] = start_x + blend * (end_x - start_x)
        foot_path[k, swing, 2] += cfg.swing_height * np.sin(np.pi*u)**2
        stance_zmp = np.array([foot_path[k, stance, 0] + 0.015 + cfg.zmp_x_offset,
                               foot_center[stance, 1] * cfg.zmp_y_scale])
        if phase < ds:
            previous_x = foot_center[swing, 0] + max(0, step - 1) * cfg.step_length + 0.015 + cfg.zmp_x_offset
            previous_y = foot_center[swing, 1] * cfg.zmp_y_scale if step else 0.0
            b = (phase / ds)**2 * (3 - 2*phase/ds)
            zmp[k] = (1-b) * np.array([previous_x, previous_y]) + b * stance_zmp
        else:
            zmp[k] = stance_zmp
    # Solve x - (z_com/g) x'' = ZMP as a stable, finite preview problem.
    alpha = (base_height + com_offset[2]) / 9.81 / dt**2
    band = np.zeros((3, n))
    band[0, 1:] = -alpha
    band[1] = 1 + 2*alpha
    band[2, :-1] = -alpha
    band[1, [0, -1]] = 1 + alpha  # zero endpoint velocity
    com = solve_banded((1, 1), band, zmp)
    pelvis = com - com_offset[:2]
    pelvis[:, 0] += cfg.com_x_offset
    return {"names": kin.names, "height": base_height, "seed": home,
            "pelvis": pelvis, "com": com + [cfg.com_x_offset, 0], "feet": foot_path,
            "rotations": rotations, "body_roll": -cfg.lean_gain*com[:, 1]}


def make_plan(cfg: GaitConfig, dt=0.02, duration=12.0, model_path="assets/MOS9/MOS9_walk.urdf"):
    reference = make_reference(cfg, dt, duration, model_path)
    kin = Kinematics(model_path)
    home = reference["seed"]
    com = reference["com"]
    pelvis = reference["pelvis"]
    rotations = reference["rotations"]
    foot_path = reference["feet"]
    base_height = reference["height"]
    n = len(com)
    q = home.copy()
    actions, errors, body_roll = [], [], []
    for k in range(n):
        desired_com = com[k]
        roll = reference["body_roll"][k]
        body_rotation = Rotation.from_euler("x", roll).as_matrix()
        target_rotations = [body_rotation.T @ r for r in rotations]
        for _ in range(4):
            targets = foot_path[k].copy()
            targets[:, :2] -= pelvis[k]
            targets = targets @ body_rotation
            q, error = kin.ik(q, targets, target_rotations)
            difference = desired_com - (pelvis[k] + (body_rotation @ kin.com(q))[:2])
            if np.max(np.abs(difference)) < 0.0002:
                break
            pelvis[k] += difference
        actions.append(q.copy())
        errors.append(error)
        body_roll.append(roll)
    return {"names": kin.names, "actions": np.array(actions, dtype=np.float32),
            "pelvis": pelvis.astype(np.float32), "height": base_height,
            "home": np.array(actions[0], dtype=np.float32), "ik_max_error": max(errors),
            "body_roll": np.array(body_roll, dtype=np.float32),
            "feet": foot_path.astype(np.float32)}
