"""Differentiable wxyz quaternion operations and authored-frame URDF kinematics."""

import math
import xml.etree.ElementTree as ET

import torch
from torch import nn


def normalize(q):
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return torch.where(q[..., :1] < 0, -q, q)


def conjugate(q):
    return torch.cat((q[..., :1], -q[..., 1:]), -1)


def multiply(a, b):
    a, b = torch.broadcast_tensors(a, b)
    w = a[..., :1] * b[..., :1] - (a[..., 1:] * b[..., 1:]).sum(-1, keepdim=True)
    v = (
        a[..., :1] * b[..., 1:]
        + b[..., :1] * a[..., 1:]
        + torch.cross(a[..., 1:], b[..., 1:], dim=-1)
    )
    return torch.cat((w, v), -1)


def rotate(q, v):
    qv, v = torch.broadcast_tensors(q[..., 1:], v)
    t = 2 * torch.cross(qv, v, dim=-1)
    return v + q[..., :1] * t + torch.cross(qv, t, dim=-1)


def rotation_vector(q):
    q = normalize(q)
    v = q[..., 1:]
    length = v.norm(dim=-1, keepdim=True)
    return v * (
        2 * torch.atan2(length, q[..., :1].clamp_min(1e-8)) / length.clamp_min(1e-8)
    )


def from_rotation_vector(v):
    angle = v.norm(dim=-1, keepdim=True)
    return torch.cat(
        (torch.cos(angle / 2), v * (0.5 * torch.sinc(angle / (2 * math.pi)))), -1
    )


def rotation_error_squared(a, b):
    # q and -q are equivalent. Finite derivatives also at a == b.
    return rotation_vector(multiply(a, conjugate(b))).square().sum(-1)


def gravity(q):
    v = torch.zeros_like(q[..., 1:])
    v[..., 2] = -1
    return rotate(conjugate(q), v)


def rotation_6d(q):
    x, y = torch.zeros_like(q[..., 1:]), torch.zeros_like(q[..., 1:])
    x[..., 0], y[..., 1] = 1, 1
    return torch.cat((rotate(q, x), rotate(q, y)), -1)


class Kinematics(nn.Module):
    def __init__(self, path, joint_names, tracked_links):
        super().__init__()
        self.joint_names, self.tracked_links = tuple(joint_names), tuple(tracked_links)
        root = ET.parse(path).getroot()
        pending = list(root.findall("joint"))
        children = {j.find("child").attrib["link"] for j in pending}
        roots = [
            l.attrib["name"]
            for l in root.findall("link")
            if l.attrib["name"] not in children
        ]
        if roots != ["base_link"]:
            raise ValueError(f"expected base_link root, got {roots}")
        self.order, known = [], {"base_link"}
        lowers, uppers = {}, {}
        while pending:
            ready = [j for j in pending if j.find("parent").attrib["link"] in known]
            if not ready:
                raise ValueError("URDF is not a connected tree")
            for j in ready:
                name, kind = j.attrib["name"], j.attrib["type"]
                if kind not in ("fixed", "revolute"):
                    raise ValueError(f"unsupported joint type {kind}")
                idx = None if kind == "fixed" else self.joint_names.index(name)
                origin = j.find("origin")
                xyz = [float(x) for x in origin.attrib.get("xyz", "0 0 0").split()]
                rpy = [float(x) for x in origin.attrib.get("rpy", "0 0 0").split()]
                quat = torch.tensor([1.0, 0, 0, 0])
                for axis, angle in zip(
                    ((0, 0, 1), (0, 1, 0), (1, 0, 0)), reversed(rpy)
                ):
                    quat = multiply(
                        quat, from_rotation_vector(torch.tensor(axis) * angle)
                    )
                axis = (
                    torch.zeros(3)
                    if idx is None
                    else torch.tensor(
                        [float(x) for x in j.find("axis").attrib["xyz"].split()]
                    )
                )
                axis = axis / axis.norm().clamp_min(1e-8)
                i = len(self.order)
                self.register_buffer(f"xyz_{i}", torch.tensor(xyz))
                self.register_buffer(f"quat_{i}", quat)
                self.register_buffer(f"axis_{i}", axis)
                parent, child = (
                    j.find("parent").attrib["link"],
                    j.find("child").attrib["link"],
                )
                self.order.append((parent, child, idx))
                known.add(child)
                pending.remove(j)
                if idx is not None:
                    lowers[name] = float(j.find("limit").attrib["lower"])
                    uppers[name] = float(j.find("limit").attrib["upper"])
        self.register_buffer("lower", torch.tensor([lowers[n] for n in joint_names]))
        self.register_buffer("upper", torch.tensor([uppers[n] for n in joint_names]))
        if not set(tracked_links) <= known:
            raise ValueError("unknown tracked links")

    def forward(self, q, dq, pos, quat, vel, ang):
        poses = {"base_link": (pos, quat, vel, ang)}
        for i, (parent, child, idx) in enumerate(self.order):
            p, r, v, w = poses[parent]
            offset = rotate(r, getattr(self, f"xyz_{i}"))
            r0 = multiply(r, getattr(self, f"quat_{i}"))
            wc = w
            rc = r0
            if idx is not None:
                axis = getattr(self, f"axis_{i}")
                rc = multiply(r0, from_rotation_vector(q[..., idx : idx + 1] * axis))
                wc = w + rotate(r0, axis) * dq[..., idx : idx + 1]
            poses[child] = (p + offset, rc, v + torch.cross(w, offset, dim=-1), wc)
        return tuple(
            torch.stack([poses[n][i] for n in self.tracked_links], -2) for i in range(4)
        )
