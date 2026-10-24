"""Bounded uniform-noise terrain and differentiable queries of its collision triangles."""

import hashlib
import xml.etree.ElementTree as ET

import numpy as np
import torch
from torch import nn

from .config import URDF
from .geometry import from_rotation_vector, multiply, rotate


class UniformTerrain(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.size, self.cell = cfg.terrain_size, cfg.terrain_cell
        # Uniform coarse knots, linear interpolation: bounded without clipping or
        # cubic overshoot. The exported fine grid is triangulated by Genesis.
        coarse_n = round(self.size / cfg.terrain_knot) + 1
        fine_n = round(self.size / self.cell) + 1
        if not np.isclose((fine_n - 1) * self.cell, self.size):
            raise ValueError("terrain_size must be a multiple of terrain_cell")
        rng = np.random.default_rng(cfg.seed + 1729)
        coarse = rng.uniform(
            -cfg.terrain_height, cfg.terrain_height, (coarse_n, coarse_n)
        )
        x = np.linspace(0, self.size, coarse_n)
        fine = np.linspace(0, self.size, fine_n)
        rows = np.stack([np.interp(fine, x, row) for row in coarse])
        heights = np.stack([np.interp(fine, x, col) for col in rows.T], axis=1)
        self.register_buffer(
            "grid", torch.tensor(heights, dtype=torch.float32), persistent=False
        )
        axis = torch.linspace(-0.2, 0.2, 5)
        patch = torch.stack(torch.meshgrid(axis, axis, indexing="ij"), -1).reshape(
            -1, 2
        )
        self.register_buffer("patch_offsets", patch, persistent=False)
        # Surface samples of the actual three collision cylinders on each foot.
        root = ET.parse(URDF).getroot()
        feet = []
        for name in ("Rfoot", "Lfoot"):
            points = []
            for collision in root.find(f"link[@name='{name}']").findall("collision"):
                cylinder = collision.find("geometry/cylinder")
                if cylinder is None:
                    raise ValueError("MOS9 spawn clearance expects foot cylinders")
                radius, half = (
                    float(cylinder.get("radius")),
                    float(cylinder.get("length")) / 2,
                )
                angle = torch.arange(32) * (2 * torch.pi / 32)
                ring = torch.stack((radius * angle.cos(), radius * angle.sin()), -1)
                local = torch.cat(
                    [
                        torch.cat((ring, torch.full((32, 1), z)), -1)
                        for z in (-half, 0, half)
                    ]
                )
                local = torch.cat(
                    (local, torch.tensor([[0.0, 0.0, -half], [0.0, 0.0, half]]))
                )
                origin = collision.find("origin")
                offset = torch.tensor(
                    [float(v) for v in origin.get("xyz", "0 0 0").split()]
                )
                rpy = [float(v) for v in origin.get("rpy", "0 0 0").split()]
                quat = torch.tensor([1.0, 0.0, 0.0, 0.0])
                for a, value in zip(((0, 0, 1), (0, 1, 0), (1, 0, 0)), reversed(rpy)):
                    quat = multiply(quat, from_rotation_vector(torch.tensor(a) * value))
                points.append(rotate(quat, local) + offset)
            feet.append(torch.cat(points))
        self.register_buffer("foot_samples", torch.stack(feet), persistent=False)

    def height(self, xy):
        uv = ((xy + self.size / 2) / self.cell).clamp(0, self.grid.shape[0] - 1 - 1e-4)
        base = uv.floor().long()
        a, b = (uv - base).unbind(-1)
        i, j = base.unbind(-1)
        z00, z10 = self.grid[i, j], self.grid[i + 1, j]
        z01, z11 = self.grid[i, j + 1], self.grid[i + 1, j + 1]
        # Same anti-diagonal as Genesis's collision and rendering mesh.
        return torch.where(
            a + b <= 1,
            z00 + a * (z10 - z00) + b * (z01 - z00),
            z11 + (1 - a) * (z01 - z11) + (1 - b) * (z10 - z11),
        )

    def features(self, xy):
        return self.height(xy[..., None, :] + self.patch_offsets) / 0.05

    def spawn_lift(self, foot_pos, foot_quat):
        points = foot_pos[..., None, :] + rotate(
            foot_quat[..., None, :], self.foot_samples
        )
        # Only the initial pose is raised; the same constant offset is applied to
        # every reference target in this episode. No support during simulation.
        return (self.height(points[..., :2]) - points[..., 2]).flatten(-2).amax(
            -1
        ) + 0.005

    def manifest(self):
        array = self.grid.detach().cpu().numpy()
        return {
            "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
            "shape": list(array.shape),
            "height_min_m": float(array.min()),
            "height_max_m": float(array.max()),
            "cell_m": self.cell,
            "size_m": self.size,
            "interpolation": "uniform coarse knots; bounded linear upsampling; collision triangles",
        }
