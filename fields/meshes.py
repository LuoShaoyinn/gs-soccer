import math
from pathlib import Path

_CACHE_DIR = Path(__file__).resolve().parent.parent / "assets" / "textures" / ".cache"


def _ensure_dir() -> Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR


def field_obj_path(half_l: float, half_w: float, z: float = 0.001) -> str:
    _ensure_dir()
    path = _CACHE_DIR / "field_quad.obj"
    if not path.exists():
        with open(path, "w") as f:
            f.write(_field_obj(half_l, half_w, z))
    return str(path)


def _field_obj(half_l: float, half_w: float, z: float) -> str:
    lines = [
        f"v {-half_l:.4f} {-half_w:.4f} {z:.4f}",
        f"v {half_l:.4f} {-half_w:.4f} {z:.4f}",
        f"v {half_l:.4f} {half_w:.4f} {z:.4f}",
        f"v {-half_l:.4f} {half_w:.4f} {z:.4f}",
        "vt 0.0 1.0",
        "vt 1.0 1.0",
        "vt 1.0 0.0",
        "vt 0.0 0.0",
        "vn 0.0 0.0 1.0",
        "f 1/1/1 2/2/1 3/3/1",
        "f 1/1/1 3/3/1 4/4/1",
    ]
    return "\n".join(lines)


def sphere_obj_path(
    radius: float,
    slices: int = 32,
    stacks: int = 16,
    inverted: bool = False,
) -> str:
    _ensure_dir()
    name = "sky_dome.obj" if inverted else "ball_sphere.obj"
    path = _CACHE_DIR / name
    if not path.exists():
        with open(path, "w") as f:
            f.write(_sphere_obj(radius, slices, stacks, inverted))
    return str(path)


def _sphere_obj(
    radius: float, slices: int, stacks: int, inverted: bool
) -> str:
    lines: list[str] = []
    sign = -1.0 if inverted else 1.0

    for i in range(stacks + 1):
        phi = math.pi * i / stacks
        sin_phi = math.sin(phi)
        cos_phi = math.cos(phi)
        for j in range(slices + 1):
            theta = 2.0 * math.pi * j / slices
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)
            x = radius * sin_phi * cos_t
            y = radius * sin_phi * sin_t
            z = radius * cos_phi
            nx = sign * sin_phi * cos_t
            ny = sign * sin_phi * sin_t
            nz = sign * cos_phi
            lines.append(f"v {x:.6f} {y:.6f} {z:.6f}")
            lines.append(f"vt {j / slices:.6f} {i / stacks:.6f}")
            lines.append(f"vn {nx:.6f} {ny:.6f} {nz:.6f}")

    for i in range(stacks):
        for j in range(slices):
            p1 = i * (slices + 1) + j + 1
            p2 = i * (slices + 1) + j + 2
            p3 = (i + 1) * (slices + 1) + j + 1
            p4 = (i + 1) * (slices + 1) + j + 2
            if inverted:
                lines.append(f"f {p1}/{p1}/{p1} {p3}/{p3}/{p3} {p2}/{p2}/{p2}")
                lines.append(f"f {p2}/{p2}/{p2} {p3}/{p3}/{p3} {p4}/{p4}/{p4}")
            else:
                lines.append(f"f {p1}/{p1}/{p1} {p2}/{p2}/{p2} {p3}/{p3}/{p3}")
                lines.append(f"f {p2}/{p2}/{p2} {p4}/{p4}/{p4} {p3}/{p3}/{p3}")

    return "\n".join(lines)
