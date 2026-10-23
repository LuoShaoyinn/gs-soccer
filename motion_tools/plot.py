"""Scientific contact plots and a labeled kinematic reference preview."""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mos9-mpl")

import matplotlib

matplotlib.use("Agg")
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FFMpegWriter

from .quality import RobotModel, inspect


def plot_quality(paths, output):
    model = RobotModel("assets/MOS9/MOS9_walk.urdf")
    fig, axes = plt.subplots(
        len(paths),
        2,
        figsize=(12, 2.8 * len(paths)),
        layout="constrained",
        squeeze=False,
    )
    for row, path in enumerate(paths):
        _metrics, data = inspect(path, model)
        times = np.arange(len(data["q"])) / 50
        h = data["foot_heights"]
        speed = data["foot_contact_speed"]
        axes[row, 0].plot(times, 100 * h[:, 0], label="right")
        axes[row, 0].plot(times, 100 * h[:, 1], label="left")
        axes[row, 0].axhline(0, color="black", lw=1)
        axes[row, 0].axhline(-2, color="red", ls="--", lw=1)
        axes[row, 0].set(
            title=path.stem + ": collision clearance", xlabel="seconds", ylabel="cm"
        )
        axes[row, 0].legend()
        lowest = h.argmin(axis=1)
        axes[row, 1].plot(times, speed[np.arange(len(times)), lowest])
        axes[row, 1].axhline(0.2, color="red", ls="--", lw=1)
        axes[row, 1].set(
            title="Lowest foot: contact-point material speed",
            xlabel="seconds",
            ylabel="m/s",
        )
    fig.suptitle("Exact MOS9 URDF geometry; kinematic references")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def preview(path, output):
    model = RobotModel("assets/MOS9/MOS9_walk.urdf")
    _, data = inspect(path, model)
    poses = model.fk(data["q"], data["root_pos"], data["root_quat"])
    fig = plt.figure(figsize=(8, 7), layout="constrained")
    ax = fig.add_subplot(projection="3d")
    ax.set_box_aspect((1, 1, 0.8))
    ax.view_init(elev=20, azim=-125)
    bones = []
    for parent, child, *_ in model.joints:
        color = "tab:orange" if child.startswith("L") else "tab:blue"
        (line,) = ax.plot([], [], [], color=color, lw=3, marker="o", markersize=3)
        bones.append((parent, child, line))
    points, _ = model.collision_points(poses, {"Rfoot", "Lfoot"})
    (feet,) = ax.plot([], [], [], "r.", markersize=7)
    (ground,) = ax.plot([], [], [], color="gray", lw=1)
    stamp = ax.text2D(0.03, 0.95, "", transform=ax.transAxes)
    ax.set(xlabel="world x (m)", ylabel="world y (m)", zlabel="z (m)")
    fig.suptitle(
        "G1 → MOS9 kinematic reference preview\nNo physics or policy in this video"
    )
    writer = FFMpegWriter(fps=25, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with writer.saving(fig, str(output), dpi=100):
        for frame in range(0, len(data["q"]), 2):
            center = data["root_pos"][frame]
            ax.set_xlim(center[0] - 0.65, center[0] + 0.65)
            ax.set_ylim(center[1] - 0.65, center[1] + 0.65)
            ax.set_zlim(-0.08, 0.95)
            for parent, child, line in bones:
                pair = np.array([poses[parent][0][frame], poses[child][0][frame]])
                line.set_data_3d(pair.T)
            feet.set_data_3d(points[frame].T)
            corners = center[:2] + np.array(
                [[-0.6, -0.6], [0.6, -0.6], [0.6, 0.6], [-0.6, 0.6], [-0.6, -0.6]]
            )
            ground.set_data_3d(corners[:, 0], corners[:, 1], np.zeros(5))
            stamp.set_text(f"{path.name}    t={frame / 50:.2f}s")
            writer.grab_frame()
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--video", type=Path, help="Preview the final input reference")
    a = p.parse_args()
    plot_quality(a.input, a.output)
    if a.video:
        preview(a.input[-1], a.video)


if __name__ == "__main__":
    main()
