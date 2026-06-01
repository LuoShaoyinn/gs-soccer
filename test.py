import os
import numpy as np
import torch
from argparse import ArgumentParser

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import genesis as gs

from fields.field import Field, FieldConfig
from robots.mos9 import MOS9, MOS9Config


IGNORED_COLLISION_GEOM_PAIRS = (
    (16, 20),
    (17, 21),
)


def ignore_collision_geom_pairs(pairs: tuple[tuple[int, int], ...]) -> None:
    from genesis.engine.solvers.rigid.collider.collider import Collider

    ignored_pairs = {tuple(sorted(pair)) for pair in pairs}
    original_compute_collision_pair_idx = Collider._compute_collision_pair_idx

    def patched_compute_collision_pair_idx(self):
        (
            n_possible_pairs,
            collision_pair_idx,
            valid_collision_pairs,
            has_terrain,
            has_non_box_plane_convex_convex,
            has_convex_specialization,
            has_nonconvex_nonterrain,
        ) = original_compute_collision_pair_idx(self)

        kept_pairs = [
            pair
            for pair in valid_collision_pairs
            if tuple(sorted((int(pair[0]), int(pair[1])))) not in ignored_pairs
        ]

        n_geoms = collision_pair_idx.shape[0]
        filtered_collision_pair_idx = np.full(
            (n_geoms, n_geoms),
            fill_value=-1,
            dtype=collision_pair_idx.dtype,
        )
        if kept_pairs:
            filtered_valid_collision_pairs = np.array(
                kept_pairs,
                dtype=valid_collision_pairs.dtype,
            )
            filtered_collision_pair_idx[
                filtered_valid_collision_pairs[:, 0],
                filtered_valid_collision_pairs[:, 1],
            ] = np.arange(len(kept_pairs), dtype=collision_pair_idx.dtype)
        else:
            filtered_valid_collision_pairs = np.empty(
                (0, 2),
                dtype=valid_collision_pairs.dtype,
            )

        return (
            len(kept_pairs),
            filtered_collision_pair_idx,
            filtered_valid_collision_pairs,
            has_terrain,
            has_non_box_plane_convex_convex,
            has_convex_specialization,
            has_nonconvex_nonterrain,
        )

    Collider._compute_collision_pair_idx = patched_compute_collision_pair_idx


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--viewer", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gs.init(
        backend=gs.gpu,  # type: ignore[attr-defined]
        performance_mode=True,
        logging_level="warning",
    )
    ignore_collision_geom_pairs(IGNORED_COLLISION_GEOM_PAIRS)

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.8, -3.0, 1.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=35,
            res=(960, 640),
            max_FPS=60,
        ),
        sim_options=gs.options.SimOptions(
            dt=0.01,
            substeps=2,
        ),
        rigid_options=gs.options.RigidOptions(
            enable_neutral_collision=True,
        ),
        show_viewer=args.viewer,
    )

    field = Field(FieldConfig(), scene)
    robot = MOS9(MOS9Config(), scene)

    field.build()
    robot.build()
    scene.build(n_envs=1, env_spacing=(0.0, 0.0))
    field.config()
    robot.config()

    envs_idx = torch.tensor([0], dtype=torch.long, device=gs.device)
    robot.reset(envs_idx=envs_idx)
    standing_q = torch.zeros((1, len(robot.cfg.joint_names)), device=gs.device)

    for _ in range(args.steps):
        robot.step(standing_q)
        scene.step()


if __name__ == "__main__":
    main()
