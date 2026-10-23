"""Unassisted Genesis reference PD replay, separate from kinematic validation.

Failure of this controller does not prove no learned tracker can follow the
motion. It prevents calling the retarget dynamically validated.
"""

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from supermimic.config import Config
from supermimic.motion import MotionLibrary, ResetSampler
from supermimic.run import init_genesis
from supermimic.task import Task


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--library", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--backend", choices=("gpu", "cpu"), default="gpu")
    p.add_argument("--actuation", choices=("nominal", "teammate"), default="teammate")
    args = p.parse_args()
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mos9-mpl")
    count = len(list(args.library.glob("*.npz")))
    cfg = Config(num_envs=count, randomize=False, sensor_noise=False)
    device = init_genesis(cfg, args.backend, False)
    task = Task(cfg).to(device)
    lib = MotionLibrary(args.library, task, device)
    sampler = ResetSampler(lib, cfg.gamma)
    from supermimic.environment import WalkingEnv

    env = WalkingEnv(task, lib, sampler)
    n = task.n
    if args.actuation == "teammate":
        small = [
            "shoulder" in name or "ankle_roll" in name
            for name in env.wrapper.cfg.joint_names
        ]
        # Values from the teammate's pinned robots/MOS9.py config.
        kp = torch.tensor(
            [47.177610 if s else 105.193621 for s in small], device=device
        )
        kd = torch.tensor([1.782347 if s else 2.629726 for s in small], device=device)
        armature = torch.tensor(
            [0.0282528 if s else 0.0478125 for s in small], device=device
        )
        effort = torch.tensor([36.0 if s else 60.0 for s in small], device=device)
        env.robot.set_dofs_kp(kp, env.indices)
        env.robot.set_dofs_kv(kd, env.indices)
        env.robot.set_dofs_armature(armature, env.indices)
        env.robot.set_dofs_force_range(-effort, effort, env.indices)
    else:
        kp = torch.full((n,), cfg.kp, device=device)
        kd = torch.full((n,), cfg.kd, device=device)
        effort = torch.full((n,), cfg.torque_limit, device=device)
    idx = torch.arange(count, device=device)
    env.reset(idx, idx, torch.zeros(count, dtype=torch.long, device=device))
    errors = [[] for _ in range(count)]
    joint_errors = [[] for _ in range(count)]
    torque_saturation = [[] for _ in range(count)]
    contact_counts = [[] for _ in range(count)]
    first_failure = [None] * count
    finished = [False] * count
    positions = [[] for _ in range(count)]
    try:
        for frame in range(int(lib.lengths.max()) - 1):
            reference = lib.reference(
                idx, torch.full((count,), frame + 1, device=device)
            )[:, 0]
            with torch.no_grad():
                env.robot.control_dofs_position_velocity(
                    reference[:, task.q], reference[:, task.dq], env.indices
                )
                env.scene.step()
                state = env.state()
                terminal = task.terminal(state, reference)
                force = env.robot.get_dofs_control_force(env.indices)
                contacts = env.robot.get_links_net_contact_force()[
                    :, env.wrapper.foot_links_idx_local
                ]
            for i in range(count):
                if finished[i]:
                    continue
                errors[i].append(
                    float((state[i, task.pos] - reference[i, task.pos]).norm())
                )
                joint_errors[i].append(
                    float((state[i, task.q] - reference[i, task.q]).square().mean())
                )
                torque_saturation[i].append(
                    float((force[i].abs() >= 0.99 * effort).float().mean())
                )
                contact_counts[i].append(int((contacts[i].norm(dim=-1) > 1).sum()))
                positions[i].append(state[i, task.pos].cpu().tolist())
                if terminal[i]:
                    first_failure[i] = (frame + 1) * cfg.dt
                    finished[i] = True
                elif frame + 1 >= int(lib.lengths[i]) - 1:
                    finished[i] = True
            if all(finished):
                break
    finally:
        env.close()
    rows = []
    for i, name in enumerate(lib.names):
        rows.append(
            {
                "clip": name,
                "first_failure_seconds": first_failure[i],
                "recorded_seconds": len(errors[i]) * cfg.dt,
                "full_clip_seconds": (int(lib.lengths[i]) - 1) * cfg.dt,
                "root_position_rmse_m": float(np.sqrt(np.mean(np.square(errors[i])))),
                "joint_rmse_rad": float(np.sqrt(np.mean(joint_errors[i]))),
                "torque_saturation_fraction": float(np.mean(torque_saturation[i])),
                "no_foot_contact_fraction": float(
                    np.mean(np.array(contact_counts[i]) == 0)
                ),
                "displacement_xy_m": (
                    np.array(positions[i][-1])[:2]
                    - lib.states[i, 0, task.pos].cpu().numpy()[:2]
                ).tolist(),
            }
        )
    result = {
        "config": asdict(cfg),
        "actuation": args.actuation,
        "kp": kp.cpu().tolist(),
        "kd": kd.cpu().tolist(),
        "fingerprint": lib.fingerprint,
        "clips": rows,
        "controller": "reference position+velocity PD; no base assistance",
        "limitation": "Controller-specific probe; not a policy evaluation or proof of infeasibility",
        "runtime_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
