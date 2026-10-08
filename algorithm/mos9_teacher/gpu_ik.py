"""Solve all walking-plan timesteps in parallel using Genesis GPU IK."""
import time
import numpy as np
import torch
import genesis as gs
from genesis.utils.geom import transform_by_quat
from scipy.spatial.transform import Rotation
from .planner import make_reference


def make_gpu_plan(cfg, model_path="assets/MOS9/MOS9_walk.urdf"):
    reference = make_reference(cfg, model_path=model_path)
    n = len(reference["feet"])
    scene = gs.Scene(show_viewer=False,
                     sim_options=gs.options.SimOptions(dt=0.02, gravity=(0, 0, 0)),
                     rigid_options=gs.options.RigidOptions(enable_collision=False))
    robot = scene.add_entity(gs.morphs.URDF(file=str(model_path),
                                           pos=(0, 0, 0.6), decimate=True,
                                           file_meshes_are_zup=True))
    print(f"Building GPU IK planner with {n} parallel poses...", flush=True)
    scene.build(n_envs=n)
    names = reference["names"]
    q_indices = [robot.get_joint(name).qs_idx_local[0] for name in names]
    dof_indices = [robot.get_joint(name).dofs_idx_local[0] for name in names if "shoulder" not in name and "elbow" not in name]
    feet = torch.as_tensor(reference["feet"], dtype=torch.float32, device=gs.device)
    feet[:, :, 2] += reference["height"]
    q = robot.get_qpos().clone()
    q[:, q_indices] = torch.as_tensor(reference["seed"], dtype=torch.float32, device=gs.device)
    q[:, :2] = torch.as_tensor(reference["pelvis"], dtype=torch.float32, device=gs.device)
    q[:, 2] = reference["height"]
    rolls = torch.as_tensor(reference["body_roll"], dtype=torch.float32, device=gs.device)
    q[:, 3:7] = torch.stack((torch.cos(rolls/2), torch.sin(rolls/2), torch.zeros_like(rolls), torch.zeros_like(rolls)), dim=1)
    rotations = [torch.as_tensor(Rotation.from_matrix(r).as_quat()[[3, 0, 1, 2]], dtype=torch.float32, device=gs.device).repeat(n, 1)
                 for r in reference["rotations"]]
    desired_com = torch.as_tensor(reference["com"], dtype=torch.float32, device=gs.device)
    links = [robot.get_link(name) for name in ("Rfoot", "Lfoot")]
    mass = robot.get_links_mass()
    local_com = robot.get_links_COM()
    started = time.perf_counter()
    for _ in range(6):
        q, error = robot.inverse_kinematics_multilink(
            links=links, poss=[feet[:, 0], feet[:, 1]], quats=rotations, init_qpos=q,
            dofs_idx_local=dof_indices, max_samples=1, max_solver_iters=60,
            damping=0.005, pos_tol=0.0001, rot_tol=0.001, return_error=True)
        robot.set_qpos(q)
        positions = robot.get_links_pos()
        quats = robot.get_links_quat()
        world_com = positions + transform_by_quat(local_com.unsqueeze(0).expand(n, -1, -1), quats)
        actual_com = (world_com*mass[None, :, None]).sum(dim=1)/mass.sum()
        difference = desired_com-actual_com[:, :2]
        if float(difference.abs().max()) < 0.0002:
            break
        q[:, :2] += difference
    # Last solve ensures returned joints match the final adjusted pelvis.
    q, error = robot.inverse_kinematics_multilink(
        links=links, poss=[feet[:, 0], feet[:, 1]], quats=rotations, init_qpos=q,
        dofs_idx_local=dof_indices, max_samples=1, max_solver_iters=60,
        damping=0.005, pos_tol=0.0001, rot_tol=0.001, return_error=True)
    elapsed = time.perf_counter()-started
    actions = q[:, q_indices].cpu().numpy()
    worst = float(error.abs().max())
    print(f"GPU IK: {n} poses in {elapsed:.2f}s; max pose residual={worst:.6g}", flush=True)
    plan = {"names": names, "actions": actions, "pelvis": q[:, :2].cpu().numpy(),
            "height": reference["height"], "home": actions[0], "ik_max_error": worst,
            "body_roll": reference["body_roll"].astype(np.float32),
            "feet": reference["feet"].astype(np.float32), "ik_seconds": elapsed,
            "ik_backend": "genesis-gpu"}
    scene.destroy()
    return plan
