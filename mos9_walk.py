"""Evaluate a classical MOS9 teacher, independently of the learner branch."""
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import torch
import genesis as gs
from algorithm.mos9_teacher.planner import GaitConfig, Kinematics, make_plan
from algorithm.mos9_teacher.controller import WalkingTeacher
from algorithm.mos9_teacher.simple import SimpleWalkingTeacher
from algorithm.mos9_teacher.model import model_path
from MDPs.mos9_walk import MOS9WalkConfig, MOS9WalkMDP
from robots.mos9 import MOS9, MOS9Config
from fields.terrain_field import TerrainField, TerrainFieldConfig
from envs.env import Env, EnvConfig


class WalkingMOS9(MOS9):
    def reset(self, envs_idx, **kwargs):
        super().reset(envs_idx, **kwargs)
        # The shared Robot reset preserves velocities; task resets must clear them.
        self.robot.set_dofs_velocity(torch.zeros((len(envs_idx), self.robot.n_dofs), device=gs.device), envs_idx=envs_idx)


class WalkingTerrain(TerrainField):
    def build(self):
        if not self.cfg.use_terrain:
            return super().build()
        # A single contact surface avoids overlapping coplanar terrain/plane contacts.
        self.terrain = self.scene.add_entity(
            gs.morphs.Terrain(n_subterrains=(1, 1), subterrain_size=self.cfg.subterrain_size,
                             horizontal_scale=self.cfg.horizontal_scale,
                             vertical_scale=self.cfg.vertical_scale,
                             subterrain_types=[[self.cfg.terrain_types]],
                             randomize=True,
                             subterrain_parameters=self.cfg.subterrain_parameters,
                             pos=self.cfg.terrain_pos),
            material=gs.materials.Rigid(friction=self.cfg.field_friction))


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--terrain", choices=("flat", "gentle"), default="gentle",
                        help="gentle: uniform random heights between 0 and terrain-height")
    parser.add_argument("--terrain-height", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("runs/mos9_teacher/eval.json"))
    parser.add_argument("--reset-xy", type=float, default=0.0)
    parser.add_argument("--reset-yaw", type=float, default=0.0)
    parser.add_argument("--feedback", type=float, default=0.2)
    parser.add_argument("--pitch-feedback", type=float, default=None,
                        help="override ankle pitch gain independently of roll")
    parser.add_argument("--kp", type=float, default=250)
    parser.add_argument("--kv", type=float, default=8)
    parser.add_argument("--feedback-sweep", action="store_true", help="evaluate eight ankle feedback gains in parallel")
    parser.add_argument("--angular-damping", type=float, default=0.0)
    parser.add_argument("--sim-freq", type=int, default=2000)
    parser.add_argument("--spawn-clearance", type=float, default=0.02,
                        help="clearance of the lowest collision point above the highest terrain")
    parser.add_argument("--self-collision", action="store_true")
    parser.add_argument("--position-feedback", type=float, default=2.0)
    parser.add_argument("--velocity-feedback", type=float, default=0.3)
    parser.add_argument("--tracking-limit", type=float, default=1.0,
                        help="maximum paired hip/ankle tracking correction in radians")
    parser.add_argument("--ik-backend", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--tracking-sweep", action="store_true")
    parser.add_argument("--online-ik", action=argparse.BooleanOptionalAction, default=False,
                        help="experimental online Cartesian feedback; offline GPU IK is the default")
    parser.add_argument("--flat-soles", action="store_true", help="use a generated URDF with flat contacts fitted to visual soles")
    for name, value in asdict(GaitConfig()).items():
        parser.add_argument("--"+name.replace("_", "-"), type=float, default=value)
    args = parser.parse_args(argv)
    if args.episodes <= 0 or args.num_envs <= 0:
        parser.error("episodes and num-envs must be positive")
    if args.spawn_clearance <= 0 or args.terrain_height < 0 or args.tracking_limit <= 0:
        parser.error("spawn-clearance and tracking-limit must be positive, and terrain-height nonnegative")
    if args.sim_freq <= 0 or args.sim_freq % 50:
        parser.error("sim-freq must be a positive multiple of 50")
    if (args.feedback_sweep or args.tracking_sweep) and args.num_envs != 8:
        parser.error("parameter sweeps require num-envs=8")
    gait = GaitConfig(**{name: getattr(args, name) for name in asdict(GaitConfig())})
    if gait.step_time <= 0 or not 0 < gait.double_support < 1 or gait.start_time < 0:
        parser.error("step-time must be positive, double-support between 0 and 1, and start-time nonnegative")
    if gait.step_length <= 0 or gait.swing_height < 0 or gait.stance_width <= 0:
        parser.error("step-length and stance-width must be positive, and swing-height nonnegative")
    return args


def build_walking_env(args, env_class=Env):
    gait = GaitConfig(**{name: getattr(args, name) for name in asdict(GaitConfig())})
    robot_urdf = model_path(args.flat_soles)
    if args.viewer:
        os.environ.pop("PYOPENGL_PLATFORM", None)
    gs.init(backend=gs.gpu, seed=args.seed, performance_mode=True, logging_level="warning")
    cache = Path("runs/mos9_teacher/plans")
    cache.mkdir(parents=True, exist_ok=True)
    import hashlib
    key = hashlib.sha256(json.dumps(asdict(gait), sort_keys=True).encode()
                         + Path("algorithm/mos9_teacher/planner.py").read_bytes()
                         + Path("algorithm/mos9_teacher/gpu_ik.py").read_bytes()
                         + args.ik_backend.encode()
                         + gs.__version__.encode()
                         + robot_urdf.read_bytes()).hexdigest()[:12]
    plan_path = cache / (key+".npz")
    if plan_path.exists():
        with np.load(plan_path) as saved:
            plan = {key: saved[key] for key in saved.files}
    else:
        print("Planning ZMP trajectory and solving URDF leg IK...", flush=True)
        if args.ik_backend == "gpu":
            from algorithm.mos9_teacher.gpu_ik import make_gpu_plan
            plan = make_gpu_plan(gait, robot_urdf)
        else:
            plan = make_plan(gait, model_path=robot_urdf)
        np.savez(plan_path, **plan)
    names = list(plan["names"])
    if float(plan["ik_max_error"]) > 0.001:
        raise RuntimeError("planned pose IK residual exceeds 1 mm/rad; revise the gait before rollout")
    kin = Kinematics(robot_urdf)
    ground_max = args.terrain_height if args.terrain == "gentle" else 0.0
    build_height = ground_max + args.spawn_clearance - kin.collision_bottom(np.zeros(len(names)))
    reset_height = ground_max + args.spawn_clearance - kin.collision_bottom(plan["home"])
    print(f"build_height={build_height:.4f} reset_height={reset_height:.4f} minimum_ground_clearance={args.spawn_clearance:.4f}", flush=True)
    print(f"plan={plan_path} height={float(plan['height']):.4f} IK_error={float(plan['ik_max_error']):.6g}", flush=True)
    n = len(names)
    cfg = MOS9Config(robot_URDF=str(robot_urdf), joint_names=names, foot_link_names=["Rfoot", "Lfoot"],
                     kp=np.full(n, args.kp, dtype=np.float32), kv=np.full(n, args.kv, dtype=np.float32),
                     armature=np.full(n, 0.01, dtype=np.float32), damping=np.zeros(n, dtype=np.float32),
                     force_range=np.array([[-100]*n, [100]*n], dtype=np.float32),
                     velocity_range=np.array([[-20]*n, [20]*n], dtype=np.float32),
                     initial_pos=np.array([0, 0, build_height], dtype=np.float32))
    field = TerrainFieldConfig(field_friction=0.8, use_terrain=args.terrain == "gentle",
                               terrain_types="random_uniform_terrain", subterrain_size=(6, 6),
                               terrain_pos=(-2, -3, 0), horizontal_scale=0.1, vertical_scale=0.001,
                               subterrain_parameters={"random_uniform_terrain": {
                                   "min_height": 0.0, "max_height": args.terrain_height,
                                   "step": 0.001, "downsampled_scale": 0.3}})
    # Planning-scene construction can consume NumPy randomness. Terrain must
    # depend on the requested seed, regardless of whether a plan was cached.
    np.random.seed(args.seed)
    env = env_class(EnvConfig(robot_cfg=cfg, robot_class=WalkingMOS9, field_cfg=field, field_class=WalkingTerrain,
                        MDP_cfg=MOS9WalkConfig(home=plan["home"], height=reset_height,
                                               reset_xy=args.reset_xy, reset_yaw=args.reset_yaw),
                        MDP_class=MOS9WalkMDP, policy_freq=50, sim_freq=args.sim_freq,
                        self_collision=args.self_collision,
                        num_envs=args.num_envs, show_viewer=args.viewer))
    env.reset()
    return env, plan, kin, build_height, reset_height, plan_path, gait


def main():
    args = parse_args()
    env, plan, kin, build_height, reset_height, plan_path, gait = build_walking_env(args)
    names = list(plan["names"])
    actions = torch.as_tensor(plan["actions"], device=gs.device)
    bounds = kin.bounds
    lower, upper = torch.tensor([bounds[name] for name in names], device=gs.device).T
    reference_roll = torch.as_tensor(plan["body_roll"], device=gs.device)
    reference_pelvis = torch.as_tensor(plan["pelvis"]-plan["pelvis"][0], device=gs.device)
    reference_velocity = torch.as_tensor(np.gradient(plan["pelvis"], 0.02, axis=0), device=gs.device)
    gains = torch.full((args.num_envs,), args.feedback, device=gs.device)
    if args.feedback_sweep:
        gains = torch.tensor([0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5], device=gs.device)
    pitch_gains = gains if args.pitch_feedback is None else torch.full_like(gains, args.pitch_feedback)
    position_gains = torch.full((args.num_envs,), args.position_feedback, device=gs.device)
    velocity_gains = torch.full((args.num_envs,), args.velocity_feedback, device=gs.device)
    if args.tracking_sweep:
        position_gains = torch.tensor([0.5, 1, 2, 3, 4, 6, 8, 10], device=gs.device)
        velocity_gains = position_gains*0.15
    teacher = (WalkingTeacher(env.robot.robot, plan, position_gains, velocity_gains, gains, args.angular_damping)
               if args.online_ik else None)
    simple_teacher = SimpleWalkingTeacher(plan, gs.device, args.position_feedback,
                                         args.velocity_feedback, args.feedback,
                                         args.feedback if args.pitch_feedback is None else args.pitch_feedback,
                                         args.tracking_limit)
    traces = []
    rows = []
    quotas = [args.episodes//args.num_envs + int(i < args.episodes % args.num_envs) for i in range(args.num_envs)]
    counts = [0]*args.num_envs
    cursor = 0
    while len(rows) < args.episodes:
        step = env.MDP.steps
        target = actions[step].clone()
        state = env.get_state(env.all_envs_idx)
        quat = state["body_quat"]
        roll = torch.atan2(2*(quat[:, 0]*quat[:, 1]+quat[:, 2]*quat[:, 3]), 1-2*(quat[:, 1]**2+quat[:, 2]**2))
        pitch = torch.asin((2*(quat[:, 0]*quat[:, 2]-quat[:, 3]*quat[:, 1])).clamp(-1, 1))
        delta = state["body_pos"][:, :2]-env.MDP.origins[:, :2]
        velocity = state["body_lin_vel"][:, :2]
        c, s = torch.cos(env.MDP.yaws), torch.sin(env.MDP.yaws)
        delta = torch.stack((c*delta[:, 0]+s*delta[:, 1], -s*delta[:, 0]+c*delta[:, 1]), dim=1)
        velocity = torch.stack((c*velocity[:, 0]+s*velocity[:, 1], -s*velocity[:, 0]+c*velocity[:, 1]), dim=1)
        tracking = position_gains[:, None]*(delta-reference_pelvis[step]) + velocity_gains[:, None]*(velocity-reference_velocity[step])
        tracking = tracking.clamp(-args.tracking_limit, args.tracking_limit)
        for side, sign in (("right", 1), ("left", -1)):
            # Paired hip/ankle adjustments translate the pelvis while keeping
            # the planned sole orientation, unlike ankle-only attitude feedback.
            target[:, names.index(side+"_hip_roll")] += tracking[:, 1]
            target[:, names.index(side+"_hip_pitch")] -= sign*tracking[:, 0]
            target[:, names.index(side+"_ankle_roll")] += gains*(roll-reference_roll[step]) + args.angular_damping*state["body_ang_vel"][:, 0] - tracking[:, 1]
            target[:, names.index(side+"_ankle_pitch")] += sign*(pitch_gains*pitch + args.angular_damping*state["body_ang_vel"][:, 1] + tracking[:, 0])
        if args.online_ik:
            target = teacher.act(state, step, env.MDP.origins, env.MDP.yaws, roll, pitch)
        elif not args.feedback_sweep and not args.tracking_sweep and args.angular_damping == 0:
            target = simple_teacher.act(state, step, env.MDP.origins, env.MDP.yaws)
        if len(traces) < 501:
            traces.append({"step": int(step[0]), "pos": state["body_pos"][0].tolist(),
                           "velocity": state["body_lin_vel"][0].tolist(),
                           "roll": float(roll[0]), "pitch": float(pitch[0]),
                           "target": target[0].tolist(), "joints": state["dofs_pos"][0].tolist(),
                           "torques": state["dofs_force"][0].tolist(),
                           "feet": state["foot_pos"][0].tolist()})
        with torch.no_grad():
            try:
                env.step(target.clamp(lower, upper))
            except Exception as error:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.with_suffix(".trace.json").write_text(json.dumps(traces)+"\n")
                args.output.with_suffix(".failure.json").write_text(json.dumps({
                    "error": str(error), "step": step.tolist(),
                    "settings": {name: str(value) if isinstance(value, Path) else value
                                 for name, value in vars(args).items()},
                    "completed": env.MDP.completed,
                }, indent=2)+"\n")
                raise
        for row in env.MDP.completed[cursor:]:
            env_id = row["env_id"]
            if counts[env_id] < quotas[env_id]:
                rows.append(row)
                counts[env_id] += 1
                print(f"episode={len(rows)} env={env_id} steps={row['steps']} "
                      f"distance={row['distance']:.3f} success={row['success']}", flush=True)
        cursor = len(env.MDP.completed)
    for row in rows:
        row["feedback"] = float(gains[row["env_id"]])
        row["position_feedback"] = float(position_gains[row["env_id"]])
        row["velocity_feedback"] = float(velocity_gains[row["env_id"]])
    result = {"genesis_version": gs.__version__, "seed": args.seed, "terrain": args.terrain,
              "terrain_height": args.terrain_height, "gait": asdict(gait), "feedback": args.feedback,
              "kp": args.kp, "kv": args.kv,
              "pitch_feedback": args.pitch_feedback,
              "angular_damping": args.angular_damping,
              "sim_freq": args.sim_freq,
              "spawn_clearance": args.spawn_clearance,
              "self_collision": args.self_collision,
              "position_feedback": args.position_feedback, "velocity_feedback": args.velocity_feedback,
              "tracking_limit": args.tracking_limit,
              "ik_backend": args.ik_backend,
              "online_ik": args.online_ik, "online_ik_max_error": teacher.max_error if teacher else 0.0,
              "flat_soles": args.flat_soles,
              "build_height": build_height, "reset_height": reset_height,
              "episodes": len(rows), "success_rate": np.mean([r["success"] for r in rows]),
              "survival_rate": np.mean([not r["fallen"] for r in rows]),
              "mean_steps": np.mean([r["steps"] for r in rows]),
              "mean_distance": np.mean([r["distance"] for r in rows]),
              "ik_max_error": float(plan["ik_max_error"]), "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    args.output.with_suffix(".trace.json").write_text(json.dumps(traces)+"\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2), flush=True)
    env.close()


if __name__ == "__main__":
    main()
