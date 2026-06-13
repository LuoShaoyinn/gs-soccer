import math
import os
import random
from argparse import ArgumentParser

import numpy as np
import torch

BALL_RADIUS = 0.07


def _random_ball_spawn(mode: str, rng: random.Random) -> tuple[float, float]:
    if mode == "walk":
        return rng.gauss(4.0, 0.2), rng.gauss(4.0, 0.2)
    if mode == "approach_kick":
        dist = rng.uniform(3.0, 4.0)
        angle = rng.uniform(-math.pi, math.pi)
        return dist * math.cos(angle), dist * math.sin(angle)
    r = rng.uniform(0.4, 1.0)
    angle = rng.gauss(0.0, 0.3)
    return r * math.cos(angle), r * math.sin(angle)


def parse_args():
    p = ArgumentParser(description="PiPlus soccer sim2sim — Genesis")
    p.add_argument("--model-dir", type=str,
                   default="refs/piplus_soccer_sim2sim/models/exported")
    p.add_argument("--mode", type=str, default="walk",
                   choices=["walk", "kick", "approach_kick"])
    p.add_argument("--kick-speed", type=float, default=2.0)
    p.add_argument("--kick-dir-deg", type=float, default=0.0)
    p.add_argument("--vel-x", type=float, default=0.5)
    p.add_argument("--vel-y", type=float, default=0.0)
    p.add_argument("--vel-yaw", type=float, default=0.0)
    p.add_argument("--num-envs", type=int, default=1)
    p.add_argument("--steps", type=int, default=2500)
    p.add_argument("--viewer", action="store_true")
    p.add_argument("--ball-seed", type=int, default=None,
                   help="Random seed for ball placement (default: random)")
    p.add_argument("--no-render", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.ball_seed)
    if args.viewer:
        os.environ.pop("PYOPENGL_PLATFORM", None)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

    import genesis as gs
    from robots.pi import PI, PIConfig
    from algorithm.observation import ObservationBuilder
    from algorithm.command import SoccerCommandBuilder
    from algorithm.policy import OnnxPolicy

    # 22-dim policy joint order (head_yaw, l_hip_pitch, ..., head_pitch, l_hip_roll, ...)
    # 20-dim genesis omits head_yaw(idx 0) and head_pitch(idx 5)
    NUM_POLICY_JOINTS = 22
    NUM_GENESIS_JOINTS = 20
    POLICY_TO_GENESIS = np.array(
        [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
        dtype=np.int32,
    )

    POLICY_DEFAULT_POS = np.array([
        0.0, -0.25, 0.0, -0.25, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.65, 0.0, 0.65, 0.0,
        -0.4, -0.4, 0.0, 0.0,
    ], dtype=np.float32)

    _POLICY_ACTION_SCALE = np.array([
        0.096, 0.098, 0.154, 0.098, 0.154, 0.096,
        0.098, 0.154, 0.098, 0.154,
        0.098, 0.154, 0.098, 0.154,
        0.098, 0.154, 0.098, 0.154,
        0.098, 0.098, 0.098, 0.098,
    ], dtype=np.float32)

    _POLICY_KP = np.array([
        7.80, 50.97, 32.51, 50.97, 32.51, 7.80,
        50.97, 32.51, 50.97, 32.51,
        50.97, 32.51, 50.97, 32.51,
        50.97, 32.51, 50.97, 32.51,
        50.97, 50.97, 50.97, 50.97,
    ], dtype=np.float32)

    _POLICY_KD = np.array([
        0.50, 3.24, 2.07, 3.24, 2.07, 0.50,
        3.24, 2.07, 3.24, 2.07,
        3.24, 2.07, 3.24, 2.07,
        3.24, 2.07, 3.24, 2.07,
        3.24, 3.24, 3.24, 3.24,
    ], dtype=np.float32)

    _POLICY_ARMATURE = np.array([
        0.001976, 0.01291, 0.008234, 0.01291, 0.008234, 0.001976,
        0.01291, 0.008234, 0.01291, 0.008234,
        0.01291, 0.008234, 0.01291, 0.008234,
        0.01291, 0.008234, 0.01291, 0.008234,
        0.01291, 0.01291, 0.01291, 0.01291,
    ], dtype=np.float32)

    DEFAULT_POS = POLICY_DEFAULT_POS[POLICY_TO_GENESIS]
    ACTION_SCALE = _POLICY_ACTION_SCALE[POLICY_TO_GENESIS]
    KP = _POLICY_KP[POLICY_TO_GENESIS]
    KD = _POLICY_KD[POLICY_TO_GENESIS]
    ARMATURE = _POLICY_ARMATURE[POLICY_TO_GENESIS]

    gs.init(backend=gs.gpu, performance_mode=True, logging_level="warning")

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.0, -3.0, 1.5),
            camera_lookat=(0.0, 0.0, 0.4),
            camera_fov=35,
            res=(960, 640),
            max_FPS=60,
            enable_help_text=False,
        ),
        sim_options=gs.options.SimOptions(dt=0.02, substeps=4),
        rigid_options=gs.options.RigidOptions(enable_neutral_collision=True),
        show_viewer=args.viewer,
    )

    robot = PI(PIConfig(
        initial_pos=np.array([0.0, 0.0, 0.351], dtype=np.float32),
        kp=KP,
        kv=KD,
        force_range=np.array([
            [-20.0] * NUM_GENESIS_JOINTS,
            [ 20.0] * NUM_GENESIS_JOINTS,
        ], dtype=np.float32),
    ), scene)
    ball_x, ball_y = _random_ball_spawn(args.mode, rng)
    ball = scene.add_entity(
        morph=gs.morphs.Sphere(radius=BALL_RADIUS, pos=(ball_x, ball_y, BALL_RADIUS + 0.05)),
        material=gs.materials.Rigid(friction=0.8),
    )
    scene.add_entity(morph=gs.morphs.Plane())

    robot.build()
    scene.build(n_envs=args.num_envs, env_spacing=(3.0, 3.0))
    robot.config()

    dofs_idx = robot.dofs_idx_local

    # Set armature to match training (critical for dynamics stability)
    robot.robot.set_dofs_armature(ARMATURE, dofs_idx_local=dofs_idx)

    # Zero URDF joint damping — training uses damping=0 on the drive
    robot.robot.set_dofs_damping(
        np.zeros(NUM_GENESIS_JOINTS, dtype=np.float32),
        dofs_idx_local=dofs_idx,
    )

    # ball config
    ball.set_mass(0.16)
    ball.set_dofs_damping(2.0, dofs_idx_local=(3, 4, 5))

    # load policy
    policy = OnnxPolicy(args.model_dir)

    # observation builder
    obs_builder = ObservationBuilder(num_envs=args.num_envs)

    # command builder (per-env: share one for single-env, vectorize later)
    cmd_builder = SoccerCommandBuilder(
        mode=args.mode,
        vel_cmd=(args.vel_x, args.vel_y, args.vel_yaw),
        kick_speed=args.kick_speed,
        kick_dir_deg=args.kick_dir_deg,
    )

    envs_idx = torch.arange(args.num_envs, dtype=torch.long, device=gs.device)

    # ---- reset ----
    robot.reset(envs_idx=envs_idx)
    ball_z = BALL_RADIUS + 0.05
    ball_pos_init = torch.zeros(args.num_envs, 3, device=gs.device)
    ball_pos_init[:, 0] = ball_x
    ball_pos_init[:, 1] = ball_y
    ball_pos_init[:, 2] = ball_z
    ball.set_pos(envs_idx=envs_idx, pos=ball_pos_init)
    ball.zero_all_dofs_velocity(envs_idx=envs_idx)

    # init joint positions to default (bent knees)
    default_q = torch.from_numpy(DEFAULT_POS).unsqueeze(0).broadcast_to(
        (args.num_envs, -1)).to(gs.device)
    robot.robot.set_dofs_position(default_q, dofs_idx_local=dofs_idx, envs_idx=envs_idx)
    robot.robot.set_dofs_velocity(
        torch.zeros((args.num_envs, len(DEFAULT_POS)), device=gs.device),
        dofs_idx_local=dofs_idx, envs_idx=envs_idx,
    )

    # init observation history
    init_cmd = np.zeros((args.num_envs, 7), dtype=np.float32)
    obs_builder.reset(envs_idx=np.arange(args.num_envs), command=init_cmd)
    last_action = np.zeros((args.num_envs, NUM_POLICY_JOINTS), dtype=np.float32)

    # Settle: hold default pose for N steps to reach equilibrium
    settle_target = torch.from_numpy(DEFAULT_POS).unsqueeze(0).broadcast_to(
        (args.num_envs, -1)).to(gs.device)
    for _ in range(50):
        robot.step(settle_target)
        scene.step()

    print(f"mode={args.mode}, kick_speed={args.kick_speed}, "
          f"ball=({ball_x:.2f},{ball_y:.2f})")

    # ---- loop ----
    for step in range(args.steps):
        # 1. read state
        state = robot.get_state(envs_idx=envs_idx)
        joint_pos_genesis = state["dofs_pos"].cpu().numpy()   # (B, 20)
        joint_vel_genesis = state["dofs_vel"].cpu().numpy()   # (B, 20)
        body_quat = state["body_quat"].cpu().numpy()          # (B, 4) wxyz
        body_ang_vel = state["body_ang_vel"].cpu().numpy()    # (B, 3) body-frame
        body_pos = state["body_pos"].cpu().numpy()            # (B, 3)
        ball_pos_now = ball.get_pos(envs_idx=envs_idx).cpu().numpy()   # (B, 3)

        # 2. map 20-dim genesis → 22-dim policy (pad head joints with zeros)
        joint_pos_policy = np.zeros((args.num_envs, NUM_POLICY_JOINTS), dtype=np.float32)
        joint_vel_policy = np.zeros((args.num_envs, NUM_POLICY_JOINTS), dtype=np.float32)
        joint_pos_policy[:, POLICY_TO_GENESIS] = joint_pos_genesis
        joint_vel_policy[:, POLICY_TO_GENESIS] = joint_vel_genesis
        joint_pos_rel = joint_pos_policy - POLICY_DEFAULT_POS
        # 3. projected gravity
        proj_grav = ObservationBuilder.quat_to_projected_gravity(body_quat)

        # 4. compute soccer command (env 0 only for now, broadcast)
        w, x, y, z = body_quat[0]
        robot_yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        cmd = cmd_builder.compute(body_pos[0], robot_yaw, ball_pos_now[0])
        cmd_batch = np.broadcast_to(cmd, (args.num_envs, 7)).copy()

        # 5. push observation
        obs_builder.push(
            ang_vel=body_ang_vel,
            proj_gravity=proj_grav,
            command=cmd_batch,
            joint_pos_rel=joint_pos_rel,
            joint_vel=joint_vel_policy,
            actions=last_action,
        )

        # 6. policy inference
        obs = obs_builder.get()  # (B, 632)
        action_policy = policy.infer(obs)  # (B, 22)

        # 7. map 22-dim policy → 20-dim genesis
        action_genesis = action_policy[:, POLICY_TO_GENESIS]

        # 8. compute target positions and apply
        target_q = DEFAULT_POS + action_genesis * ACTION_SCALE
        target_t = torch.from_numpy(target_q).to(gs.device)
        robot.step(target_t)

        # 9. step physics
        scene.step()

        last_action = action_policy

        if step % 50 == 0:
            bp = ball_pos_now[0]
            bv = ball.get_vel(envs_idx=envs_idx).cpu().numpy()
            print(
                f"  t={step:4d}  "
                f"robot=[{body_pos[0,0]:.2f},{body_pos[0,1]:.2f},{body_pos[0,2]:.2f}]  "
                f"ball=[{bp[0]:.2f},{bp[1]:.2f}]  "
                f"|v_ball|={np.linalg.norm(bv[0,:3]):.2f}  "
                f"cmd=[{cmd[0]:.2f},{cmd[1]:.2f},{cmd[2]:.2f},"
                f"{cmd[3]:.2f},{cmd[4]:.2f},{cmd[5]:.2f},{cmd[6]:.2f}]"
            )

    print(f"\ndone. steps={args.steps}")


if __name__ == "__main__":
    main()
