# PiPlus Soccer Sim2Sim Experiment Notes

This document covers how to run the Genesis-based sim2sim transfer, the key
algorithm details that must be matched exactly, and the training recipe used
for the original policy.

---

## 1. Running the Sim2Sim

### Prerequisites

```bash
uv sync --extra rocm   # or --extra cu121 for NVIDIA
```

The converted torch checkpoint must exist at `runs/pi_plus_actor.pt`. If you
only have the original ONNX (`refs/.../actor.onnx`), convert it first — see
section 2.5.

### Commands

```bash
# Walk forward at 0.5 m/s, headless
uv run --extra rocm python main.py --mode walk --vel-x 0.5 --no-viewer

# Walk with GUI viewer
uv run --extra rocm python main.py --mode walk --vel-x 0.5

# Approach ball then kick at 2.0 m/s toward 0 degrees
uv run --extra rocm python main.py --mode approach_kick --kick-speed 2.0 --kick-dir-deg 0

# Direct kick (robot spawns close to ball)
uv run --extra rocm python main.py --mode kick --kick-speed 2.0

# Multi-env stress test
uv run --extra rocm python main.py --mode walk --num-envs 64 --steps 2500 --no-viewer
```

### CLI Arguments

| Flag             | Default | Description                                |
|------------------|---------|--------------------------------------------|
| `--mode`         | walk    | `walk`, `kick`, or `approach_kick`         |
| `--vel-x`        | 0.5     | Forward velocity command (m/s)             |
| `--vel-y`        | 0.0     | Lateral velocity command (m/s)             |
| `--vel-yaw`      | 0.0     | Yaw rate command (rad/s)                   |
| `--kick-speed`   | 2.0     | Desired ball exit speed (m/s)              |
| `--kick-dir-deg` | 0.0     | Kick direction in world frame (degrees)    |
| `--num-envs`     | 1       | Number of parallel environments            |
| `--steps`        | 2500    | Policy steps after settle                  |
| `--no-viewer`    | off     | Disable GUI (set `PYOPENGL_PLATFORM=egl`)  |

### Console Output Format

```
t= 200  robot=[3.34,0.03,0.34]  ball=[4.10,-0.22]  |v_ball|=0.00  cmd=[0.50,0.00,0.00,0.79,-0.15,0.00,0.00]
```

- `robot`: base link position [x, y, z] — z should stay ~0.35
- `ball`: ball position [x, y]
- `|v_ball|`: ball speed (nonzero = kicked)
- `cmd`: soccer command [vx, vy, wz, ball_x_body, ball_y_body, kick_dir_rel, kick_speed]

---

## 2. Algorithm Details

These details must match the training environment exactly. Any mismatch
causes the policy to produce garbage actions and the robot will fall.

### 2.1 Joint Mapping

The policy operates on **22 joints** in a specific order. The Genesis URDF
has 21 joints (index 0 = head_yaw is excluded from policy control). The
mapping is:

```
Policy index → Genesis URDF index:
[1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
```

Defined in `algorithm/actor.py:POLICY_TO_GENESIS`.

### 2.2 Observation (632-dim, term-major history)

The observation is a stack of **8 frames** of proprioceptive data, laid out
in **term-major** order (all frames of term 0, then all frames of term 1,
etc.):

```
[ ang_vel(8×3) | gravity(8×3) | cmd(8×7) | jpos(8×22) | jvel(8×22) | act(8×22) ]
  = 24 + 24 + 56 + 176 + 176 + 176 = 632
```

| Term          | Size  | Scale | Source                          |
|---------------|-------|-------|---------------------------------|
| base_ang_vel  | 3     | 0.25  | IMU angular velocity            |
| proj_gravity  | 3     | 1.0   | Gravity in body frame           |
| soccer_cmd    | 7     | 1.0   | [vx, vy, wz, bx, by, kdir, ksp] |
| joint_pos_rel | 22    | 1.0   | Joint pos - default pos         |
| joint_vel     | 22    | 0.05  | Joint velocity                  |
| actions       | 22    | 1.0   | Last policy action (raw)        |

**Critical**: term-major layout means `reshape(B, H, -1)` then flatten, NOT
`reshape(B, -1, H)`. Frame-major layout causes the robot to fall immediately.

### 2.3 Action Mapping

```python
target_q = default_pos + action[policy_idx] * action_scale[policy_idx]
```

The action scale varies per joint group (from training `env.yaml`):

| Joint group    | Scale   |
|----------------|---------|
| Thigh/hip/calf| 0.098   |
| Ankle          | 0.098   |
| Head           | 0.096   |
| Shoulder/arm   | 0.154   |

Default positions (policy order):

```python
[0, -0.25, 0, -0.25, 0, 0,    # legs (hip_pitch, calf pattern)
 0, 0, 0, 0,                    # arms upper
 0, 0, 0, 0,                    # arms lower
 0.65, 0, 0.65, 0,              # calf, ankle
 -0.4, -0.4, 0, 0]              # ankle roll, etc.
```

### 2.4 Soccer Command Builder

The 7-dim command depends on `--mode`:

- **walk**: `[vx, vy, wz, ball_x_body, ball_y_body, 0, 0]`
  - vx/vy/wz are the user velocity command
  - ball position in body frame is always filled

- **approach_kick**: Walk toward ball using proportional yaw control until
  within `kick_zone_radius` (1.0 m), then switch to kick command.
  - Pre-kick: `vx = cos(yaw_err) * 1.2`, `wz = clamp(2.0 * yaw_err, ±1.5)`
  - Post-kick: `[0, 0, 0, bx, by, kick_dir_rel, kick_speed]`

- **kick**: Immediately enter kick mode. Computes kick direction relative to
  robot yaw: `kick_dir_rel = wrap(angle_to_target - robot_yaw)`.

### 2.5 Policy Network

```
ActorMLP: Linear(632→512) → ELU → Linear(512→256) → ELU → Linear(256→128) → ELU → Linear(128→22)
```

Converted from ONNX to a torch state_dict at `runs/pi_plus_actor.pt`. The
conversion was verified to match ONNX output within 1.5e-5 max difference.

### 2.6 PD Gains & Armature

From the training config `env.yaml`:

| Joint group    | KP     | KV    | Armature  |
|----------------|--------|-------|-----------|
| Legs (thigh/hip/calf) | 50.97 | 3.24 | 0.01291 |
| Ankles         | 50.97  | 3.24  | 0.01291   |
| Arms (shoulder/elbow) | 32.51 | 2.07 | 0.008234 |
| Head           | 7.80   | 0.50  | 0.001976  |

**Note**: The Genesis URDF sets `damping=0.08` on all joints. This must be
overridden to `0.0` (PD damping is handled by KV, not joint damping).

These are set via `RobotConfig.armature` and `RobotConfig.damping` fields
(implemented on `main` branch).

### 2.7 Settle Phase

Before running the policy, the robot must settle:
1. **50 steps** holding `default_pos` at 50 Hz (1 second) — lets the robot
   drop to the ground and stabilize
2. **1 prime step**: `env.step(zeros_22)` — refreshes stale observation
   buffers with actual sensor data

Skipping settle or prime causes the history buffer to be all-zeros, which
the policy interprets as extreme tilt and produces corrective actions that
topple the robot.

### 2.8 Action History Timing

The observation's `actions` term at step `t` must contain the action from
step `t-1`, NOT step `t`. This is implemented via a `_pending_action`
mechanism:

1. `preprocess_action(action_t)` → saves to `_pending_action`, returns
   mapped joint targets
2. `build_observation()` → pushes `_last_action` (from t-1) into history,
   then promotes `_pending_action` → `_last_action`

This one-step delay matches IsaacLab's `last_action` observation term.

---

## 3. Training Techniques

The original policy was trained in **IsaacLab** (Isaac Sim 5.1, PhysX) using
a custom framework called **HT_lab** / **instinctlab**. All details below are
extracted from `refs/.../models/params/env.yaml` and `agent.yaml`.

### 3.1 Algorithm: WasabiPPO + AMP

- **Base**: PPO with adaptive learning rate (targets KL divergence of 0.01)
- **AMP**: Adversarial Motion Priors — a discriminator rewards motions that
  match mocap data, added to the task reward (coef 0.25, quadratic form)
- **Discriminator**: 2-layer MLP (1024→512), ReLU, gradient penalty coef 5.0,
  weight decay 3e-4 (body) / 4e-2 (logit), lr 1e-4 (AdamW)
- **Entropy coef**: 0.006
- **Experiment**: `pi_plus_soccer_target_amp`

### 3.2 PPO Hyperparameters

| Parameter              | Value              |
|------------------------|--------------------|
| Clip param             | 0.2                |
| Num learning epochs    | 5                  |
| Num mini batches       | 4                  |
| Steps per env          | 24                 |
| Gamma                  | 0.99               |
| Lambda (GAE)           | 0.95               |
| Max grad norm          | 1.0                |
| Learning rate          | 1e-3 (AdamW, adaptive) |
| Init noise std         | 0.6                |
| Clip min std           | 1e-12              |
| Advantage mixing       | 1.0                |
| Value loss coef        | 1.0 (clipped)      |
| Save interval          | 1000 iters         |
| Max iterations         | 300,000            |

### 3.3 Simulation Setup

| Parameter              | Value              |
|------------------------|--------------------|
| Physics engine         | PhysX (Isaac Sim)  |
| dt                     | 0.005 s            |
| Decimation             | 4 (control at 50 Hz / 20 ms) |
| Num envs               | 4096               |
| Env spacing            | 10 m               |
| Episode length         | 10 s (500 steps)   |
| Seed                   | 42                 |
| Self-collision         | false              |
| Filter collisions      | true (between envs)|
| Solver pos iterations  | 8                  |
| Solver vel iterations  | 4                  |
| Bounce threshold vel   | 0.5 m/s            |
| Soft joint pos limit   | 0.9                |
| CCD                    | disabled           |

### 3.4 Actuator Derivation

PD gains are derived from per-joint armature via second-order system formulas:

```
Natural frequency:  ωn = 10.0 × 2π = 62.83 rad/s
Damping ratio:      ζ = 2.0
Stiffness (KP):     k = armature × ωn²
Damping (KV):       d = 2 × ζ × armature × ωn
```

Three motor groups based on motor size:

| Motor     | Armature  | KP = k     | KV = d    | Effort limit | Action scale = 0.25×effort/kp |
|-----------|-----------|------------|-----------|--------------|-------------------------------|
| 5047 (legs/ankles) | 0.01291 | 50.967 | 3.245 | 20.0 N·m | 0.0981 |
| 4438 (arms)        | 0.008234 | 32.507 | 2.069 | 20.0 N·m | 0.1538 |
| 3536 (head)        | 0.001976 | 7.801 | 0.497 | 3.0 N·m  | 0.0961 |

Head uses `ImplicitActuator` (no torque-speed curve); legs and arms use
`HTMotor` with torque-speed saturation curves:

| Group | curve_a | curve_b | curve_c | max_torque | max_vel |
|-------|---------|---------|---------|------------|---------|
| Legs  | -0.0141 | -0.0709 | 6.2756  | 20.0       | 6.0     |
| Arms  | -0.1284 | -0.6996 | 19.833  | 10.0       | 20.0    |

### 3.5 Motion Reference (AMP)

- **Source**: AMASS soccer walk/run mocap data, retargeted to PiPlus 22-DOF
- **Frame rate**: 50 Hz (matched to control rate), interpolated bilinearly
- **Assumed source framerate**: 120 Hz
- **Dual buffer system**:
  - `soccer` buffer: main locomotion clips from `soccer_walk_run_piplus`
  - `walk_init` buffer: initialization clips from `soccer_init_piplus`
- **Motion start**: sampled from middle 0–50% of clip
- **Velocity estimation**: frontward differencing
- **Link-of-interest tracking**: 14 links (base, shoulders, elbows, wrists,
  hips, calves, ankles)

### 3.6 Symmetric Augmentation

Left-right mirror doubles effective training data. Joint mapping swaps
left/right pairs; sign flips applied to roll/yaw joints:

```
joint_mapping:     [0,3,4,1,2,5, 8,9,6,7, 12,13,10,11, 16,17,14,15, 19,18, 21,20]
sign_flip:         [-,+5, +5, +5, -5, -5, -5, -5, -5, -5, +4, +4]
```

Indices that flip sign: 0 (base yaw), 6-13 (hip roll + ankle roll),
19-20 (head yaw, shoulder roll).

### 3.7 Kick Role Assignment (Curriculum)

At each environment reset, roles are assigned via `assign_kick_role`:

| Role           | Fraction | Description                              |
|----------------|----------|------------------------------------------|
| walk           | 35%      | Pure locomotion, ball far away           |
| kick           | 35%      | Ball close, immediate kick command       |
| approach_kick  | 30%      | Walk toward ball, kick when in range     |

### 3.8 Soccer Command (SoccerCommand)

The 7-dim command is managed by `HT_lab.commands.soccer_command:SoccerCommand`:

| Parameter                | Value            |
|--------------------------|------------------|
| Resampling time          | 10 s (full episode) |
| Forward probability      | 0.8              |
| Rel standing envs        | 0.2              |
| Rel rotation-only envs   | 0.15             |
| Rel lateral-only envs    | 0.15             |

Velocity command ranges:

| Axis         | Range            |
|--------------|------------------|
| lin_vel_x    | [-0.6, 1.5] m/s  |
| lin_vel_y    | [-0.6, 0.6] m/s  |
| ang_vel_z    | [-1.2, 1.2] rad/s|
| forward_cone | [-90°, 90°]      |

Kick speed sampling:

| Category  | Range (m/s) | Fraction |
|-----------|-------------|----------|
| zero      | —           | 0%       |
| soft      | [0.5, 2.0]  | 20%      |
| hard      | [2.0, 5.0]  | 80%      |

Approach-kick parameters:

| Parameter              | Value  |
|------------------------|--------|
| Kick zone radius       | 1.0 m  |
| Approach speed (vx)    | 1.2 m/s|
| Approach yaw gain (wz) | 2.0    |
| Approach max yaw rate  | 1.5 rad/s |
| Clear distance         | 0.2 m  |
| Contact force threshold| 0.1 N  |

### 3.9 Reward Configuration

#### Task Rewards (positive)

| Reward term                  | Weight | Scope    | Key params                          |
|------------------------------|--------|----------|-------------------------------------|
| `is_alive`                   | +1.0   | all      | Survival bonus                      |
| `track_lin_vel_xy_exp`       | +2.0   | walk     | exp kernel, std=0.5                 |
| `track_ang_vel_z_exp`        | +2.0   | walk     | exp kernel, std=0.5                 |
| `feet_air_time_walk`         | +0.5   | walk     | vel_threshold=0.15                  |
| `feet_close_xy`              | +0.4   | all      | gaussian, threshold=0.12, std=0.224 |
| `lower_limb_symmetry_walk`   | +500.0 | walk     | alpha=0.005, vel/lateral/ang thresh |
| `kick_direction_speed`       | +20.0  | kick     | exp kernel, std=1.0, speed_thresh=0.3 |
| `face_ball`                  | +1.0   | kick     | exp kernel, std=0.6, freeze=30°     |
| `approach_ball`              | +1.0   | kick     | target_speed=0.7, std=0.5, act_dist=0.5 |

#### Regularization Penalties (negative)

| Reward term                  | Weight   | Scope    | Key params                          |
|------------------------------|----------|----------|-------------------------------------|
| `flat_orientation_l2`        | -6.0     | all      | Gravity alignment penalty           |
| `pelvis_orientation_l2_walk` | -6.0     | walk     | Base link orientation               |
| `undesired_contacts`         | -1.0     | all      | Non-foot contact, thresh=1.0        |
| `dof_pos_limits`             | -1.0     | all      | Joint position limits               |
| `dof_vel_limits`             | -1.0     | all      | soft_ratio=0.9                      |
| `post_clear_default_pose`    | -1.0     | all      | Deviation from default after clear  |
| `heading_error_walk`         | -1.0     | walk     | Heading vs command direction        |
| `joint_deviation_hip`        | -0.5     | all      | Hip pitch + roll (squared)          |
| `dont_wait_walk`             | -0.5     | walk     | Penalize no movement toward ball    |
| `feet_slide`                 | -0.4     | all      | Contact sliding, thresh=1.0         |
| `feet_flat_ori_walk`         | -0.4     | walk     | Feet flat on ground                 |
| `stand_still_walk`           | -0.3     | walk     | Penalize stillness at offset=4.0m   |
| `ball_z_speed`               | -0.2     | kick     | Vertical ball velocity              |
| `ang_vel_xy_l2`              | -0.1     | all      | Horizontal angular velocity         |
| `torque_limits`              | -0.01    | all      | limit_ratio=0.8                     |
| `action_rate_l2`             | -0.01    | all      | Action smoothness                   |
| `dof_vel_l2`                 | -1e-4    | all      | Joint velocity regularization       |
| `energy`                      | -5e-5   | legs     | Motor power², normalized by stiffness |
| `dof_torques_l2`             | -1.5e-7  | legs     | Joint torque (hip/thigh/ankle)      |
| `dof_acc_l2`                 | -1.25e-7 | all      | Joint acceleration                  |

Note: "walk" scope = applied only to walk-role envs; "kick" scope = kick-role
envs; "all" = all envs. The `lower_limb_symmetry_walk` weight of 500 is large
but scaled internally by alpha=0.005 and velocity gates.

#### AMP Reward

In addition to the task rewards above, the AMP discriminator adds a style
reward (coef 0.25, quadratic form) that encourages motions matching the
mocap reference. The discriminator observation is a 10-frame history of
projected_gravity(3) + joint_pos_rel(22) + joint_vel(22) + base_lin_vel(3) +
base_ang_vel(3) = 530 dims per frame.

### 3.10 Termination Conditions

| Termination       | Type     | Condition                              |
|-------------------|----------|----------------------------------------|
| `time_out`        | timeout  | Episode reaches 10 s                   |
| `terrain_out_bound`| timeout | Robot moves >2 m beyond terrain bounds |
| `base_contact`    | failure  | Contact force >1.0 N on base_link,     |
|                   |          | head, thigh, hip, shoulder, elbow,     |
|                   |          | or upper_arm links                     |
| `root_height`     | failure  | Base height drops below 0.1 m          |

### 3.11 Domain Randomization

All applied at `startup` or `reset` (IsaacLab events):

| Parameter                  | Mode    | Distribution    | Range          |
|----------------------------|---------|-----------------|----------------|
| Robot static friction      | startup | uniform         | [0.2, 2.0]     |
| Robot dynamic friction     | startup | uniform         | [0.2, 2.0]     |
| Robot restitution          | startup | uniform         | [0.0, 0.5]     |
| Link mass scale            | startup | uniform         | [0.85, 1.15]   |
| Joint friction             | startup | gaussian scale  | [0.5, 1.5]     |
| Actuator stiffness (KP)    | startup | log_uniform     | [0.75, 1.25]   |
| Actuator damping (KV)      | startup | log_uniform     | [0.75, 1.25]   |
| COM shift (x, y, z)        | startup | uniform         | [-0.02, 0.02]  |
| Ball mass                  | startup | uniform (abs)   | [0.20, 0.28]   |
| Ball static friction       | startup | uniform         | [0.3, 1.0]     |
| Ball dynamic friction      | startup | uniform         | [0.2, 0.8]     |
| Ball restitution           | startup | uniform         | [0.4, 0.8]     |

### 3.12 Observation Noise

Uniform additive noise on policy observations during training (noise is NOT
applied to critic or AMP discriminator observations):

| Term             | Noise range |
|------------------|-------------|
| base_ang_vel     | ±0.2        |
| projected_gravity| ±0.05       |
| soccer_command   | ±0.05       |
| joint_pos        | ±0.01       |
| joint_vel        | ±0.5        |

### 3.13 Ball Reset Configuration

Ball spawn position depends on kick role:

| Mode          | r_min | r_max | Angle range | z     | Notes                    |
|---------------|-------|-------|-------------|-------|--------------------------|
| walk          | —     | —     | —           | 0.07  | xy_offset=[4.0,4.0], noise=0.2 |
| kick          | 0.4 m | 1.0 m | [0°, 0°]   | 0.07  | In front of robot        |
| approach_kick | 3.0 m | 4.0 m | —           | 0.07  | Lateral noise = 0        |

### 3.14 Ball Properties (Training)

| Property          | Value  |
|-------------------|--------|
| Radius            | 0.07 m |
| Mass (nominal)    | 0.25 kg (randomized [0.20, 0.28]) |
| Linear damping    | 0.05   |
| Angular damping   | 0.05   |
| Static friction   | 0.8    |
| Dynamic friction  | 0.6    |
| Restitution       | 0.82   |
| Friction combine  | average|
| Restitution combine| average|

**Sim2sim adjustments**: Our Genesis sim2sim uses mass=0.16 kg and zero
damping as a closer match to a real size-1 soccer ball. This is a deliberate
sim2sim gap from training.

### 3.15 Contact Sensors

Two contact sensor groups track robot interactions:

| Sensor              | Body filter             | History | Purpose               |
|---------------------|-------------------------|---------|-----------------------|
| `contact_forces`    | All robot links         | 3 frames| General contact, feet_slide, undesired_contact |
| `foot_ball_contact` | `.*_ankle_roll_link`    | 5 frames| Ball-foot contact detection (filtered to Ball only) |

### 3.16 AMP Discriminator Observation

The AMP discriminator uses a separate 10-frame history (different from the
policy's 8-frame):

```
10 × (projected_gravity(3) + joint_pos_rel(22) + joint_vel(22×0.05) + base_lin_vel(3) + base_ang_vel(3))
= 10 × 73 = 730 dims
```

Note: the AMP obs uses `projected_gravity` (from robot root) not
`imu_projected_gravity`, and includes `base_lin_vel` which the policy does
not see.

---

## Architecture Map

```
main.py                      Entry point — builds Env + Actor, calls actor.eval()
├── algorithm/actor.py       Actor(Algorithm): network, infer, map_action, eval loop
│   ├── ActorMLP             632→512→256→128→22, ELU
│   └── ActorConfig          model path, settle/max steps
├── models/sim2sim_soccer.py Sim2SimSoccerModel(Model): MDP definition
│   ├── build_observation    Term-major 8-frame history, _pending_action timing
│   ├── preprocess_action    Action → joint targets (default + action × scale)
│   ├── _compute_soccer_cmd  walk / approach_kick / kick command builder
│   └── build_info           Logging telemetry
├── envs/env.py              Env: flat environment (scene + robot + field + model)
├── fields/ball_field.py     BallField: ball entity, reset, physics props
├── robots/robot.py          Robot base: build, config, randomize, step, state
├── robots/pi.py             PI: 20-joint humanoid URDF config
└── runs/pi_plus_actor.pt    Torch checkpoint (converted from ONNX)
```
