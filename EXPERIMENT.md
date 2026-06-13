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

The total reward at each step is:

```
R_total = Σ_i  w_i · r_i(env)  +  w_amp · r_amp(discriminator)
```

where `r_i` are the per-term raw values (below), `w_i` are the weights from
`env.yaml`, and `r_amp` is the AMP style reward.

Many terms are **role-gated**: `_walk_envs` suffix means the term returns 0
for kick/approach_kick envs; `_kick_envs` means it returns 0 for walk envs.
Terms without a suffix apply to all envs.

Notation: `N` = num_envs, subscripts `i` range over joints or feet, `t`
is the current timestep. Unless noted, `||·||` is the L2 norm.

---

#### 3.9.1 Survival

**`is_alive`** — weight +1.0, all envs

```
r = 1.0  if not terminated
r = 0.0  if terminated
```

---

#### 3.9.2 Velocity Tracking (walk envs)

**`track_lin_vel_xy_exp`** — weight +2.0, std=0.5

Tracks commanded xy linear velocity in the body frame:

```
lin_vel_error = Σ_{k=x,y} (cmd_k − v_b,k)²
r = exp(−lin_vel_error / σ²)          where σ = 0.5
```

**`track_ang_vel_z_exp`** — weight +2.0, std=0.5

Tracks commanded yaw angular velocity:

```
ang_vel_error = (cmd_wz − ω_b,z)²
r = exp(−ang_vel_error / σ²)          where σ = 0.5
```

---

#### 3.9.3 Gait Rewards (walk envs)

**`feet_air_time_walk`** — weight +0.5, vel_threshold=0.15

Rewards longer air time, credited on first contact after flight:

```
for each foot f:
    if |cmd_vx| > vel_threshold:        # only when moving
        air_time_f = clamp(time_since_last_contact, 0, 0.5)
        reward_f   = air_time_f × first_contact_f
r = Σ_f reward_f
```

**`lower_limb_symmetry_walk`** — weight +500.0, alpha=0.005,
vel_threshold=0.15, lateral_threshold=0.1, ang_threshold=0.1

Penalizes gait asymmetry between left and right legs (hip/calf phase
difference), gated by forward motion:

```
gate = (|cmd_vx| > vel_threshold) ∧ (|cmd_vy| < lateral_threshold) ∧ (|cmd_wz| < ang_threshold)
symmetry_error = || left_hip_phase − right_hip_phase ||²
                 + || left_calf_phase − right_calf_phase ||²
r = −alpha × symmetry_error × gate     where alpha = 0.005
```

Effective weight = 500 × 0.005 = **2.5** per unit symmetry error.

---

#### 3.9.4 Kick Rewards (kick envs)

**`kick_direction_speed`** — weight +20.0, std=1.0, speed_threshold=0.3

Rewards the ball moving in the commanded direction at the commanded speed.
Evaluated when ball speed exceeds threshold:

```
gate = (|v_ball| > speed_threshold)
ball_dir = atan2(v_ball_y, v_ball_x)
dir_error = (ball_dir − target_dir)²
speed_error = (|v_ball| − kick_speed_cmd)²
r = exp(−(dir_error + speed_error) / σ²) × gate     where σ = 1.0
```

**`face_ball`** — weight +1.0, std=0.6, freeze_angle=30° (0.524 rad)

Rewards the robot facing the ball, with a freeze cutoff:

```
angle_to_ball = |atan2(ball_y_b, ball_x_b)|
r = exp(−angle_to_ball² / σ²)    if angle_to_ball < freeze_angle
r = 0                             otherwise             where σ = 0.6
```

**`approach_ball`** — weight +1.0, target_speed=0.7, std=0.5,
activate_distance=0.5

Rewards decreasing distance to ball (ball approaching robot in body frame):

```
gate = (dist_to_ball < activate_distance)
ball_speed_toward_robot = −d(dist)/dt      # closing rate
r = exp(−(ball_speed_toward_robot − target_speed)² / σ²) × gate
                                                              where σ = 0.5
```

**`ball_z_speed`** — weight −0.2

Penalizes vertical ball velocity (encourages ground-level kicks):

```
r = |v_ball_z|
```

---

#### 3.9.5 Orientation Penalties

**`flat_orientation_l2`** — weight −6.0, all envs

Penalizes non-flat base orientation via projected gravity xy components:

```
g = projected_gravity_b = R^T · [0, 0, −1]^T
r = g_x² + g_y²
```

**`pelvis_orientation_l2_walk`** — weight −6.0, walk envs

Identical to `flat_orientation_l2` but applied to walk-role envs only:

```
r = g_x² + g_y²        (walk envs only)
```

**`feet_flat_ori_walk`** — weight −0.4, walk envs

Penalizes foot links tilting (non-flat foot contact). Computed on
`.*_ankle_roll_link` projected gravity:

```
r = Σ_f |projected_gravity_f,x| + |projected_gravity_f,y|     (walk envs)
```

---

#### 3.9.6 Velocity/Smoothness Penalties

**`ang_vel_xy_l2`** — weight −0.1, all envs

Penalizes roll/pitch angular velocity:

```
r = ω_b,x² + ω_b,y²
```

**`action_rate_l2`** — weight −0.01, all envs

Penalizes rapid action changes between consecutive steps:

```
r = Σ_{j=1..22} (a_t,j − a_{t−1},j)²
```

---

#### 3.9.7 Joint Regularization

**`dof_vel_l2`** — weight −1e-4, all envs

```
r = Σ_j q̇_j²
```

**`dof_acc_l2`** — weight −1.25e-7, all envs

```
r = Σ_j q̈_j²         where q̈ = (q̇_t − q̇_{t−1}) / dt
```

**`dof_torques_l2`** — weight −1.5e-7, hip/thigh/ankle joints

```
r = Σ_{j ∈ hip,thigh,ankle} τ_j²
```

**`joint_deviation_hip`** — weight −0.5, hip_pitch + hip_roll

Squared deviation from default position:

```
r = Σ_{j ∈ hip_pitch, hip_roll} (q_j − q_default,j)²
```

**`post_clear_default_pose`** — weight −1.0, all envs

Penalizes full-body deviation from default pose after the ball has been
cleared (moved beyond `clear_distance = 0.2 m` from robot):

```
gate = (dist(robot, ball) > clear_distance)
r = Σ_j |q_j − q_default,j| × gate
```

---

#### 3.9.8 Limit Penalties

**`dof_pos_limits`** — weight −1.0, all envs

Sum of violations beyond soft joint position limits:

```
r = Σ_j [max(0, q_min_soft,j − q_j) + max(0, q_j − q_max_soft,j)]
where q_soft = 0.9 × q_limit
```

**`dof_vel_limits`** — weight −1.0, soft_ratio=0.9, all envs

Sum of violations beyond soft joint velocity limits (clipped at 1 rad/s):

```
r = Σ_j clamp(|q̇_j| − 0.9 × q̇_limit,j, 0, 1.0)
```

**`torque_limits`** (a.k.a. `applied_torque_limits_by_ratio`) — weight −0.01,
limit_ratio=0.8, all envs

Penalizes applied torques exceeding a ratio of the effort limit:

```
r = Σ_j max(0, |τ_applied,j| − 0.8 × τ_limit,j)
```

---

#### 3.9.9 Energy Penalty

**`energy`** (a.k.a. `motors_power_square`) — weight −5e-5, hip/thigh/ankle,
normalize_by_stiffness=true

Penalizes squared mechanical power, normalized by joint stiffness:

```
power_j = τ_j × q̇_j
r = Σ_{j ∈ legs} (power_j / k_j)²          where k_j = KP_j
```

The normalization by stiffness makes the penalty scale-invariant across
different actuator groups.

---

#### 3.9.10 Contact Penalties

**`undesired_contacts`** — weight −1.0, threshold=1.0 N, all envs except
ankle_roll_link

Counts body parts with contact force above threshold (excluding feet):

```
is_contact_i = max_t(net_force_i) > 1.0 N
r = Σ_{i ∉ ankle_roll} is_contact_i
```

**`feet_slide`** (a.k.a. `contact_slide`) — weight −0.4, threshold=1.0 N,
ankle_roll_link

Penalizes foot sliding velocity when foot is in contact:

```
contact_f = (net_force_f > 1.0 N)
slide_f   = √(v_f,x² + v_f,y²)           # foot xy speed
r = Σ_f √(slide_f) × contact_f
```

The square root provides a softer penalty for small slides.

---

#### 3.9.11 Locomotion Style Rewards

**`feet_close_xy`** — weight +0.4, threshold=0.12, std=0.224

Rewards feet being close together in xy plane (gaussian kernel):

```
foot_dist = ||foot_L_xy − foot_R_xy||
r = exp(−(max(0, foot_dist − threshold))² / σ²)    where σ = 0.224
```

**`heading_error_walk`** — weight −1.0, walk envs

Penalizes heading direction mismatch between commanded velocity and actual
walking direction:

```
heading_actual = atan2(v_y, v_x)
heading_cmd    = atan2(cmd_vy, cmd_vx)
r = |wrap(heading_actual − heading_cmd)|
```

**`dont_wait_walk`** — weight −0.5, walk envs

Penalizes standing still when there is a non-zero velocity command and the
robot is far from the ball:

```
gate = (||cmd_v|| > 0.1) ∧ (|v_actual| < threshold)
r = gate × 1.0
```

**`stand_still_walk`** — weight −0.3, offset=4.0, walk envs

Penalizes being too slow when the ball is far away (beyond offset distance):

```
gate = (dist_to_ball > offset)
r = gate × max(0, offset_speed − |v_actual|)
```

---

#### 3.9.12 AMP Style Reward

The AMP discriminator provides an additional reward based on adversarial
training against mocap reference motions:

```
D(s) = MLP_2layer(observation_amp)          # discriminator output ∈ ℝ
r_amp = clamp(−α · (D(s) − 1)² / 4, 0, 1)   # quadratic form, α = 0.25
```

This is the standard AMP reward formula from the original paper (Peng et al.,
2021). `s` is the 10-frame AMP observation (730 dims, see section 3.16).
The quadratic form gives maximum reward (1.0) when `D(s) = 1` (motion
indistinguishable from reference).

Discriminator training uses gradient penalty (coef 5.0) and weight decay
(3e-4 body, 4e-2 logits).

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
