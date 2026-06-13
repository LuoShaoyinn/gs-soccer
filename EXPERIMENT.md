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

The original policy was trained in IsaacLab with the following recipe
(from `agent.yaml` and `env.yaml`):

### 3.1 Algorithm: WasabiPPO + AMP

- **Base**: PPO with adaptive learning rate (targets KL divergence of 0.01)
- **AMP**: Adversarial Motion Priors — a discriminator rewards motions that
  match mocap data, added to the task reward
- **Discriminator**: 2-layer MLP (1024→512), ReLU, gradient penalty (coef 5.0)
- **Discriminator reward coef**: 0.25 (quadratic form)
- **Entropy coef**: 0.006

### 3.2 PPO Hyperparameters

| Parameter              | Value     |
|------------------------|-----------|
| Clip param             | 0.2       |
| Num learning epochs    | 5         |
| Num mini batches       | 4         |
| Steps per env          | 24        |
| Gamma                  | 0.99      |
| Lambda (GAE)           | 0.95      |
| Max grad norm          | 1.0       |
| Learning rate          | 1e-3 (AdamW, adaptive) |
| Init noise std         | 0.6       |

### 3.3 Simulation Setup

| Parameter         | Value          |
|-------------------|----------------|
| Physics engine    | PhysX (Isaac)  |
| dt                | 0.005 s        |
| Decimation        | 4 (control at 50 Hz) |
| Num envs          | 4096           |
| Env spacing       | 10 m           |
| Max iterations    | 300,000        |

### 3.4 Motion Reference (AMP)

- **Source**: AMASS soccer walk/run mocap data, retargeted to PiPlus
- **Frame rate**: 50 Hz (matched to control rate)
- **Dual buffer**: `soccer` buffer for locomotion, `walk_init` buffer for
  reset initialization
- **Symmetric augmentation**: Left-right mirror mapping doubles effective
  data, with sign flips on roll/yaw joints

### 3.5 Curriculum: Kick Role Assignment

At each reset, environments are split into:
- **35%** pure walking (`walk`)
- **35%** immediate kick (`kick`)
- **30%** approach-then-kick (`approach_kick`)

This ensures the policy learns all three behaviors simultaneously.

### 3.6 Domain Randomization

Applied at reset (IsaacLab events):

| Parameter              | Range          |
|------------------------|----------------|
| Static friction        | [0.2, 2.0]     |
| Dynamic friction       | [0.2, 2.0]     |
| Restitution            | [0.0, 0.5]     |
| Link mass shift        | randomized     |
| COM shift              | randomized     |
| KP ratio               | randomized     |
| KV ratio               | randomized     |

### 3.7 Observation Noise

Uniform additive noise applied to policy observations during training:

| Term          | Noise range |
|---------------|-------------|
| base_ang_vel  | ±0.2        |
| proj_gravity  | ±0.05       |
| soccer_cmd    | ±0.05       |
| joint_pos     | ±0.01       |
| joint_vel     | ±0.5        |

This makes the policy robust to sensor noise, which is critical for sim2sim
transfer where IMU readings differ from training.

### 3.8 Ball Properties (Training)

| Property          | Value  |
|-------------------|--------|
| Radius            | 0.07 m |
| Mass              | 0.25 kg |
| Linear damping    | 0.05   |
| Angular damping   | 0.05   |
| Static friction   | 0.8    |
| Dynamic friction  | 0.6    |
| Restitution       | 0.82   |

**Note**: Our sim2sim uses mass=0.16 kg and zero damping as a closer match
to a real size-1 soccer ball. This is a deliberate sim2sim adjustment.

### 3.9 Actuator Model: HTMotor

Training used a custom actuator model (`HTMotor`) with a torque-speed curve:

```
Groups: legs/feet:  curve = [-0.0141, -0.0709, 6.2756], max_torque=20, max_vel=6.0
        arms:       curve = [-0.1284, -0.6996, 19.83],  max_torque=10, max_vel=20.0
```

The sim2sim uses ideal PD control (no torque-speed curve) since Genesis
doesn't have the HTMotor model. This is a known sim2sim gap.

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
