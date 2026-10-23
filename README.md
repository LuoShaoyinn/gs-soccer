# gs-soccer

Genesis-based environment scaffolding for humanoid soccer simulation.

This branch keeps only the reusable, abstract framework surface. The physics
layer (`envs/`, `robots/`, `fields/`) is frozen; experiments fork and provide a
concrete `MDP`, an `Algorithm`, and the `main.py` entry point.

## Project structure

- `envs/env.py`: `Env` — single generic orchestrator owning the Genesis scene.
  Delegates observation / reward / termination / info to an injected `MDP`.
- `robots/`: robot definitions (`Robot`, `PI`, `MOS9`, `FloatingCameraRobot`).
- `fields/`: field definitions (`Field` -> `TerrainField` -> `BallField` —
  a plane, optionally with terrain, plus a ball).
- `MDPs/MDP.py`: `MDP` — abstract task contract (spaces, observation,
  reward, termination, truncation, info, and reset).
- `MDPs/dummy.py`: `DummyMDP` — a standing-task example showing how to
  subclass `MDP` and implement the full contract (including `reset`).
- `algorithm/algorithm.py`: `Algorithm` — abstract base with `train()` / `eval()`.
- `main.py`: composition root. Selects robot + field + MDP (+ algorithm) and
  runs. The shipped template renders the viewer with a standing robot.
- `assets/`: robot URDFs and meshes.

## Core abstractions

- `Env`: owns the Genesis scene; handles stepping, reset, observation, reward,
  termination, truncation, and info flow through a supplied `MDP`.
- `MDP` (formerly `Model` — renamed): abstract task contract —
  observation/action spaces, observation, reward, termination, truncation,
  and info. The `Env` no longer resets the robot/field itself; instead the
  MDP must provide `reset(envs_idx, robot_reset_fn, field_reset_fn)` and own
  the task-level reset logic, calling the injected `robot_reset_fn` /
  `field_reset_fn` with the desired joint / base / ball poses. Concrete MDPs
  are provided per experiment (see `MDPs/dummy.py`, wired up in `main.py`).
- `Robot`: wraps a Genesis URDF robot; exposes actuator, reset, and state APIs.
- `Field`: owns field entities (and the ball, if any) and exposes reset/state.
- `Algorithm`: abstract training / evaluation interface.

## Experiment workflow

1. Fork a branch from `main`.
2. Provide a concrete `MDP` subclass (task: observation, reward, termination,
   and `reset()` — where to place the robot/ball; plus domain randomization
   such as `cmd_vel` / `target_ball_pos`). See `MDPs/dummy.py` for a template.
3. Provide a concrete `Algorithm` (`train()` / `eval()`).
4. Wire them together in `main.py`.
5. Leave `envs/`, `robots/`, `fields/` untouched.

## Run the template

```bash
uv run --extra rocm python main.py                  # viewer on, standing robot
uv run --extra rocm python main.py --no-viewer      # headless
```

> Note: `Model` has been renamed to `MDP`. New tasks subclass `MDP` (in
> `MDPs/`) and must implement `reset()`. `DummyMDP` in `MDPs/dummy.py` is the
> reference example.

## MOS9 tracking experiment

The demo and learning implementation is in `algorithm/mos9/`. It uses the
shipped TorchScript teacher on the GPU, with the teacher's 63 observations,
18 raw policy outputs converted to absolute joint targets, and 50 Hz policy /
500 Hz physics. No IK or gait
planner is involved. Both controllers use the shipped 50-step standing reset
protocol. The default trainable actor has the shipped teacher's ELU architecture
and weights; online initialization copies the pretrained IQL actor. Both preserve
the raw policy output in the observation before clipping joint targets;
reconstructing it from executed targets loses essential teacher feedback. Exploration
is 0.005 rad, outside the standing phase.

Commands are sampled independently in `[0.3, 0.6]^3` at reset. Success requires
five consecutive seconds of the calibrated v2 tracking/base-hold predicate,
completed within 20 seconds. Falls and missed deadlines fail. The predicate
uses a full two-second causal velocity window, xy errors at most
`0.05 + 0.70 * command`, yaw error at most `0.10 + 0.60 * command`, at least
50% progress along commanded xy and 25% commanded yaw, tilt at most 25 degrees,
height within 6 cm, and filtered vertical speed at most 0.15 m/s. This is coarse
tracking calibrated to the teacher, not precise tracking of all three axes.

Step rewards combine velocity tracking with bounded torque, tilt, angular-rate,
height, vertical-velocity, and yaw costs. Ordinary rewards are `[-0.01, 0.02]`;
a fall gives `-5` once and ends the episode. Success has no extra reward bonus.
The 63 inputs omit the hold timer, filter history, and deadline, so the learner
has a partially observed task despite sharing the teacher's inputs.

Collect 1024 complete successes with independent small terrain meshes and
randomized resets/dynamics in 512 environments:

```bash
uv run --extra rocm python collect_mos9_demos.py \
  --num-envs 512 --success-episodes 1024 \
  --output runs/mos9_demos/teacher_tracking_v2_rewrite
```

Add `--resume` to preserve and extend completed episodes. Unfinished attempts
are discarded on interruption; failed attempts never enter the demo dataset.
Each environment has a different 6 m terrain geometry through Genesis mesh
variants. This avoids both a huge terrain atlas and the shared heightfield of
multiple Genesis Terrain entities. Mesh contacts need their own teacher
validation; the nominal `--nominal` probe uses the original Terrain collider.

Start training from the existing 1024-demo FP16 dataset:

```bash
uv run --extra rocm python train_mos9_limit_action.py \
  --demo-dir runs/mos9_demos/teacher_tracking_v2_fp16 \
  --run-dir runs/mos9_limit_action/rewrite \
  --num-envs 4 --batch-size 4096 --hidden-dim 512 \
  --pretrain-updates 2000 --utd 64 \
  --capacity 100000000 --ram-blocks 4096 --gpu-cache-gib 16 \
  --takeover-probability 0.01 --checkpoint-every 4096
```

The learner uses the teacher's raw policy outputs as its 18-dimensional actions.
Clipped actuator targets are retained separately: identical clipped targets can
carry different raw feedback and produce different next observations. Existing
demos provide the raw action in their physical next observation, so import
recovers it exactly at their stored precision. The standing-phase flags guide
bootstrap actions and actor updates without adding teacher-incompatible inputs.

Existing FP32 and successor-compressed FP16 demos are accepted after task,
teacher, checksums, chronology, and terminal-label validation. New replay rows
store both states and actions as FP16; network inputs and rewards use FP32.
The **shipped `torch-block-replay`** owns disk, RAM, and the GPU pool. Its full
16 GiB pool is allocated on first append, even with a small dataset. Startup
checks the actual reservation. There is no alternative compact backend.

Every learner component samples that same physical store. SAC learns from
all actual transitions. IQL, action likeness, and the floor use only confirmed
successful contiguous teacher suffixes. Failed, incomplete, and student
suffixes cannot be confirmed. A factual-return annotation index is updated
when success becomes known; it contains no duplicate observations/actions.
Ten percent of seed episodes are held out from reference training, and
pretraining must pass return/base-head/action calibration before collection.

The 1000-head critics use linear outputs, with task bounds on Bellman targets.
IQL retains its expectile value and advantage-weighted actor updates, while
observed discounted returns anchor its critic at every horizon. The floor
constrains the critic on recorded raw teacher actions, with an additional measured
return anchor and an action-likeness constraint on unsupported proposals. It
is an empirical successful-teacher reference, not Q-star or a hard guarantee
that the learned actor succeeds. The online update retains this project's
deterministic fixed-exploration SAC variant; it does not implement automatic
entropy-temperature tuning.

The first half of environments stays autonomous. In the other half, a 1%
per-step rescue trigger switches to the teacher until the episode ends. There
is no takeover penalty. Only successful teacher suffixes become references.

TensorBoard starts during import/pretraining:

```bash
uv run --extra rocm tensorboard --logdir runs/mos9_limit_action/rewrite/tensorboard
```

`returns/` compares measured teacher-demo and autonomous-actor returns with
the actor critic at episode start. `reference/` reports held-out calibration;
`learning/` reports TD, floor, action-likeness, and actor losses. Reward terms
and replay reservation are logged separately. Every scalar is reduced with
`.mean().detach()` before logging.

Checkpoints are atomic, with immutable replay-index snapshots. They save every
4096 online transitions, every 250 pretraining updates, and on SIGINT/SIGTERM.
Resume into a new empty directory:

```bash
uv run --extra rocm python train_mos9_limit_action.py \
  --resume runs/mos9_limit_action/rewrite/latest.pt \
  --run-dir runs/mos9_limit_action/rewrite_resumed
```

Weights, optimizers, accepted replay prefix, reference/calibration labels,
update budget, and RNG state are restored. Physics restarts at episode reset;
unfinished prior suffixes remain ordinary replay and cannot acquire success
from the new episode. The old implementation's checkpoints are incompatible.
Existing demo and run artifacts are preserved.

If the converted teacher artifact is missing:

```bash
uv run --extra rocm --with onnx python -m tools.export_mos9_teacher
```

Validate the task/replay/learning contracts and the actual GPU pool:

```bash
uv run --extra rocm python -m unittest discover -s tests -v
uv run --extra rocm python -m tools.validate_mos9_replay
```
