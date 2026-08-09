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

## Kick sim2sim

The `tmp/kick-sim2sim` experiment loads the 646-input / 22-action actor from
`refs/kick_ball_0625`, including its command/history buffers and normalized
joint-position action mapping. The ONNX export is converted once to a native
Torch `actor.pt` checkpoint and inference then runs on the selected Torch
device (including ROCm), not through CPU ONNX Runtime.

```bash
uv run python kick_sim2sim.py --no-viewer
```

## Floor-IQL teacher environment

This branch adds `FloorIQLMDP` and `TeacherActor`. The floor task places the
ball at `x=1.0 m`, commands a forward kick, succeeds when `delta_x > 0.1 m`,
returns `1.0` on success and `-1/350` per non-success step, and truncates after
350 policy steps. Each reset randomizes robot `x/y` by ±0.05 m and yaw by ±10°;
the ball is placed 1 m along the robot's initial forward frame. Its
learner-facing observation is the 646-D teacher input plus a
`[ready, kicking, terminal]` phase one-hot; the teacher itself still receives
exactly 646 values.

The environment `info` contains `success`, `timeout`, forward-frame `delta_x`,
`phase`, `step_count`, and cumulative `episode_return`.

```bash
uv run python floor_iql.py --no-viewer --diagnostics
```

## Grounded vector SAC with teacher-action limit

`algorithm/grounded_sac/` cleanly separates the experiment into direct
observation MLP models, a single transition replay with a successful-teacher
suffix view, and the learner. It has no image encoder or state-familiarity
ensemble. SAC always controls the rollout.

The learner uses 350-head Q/V vectors (one head per policy step to timeout),
moving human-only IQL (`τ=0.7`), a
detached executable reference `min(Q_IQL,1,Q_IQL,2)(s, π_IQL(s))`, standard
vector SAC TD, squared reference-floor violations, and horizon ranking. A
moving action-likeness model `H(s,a)` is trained on successful teacher suffixes
and action-standardized Gaussian perturbations. When the current SAC proposal
has `H(s,a)<0.7`, the critic receives the relative constraint
`Q(s,a_IQL) >= Q(s,a_SAC) + 1/350`; `H` and IQL remain detached and the SAC
actor still optimizes only SAC Q. There is deliberately no intervention-start
buffer or direct `Q(s,a_H)>Q(s,a_SAC)` boundary loss.

The initial 2,000 successful teacher demonstrations fit and freeze the observation
and raw-action normalizers; IQL and `H` continue to train only from successful
teacher suffixes and never from ordinary SAC replay. A fake human intervention is
sticky: an exploring environment triggers with probability `p` each step, and
the teacher then controls that environment through the end of its episode.

```bash
# Optional explicit conversion; normal teacher loading performs it if actor.pt is absent.
uv run --extra rocm python convert_kick_actor.py refs/kick_ball_0625/.../actor.onnx

# Uses 256 Genesis environments: 128 autonomous and 128 rescue-enabled.
uv run --extra rocm python train_grounded_sac.py --no-viewer \
  --teacher-intervention-prob 0.01 \
  --updates-per-vector-step 4 \
  --replay-capacity 3000000 \
  --action-likeness-threshold 0.95
```

Replay checkpoints are directories containing 100 shards plus `manifest.pt`,
rather than one multi-gigabyte file. Initial successful teacher trajectories
are also saved separately under `<run>/teacher_buffer`. Reuse them without a
copy by symlinking that directory and passing the link as the restore source:

```bash
ln -s "$(realpath runs/limit_action_1/teacher_buffer)" runs/kick_teacher_buffer
uv run --extra rocm python train_grounded_sac.py --no-viewer \
  --restore-human-buffer runs/kick_teacher_buffer \
  --logdir runs/limit_action_2
```

Legacy monolithic `buffer_*.pt` snapshots remain loadable.
