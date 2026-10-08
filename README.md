# gs-soccer

## MOS9 planned walking experiment

Branch: `experiment/mos9-planned-walk`. The current stage builds and evaluates
a traditional walking teacher before learning a policy.

```bash
HIP_VISIBLE_DEVICES=0 uv run --no-sync --extra rocm python mos9_walk.py
HIP_VISIBLE_DEVICES=0 uv run --no-sync --extra rocm python mos9_walk.py --viewer --num-envs 1 --episodes 1
HIP_VISIBLE_DEVICES=0 uv run --no-sync --extra rocm python mos9_walk.py --flat-soles
```

The task lasts 500 steps at 50 Hz (10 seconds), ending early on a fall.
Reward is zero during the episode, -1 on a fall, and +1 at the horizon if
the robot has walked at least 0.3 m forward. Survival and distance are
reported separately so standing still cannot pass the walking evaluation.

The default field is uniform random terrain between 0 and 3 mm high,
interpolated from a 0.3 m grid onto 0.1 m samples. `--terrain flat` selects
a flat baseline; `--terrain-height` changes the terrain amplitude in metres.
Each scene has one ground surface. `--seed` controls terrain generation,
simulation, and reset randomization.

Both the build pose and every reset put the lowest URDF collision point
20 mm above the terrain maximum. This clearance is configurable with
`--spawn-clearance`; the robot settles under gravity and joint PD control.

The teacher plans a footstep/ZMP sequence, solves a bounded LIPM trajectory,
and computes leg IK using the MOS9 URDF, including joint limits and full-body
COM. Planned torso lean accommodates the asymmetric hip-roll limits.
Genesis 1.4.3 solves the full trajectory as a batch of 601 GPU IK problems.
Paired hip/ankle tracking and ankle attitude feedback adjust the targets.
`--online-ik` enables an experimental Cartesian feedback controller.
All movements are
ordinary joint position commands; no learned teacher or root-pose control
is used during simulation.

The task and robot/field adapters are branch-local. Shared `envs/`, `robots/`,
`fields/`, and URDF assets remain unchanged. Plans, traces, and evaluation
JSON files are written to ignored `runs/mos9_teacher/`.

`--flat-soles` generates an optional contact-model variant in `runs/`: two
boxes fitted to the visual sole plates replace the foot collision cylinders.
The original URDF, visuals, joint limits, and inertials are preserved.
See [MOS9_TEACHER_REPORT.md](MOS9_TEACHER_REPORT.md) for measured performance
and limitations; this teacher is still a baseline with falls.

API reference: [Genesis terrain documentation](https://genesis-world.readthedocs.io/en/latest/user_guide/getting_started/terrain.html).
The implementation is checked against the installed Genesis 1.4.3 source.

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

MOS9 limit-action training uses the branch-local learner copied from
`experiment/limit-action`, with factual teacher transitions (including falls)
for IQL and action likeness. Terminal states stop both reference and online
bootstrapping. The critic floor and rejected-action ranking remain enabled;
the fence does not override the actor at execution time. Each episode uses
teacher control with probability 0.5, otherwise the learned actor. This differs
from the earlier success-only reference dataset because the simple walking
teacher does not yet reliably complete the task.

```bash
.venv/bin/python launch_mos9_limit_action.py \
  --run-dir runs/mos9_limit_action/new_run --utd 64 --batch-size 256
```

Training defaults to the generated flat-sole contact model used by the workable
teacher, gentle 0–3 mm uniform terrain, 20 mm reset clearance, 50 Hz control,
and 500 steps per episode. Training uses the full `implicitfast` integrator,
CG constraints, 50 solver iterations, tolerance 1e-5, full scene resets at
episode boundaries, and 4000 Hz physics and limits learner
joint-target changes to 0.05 radians per control step. Use `--no-flat-soles` for original URDF contacts.
Rewards are zero except -1 on falling and +1 on completing 10 seconds with
at least 0.3 m forward progress. Gamma is 1 for this finite-horizon task.

Here UTD means primary SAC TD minibatch rows divided by newly collected online
transitions: updates accrue at `utd * num_envs / batch_size` per vector step,
with fractional credit carried forward. Thus UTD 64 and batch 256 produce one
update per four transitions, independent of environment count. Every combined
update also samples an IQL batch and a floor batch, and trains action likeness;
those auxiliary samples are excluded from the named UTD. Initial IQL/H
pretraining (500 updates after 2048 teacher transitions) is reported separately.
`metrics.jsonl` records achieved online UTD and factual episode outcomes.
`checkpoint.pt` stores networks, optimizers, replay, counters, update credit,
and Torch RNG state. The trainer saves on completion, interruption, and failure;
`failure.json` records errors. A checkpoint is a training artifact, not an exact
physics-state resume; automatic resume is not implemented. Existing run
folders are rejected to protect artifacts. Shared infrastructure is unchanged.

The initial approximate-integrator and Newton-solver runs encountered Genesis
constraint-force NaNs during teacher collection and are retained as failed
artifacts. The current `run3` uses CG and full scene resets, importing factual
teacher rows from `run2/checkpoint.pt` with `--teacher-checkpoint`. This option
imports teacher-only replay from a matching task/model, not optimizer or
physics state; outcomes are preserved. Physics solver changes are recorded
in the new config.

TensorBoard logs are written under each run directory in `tensorboard/`.
Training scalars use collected online transitions as the x-axis; offline
pretraining uses pretraining updates. Losses, target/achieved UTD, update
counts, throughput, and separate teacher/learner episode distance, duration,
fall and success rates are logged. Rolling episode metrics use the latest
100 episodes per controller. The default effective UTD is 64, matching
`experiment/limit-action` (4 updates × 4096 batch / 256 environments);
this run uses one environment and minibatch 256.

```bash
.venv/bin/tensorboard --logdir runs/mos9_limit_action --port 6006
```

The URSoccerLab ONNX walking policy can be tested here as a separate reference:

```sh
uv pip install --python .venv/bin/python onnxruntime
HIP_VISIBLE_DEVICES=0 .venv/bin/python test_mos9_onnx.py --render --episodes 3
```

The evaluator reads the local URSoccerLab policy and MOS9 actuator settings,
uses its 63-value observation, arms-down default pose, raw previous action,
and joint target scaling. It settles for one second, then evaluates the same
500-step walking task. JSON and PNG artifacts are saved under
`runs/mos9_teacher/onnx_reference/`. The default uses the original URDF contact
geometry; `--flat-soles` explicitly selects the experimental sole variant.
This learned reference is separate from the traditional planning teacher.

Fresh training with the validated ONNX reference and higher uniform terrain:

```sh
.venv/bin/python launch_mos9_limit_action.py \
  --run-dir runs/mos9_limit_action/onnx_10mm_utd64 \
  --teacher onnx --no-flat-soles --terrain-height 0.01 \
  --num-envs 1 --utd 64 --replay-capacity 3000000
```

The replay capacity includes initial teacher collection and online transitions.
Training stops at capacity, saves the final checkpoint and `status.json`, and
never overwrites rows. A final vector batch is truncated to the remaining space;
UTD uses the number of rows actually inserted. `--transitions` optionally sets
an earlier online limit; by default the buffer determines the stopping point.
The ONNX teacher holds its default pose during the first 50 steps **inside**
the 500-step horizon, then walks. ONNX inference uses one CPU thread; simulation
and learning run on GPU 0. Teacher state tracks the learner's actually applied
joint targets during learner control.
