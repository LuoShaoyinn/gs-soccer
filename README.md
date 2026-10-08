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
`experiment/limit-action`, with only success-confirmed teacher suffixes
for IQL and action likeness. Reference IQL matches the old success-only update (no done masking); SAC
uses physical terminal masking. The critic floor and rejected-action ranking remain enabled;
the fence does not override the actor at execution time. Every online episode starts with the learner. A fixed half of environments
allow per-step probabilistic teacher takeover, which lasts until episode end;
the other half remains autonomous. The first action is always the learner's. Initial demos must be complete successful teacher episodes. Failed rescues
and learner prefixes stay in ordinary replay, excluded from IQL/H/floor.

```bash
.venv/bin/python launch_mos9_limit_action.py \
  --run-dir runs/mos9_limit_action/new_run --utd 64 --batch-size 4096 \
  --teacher onnx --no-flat-soles --terrain-height .01
```

Training defaults to the generated flat-sole contact model used by the workable
teacher, gentle 0–1 mm uniform terrain, 20 mm reset clearance, 50 Hz control,
and 500 steps per episode. Training uses the full `implicitfast` integrator,
CG constraints, 50 solver iterations, tolerance 1e-5, full scene resets at
episode boundaries, and 4000 Hz physics. Extra action slew limiting is disabled. Use `--no-flat-soles` for original URDF contacts.
Rewards are zero except -1 on falling and +1 on completing 10 seconds with
at least 0.3 m forward progress. Gamma remains 0.99, matching the older algorithm branch.

Here UTD means primary SAC TD minibatch rows divided by newly collected online
transitions: updates accrue at `utd * num_envs / batch_size` per vector step,
with fractional credit carried forward. Thus UTD 64 and batch 4096 produce one
update per 64 transitions, or 16 vector steps with four environments. Every combined
update also samples an IQL batch and a floor batch, and trains action likeness;
those auxiliary samples are excluded from the named UTD. Initial IQL/H
pretraining (2000 updates after 2000 successful teacher episodes) is reported
separately. The independently initialized SAC actor then collects 65536
warmup transitions without optimizer updates.
`metrics.jsonl` records achieved online UTD and factual episode outcomes.
`checkpoint.pt` stores networks, optimizers, replay metadata, counters, update credit,
and Torch RNG state. The trainer saves on completion, interruption, and failure;
`failure.json` records errors. A checkpoint is a training artifact, not an exact
physics-state resume; automatic resume is not implemented. Existing run
folders are rejected to protect artifacts. Shared infrastructure is unchanged.

The initial approximate-integrator and Newton-solver runs encountered Genesis
constraint-force NaNs during teacher collection and are retained as failed
artifacts. The historical `run3` used CG and full scene resets with imported
teacher rows. It predates the full fidelity correction and cannot seed the
audited experiment. The corrected run uses fresh weights and fresh data.

TensorBoard logs are written under each run directory in `tensorboard/`.
The dashboard contains only the 16 selected online scalar tags listed below.
Initial collection and offline pretraining remain in JSON and console logs.
Online scalars use collected online transitions as the x-axis. Rolling episode metrics use the latest
100 episodes per controller. The default effective UTD is 64, matching
`experiment/limit-action` (4 updates × 4096 batch / 256 environments);
the audited setup uses four environments and minibatch 4096.

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


The current four-environment run uses disk replay from the pinned private
`torch-block-replay` package. Install it without changing the existing ROCm
PyTorch build:

```sh
uv pip install --python .venv/bin/python --no-deps \
  'git+ssh://git@github.com/LuoShaoyinn/torch-block-replay.git@f276e76ef480ea209d4975c6c841a4301814097b'
.venv/bin/python launch_mos9_limit_action.py \
  --run-dir runs/mos9_limit_action/onnx_10mm_block4_old_branch_fidelity \
  --teacher onnx --no-flat-soles --terrain-height .01 --num-envs 4 \
  --utd 64 --teacher-intervention-prob .01 --replay-backend block \
  --replay-capacity 100000000 --replay-block-size 4096 \
  --replay-ram-blocks 4096 --replay-gpu-cache-mib 16384 \
  --checkpoint-every 4096
```

Environments 0 and 1 stay autonomous; 2 and 3 allow sticky takeover. Initial
collection retains 2000 complete successful teacher episodes, followed by
2000 reference pretraining updates and 65536 learner-first warmup transitions
without optimizer updates. SAC stays independently initialized.
`training/teacher_fraction` now measures cumulative online teacher rows, with detailed controller masks retained in JSON logs. Episode records include takeover step and actual teacher-step fraction.

Replay has one physical disk store with a success-confirmed suffix index over
immutable cached rows. Sampling is uniform within the current caches rather
than exact uniform sampling across the entire disk history. Newly collected
rows become sampleable when a block is published; checkpoints flush partial
blocks. Extra disk block slots prevent pruning of partial checkpoint blocks
before the 100-million-row quota is reached. Network checkpoints reference
the durable disk replay instead of copying it into the checkpoint. The package
currently cannot reopen replay for resume; this run starts fresh.

Each block row is 631 bytes (including a global row ID): the RAM cache budget is about 9.9 GiB, the GPU pool
budget is 16 GiB, and the full disk replay is about 59 GiB before serialization
overhead. These budgets exclude simulator, learner, and staging allocations.
A four-environment preflight verified GPU sampling, teacher-only sampling,
learner-first sticky takeover, UTD 64, and clean stop exactly at buffer capacity.

TensorBoard is limited to 16 scalar observations: rolling success, episode
length and distance for autonomous and intervened episodes; online teacher
fraction; SAC critic MSE and actor loss; IQL critic TD; action-likeness loss;
floor RMS violation; action-limit loss; effective UTD; replay size; and throughput.
Full diagnostics and configuration remain in the JSON logs.

New trainers write the selected 16 tags directly into `tensorboard/`.
The dashboard must point to the active audited run.

The prior all-teacher-reference runs are invalid as algorithm experiments.
They have been stopped and archived with explicit invalid-run metadata.
See [ALGORITHM_CONTRACT.md](ALGORITHM_CONTRACT.md) for the restored dataset
and update rules. A separate persisted suffix index confirms row eligibility
only after success, including rows already published or cached on GPU.

The complete comparison and remaining explicit task/resource differences are
recorded in [MOS9_ALGORITHM_AUDIT.md](MOS9_ALGORITHM_AUDIT.md).
