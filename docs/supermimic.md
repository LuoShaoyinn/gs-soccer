# MOS9 SuperMimic experiment

Branch: `experiments/mos9-walk/supermimic`, based on `infra`.

This is an independent implementation from the
[SuperMimic paper](https://arxiv.org/abs/2610.28480), especially equations
1–3 and appendices A–D. The [authors' project](https://supermimic.github.io/)
did not link official training code when checked on 2026-10-23. This branch
adapts the method to Genesis 1.4.5 and the repository's MOS9 model. It does
not contain a trained walking controller.

## Required motion data

Provide one or more flat-ground walking trajectories **retargeted to
`assets/MOS9/MOS9_walk.urdf`**. A directory of `.npz` files is a motion library;
a single `.npz` also works. Arrays must be numeric or Unicode, with no pickled
Python objects. Minimum length is 51 frames for the default horizons; several
seconds of clean walking per clip is preferable. Every clip must supply:

| Key | Shape/type | Meaning |
| --- | --- | --- |
| `robot` | scalar Unicode `"MOS9"` | Identifies the target morphology |
| `fps` | scalar `50.0` | Control/reference sampling rate |
| `joint_names` | Unicode `[18]` | Every actuated URDF joint exactly once |
| `joint_pos` | float `[T,18]` | Joint angles in radians, in named order |
| `root_pos` | float `[T,3]` | Authored `base_link` origin, world xyz in meters |
| `root_quat` | float `[T,4]` | Unit **wxyz** root quaternion, body to world |
| `joint_vel` | optional float `[T,18]` | Joint velocities, rad/s, same named order |
| `root_vel` | optional float `[T,3]` | World velocity of the authored root origin, m/s |
| `root_ang_vel` | optional float `[T,3]` | World angular velocity, rad/s |

Missing velocities are computed from the trajectory. Coordinates are right
handed with z up and the floor at z=0. Root xy is translated to start at zero;
root z is preserved. Joint-name order is checked and reordered on import.
Do not use a pelvis COM position in place of the authored base-link origin.
The six arm joints are active alongside the twelve leg joints. The model uses
the original URDF collision geometry, including its foot primitives.

Print the canonical joint order:

```bash
uv run --extra rocm python -c 'from supermimic.config import JOINT_NAMES; print(JOINT_NAMES)'
```

For an existing MOS9 retarget:

```python
import numpy as np

np.savez("walk.npz", robot="MOS9", fps=50., joint_names=joint_names,
         joint_pos=joint_angles, root_pos=base_positions, root_quat=base_quaternions)
```

The loader rejects wrong robots, missing/duplicate joints, invalid shapes,
nonfinite values, non-unit quaternions, inconsistent sampling rates, short
clips, and joint-limit violations greater than 0.05 rad. This is a numeric
preflight, not proof that a retarget is physically feasible: inspect foot
contact, ground penetration, and the root height during retargeting. At scene
construction, the experiment compares its FK and link velocities with Genesis
and refuses to run if the maximum discrepancy exceeds 1 mm or 0.001 rad,
m/s, or rad/s respectively.

**Human data/retargeting status:** no suitable MOS9 walking motion was found
in the repository or its local `refs/`. The available recovery motions target
PI Plus, a different morphology. Supply either MOS9 trajectories in the format
above, or human walking data such as BVH/SMPL plus the source skeleton/model
information so that a MOS9 retarget can be developed. Raw human data and G1
retargets cannot be consumed directly. This branch does not invent a gait or
silently use another robot's motions.

## Training and playback

```bash
uv sync --extra rocm                 # use --extra cuda on NVIDIA
uv run --extra rocm python -m supermimic validate --motion /path/to/mos9_walks
uv run --extra rocm python -m supermimic train \
  --motion /path/to/mos9_walks --num-envs 128 --iterations 10000 \
  --run-dir runs/mos9_supermimic/walk
uv run --extra rocm tensorboard --logdir runs/mos9_supermimic/walk/tensorboard
uv run --extra rocm python -m supermimic eval \
  --checkpoint runs/mos9_supermimic/walk/latest.pt \
  --output runs/mos9_supermimic/walk/evaluation.json
```

Evaluation plays every clip from its first frame in one environment, with
friction 1, no randomization, and no sensor/action noise. Reports include
failures, full-clip normalized return, root-position RMSE, and xy displacement.
Early failures retain the full clip's normalization denominator. `--motion`
can select an additional library; the report distinguishes the training library
by checksum. Add `--viewer` for playback. This is a motion-conditioned tracker;
it requires future motion targets and a root-pose estimate at execution time.
There is no standalone velocity-command interface or hardware deployment here.

Start every run in a new or empty directory. Checkpoints save the actor,
world model, critic, optimizers, replay, return sampler, learning-rate position,
update budget, data/URDF hashes, and Python/NumPy/Torch RNG states atomically.
Resume with:

```bash
uv run --extra rocm python -m supermimic train \
  --resume runs/mos9_supermimic/walk/latest.pt --iterations 10000 \
  --run-dir runs/mos9_supermimic/walk_resumed
```

`--iterations` means additional iterations when resuming. The checkpoint's
configuration and original motion path are restored; `--motion` can relocate
identical data, whose checksums must match. The physical scene starts new
episodes and cuts replay continuity at the old collection boundary. This is
not bit-exact restoration of simulator contacts. SIGINT/SIGTERM finish the
current iteration and save. Other runtime errors propagate without overwriting
the last valid checkpoint.

## Algorithm and MOS9 adaptations

Each iteration collects five vector steps, then spends a fractional budget of
0.4 learner updates per vector step. Each learner update fits ten random
eight-step autoregressive model batches, then makes one policy update. The
policy imagines 32 steps from replay states with their recorded context and
observation history. World-model and critic parameters are frozen during the
policy backward pass, while derivatives through their inputs remain active.
The value head bootstraps the horizon. The critic learns detached lambda
returns with gamma=0.99 and lambda=0.95, over five epochs and four minibatches.
No simulator gradient or PPO likelihood-ratio update is used.

| Component | Implementation |
| --- | --- |
| State | 18 joint positions/velocities, root pose and world velocities, previous executed normalized action |
| Observation | Ten readings of relative joint positions, joint velocities, body angular velocity, projected gravity, previous raw action |
| Goal | Ten reference frames: joints/velocities and root-relative position, rotation, linear/angular velocity |
| World model | Residual physical-state predictor with kinematic integration prior, quaternion normalization, and learned termination probability |
| Analytic heads | Differentiable URDF FK, body velocities, observation, and tracking reward |
| Reward | Paper A.2 body/root tracking and action/joint-limit costs, multiplied by 0.02 s; maximum 0.1 per step |
| Failure | Root height error above 0.15 m or projected-gravity z error above 0.8 |
| Context | Friction U[0.3,1.2], actuator calibration U[-.01,.01] rad, root COM offset within [.025,.05,.05] m |
| Collection noise | Gaussian action noise, std=0.1, stored after adding noise |
| Sensor noise | Paper A.1 ranges; added outside the deterministic model |
| Reset sampling | One-second bins, 50% uniform, 40% softmax of negative returns, 10% historical softmax weights |
| Replay | CPU vector-step ring, physical terminal next states, no windows spanning resets or stale ring chronology |
| Optimizers | Adam at 2e-4; actor beta=(.9,.95); global clipping at 1, actor element clipping at .1; paper cosine LR schedule |

Relative to the G1 paper, this uses 18 rather than 29 joints, 15 tracked
MOS9 links, a 0.15 m height-error threshold rather than 0.25 m, and Genesis
with ten 2 ms physics substeps per control period. The model predicts
generalized state and computes body states through URDF FK. The paper does
not specify all state encodings or network activations; the residual prior,
fixed feature scales, ELU activations, and root-relative goal encoding are
implementation choices.

Defaults are reduced to fit ordinary GPUs: 128 environments, batches of 128,
world-model widths (512,512,512), actor encoder/decoder widths (512,256,128).
`--paper-networks` selects the paper's world-model widths (2048,2048,2048)
and encoder/decoder widths (2048,1024,512,256), with a tanh 32-dimensional
latent. The critic uses (256,128). Batch/environment sizes are independently
configurable with `--num-envs`, `--batch-size`, and `--imagination-batch`;
the paper uses 4096/2048/4096 respectively. The CPU replay stores complete
observation histories; estimate host RAM before using the paper's scale.

Initial MOS9 actuation settings are kp=40, kd=2, torque limit=20 Nm, action
scale=0.5 rad, and armature=0.01. These are experiment defaults, not measured
MOS9 hardware calibration. Configuration lives in `supermimic/config.py` and
is recorded in each run. Walking performance, world-model accuracy on real
reference data, and actuator settings still need validation after data arrives.
There is no claim of reproducing the paper's G1 performance or speedup.

## Verification

```bash
uv run --extra rocm python -m unittest discover -s tests -v
GS_SUPERMIMIC_GPU_TEST=1 uv run --extra rocm python -m unittest discover -s tests -v
uv run --extra rocm python -m supermimic train --smoke-test --iterations 2 \
  --run-dir runs/mos9_supermimic/smoke
```

The opt-in GPU test checks reference reset velocities, Genesis/FK agreement,
partial reset isolation, raw executed actions, and physical terminal next
states before autoreset. The synthetic smoke motion contains small kinematic
oscillations, is marked `smoke_only`, and is refused by ordinary data loading.
Smoke checkpoints and evaluation reports retain that label.

See [the recorded validation](supermimic_validation.md) for results from the
development machine. Passing these checks establishes executable code and
specific contracts, not a learned walking gait.
