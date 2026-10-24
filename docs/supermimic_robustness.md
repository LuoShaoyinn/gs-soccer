# MOS9 uniform-terrain robustness experiment

The flat-ground policy is used to initialize a new dynamics distribution on
`experiments/mos9-walk/supermimic`. The retargeting branch remains separate.
Configuration: [`mos9_uniform_5cm.json`](../configs/supermimic/mos9_uniform_5cm.json).

## Terrain and randomization

| Quantity | Distribution |
|---|---|
| Terrain coarse heights | Independent continuous U[-0.05, +0.05] m |
| Coarse knot spacing | 0.25 m; bounded linear upsampling, without clipping or cubic overshoot |
| Collision grid | 0.10 m, triangulated height field over 12 x 12 m |
| Episode terrain location | Independent x/y offsets U[-2, +2] m on the shared terrain |
| Kp | Per joint/episode U[0.5, 1.5] times 40 Nm/rad = 20–60 |
| Kd | Per joint/episode U[0.5, 2.0] times 2 = 1–4 |
| Root local COG offset | Independent U[-0.05,+0.05], U[-0.08,+0.08], U[-0.08,+0.08] m |
| Contact friction | Per episode U[0.2,1.5], applied to robot and terrain |
| Joint calibration offset | Per joint/episode U[-0.02,+0.02] rad |
| Push timing | Independent integer U{2,3,4,5,6} chunks; each chunk is 32 control steps (0.64 s) |
| Push | Additive world delta-v x/y U[-0.8,+0.8] m/s per axis; world delta-omega z U[-0.8,+0.8] rad/s |

Thus push intervals are 1.28–3.84 s and maximum planar speed increment is
sqrt(2)*0.8 = 1.13 m/s. Each event is an instantaneous velocity impulse,
without changing root pose, zeroing momentum, or ending the episode. There
is no persistent support force. Independent countdowns restart on episode
reset and after each kick. Short random-phase episodes may end before a push.
Push counts and maximum applied planar delta-v are recorded in training metrics.

The ±5 cm range is the interpretation of the requested "5 cm random height";
it represents a 10 cm total elevation span. Uniformity applies to the coarse
knots; the interpolated grid has a smoother marginal height distribution.
Each run saves its exact height field, SHA256, and measured grid min/max.
Environments share one physical field, with different sampled locations,
rather than receiving independently rebuilt collision meshes at each reset.
No spawn region is flattened or excluded from the requested distribution.

At reset only, the reference is translated horizontally to its sampled terrain
location, and raised by the highest sampled collision-surface clearance plus
5 mm. The same constant translation applies to the entire reference episode,
including all future goals and model targets. This prevents starting with a
foot inside a bump, without teleporting during walking or pretending the
original flat-ground motion was terrain-retargeted. The policy must learn to
adapt to varying contact heights while tracking this translated motion.

## Learning and scaling

The actor observation/history stays unchanged: this is a blind terrain tracker.
The learned dynamics and privileged critic receive per-joint gain ratios,
episode translation, and a differentiable 5 x 5 local terrain-height patch.
Queries match Genesis's collision triangle diagonal, rather than approximating
the surface with a different interpolation during imagination.

Warm initialization copies the successful flat-ground actor, world model and
critic. Newly added privileged input columns are zero initialized, preserving
the original model/critic predictions exactly at initialization. Optimizers,
replay, return sampler and update counters start fresh. This is a new training
stage, not an exact resume of the old physics distribution.

An external kick cuts the corresponding replay window at the preceding step.
It does not label a fall or reset the physical episode. The following replay
state includes the imposed velocity, and the current observation history is
updated before choosing an action. Multi-step model fitting therefore never
attributes an unobserved external impulse to the actor's preceding action.

The configuration uses 4096 environments, model/imagination batches of 512,
and 128 replay vector steps (524,288 transitions at capacity). Replay remains
in host RAM. This is a fourfold batch-size increase over the first experiment;
the learner still performs 0.4 updates per collected vector step, each with
ten eight-step model batches and one 32-step imagined policy update.
The shorter vector replay capacity avoids the large RAM/checkpoint overhead
of storing 4096 complete observation histories for 1000 vector steps.

## Run

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MPLCONFIGDIR=/tmp/mos9-mpl \
.venv/bin/python -m supermimic train \
  --robust-terrain --config configs/supermimic/mos9_uniform_5cm.json \
  --init-from runs/mos9_supermimic/local_walk1_20261024/training/latest.pt \
  --motion runs/g1_mos9_retarget/mos9_fk_retarget/walk1_subject1.npz \
  --iterations 10000 --checkpoint-every 100 \
  --run-dir runs/mos9_supermimic/robust_uniform_5cm_20261024/training
```

The original clip is retained for a controlled robustness experiment. The second
clip remains unused; success here will not establish broader motion coverage.
To continue a saved stage into a new directory, use `--resume` and omit preset,
configuration and warm-start flags. `--iterations` then means additional steps.
SIGTERM requests a graceful iteration boundary and checkpoint save.

## Validation

CPU contracts check bounded reproducible terrain, the terrain height Jacobian,
exact preservation under warm initialization, translated future targets,
and per-environment replay cuts without false fall labels.

An actual GPU physics test checks triangle-height agreement with Genesis,
per-environment Kp/Kd readback, isolation of partial resets, unchanged pose and
unaffected neighbors under an impulse, exact world xy/yaw velocity increments,
and finite subsequent physics. The efficient 10 cm collision mesh passed in
48.6 seconds on the RX 7900 XTX. This validates the perturbation implementation,
not robust learned walking.

Terrain collision and query behavior follows the installed Genesis 1.4.5
implementation; the [official terrain guide](https://genesis-world.readthedocs.io/en/v1.4.0/user_guide/physics/terrain.html)
describes user-supplied height fields. Our bounded terrain generator supplies
its own field rather than relying on generator defaults.

## Capacity measurement on this computer

The 4096-environment pilot completed 20 iterations, 409,600 transitions and
39 learner updates in 84.1 seconds of rollout/learning time (4,873 real
transitions/s). It applied 861 random kicks. Measured total board VRAM during
learning was about 6.44 GB and GPU utilization was 100%; the PyTorch-only
allocator metric excludes Genesis/Quadrants allocations. The measured terrain
grid range was -0.049827 to +0.049961 m. Dynamics extrema reached the configured
Kp/Kd/friction/COG ranges. The pilot validates capacity and data collection;
its initial episode failures do not establish learned robustness.

The sustained run continues this pilot checkpoint to 10,000 total iterations
(204,800,000 total transitions, 19,999 policy updates). The planned additional
9980 iterations take roughly 11.7 hours at the pilot rate. Final throughput
and runtime may vary as episodes and resets change.

A fresh-terrain challenge baseline uses seed 101, four full-clip attempts,
randomized dynamics/sensor noise, and pushes. Repeat exactly after training:

```bash
.venv/bin/python -m supermimic eval \
  --checkpoint runs/mos9_supermimic/robust_uniform_5cm_20261024/training/latest.pt \
  --randomized --pushes --trials 4 --seed 101 \
  --output runs/mos9_supermimic/robust_uniform_5cm_20261024/final_robustness.json
```

Evaluation without those switches is a nominal-gain, no-noise, no-push terrain
control. `--flat` supplies flat terrain instead, `--start-frame` selects a later
reset phase, and `--seed` changes both the terrain realization and random draws.
Every report records its effective settings and terrain hash.
