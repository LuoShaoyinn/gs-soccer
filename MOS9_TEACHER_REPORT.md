# MOS9 traditional walking teacher

Branch: `experiment/mos9-planned-walk`, based on shared infrastructure at
`main` (`9da7a23`). The existing limit-action branch is unchanged.

## Task and implementation

- MOS9, 18 joint position targets, 50 Hz, maximum 500 control steps.
- Terminate when base height drops below 60% of reset height or its upright
  cosine drops below 0.65; otherwise truncate at 10 seconds.
- Sparse reward: -1 on fall, +1 at the horizon after at least 0.3 m forward,
  zero otherwise. Walking distance and survival are separate measurements.
- Uniform random terrain, heights 0–3 mm, 0.3 m interpolation grid and
  0.1 m samples. Seeded generation, one ground surface.
- Build and reset clearance: lowest collision point 20 mm above terrain max.
- ZMP footsteps, bounded LIPM preview, URDF whole-body COM, and constrained
  leg IK. Genesis GPU IK solves 601 poses together. Execution uses joint PD
  control, with hip/ankle tracking feedback; no base teleportation or forces.

Tested with Genesis 1.4.3, Torch 2.11.0+rocm7.2.4, RX 7900 XTX,
`HIP_VISIBLE_DEVICES=0`, 2 kHz physics, self-collision disabled.

The reusable default is `algorithm/mos9_teacher/simple.py`:
`SimpleWalkingTeacher(plan, device).act(state, steps, origins, yaws)` returns
18 joint targets. The caller clamps joint limits before applying PD control.
It needs no online optimization or trained weights. Episode steps and reset
origins are explicit inputs, so per-environment resets do not require hidden
teacher state. GPU IK is used once to build the plan, which is cached.

## Measured baseline

These are individual rollouts, not an aggregate success-rate claim.

| Contact model / setting | Terrain | Steps before fall | Time | Forward distance |
| --- | --- | ---: | ---: | ---: |
| Original cylinders, earlier gait | Flat | 225 | 4.50 s | 0.118 m |
| Original cylinders, current gait | 0–3 mm | 174 | 3.48 s | 0.061 m |
| Optional flat soles, earlier feedback | Flat | 239 | 4.78 s | 0.135 m |
| Optional flat soles, conservative gait | 0–3 mm | 230 | 4.60 s | 0.159 m |
| Optional flat soles, current gait, no angular damping | 0–3 mm | 433 | 8.66 s | 0.403 m |

The last row is `runs/mos9_teacher/no_angular.json`, with the corresponding
log and trace. It reaches a useful walking distance but falls before the
10-second horizon: **it is not a successful full episode**.
GPU IK residual for this plan was 8.27e-7, whereas the executed gait still
has growing lateral sway. Small IK residual does not establish dynamic balance.

A three-episode repeat (`baseline.failure.json`) completed two failed walks:
299 steps / 0.303 m and 302 steps / 0.313 m. The third rollout encountered a
constraint-force NaN at step 100. The trajectories in its regenerated plan
were identical to the 433-step plan; planning-scene construction had consumed
terrain RNG state. Terrain is now explicitly reseeded before scene construction
so cache misses do not change its heightfield. Historical rows above retain
their original measured terrain and should not imply repeatable success.
After the RNG correction, `reproduce.json` independently reproduced the
433-step / 0.403 m result exactly at seed 0. This checks repeatability for one
heightfield, not generalization across terrain seeds.
Nearby gain checks at the same seed were worse: roll gain 0.15 reached 302
steps / 0.122 m; velocity gain 0.2 reached 185 / 0.043 m; position gain 1.5
reached 282 / 0.247 m. The baseline gains remain unchanged. This teacher can
provide walking actions and partial demonstrations even when an episode fails;
failed trajectories must retain their terminal failure label.

The flat-sole variant replaces only foot collision cylinders with boxes
fitted to the existing visual sole plates. It is opt-in, generated under
ignored `runs/`, and preserves original assets, visuals, inertials, and joints.
Performance on this variant should not be presented as original-URDF performance.

Reproduce the current baseline:

```bash
HIP_VISIBLE_DEVICES=0 .venv/bin/python mos9_walk.py --flat-soles \
  --episodes 1 --output runs/mos9_teacher/reproduce.json
```

Remove `--flat-soles` to evaluate original contacts. Add `--viewer` to inspect
the gait, or `--terrain flat` for a flat baseline. The default path uses
offline GPU IK; `--online-ik` remains experimental and performed worse in
initial trials. Multiple-environment terrain trials also encountered Genesis
constraint-force NaNs; the default evaluation uses one environment. Failures
write a trace and `.failure.json` and exit with an error rather than count as
successful episodes.

Shared `envs/`, `robots/`, and `fields/` remain unchanged. The branch now includes `train_mos9_limit_action.py` and a detached launcher.
Its reference dataset includes factual teacher failures, with terminal
bootstrapping masked, rather than requiring success-only seed episodes.
See README for UTD accounting and training settings.

Validation: Python compilation and `git diff --check` passed. A focused task
check verified the 500-action clock, positive horizon reward, and negative
fall reward. Actual Genesis rollouts provide the walking measurements above.
The extracted simple teacher reproduced recorded baseline joint commands
within 5.96e-8 radians. Its independent simulator run is saved as
`runs/mos9_teacher/simple_teacher.json`: 360 steps (7.2 s), 0.346 m, terminal
fall. This is useful guidance, not a successful 10-second demonstration.
