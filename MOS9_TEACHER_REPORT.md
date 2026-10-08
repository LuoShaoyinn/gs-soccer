> Algorithm correction: earlier runs described below that used all teacher
> transitions as reference data are invalid algorithm experiments. They are
> stopped and archived. The current contract is success-confirmed teacher
> suffixes only; see ALGORITHM_CONTRACT.md. Historical measurements are retained
> as records, not evidence for the corrected algorithm.

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

## URSoccerLab reference policy and arm posture

The local `py_example/examples/mos9_walk/mos9_walk.py` loads
`mos9_walk_v11_5500.onnx`. Its default shoulder rolls are right -1.4 rad and
left +1.4 rad. Rendering the previous planner home pose confirmed that zero
shoulder rolls produce a T pose. The planner now includes the arms-down pose
before COM calculation and leg IK; this changes the plan cache key. The online
IK controller also preserves the planned arm pose.

`test_mos9_onnx.py` is a separate learned-policy reference evaluator. It uses
the source observation/action contract and actuator gains/torque limits from
`external/robots/mos9/model.xml`, with zero armature. It holds the default pose
for one simulated second before starting the 500-step MDP. Reference results
are not evidence that the classical teacher succeeds.

The first reference episode on 0–1 mm uniform terrain, original URDF contacts,
50 Hz control and 4000 Hz physics survived 500 steps and moved 3.2704 m.
Its result and T-pose comparison render are preserved in
`runs/mos9_teacher/onnx_reference_first/`. Learner training remains stopped.

The higher-terrain reference test on 0–10 mm uniform terrain completed a full
500-step episode without falling, with 3.0838 m forward displacement. See
`runs/mos9_teacher/onnx_10mm/result.json` and its pose PNGs. This test was stopped
after the first complete episode to release GPU 0 for the requested fresh run;
it does not establish success across terrain seeds.

Fresh run: `runs/mos9_limit_action/onnx_10mm_utd64`, explicitly using the learned
ONNX teacher, original URDF contacts, 0–10 mm terrain, UTD 64, one environment,
and 3,000,000 replay rows. Initial teacher data is newly collected; no old
checkpoint or replay is imported. The teacher's 50-step standing phase is
included in the 500-step training horizon. The replay stops at capacity,
never overwrites rows, saves the final checkpoint and records `buffer_full`
in `status.json`. Older runs were moved to `runs/archive/`.

The first teacher episode inside the fresh training process also succeeded:
500 steps, no fall, 2.9307 m forward displacement, including the standing phase.
This is logged in the run's `teacher_collection.jsonl` and TensorBoard. At the
handoff the process is collecting its initial 2048 teacher transitions, before
IQL pretraining and online learning.

## Four environments and block replay

The older branch's runtime takeover condition is a per-step random trigger,
restricted to a fixed rescue-enabled half of environments, sticky until terminal.
Its CLI default probability is zero; H threshold 0.95 is a critic-training
constraint rather than an execution takeover gate. This branch uses p=0.01,
and adds an explicit first-action learner guard as requested.

The 4-env preflight (`runs/archive/mos9_before_block4/block4_preflight`) verified
online masks starting [learner, learner, learner, learner], then remaining
[learner, learner, teacher, teacher] with p=1. It stopped exactly at 832 rows,
with 320 online transitions and 80 updates: UTD 64. CPU and ROCm checks covered
filtered teacher sampling, immutable terminal next observations, and availability
of the teacher view when the random RAM cache contains no teacher rows.

Historical invalid run: `runs/mos9_limit_action/onnx_10mm_block4_utd64`, p=0.01, 4 envs,
100 million rows, 4096-row blocks, 4096 RAM blocks, and 16 GiB GPU pool. RAM
cache maximum is 9.7 GiB; raw disk data maximum is 58 GiB. On 2026-10-09 02:40
local time, after GPU pool allocation, total VRAM usage was 17.0/24.0 GiB,
host available RAM was 18.9/30.5 GiB, and the trainer RSS was about 2.8 GiB.
RAM cache grows as replay rows arrive; GPU pool is allocated at schema setup.
A timestamped live snapshot is saved in the run's `resource_snapshot.json`.

## Success-only suffix correction

The previous all-teacher reference implementation was wrong for this algorithm.
It has been stopped, marked invalid, and archived; its replay and trained weights
are not reused. The learner computations were compared against the old branch:
after removing docstrings and the monitored-horizon constant, their ASTs match.
Reference IQL's original unmasked target and raw SAC proposals are restored.

Regression tests passed on RAM, CPU disk, and GPU disk replay: failed rescues,
autonomous successes, learner prefixes and unfinished rescues remain ineligible;
a successful contiguous teacher suffix becomes eligible only after terminal
success, including rows already published/cached. The authoritative persisted
suffix index is separate from immutable physical transitions. Initial collection
retains only complete successful teacher demonstrations.

The full 4-env preflight saved exactly 1032 rows: a complete successful 500-row
teacher demo plus 532 ordinary online rows. Its suffix index contains exactly
the first 500 rows; failed/unfinished rescues are excluded. It stopped at capacity
with 133 online updates, giving UTD 64. Artifacts are preserved under
`runs/archive/success_suffix_preflight_verified/`.

Superseded partial correction: `runs/mos9_limit_action/onnx_10mm_block4_success_suffix`.
It keeps the 100-million-row limit, 16 GiB GPU cache, roughly 9.9 GiB RAM cache,
4 environments, 10 mm terrain, UTD 64, sticky p=0.01 takeover, and 16 dashboard
signals. Each disk block row is now 631 bytes including its global row ID;
a 100 MB CPU suffix bitmap gives post-success eligibility without duplicating
physical transitions. Checkpoints persist the suffix index. See
`ALGORITHM_CONTRACT.md` for the exact data and update contract.

## Additional algorithm fidelity correction

The success-suffix correction alone did not establish full algorithm fidelity.
The subsequent audit found SAC initialization copied from IQL, missing online
warmup, modified actor constraints, and changed optimization defaults. Those
runs are superseded and their checkpoints must not seed the corrected run.
The restored source matches learner/network computations from 9778c66 and
restores independent SAC initialization, warmup, and old optimization defaults.
The runtime preflight is recorded separately in
`runs/mos9_limit_action/fidelity_v2_preflight`; full verification passed.
The recorded eligibility index exactly matches 2496 successful teacher rows
reconstructed from actual episodes. All learner/optimizer state stayed unchanged
during 4096 warmup transitions. Four online updates followed, with a clean stop
at 4860 rows. See `MOS9_ALGORITHM_AUDIT.md` for details.
