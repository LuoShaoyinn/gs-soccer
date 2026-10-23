# G1 walking references retargeted to MOS9

Recorded on 2026-10-24, Asia/Shanghai. Development branch:
`experiments/MOS9-retarget`, based on the SuperMimic branch's MOS9 data audit.
The `infra` branch is unchanged.

## Outcome

Two eight-second, 401-frame references at 50 Hz have been generated from
public Unitree G1 walking sequences. Their named joint arrays, root poses,
and velocities load directly into this repository's SuperMimic trainer.

| Metric | G1 walk1 → MOS9 | G1 walk2 → MOS9 |
| --- | ---: | ---: |
| Lowest collision height, minimum | +0.24 mm | +0.015 mm |
| Lowest support-point speed, median | 0.0083 m/s | 0.0052 m/s |
| Lowest support-point speed, p95 | 0.0573 m/s | 0.0228 m/s |
| Support speed above 0.2 m/s | 1.25% | 0.91% |
| Joint-angle limit violations | 0 | 0 |
| Motor-speed limit violations | 0 | 0 |
| Foot target-position error, p95 | 1.75 cm | 1.51 cm |

For comparison, the teammate's two simple straight-walking references reach
2.73 and 4.63 cm of floor penetration, with support-point speed p95 of 2.86
and 2.37 m/s under the same geometry/gates. Some of their lateral-stepping
references are considerably cleaner; see [the audit](mos9_amp_quality.md).

**These are kinematic candidates.** They have not passed dynamic walking
validation. In a common unassisted Genesis position-and-velocity PD probe,
the first and second retarget lose balance after 1.04 and 0.88 s. The teammate
straight clips also fail, at 0.96 and 1.68 s. The lateral clip fails at 1.68 s
of its 1.70 s duration. These times characterize this particular controller;
they do not prove that a learned balance controller cannot track the motions.

The probe uses the teammate's published kp/kd, armature, and torque limits,
with all references reset to their first pose and velocity. There is no base
wrench, pose teleporting during rollout, or external support. Joint RMSE for
the retargets is about 0.038 and 0.066 rad, while root-position RMSE before
termination is about 0.232 and 0.306 m. The main observed failure is whole-body
balance/root tracking, despite substantially cleaner reference foot geometry.

A real-data SuperMimic integration probe completed two iterations with 16
environments, 256 transitions, and six model/actor/critic updates. It fitted
eight-step model windows and backpropagated through 32-step imagined rollouts.
This verifies loading, simulation collection, learning, logging, and checkpoint
integration. It is too short to assess learned walking or model accuracy.

## Source and method

Source: [Unitree LAFAN1 retargeted dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset),
revision `ce1572906efe6157840e8474d5a0d7aa87481e74`. Files:
`g1/walk1_subject1.csv` and `g1/walk2_subject1.csv`. The original format is
30 Hz xyz + xyzw + 29 G1 joints. The source model is the checked-in
`assets/g1/g1_29dof.urdf`; the target is `assets/MOS9/MOS9_walk.urdf`.

Segments are selected before target fitting, using source-only gates for
upright root orientation, active walking speed, and at least one detected
support foot. Selection is deterministic. The selected inclusive source-frame
intervals are 446–686 (14.867–22.867 s) and 1263–1503 (42.1–50.1 s). They
are resampled to 50 Hz with cubic position/joint interpolation and quaternion
SLERP; the original eight-second timing is retained.

The method transfers lower-body/root motion. Six arm joints hold a MOS9
rest pose; it does not attempt a full G1 arm/waist retarget. Root xy and heading
start at zero. Leg-length scale is 0.667755792; stance width is adjusted using
the two robots' neutral sole positions. Source stance segments become fixed
world foot targets. Swing targets use a smooth horizontal interpolation and
up to 5 cm clearance. Foot targets are flat with source heading.

A joint trajectory and root translation are optimized together through
differentiable URDF FK. Joint angles use a bounded tanh parameterization.
Objectives include foot poses, exact cylinder floor distance, stance velocity,
root/velocity priors, joint/root smoothness, and motor speed excess. MOS9 speed
limits follow the teammate configuration: 9.32 rad/s for shoulders/ankle roll
and 15.60 rad/s for the other joints. No framewise root-height projection is
used to conceal floor errors. Full FK/body velocities are recomputed from the
optimized trajectory before export.

Exported `.npz` files carry source/model hashes and
`quality_status=kinematic_candidate_not_dynamic_validation`. A per-clip JSON
records optimization, source interval, floor/slip metrics, continuity, and
the retargeter source-code hash. The raw CSVs and teammate motions remain
unchanged; generated trajectories live under `runs/`.

## Reproduce

```bash
uv sync --extra rocm --extra motion
uv run --extra rocm --extra motion python -m motion_tools.fetch g1 \
  --output runs/g1_mos9_retarget/source

uv run --extra rocm --extra motion python -m motion_tools.retarget \
  --input runs/g1_mos9_retarget/source/g1/walk1_subject1.csv \
  --output runs/g1_mos9_retarget/mos9_fk_retarget/walk1_subject1.npz
uv run --extra rocm --extra motion python -m motion_tools.retarget \
  --input runs/g1_mos9_retarget/source/g1/walk2_subject1.csv \
  --output runs/g1_mos9_retarget/mos9_fk_retarget/walk2_subject1.npz

uv run --extra rocm --extra motion python -m supermimic validate \
  --motion runs/g1_mos9_retarget/mos9_fk_retarget
uv run --extra rocm --extra motion python -m motion_tools.quality \
  --input runs/g1_mos9_retarget/mos9_fk_retarget \
  --output runs/g1_mos9_retarget/quality.json
uv run --extra rocm --extra motion python -m motion_tools.plot \
  --input runs/g1_mos9_retarget/mos9_fk_retarget/walk1_subject1.npz \
  --output runs/g1_mos9_retarget/contact_plot.png \
  --video runs/g1_mos9_retarget/kinematic_preview.mp4

uv run --extra rocm --extra motion python -m unittest discover -s tests -v
GS_SUPERMIMIC_GPU_TEST=1 uv run --extra rocm --extra motion \
  python -m unittest discover -s tests -p test_supermimic_genesis.py -v
```

The kinematic optimizer runs on CPU by default; `--device cuda` is available.
`--start` selects an explicit source time, `--seconds` selects duration, and
`--iterations` defaults to 1000 L-BFGS iterations. To prepare an IsaacLab FK
reference for the trainer/replayer, use `python -m motion_tools.convert
--input <file.npz> --output <library/file.npz>`. Then run `python -m
motion_tools.replay --library <library> --output <report.json>`. Use separate
directories for reference clips and diagnostics; the exporter does this
automatically.

The short integration probe used:

```bash
uv run --extra rocm --extra motion python -m supermimic train \
  --motion runs/g1_mos9_retarget/mos9_fk_retarget --num-envs 16 \
  --batch-size 16 --imagination-batch 16 --model-batches 2 \
  --collect-steps 8 --iterations 2 \
  --run-dir runs/g1_mos9_retarget/supermimic_final_probe
```

## Evidence and validation

Checked-in measurements:

- [Teammate audit, 44 files](motion_quality/mos9_amp_audit.json)
- [G1 source revisions and checksums](motion_quality/g1_source_manifest.json)
- [Retarget fit/quality/provenance](motion_quality/g1_retarget_results.json)
- [Common-controller Genesis replay](motion_quality/genesis_pd_replay.json)
- [SuperMimic integration probe](motion_quality/supermimic_probe.json)

Local review artifacts: `runs/g1_mos9_retarget/contact_comparison.png`,
`kinematic_preview.mp4`, and the original per-clip `.npz`/`.json` files.
The preview shows FK reference motion and is labeled accordingly.

All 19 CPU tensor/geometry tests passed, and the separate real Genesis GPU
regression passed. New tests cover primitive floor support, independent FK,
finite-difference retargeter Jacobians, bounded stance fitting, and export
compatibility with the training loader. Lint, formatting, and lockfile checks
passed. The final replay/training manifests were checked against the exact
on-disk artifact hashes.

The next performance question is whether the learned tracker can stabilize
these references over full clips. The current evidence does not establish
that outcome, hardware transfer, or superiority over the teammate's AMP policy.
