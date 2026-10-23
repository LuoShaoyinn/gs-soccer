# MOS9-AMP reference audit

Audited on 2026-10-24 (Asia/Shanghai), against the exact current
`assets/MOS9/MOS9_walk.urdf`.

Source: [THMOS2025/MOS9-AMP](https://github.com/THMOS2025/MOS9-AMP), commit
`22aa35fd80b65d2064cbc790163e59f9667266bd`. The download manifest records each
file's Git blob and SHA256; raw files are retained unchanged under
`runs/mos9_amp_audit/raw/`. The audit covers 30 full FK motions, 11 simple
clipped files, and three rotation clips. It does not treat derived copies as
independent demonstrations.

The coordinate conventions are consistent: recomputing all named link poses
from the stored joints and authored root pose agrees with the stored FK poses
to submicrometer accuracy. This is not a joint-order or robot-scale failure.

The references should not be imported unchanged as the primary forward-walking
library. The two provided straight-walking clips have ground penetration and
sliding near contact. A constant floor-height correction does not remove the
problem. Some sidestep clips are substantially cleaner and remain useful
candidates for a separate lateral-stepping task.
These checks assess full motion tracking in the current Genesis model; they
do not evaluate the teammate's trained AMP policy.

| Simple clip | Worst penetration | Lowest support point's horizontal speed, p95 | Fraction above 0.2 m/s |
| --- | ---: | ---: | ---: |
| `walk_straight1.npz` | 2.73 cm | 2.86 m/s | 35.4% |
| `walk_straight2.npz` | 4.63 cm | 2.37 m/s | 41.7% |
| `walk_backwards.npz` | 3.94 cm | 2.26 m/s | 40.8% |
| `step_left1.npz` | 0.96 cm | 0.129 m/s | 3.6% |
| `step_right1.npz` | 0.91 cm | 0.053 m/s | 0% |

Contact points use exact support functions for the authored foot cylinders,
not foot-link origins. Their material-point velocity is computed from the
link's finite-difference translation and angular velocity. The support gate
requires a point within 1.5 cm of a calibrated constant floor and within 3 mm
of the lowest foot. A second, broader near-floor speed statistic is also
reported. These are geometric screening gates, not measured contact forces
and not proof that every flagged frame bears body weight. Fast touchdown and
swing phases can contribute. Ground penetration itself is computed exactly
against the floor plane using the original collision primitives.

The simple-clipped folder has eleven files but seven distinct trajectories:
`step_left1_copy1/2` duplicate `step_left1`, and `step_right1_copy1/2` duplicate
`step_right1`. The backward clip is also an exact copy of its full FK source.
Some other full/rotation clips exceed the current joint limits; the detailed
report includes per-clip violations, floor offsets, contact statistics,
continuity, velocities, and hashes.

## Reproduce

```bash
uv run --extra rocm python -m motion_tools.fetch mos9 --output runs/mos9_amp_audit
uv run --extra rocm python -m motion_tools.quality \
  --input runs/mos9_amp_audit/raw/data/motions \
  --output runs/mos9_amp_audit/current_urdf_quality.json
uv run --extra rocm python -m unittest discover -s tests -p test_motion_quality.py -v
```

Local evidence: `runs/mos9_amp_audit/upstream_manifest.json`,
`current_urdf_quality.json`, and `contact_quality.png`. Tests independently
check the primitive support function and compare this auditor's NumPy FK
with the Torch FK already validated against Genesis.

## Decision

Keep the teammate's original files and audit evidence. Develop the public
Unitree G1 fallback on `experiments/MOS9-retarget`, with explicit morphology
mapping, stance-foot locking, joint bounds, and the same collision audit.
Kinematic retargeting and simulator tracking must be reported separately;
passing geometric checks alone is not a trained or dynamically validated gait.
