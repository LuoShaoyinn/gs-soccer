# Validation recorded on 2026-10-23

Machine: AMD Radeon RX 7900 XTX, Genesis 1.4.5, PyTorch
2.11.0+rocm7.2.4.git5fbd98f3, Python 3.12.

The tested runtime source SHA256 is
`5fc42ceb9bdb08dda9af1262b07f292dc231cc00271ac6e3077dbf3135ae45c8`.
It is computed by `supermimic.run.provenance()` over sorted Python filenames
and contents in `supermimic/`. Runs were made before the implementation commit,
on infrastructure parent `630bc3ad5e66c5d6851628eb2787589b75482cb9`;
the source hash identifies the implementation independently of that parent.

## Checks

- `uv lock --check`: passed.
- `uvx ruff check supermimic tests`: passed.
- `uvx ruff format --check supermimic tests`: passed.
- `git diff --check`: passed.
- `GS_SUPERMIMIC_GPU_TEST=1 uv run --extra rocm python -m unittest discover -s tests -v`:
  all **15 tests passed**, including real GPU simulation (30.397 s total).
- Ordinary motion validation refused the synthetic smoke file with
  `smoke-only reference cannot be used for walking training`.

The GPU test compared the analytical authored-frame FK and link velocities
against actual Genesis state at nonzero reference joint/root velocities:

| Maximum absolute discrepancy | Measured |
| --- | ---: |
| Link position | 8.94e-8 m |
| Link orientation angle | 3.11e-7 rad |
| Link linear velocity | 1.86e-8 m/s |
| Link angular velocity | 2.98e-8 rad/s |

It also checked full pose/velocity resets, partial reset isolation, logging
the raw normalized action even when setpoints clip, and retaining the physical
terminal next state before autoreset. CPU tensor tests cover 32-step actor
updates with frozen world-model parameters, finite-difference Jacobians,
eight-step autoregressive fitting, GAE bootstrap/termination, replay windows
across episode and ring boundaries, motion validation, checkpoint/RNG round
trips, and return-based sampling.

## End-to-end GPU smoke run

```bash
uv run --extra rocm python -m supermimic train --smoke-test \
  --run-dir runs/supermimic-final-smoke
uv run --extra rocm python -m supermimic train \
  --resume runs/supermimic-final-smoke/latest.pt --iterations 1 \
  --run-dir runs/supermimic-final-resume
uv run --extra rocm python -m supermimic eval \
  --checkpoint runs/supermimic-final-smoke/latest.pt \
  --output runs/supermimic-final-smoke/evaluation.json
```

The first command collected 64 transitions across two environments and made
two world-model/actor/critic updates with eight-step model windows and
32-step imagination. State losses were 0.0105200 and 0.00571334; these are
tiny-batch smoke losses, not calibrated prediction accuracy. The resumed
command continued to iteration 3, update 3, and 96 total transitions. Both
saved readable checkpoints containing replay and optimizer state. Model-head
survival predictions are still untrained after these few updates.

The generated reference is a synthetic kinematic oscillation with no sustained
walking translation. Its `.npz` SHA256 is
`3854feac9d4275f713643024a3a2d987564b5fdc4184516bf1f25e9664e6c561`.
The MOS9 URDF SHA256 is
`1e8e4479e4b5319c4899337d4266c35775f52fa9c3589a4eda6cca4cc50a532b`.
Checkpoints, metadata, TensorBoard events, and evaluation JSON remain in the
ignored local `runs/` paths above; they are not distributed as trained models.

Deterministic evaluation correctly reports a failure on the synthetic clip.
The policy falls; forward displacement during that fall is not evidence of a
walking gait. See the local evaluation JSON for exact rollout measurements.

**Walking remains unvalidated:** there are no MOS9-retargeted walking clips in
this repository. The next required input is real walking reference data in the
[documented format](supermimic.md). Training and evaluating a walking policy,
calibrating actuator gains, and measuring tracking quality must follow that
input; the smoke checks do not establish those outcomes.
