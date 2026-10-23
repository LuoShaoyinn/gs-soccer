# MOS9 walking with SuperMimic

This experiment implements a MOS9/Genesis adaptation of
[SuperMimic (Liao et al., CoRL 2026)](https://arxiv.org/abs/2610.28480).
It alternates simulator collection, autoregressive world-model fitting, and
analytic policy-gradient updates through the learned model. It includes
motion tracking, reference-state resets, return-based reset sampling,
checkpoint/resume, TensorBoard, and deterministic full-clip evaluation.

**Walking reference data is still required.** No MOS9 walking dataset ships
with this branch. Human or G1 motions must be retargeted to the MOS9 URDF.
The synthetic smoke mode verifies execution only; it cannot establish walking.

```bash
uv sync --extra rocm
uv run --extra rocm python -m supermimic validate --motion /path/to/mos9_walks
uv run --extra rocm python -m supermimic train \
  --motion /path/to/mos9_walks --run-dir runs/mos9_supermimic/walk
```

See [the experiment guide](docs/supermimic.md) for data, method, testing,
evaluation, and resume instructions.

## Shared infrastructure

`infra` is the default branch. Its tip contains the shared Genesis dependency,
robot definitions in `robots/`, and URDFs/meshes in `assets/`, plus project
metadata. Genesis is installed as the pinned `genesis-world` package, not
vendored into this repository.

```bash
uv sync --extra rocm     # AMD GPUs
# or: uv sync --extra cuda
```

Training algorithms, task environments, reference motions, and their tests
belong on experiment branches, including this branch:

```bash
git switch -c experiments/mos9-walk/supermimic infra
```

Existing experiment branches retain their histories. Local `refs/`, `runs/`,
and `.env` are ignored and do not form part of the infrastructure.
