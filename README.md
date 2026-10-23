# MOS9 motion audit and G1 retargeting

Branch: `experiments/MOS9-retarget`.

This branch audits the team's public MOS9 references and retargets public
Unitree G1 walking clips to the exact MOS9 URDF. It includes contact-aware
trajectory fitting, joint/motor limits, a format converter, contact plots,
kinematic preview, and unassisted Genesis PD replay.

Two eight-second G1 retargets have been produced locally and pass the
kinematic screening gates. They still lose balance in the reference PD
probe. A short SuperMimic integration run succeeds; walking performance
with a learned tracker remains unvalidated.

See [the teammate-data audit](docs/mos9_amp_quality.md) and
[retargeting results and reproduction commands](docs/mos9_retarget.md).

```bash
uv sync --extra rocm --extra motion
uv run --extra rocm --extra motion python -m motion_tools.fetch g1 \
  --output runs/g1_mos9_retarget/source
uv run --extra rocm --extra motion python -m motion_tools.retarget \
  --input runs/g1_mos9_retarget/source/g1/walk1_subject1.csv \
  --output runs/g1_mos9_retarget/mos9_fk_retarget/walk1_subject1.npz
```

## SuperMimic integration

This experiment implements a MOS9/Genesis adaptation of
[SuperMimic (Liao et al., CoRL 2026)](https://arxiv.org/abs/2610.28480).
It alternates simulator collection, autoregressive world-model fitting, and
analytic policy-gradient updates through the learned model. It includes
motion tracking, reference-state resets, return-based reset sampling,
checkpoint/resume, TensorBoard, and deterministic full-clip evaluation.

Reference data is fetched/generated locally under ignored `runs/` paths.
Source datasets are not bundled into Git. The retargeted outputs use the
trainer's named-joint, 50 Hz schema.

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
