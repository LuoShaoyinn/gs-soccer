# gs-soccer infrastructure

`infra` is the default branch. It contains the shared Genesis dependency,
robot definitions in `robots/`, and URDFs/meshes in `assets/`, plus project
metadata. Genesis is installed as the pinned `genesis-world` package, not
vendored into this repository.

```bash
uv sync --extra rocm     # AMD GPUs
# or: uv sync --extra cuda
```

Keep training algorithms, task environments, reference motions, and their tests
on experiment branches. Create each experiment from `infra`, for example:

```bash
git switch -c experiments/mos9-walk/supermimic infra
```

Existing experiment branches retain their histories. Local `refs/`, `runs/`,
and `.env` are ignored and do not form part of the infrastructure.
