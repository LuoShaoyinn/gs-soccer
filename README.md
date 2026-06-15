# gs-soccer

Genesis-based environment scaffolding for humanoid soccer simulation.

This branch keeps only the reusable, abstract framework surface. The physics
layer (`envs/`, `robots/`, `fields/`) is frozen; experiments fork and provide a
concrete `MDP`, an `Algorithm`, and the `main.py` entry point.

## Project structure

- `envs/env.py`: `Env` — single generic orchestrator owning the Genesis scene.
  Delegates observation / reward / termination / info to an injected `MDP`.
- `robots/`: robot definitions (`Robot`, `PI`, `MOS9`, `FloatingCameraRobot`).
- `fields/`: field definitions (`Field`, `BallField` — a plane plus a ball).
- `MDPs/MDP.py`: `MDP` — abstract task contract (spaces, observation,
  reward, termination, truncation, info, and reset).
- `MDPs/dummy.py`: `DummyMDP` — a standing-task example showing how to
  subclass `MDP` and implement the full contract (including `reset`).
- `algorithm/algorithm.py`: `Algorithm` — abstract base with `train()` / `eval()`.
- `main.py`: composition root. Selects robot + field + MDP (+ algorithm) and
  runs. The shipped template renders the viewer with a standing robot.
- `assets/`: robot URDFs and meshes.

## Core abstractions

- `Env`: owns the Genesis scene; handles stepping, reset, observation, reward,
  termination, truncation, and info flow through a supplied `MDP`.
- `MDP` (formerly `Model` — renamed): abstract task contract —
  observation/action spaces, observation, reward, termination, truncation,
  and info. The `Env` no longer resets the robot/field itself; instead the
  MDP must provide `reset(envs_idx, robot_reset_fn, field_reset_fn)` and own
  the task-level reset logic, calling the injected `robot_reset_fn` /
  `field_reset_fn` with the desired joint / base / ball poses. Concrete MDPs
  are provided per experiment (see `MDPs/dummy.py`, wired up in `main.py`).
- `Robot`: wraps a Genesis URDF robot; exposes actuator, reset, and state APIs.
- `Field`: owns field entities (and the ball, if any) and exposes reset/state.
- `Algorithm`: abstract training / evaluation interface.

## Experiment workflow

1. Fork a branch from `main`.
2. Provide a concrete `MDP` subclass (task: observation, reward, termination,
   and `reset()` — where to place the robot/ball; plus domain randomization
   such as `cmd_vel` / `target_ball_pos`). See `MDPs/dummy.py` for a template.
3. Provide a concrete `Algorithm` (`train()` / `eval()`).
4. Wire them together in `main.py`.
5. Leave `envs/`, `robots/`, `fields/` untouched.

## Run the template

```bash
uv run --extra rocm python main.py                  # viewer on, standing robot
uv run --extra rocm python main.py --no-viewer      # headless
```

> Note: `Model` has been renamed to `MDP`. New tasks subclass `MDP` (in
> `MDPs/`) and must implement `reset()`. `DummyMDP` in `MDPs/dummy.py` is the
> reference example.
