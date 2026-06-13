# gs-soccer

Genesis-based environment scaffolding for humanoid soccer simulation.

This branch keeps only the reusable, abstract framework surface. The physics
layer (`envs/`, `robots/`, `fields/`) is frozen; experiments fork and provide a
concrete `Model`, an `Algorithm`, and the `main.py` entry point.

## Project structure

- `envs/env.py`: `Env` — single generic orchestrator owning the Genesis scene.
  Delegates observation / reward / termination / info to an injected `Model`.
- `robots/`: robot definitions (`Robot`, `PI`, `MOS9`, `FloatingCameraRobot`).
- `fields/`: field definitions (`Field`, `BallField` — a plane plus a ball).
- `models/model.py`: `Model` — abstract task contract (spaces, observation,
  reward, termination, truncation, info).
- `algorithm/algorithm.py`: `Algorithm` — abstract base with `train()` / `eval()`.
- `main.py`: composition root. Selects robot + field + model (+ algorithm) and
  runs. The shipped template renders the viewer with a standing robot.
- `assets/`: robot URDFs and meshes.

## Core abstractions

- `Env`: owns the Genesis scene; handles stepping, reset, observation, reward,
  termination, truncation, and info flow through a supplied `Model`.
- `Model`: abstract task contract (observation/action spaces, observation,
  reward, termination, truncation, info). Concrete models are provided per
  experiment (e.g. inline in `main.py`).
- `Robot`: wraps a Genesis URDF robot; exposes actuator, reset, and state APIs.
- `Field`: owns field entities (and the ball, if any) and exposes reset/state.
- `Algorithm`: abstract training / evaluation interface.

## Experiment workflow

1. Fork a branch from `main`.
2. Provide a concrete `Model` (task: observation, reward, termination, domain
   randomization such as `cmd_vel` / `target_ball_pos`).
3. Provide a concrete `Algorithm` (`train()` / `eval()`).
4. Wire them together in `main.py`.
5. Leave `envs/`, `robots/`, `fields/` untouched.

## Run the template

```bash
python main.py                 # viewer on, standing robot
python main.py --no-viewer     # headless
```
