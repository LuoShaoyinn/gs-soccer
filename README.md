# gs-soccer

Genesis-based environment scaffolding for humanoid soccer simulation.

This branch keeps only the reusable environment surface: robots, fields, envs,
and abstract task interfaces. Training algorithms, experiment runners, scripted
policies, checkpoints, and concrete task models have been removed.

## Project structure

- `envs/`: Genesis environment orchestration (`Env`, `WalkEnv`, `DribbleEnv`, `GameEnv`)
- `robots/`: robot definitions and wrappers (`Robot`, `PI`, `MOS9`, `ControlledRobotWrapper`, `TeamedRobot`)
- `fields/`: field definitions (`Field`, `SoccerField`)
- `models/model.py`: abstract model interface used by environments
- `assets/`: robot URDFs and meshes

## Core abstractions

- `Env`: owns the Genesis scene and handles stepping, reset, observation,
  reward, termination, truncation, and info flow through a supplied model.
- `Model`: defines the task contract for observation/action spaces, action
  preprocessing, observations, reward, termination, truncation, and info.
- `Robot`: wraps a Genesis URDF robot and exposes actuator, reset, and state APIs.
- `Field`: owns field entities and exposes reset/state APIs.

## Environment classes

- `Env`: single-robot base environment.
- `WalkEnv`: single walking robot environment shell.
- `DribbleEnv`: single controlled robot plus ball environment shell.
- `GameEnv`: red-vs-blue soccer environment shell.

These classes are framework pieces. To run a concrete task, provide a concrete
`Model` implementation and matching config objects from your application code.
