# gs-soccer

Genesis-based RL environments for humanoid locomotion, dribbling, and 1v1 soccer.

## Project structure

- `envs/`: environment wrappers (`Env`, `WalkEnv`, `DribbleEnv`, `GameEnv`)
- `models/`: task models (obs/action/reward/termination logic)
- `robots/`: robot definitions and wrappers (`PI`, `MOS9`, `ControlledRobotWrapper`)
- `fields/`: field/ball definitions (`Field`, `SoccerField`)
- `algorithms/`: PPO/SAC/AlphaStar-style training loops
- `policies/`: traditional/scripted policies (e.g. PID go-to-ball)
- `run_*.py`: entry scripts

## Core abstractions

- `Env`: owns Genesis scene + stepping/reset orchestration
- `Model`: defines task interface:
  - observation space
  - action space
  - action preprocessing
  - observation building
  - reward
  - terminated/truncated
  - info
- `Robot`: low-level actuator interface over URDF robot
- `ControlledRobotWrapper`: uses a walking policy (TorchScript) to transform high-level `cmd_vel` to joint targets

## Environment ↔ model mapping

### 1) Walk

- Env: `envs/walker.py` (`WalkEnv`)
- Model options:
  - `models/mos9_walk_model.py` (`MOS9WalkModel`)
  - `models/pi_walk_model.py` (`PIWalkModel`)
- Typical runners:
  - `run_walk.py` (MOS9)
  - `run_walk_pi.py` (PI)

### 2) Dribble

- Env: `envs/dribble.py` (`DribbleEnv`)
- Model: `models/dribble_model.py` (`DribbleModel`)
- Runner: `run_dribble_ppo.py`

### 3) Soccer 1v1

- Env: `envs/game.py` (`GameEnv`)
- Model: `models/game_model.py` (`GameModel`)
- Field: `fields/soccer_field.py` (`SoccerField`)
- Algorithm runner: `run_soccer_1v1_alphastar.py`
- Scripted policy runner: `run_soccer_1v1_policies.py`

## Robots and control

### Raw robots

- `robots/pi.py` (`PI`)
- `robots/mos9.py` (`MOS9`)

### Wrapped robots for high-level control

- `robots/controlled_robot.py` (`ControlledRobotWrapper`)
  - input action: high-level velocity command `(lin_x, lin_y, ang_z)`
  - internally calls a walking control model + TorchScript policy (`ctrl_policy_path`)
  - outputs joint-level action to the underlying robot

## Current soccer schema (important)

`GameEnv` + `GameModel` use team-keyed dictionaries:

- observations: `{"red": Tensor, "blue": Tensor}`
- rewards: `{"red": Tensor, "blue": Tensor}`
- terminated: `{"red": Tensor, "blue": Tensor}`
- truncated: `{"red": Tensor, "blue": Tensor}`
- actions passed to env: `{"red": Tensor, "blue": Tensor}`

`info` includes at least:

- `extra`: per-team reward breakdown dict
- `goal_team_red`: `BoolTensor`
- `goal_team_blue`: `BoolTensor`
- `ball_out`: `BoolTensor`

## Soccer observation/action definition

From `GameModel`:

- action (per team): shape `(3,)`, range `[-1, 1]`
  - interpreted as `(lin_x, lin_y, ang_z)` command

- observation (per team), shape `(18)` for `num_robots_in_team=1`
  - `self_heading` (2)
  - `self_vel_2d` (`2 * num_robots_in_team`)
  - `self_ang_z` (1)
  - `opp_rel_pos` (2)
  - `opp_rel_vel` (2)
  - `ball_rel_pos` (2)
  - `ball_rel_vel` (2)
  - `ball_to_goal` (2)
  - `last_cmd` (3)

If `num_robots_in_team` changes, `self_vel_2d` expands accordingly.

## Soccer reward/termination definition

### Reward (team-symmetric)

Main terms in `GameModel`:

- goal/concede
- ball-progress-to-goal
- possession-like distance advantage
- close-to-ball, facing-ball, contact
- command smoothness penalty (`-|Δcmd|` style)
- ball-out advantage + ball-out penalty
- anti-stall penalty (ball speed too low)
- timeout penalty

Dense competitive terms are converted to pairwise advantage (red vs blue) so shaping is closer to zero-sum.

### Terminated

Episode terminates if any:

- goal for red or blue
- ball out of field (non-goal out)
- robot fall (either team base height below threshold)
- timeout reached (`timeout_steps_limit`)

### Truncated

Currently always false (zeros) for soccer.

## Scripted/traditional policies

`policies/go_to_ball_pid.py`:

- `GoToBallPIDPolicy`: orientation-aware go-to-ball policy using the same observation layout
- used as scripted opponent in AlphaStar flow (`--scripted-opponent pid`)

`policies/advanced_dribble/`:

- `AdvancedDribblePolicy`: non-RL dribble controller adapted from the mos-brain reference
- consumes the same `GameModel` concatenated observation (currently hard-coded for 1 robot per team)

## How to run

### Train soccer (AlphaStar-style league)

```bash
python run_soccer_1v1_alphastar.py \
  --num-envs 4096 \
  --timesteps-per-gen 65536 \
  --scripted-opponent pid
```

### Evaluate soccer

```bash
python run_soccer_1v1_alphastar.py --eval --headless
```

### Run scripted policy vs scripted policy

```bash
python run_soccer_1v1_policies.py \
  --red-policy go_to_ball_pid \
  --blue-policy advanced_dribble \
  --num-envs 1
```

### Train dribble PPO

```bash
python run_dribble_ppo.py
```

## Notes

- For large `num_envs`, collision capacity is controlled by `EnvConfig`:
  - `max_collision_pairs`
  - `multiplier_collision_broad_phase`
- Soccer runner already sets higher values suitable for large parallel simulation.
