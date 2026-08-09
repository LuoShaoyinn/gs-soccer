# Limit-Action Experiment 2 Report

## Purpose

This experiment tests whether a SAC policy can improve beyond a weak kick
teacher while remaining constrained by successful teacher-supported actions.
The state-based familiarity fence and intervention-start correction ranking are
not used.

The teacher retains its original interface:

- teacher observation: 646 dimensions;
- learner observation: the 646 teacher dimensions plus a three-dimensional
  phase one-hot, for 649 dimensions total;
- action: 22 raw joint-target multipliers.

## Algorithm

The learner contains three coupled but gradient-isolated components:

1. Moving IQL is trained only from complete successful initial teacher
   trajectories and successful later teacher-intervention suffixes. Ordinary
   SAC transitions and unsuccessful intervention suffixes do not train IQL.
2. The SAC critic receives an IQL action floor on successful teacher-supported
   states.
3. A moving action-likeness model `H(s,a)` is trained from successful teacher
   actions and standardized Gaussian action perturbations. If
   `H(s,a_SAC) < 0.95`, the critic is trained toward the relative inequality
   `Q(s,a_IQL) >= Q(s,a_SAC) + 1/350`.

The SAC actor has no BC, IQL, teacher, or `H` gradient. It always optimizes SAC
Q and always controls autonomous rollout. The action constraint changes only
the critic landscape.

## Training configuration

- Genesis environments: 256;
- autonomous environments: 128;
- rescue-enabled environments: 128;
- successful initial teacher trajectories: 2,000;
- IQL pretraining updates: 2,000;
- online vector steps: 10,000;
- updates per vector step: 4;
- online learner updates: 40,000;
- final learner update: 42,000;
- replay batch size: 4,096;
- effective sampled-row UTD per individual loss: 64;
- replay capacity: 3,000,000 rows;
- final replay size: 2,932,975 rows;
- IQL expectile: 0.7;
- exploration standard deviation during training: 0.05;
- teacher-intervention probability: 0.01;
- action-likeness threshold: 0.95;
- wall-clock training duration from TensorBoard timestamps: approximately
  104.5 minutes.

Replay checkpoints are stored as 100 shards plus a manifest. The reusable
initial successful-teacher replay is stored at
`runs/limit_action_1/teacher_buffer` and contains 307,439 rows.

## Training result

Autonomous success increased across ten chronological bins:

```text
9.9% -> 31.3% -> 41.1% -> 79.4% -> 88.0%
     -> 74.3% -> 71.3% -> 79.3% -> 79.5% -> 85.4%
```

Across the full second run, autonomous success was 65.2%. It was 84.2% over
the final 1,000 completed autonomous episodes. Late failures were primarily
timeouts rather than falls.

## Fair frozen-policy evaluation

The frozen teacher, final IQL actor, and final deterministic SAC actor were
each evaluated for 5,000 fresh episodes with the same task randomization and
seed. There was no intervention or exploration noise.

| Policy | Success | Falls | Timeouts | Mean episode steps | Successful episode steps | Mean return |
|---|---:|---:|---:|---:|---:|---:|
| Teacher | 23.2% | 6.58% | 70.22% | 297.9 | 155.3 | -0.684 |
| IQL reference | 22.7% | 2.84% | 74.46% | 304.9 | 162.6 | -0.672 |
| Final SAC | **92.1%** | **0.12%** | **7.78%** | **155.7** | **139.2** | **+0.478** |

The final SAC improves absolute success by 68.9 percentage points and reaches
approximately four times the teacher success rate. Its successful episodes are
about 10.4% faster than the teacher's successful episodes.

Evaluation artifacts:

- `runs/limit_action_2/eval_teacher.json`;
- `runs/limit_action_2/eval_iql.json`;
- `runs/limit_action_2/eval_sac.json`.

## Evidence that SAC did not fall back to IQL

On 61,487 states sampled from successful initial teacher trajectories:

- standardized IQL-to-teacher action RMSE: 0.038;
- standardized SAC-to-teacher action RMSE: 0.290;
- raw SAC-to-IQL action RMSE: 0.853;
- IQL/teacher cosine similarity: 0.9996;
- SAC/teacher cosine similarity: 0.9776;
- SAC critic mean preference `Q(s,a_SAC)-Q(s,a_IQL)`: +0.00675.

The IQL actor reproduces teacher-level performance, while SAC is measurably
different and succeeds much more often. This supports constrained policy
improvement rather than behavior-cloning fallback.

## Action-fence diagnostics

Late in training:

- mean `H(s,a_SAC)` on general replay: 0.957;
- rejected SAC proposal fraction: 20.7%;
- `H(s,a_SAC)` on successful teacher states: 0.9986;
- mean relative-constraint violation: 0.0053.

The fence remains active on about one fifth of general proposals while allowing
SAC deviations in strongly supported regions.

## Remaining mathematical issue

IQL value is not a calibrated factual lower bound. On the 307,439 initial
successful-teacher rows:

- factual mean discounted return: 0.335;
- final IQL `Q_h350` mean: 0.562;
- mean bias: +0.227;
- MAE: 0.279;
- prediction/return correlation: 0.606.

An offline expectile-0.99 test was worse: `Q_h350` saturated at 0.987 and had
correlation -0.366. Therefore the main experiment keeps expectile 0.7. Exact
multi-horizon return targets or explicit return anchors on complete successful
suffixes remain the primary value-calibration improvement.

Rare large floor violations also remain despite a small mean violation. The
late mean violation was 0.0035, but the late average maximum violation was
0.826. The floor should therefore be interpreted as a soft training constraint,
not a hard guarantee.

## Reproduce deterministic evaluation

```bash
uv run --extra rocm python evaluate_floor_policy.py \
  --policy sac \
  --episodes 5000 \
  --num-envs 256 \
  --seed 123 \
  --sac-checkpoint runs/limit_action_2/sac/model_42000.pt \
  --iql-checkpoint runs/limit_action_2/iql/model_42000.pt \
  --output runs/limit_action_2/eval_sac.json
```
