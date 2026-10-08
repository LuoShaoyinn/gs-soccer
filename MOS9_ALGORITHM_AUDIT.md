# MOS9 algorithm fidelity audit

The reference is `experiment/limit-action` at `9778c66`. This experiment ports
that implementation to MOS9 walking; it does not claim to implement a newer
paper formulation. The earlier runs are not valid evidence for the algorithm.

| Component | Audited behavior and evidence |
| --- | --- |
| Learner and networks | Computation ASTs match the pinned reference. Only documentation and monitored horizons differ. Regression checks cover actor, critics, Bellman targets, IQL, H, detached floor, optimizer steps, and target updates. |
| Optimization defaults | Gamma 0.99, width 512, batch 4096, exploration 0.05, and all loss weights, learning rates, expectile, thresholds, and target coefficients match. A configuration regression test enumerates the remaining task/resource differences. |
| Initial demonstrations | Stage entire episodes, discard failures and unfinished attempts, admit only complete successful teacher episodes. Default budget is 2000 episodes. |
| Reference eligibility | Confirm only the contiguous teacher suffix ending in an actual episode success. Failed rescues, learner prefixes, autonomous successes, and unfinished rescues cannot train IQL, H, or the floor. Tests exercise memory, disk CPU, and disk GPU sampling, including confirmation after a block is cached. |
| SAC replay | All real online transitions, including failures and learner prefixes, remain eligible. No fake success labels or fabricated transitions. |
| Initialization | IQL/H pretraining is 2000 updates. The SAC actor remains independently initialized; it is not copied from IQL. |
| Warmup | Collect 65536 learner-first transitions without optimizer updates. Continue the same collector into online learning without resetting at this boundary. |
| Takeover | Fixed autonomous half and rescue-enabled half. A per-step draw triggers sticky takeover through episode end. Probability 0.01 is the explicit MOS9 setting; the old CLI default was zero. The user's learner-first requirement adds a first-action guard. H is never an execution gate. |
| UTD | 64 primary SAC minibatch rows per new online transition. Four environments and batch 4096 accrue one update every 16 vector steps. Initial pretraining, warmup, and auxiliary samples are excluded from this denominator. |
| Capacity | Append until 100 million accepted rows; do not overwrite. Stop before another optimizer update when full, matching the old stopping order. Record pending fractional update credit. |
| Actions | Preserve the original scalar actor formula. MOS9 uses absolute joint targets and a scalar 3-radian action bound. Added network per-joint projection and execution slew limiting are removed from the audited setup. The explicit teacher retains its own valid joint bounds. |
| Terminal observations | Capture the physical observation before auto-reset. Ordinary replay stores this observation, not the next episode's reset state. SAC masks terminal continuation; successful reference terminals override targets to one as in the baseline. |
| Checkpoints | Atomic weight checkpoint references an immutable, versioned suffix bitmap. Prune obsolete bitmap snapshots only after the weight checkpoint commits. Persist collection RNG and counters. |
| Shared infrastructure | `envs/`, `robots/`, `fields/`, and original `assets/` match `main`; scene and terminal-observation adaptations are branch-local. |
| Dashboard | Exactly 16 allowed scalar tags. Detailed configuration and outcomes remain in JSON files. |

The requested task adaptations are 18 MOS9 joint actions, 68 observations,
500 steps at 50 Hz, sparse terminal reward, task lower bound -1, a workable
explicit ONNX teacher, 20 mm spawn clearance, and uniform 0–10 mm terrain
with original URDF contacts. Horizon-dependent constants use 500 steps.

The requested block replay changes sampling: batches are uniform within
the current RAM/GPU caches, rather than exactly uniform over all disk rows.
The success-confirmed reference view uses the same physical disk store,
with a CPU row-ID bitmap as the authoritative eligibility index. The pinned
package is `torch-block-replay` at
`f276e76ef480ea209d4975c6c841a4301814097b`. It cannot reopen replay for resume;
the corrected experiment must start from new weights and new data.

Resource budgets are four environments, approximately 9.9 GiB RAM replay
cache, 16 GiB GPU replay pool, and approximately 59 GiB disk payload at
capacity, before serialization overhead. Simulator and learner allocations
are additional. Full training retains the old collection/pretraining/warmup
budgets; reduced budgets are used only for the explicit runtime preflight.

The all-teacher-reference runs are archived under
`runs/archive/invalid_all_teacher_reference_20261009/`. The subsequent
success-suffix run is also superseded, because its SAC initialization,
warmup, action constraints, and defaults still differed. It is archived under
`runs/archive/superseded_algorithm_protocol_20261009/`. Neither may seed the
audited experiment.

Runtime evidence is recorded separately under
`runs/mos9_limit_action/fidelity_v2_preflight/`. Its full collection,
warmup, online update, suffix reconstruction, and capacity checks must pass
before the fresh full-budget experiment is reported as restarted.

## Verified runtime result

The completed preflight retained 500 initial successful demo rows, collected
4096 warmup rows without changing any learner or optimizer state, then
collected 264 online rows with four online updates. It stopped at exactly
4860 rows. UTD was 64 at 256 online rows; the final stop left 0.125 update
credit, so final measured UTD was 62.0606. No update was forced after capacity.

Reconstruction from actual episode outcomes and takeover steps matched all
2496 reference-eligible rows: the initial 500 rows plus four completed
successful 499-step teacher suffixes. Learner prefixes and unfinished
rescues were excluded. Physical terminal observations and all 16 dashboard
tags passed verification. Results are in `algorithm_verification.json`.

A separate full-cache allocation check used real preflight transitions and
completed a full-width, 4096-row learner update with the requested 16 GiB GPU
pool. Total live GPU use was 18.17 GiB of 23.98 GiB on the RX 7900 XTX, with
20.09 GiB host RAM available and 539.77 GiB disk free. This isolated check
excluded the simulator; the fresh run also records its live resources. The
RAM cache is a maximum budget and grows as replay data arrives.
