# MOS9 limit-action algorithm contract

Baseline: `experiment/limit-action` (`9778c66`). This remains the older branch's
algorithm, adapted to the MOS9 task; it is not claimed to be the final paper's
newer formulation.

- Initial data: stage full teacher episodes; retain only complete successes.
  Failed and unfinished initial attempts do not enter the demonstration replay.
- Online data: retain every real transition for SAC, including failures,
  autonomous successes, and learner prefixes of rescued episodes.
- Reference eligibility: after a real episode success, walk backward from its
  terminal transition and mark only the contiguous teacher-controlled suffix.
  Failed or unfinished rescues are never eligible. A pure learner success has
  no teacher suffix. A full successful teacher demo is entirely eligible.
- IQL, H, and the detached critic floor sample only this confirmed suffix view.
  SAC TD samples ordinary replay. There is one physical transition store.
- IQL Bellman targets match the old branch's success-only update with
  `mask_terminal=False`. SAC keeps physical terminal masking. Real success is
  the terminal success label; no fake successful labels or fabricated suffixes.
- H threshold 0.95 selects rejected learner proposals for the critic inequality;
  it does not cause execution takeover. The actor objective and detached
  reference-floor/action-limit calculations match the old learner.
- Online control: learner first, with a per-step random trigger on the fixed
  rescue-enabled half of environments; takeover lasts through terminal. The
  other half stays autonomous. A first-action learner guard implements the
  user's explicit requirement. Current probability is 0.01; old CLI default 0.
- UTD: 64 primary SAC TD rows per newly collected online transition; initial
  pretraining and auxiliary IQL/H/floor samples are excluded.
- Stop at replay capacity; never overwrite stored rows. An unfinished episode
  at stopping time has no confirmed suffix. Disk replay keeps an authoritative
  CPU suffix bitmap keyed by stored global row IDs, persisted as suffix_index.pt.
  GPU/RAM sampling consults this index, so cached rows cannot become eligible
  prematurely and later confirmation does not require a second transition store.

Task/interface changes: 18 MOS9 joints, 68 observations, 500 steps at 50 Hz,
sparse terminal reward, gamma 1, task lower bound -1, hidden width 128,
batch size 256, ONNX teacher, original sole contacts, and 0–10 mm terrain.
Executed learner targets have the existing 0.05-rad step limit in the robot
interface; algorithm proposals remain the raw actor output, as in the baseline.
Disk-block sampling is uniform within its current cache, not exactly uniform
across the entire disk history. The block library currently cannot reopen
replay for resume, so corrected training starts from new weights and data.

The previous all-teacher-reference runs violated this contract. Their weights
and replay are archived as invalid and must not initialize a corrected run.
