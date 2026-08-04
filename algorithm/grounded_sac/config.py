from dataclasses import dataclass


@dataclass(kw_only=True)
class GroundedSACConfig:
    """Configuration for the first no-fence grounded SAC experiment.

    The familiarity ensemble/controller is deliberately absent.  Consequently
    every replay state is treated as familiar, the SAC controller is always
    selected, and the outside-fence field has zero weight.  The data fields and
    loss boundary are kept explicit so the frozen fence can be added later
    without changing the replay format.
    """

    observation_dim: int = 649
    action_dim: int = 22
    # The teacher actions are raw joint-target multipliers, not normalized
    # actions.  They are clipped to this physical-interface limit by the MDP.
    # Keep the learned actors in the same units as replay.
    action_limit: float = 100.0
    # Match the physical FloorIQLMDP timeout exactly.
    horizons: int = 350
    hidden_dim: int = 512
    gamma: float = 0.99
    step_penalty: float = -1.0 / 350.0
    # A fall adds a terminal -1 after any preceding timeout-equivalent step
    # costs, so every vector head must be able to express the task range.
    value_lower_bound: float = -2.0
    # Extending a horizon by one control step can now include the terminal
    # fall cost rather than only the ordinary step penalty.
    max_horizon_drop: float = 1.0
    expectile: float = 0.7
    awr_beta: float = 3.0
    awr_max_weight: float = 100.0
    batch_size: int = 4_096
    exploration_std: float = 0.05
    max_grad_norm: float = 10.0
    # Lean GPU replay: ~9.85 GiB at 2M rows for 649-D obs/next-obs, one 22-D
    # executed action, scalar reward, and three boolean labels.
    replay_capacity: int = 2_000_000
    learning_rate: float = 3e-4
    iql_learning_rate: float = 3e-4
    actor_learning_rate: float = 3e-4
    target_tau: float = 0.005
    rank_weight: float = 0.1
    floor_weight: float = 1.0
    outside_weight: float = 0.0
    action_penalty: float = 1.0
    updates_per_env_step: float = 1.0
    iql_pretrain_updates: int = 2_000
    warmup_transitions: int = 20_000
    device: str = "cuda"
