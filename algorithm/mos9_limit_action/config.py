from dataclasses import dataclass


@dataclass(kw_only=True)
class GroundedSACConfig:
    """Moving human-IQL floor plus an action-likeness critic constraint."""

    observation_dim: int = 68
    action_dim: int = 18
    action_limit: float = 3.0
    action_slew: float = 0.0
    horizons: int = 500
    hidden_dim: int = 512
    gamma: float = 0.99
    step_penalty: float = 0.0
    value_lower_bound: float = -1.0
    max_horizon_drop: float = 1.0
    expectile: float = 0.7
    awr_beta: float = 3.0
    awr_max_weight: float = 100.0
    batch_size: int = 4096
    exploration_std: float = 0.05
    max_grad_norm: float = 10.0
    replay_capacity: int = 100_000_000
    learning_rate: float = 3e-4
    iql_learning_rate: float = 3e-4
    actor_learning_rate: float = 3e-4
    td_max_head_weight: float = 0.1
    target_tau: float = 0.005
    rank_weight: float = 0.1
    floor_weight: float = 1.0
    action_likeness_weight: float = 1.0
    action_likeness_noise_std: float = 1.0
    action_likeness_noise_samples: int = 2
    action_likeness_threshold: float = 0.95
    action_limit_weight: float = 1.0
    action_limit_margin: float = 1.0 / 500.0
    iql_pretrain_updates: int = 2000
    warmup_transitions: int = 65_536
    device: str = "cuda"
