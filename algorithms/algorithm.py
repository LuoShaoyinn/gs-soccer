from dataclasses import dataclass


@dataclass(kw_only=True)
class AlgorithmConfig:
    device: str = "cuda"
    timesteps: int = 1_000_000
    headless: bool = True
    environment_info: str = "extra"
    experiment_name: str = "experiment"
    experiment_directory: str = "runs"
    write_interval: int = 100
    checkpoint_interval: int = 1000
    discount_factor: float = 0.99
    mixed_precision: bool = True
    init_method_name: str = "normal_"
    init_mean: float = 0.0
    init_std: float = 0.1
    compile_policy: bool = False
    resume: bool = False
    checkpoint_path: str | None = None
    eval_steps: int = 2000


class Algorithm:
    def __init__(self, env, cfg: AlgorithmConfig):
        self.env = env
        self.cfg = cfg
