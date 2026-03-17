# algorithm.py
#   Base class for all algorithms

from dataclasses import dataclass, field
from envs.env import EnvConfig, Env

@dataclass(kw_only = True)
class AlgorithmConfig:
    env_cfg:            EnvConfig
    env_class:          Env
    models:             dict
    memory_size:        int     = 8192
    total_timesteps:    int     = 30000 
    experiment_name:    str
    experiment_dir:     str     = "runs"
    discount_factor:    float   = 0.97
    mixed_precision:    bool    = True
    write_interval:     int     = 50
    checkpoint_interval:int     = 1000

class Algorithm:
    def __init__(self, cfg: AlgorithmConfig, env: Env | None = None):
        self.cfg = cfg
        self.env = env if env is not None else cfg.env_class(cfg.env_cfg)
