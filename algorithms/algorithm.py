from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import genesis as gs
import torch
from envs.env import Env


@dataclass(kw_only=True)
class RunConfig:
    checkpoint_path: Optional[str] = None
    resume: bool = True
    load_state_dict: bool = True
    checkpoint_is_compiled: bool = True
    compile_policy: bool = True
    eval_steps: int = 1000
    max_steps: int | None = None


@dataclass(kw_only=True)
class AlgorithmConfig(RunConfig):
    device: str = "cuda"
    timesteps: int = 30000
    headless: bool = True
    environment_info: str = "extra"

    experiment_name: str = "experiment"
    experiment_directory: str = "runs"
    write_interval: int = 50
    checkpoint_interval: int = 1000

    discount_factor: float = 0.97
    mixed_precision: bool = True
    init_method_name: str = "normal_"
    init_mean: float = 0.0
    init_std: float = 0.1


class Algorithm:
    def __init__(self, env: Env, cfg: RunConfig):
        self.env = env
        self.cfg = cfg

    def _load_model(self) -> None:
        if not self.cfg.resume:
            raise ValueError("resume must be True when using base Algorithm loader")
        if not self.cfg.load_state_dict:
            raise ValueError("The checkpoint must be a whole model when using base Algorithm loader")

        if self.cfg.checkpoint_is_compiled:
            self.model = torch.jit.load(self.cfg.checkpoint_path).to(gs.device)  # type: ignore[arg-type]
        else:
            self.model = torch.load(self.cfg.checkpoint_path, map_location=gs.device)  # type: ignore[arg-type]
            if hasattr(self.model, "to"):
                self.model = self.model.to(gs.device)

        if self.cfg.compile_policy:
            self.model = torch.compile(self.model)

    def train(self) -> None:
        raise NotImplementedError("train() is not implemented for this algorithm")

    def eval(self) -> None:
        self._load_model()
        obs, _ = self.env.reset()
        action_dim = self.env.action_space.shape[0]
        actions = torch.zeros((self.env.num_envs, action_dim), dtype=torch.float, device=gs.device)

        if self.cfg.max_steps is None:
            while True:
                actions = self.model(obs)
                obs, _, _, _, _ = self.env.step(actions)
            return

        for _ in range(self.cfg.max_steps):
            actions = self.model(obs)
            obs, _, _, _, _ = self.env.step(actions)
