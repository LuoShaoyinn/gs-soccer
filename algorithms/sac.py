from __future__ import annotations
from dataclasses import dataclass

import torch

from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer

from network import Policy, QNetwork

from .algorithm import Algorithm, AlgorithmConfig


@dataclass(kw_only=True)
class SACAlgorithmConfig(AlgorithmConfig):
    memory_size: int = 65536
    timesteps: int = 30000000

    experiment_name: str = "dribble_sac"
    experiment_directory: str = "runs/SAC_Walker"

    batch_size: int = 4096
    random_timesteps: int = 20000
    learning_starts: int = 20000
    gradient_steps: int = 8


class SACAlgorithm(Algorithm):
    def __init__(self, env, cfg: SACAlgorithmConfig):
        super().__init__(env, cfg)

        self.models = {
            "policy": Policy(env.observation_space, env.action_space, cfg.device),
            "critic_1": QNetwork(env.observation_space, env.action_space, cfg.device),
            "critic_2": QNetwork(env.observation_space, env.action_space, cfg.device),
            "target_critic_1": QNetwork(env.observation_space, env.action_space, cfg.device),
            "target_critic_2": QNetwork(env.observation_space, env.action_space, cfg.device),
        }
        for model in self.models.values():
            model.init_parameters(
                method_name=cfg.init_method_name,
                mean=cfg.init_mean,
                std=cfg.init_std,
            )

        agent_cfg = SAC_DEFAULT_CONFIG.copy()
        agent_cfg["discount_factor"] = cfg.discount_factor
        agent_cfg["batch_size"] = cfg.batch_size
        agent_cfg["random_timesteps"] = cfg.random_timesteps
        agent_cfg["learning_starts"] = cfg.learning_starts
        agent_cfg["gradient_steps"] = cfg.gradient_steps
        agent_cfg["mixed_precision"] = cfg.mixed_precision

        agent_cfg["experiment"]["directory"] = cfg.experiment_directory  # type: ignore[index]
        agent_cfg["experiment"]["write_interval"] = cfg.write_interval  # type: ignore[index]
        agent_cfg["experiment"]["checkpoint_interval"] = cfg.checkpoint_interval  # type: ignore[index]
        agent_cfg["experiment"]["experiment_name"] = cfg.experiment_name  # type: ignore[index]

        self.agent = SAC(
            models=self.models,
            memory=RandomMemory(
                memory_size=cfg.memory_size,
                num_envs=self.env.num_envs,
                device=cfg.device,
            ),
            cfg=agent_cfg,
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            device=cfg.device,
        )

        trainer_cfg = {
            "timesteps": cfg.timesteps,
            "headless": cfg.headless,
            "environment_info": cfg.environment_info,
        }
        self.trainer = SequentialTrainer(cfg=trainer_cfg, env=self.env, agents=[self.agent])  # type: ignore[arg-type]
        if self.cfg.resume:
            self.agent.load(self.cfg.checkpoint_path)  # type: ignore[arg-type]
            print(f"Model loaded from {self.cfg.checkpoint_path}")

    def _maybe_compile_policy(self) -> None:
        if self.cfg.compile_policy:
            self.agent.policy = torch.compile(self.agent.policy)  # type: ignore[assignment]

    def train(self) -> None:
        self._maybe_compile_policy()
        self.trainer.train()

    def eval(self) -> None:
        self._maybe_compile_policy()
        self.agent.policy.eval()  # type: ignore[union-attr]
        states, _ = self.env.reset()
        with torch.no_grad():
            for _ in range(self.cfg.eval_steps):
                actions, _, _ = self.agent.act(states, timestep=0, timesteps=0)
                states, _, _, _, _ = self.env.step(actions)
