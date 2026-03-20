from __future__ import annotations
from dataclasses import dataclass

import torch

from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer

from network import Policy, Value

from .algorithm import Algorithm, AlgorithmConfig


@dataclass(kw_only=True)
class PPOAlgorithmConfig(AlgorithmConfig):
    rollout_steps: int = 32
    experiment_name: str = "dribble_ppo"
    learning_epochs: int = 8
    mini_batches: int = 4


class PPOAlgorithm(Algorithm):
    def __init__(self, env, cfg: PPOAlgorithmConfig):
        super().__init__(env, cfg)

        self.models = {
            "policy": Policy(env.observation_space, env.action_space, cfg.device),
            "value": Value(env.observation_space, env.action_space, cfg.device),
        }
        for model in self.models.values():
            model.init_parameters(
                method_name=cfg.init_method_name,
                mean=cfg.init_mean,
                std=cfg.init_std,
            )

        agent_cfg = PPO_DEFAULT_CONFIG.copy()
        agent_cfg["rollouts"] = cfg.rollout_steps
        agent_cfg["discount_factor"] = cfg.discount_factor
        agent_cfg["learning_epochs"] = cfg.learning_epochs
        agent_cfg["mixed_precision"] = cfg.mixed_precision
        agent_cfg["mini_batches"] = cfg.mini_batches

        agent_cfg["experiment"]["directory"] = cfg.experiment_directory  # type: ignore[index]
        agent_cfg["experiment"]["write_interval"] = cfg.write_interval  # type: ignore[index]
        agent_cfg["experiment"]["checkpoint_interval"] = cfg.checkpoint_interval  # type: ignore[index]
        agent_cfg["experiment"]["experiment_name"] = cfg.experiment_name  # type: ignore[index]

        self.agent = PPO(
            models=self.models,
            memory=RandomMemory(
                memory_size=cfg.rollout_steps,
                num_envs=self.env.num_envs,
                device=cfg.device,
            ),
            cfg=agent_cfg,
            observation_space=self.env.observation_space,  # type: ignore[arg-type]
            action_space=self.env.action_space,  # type: ignore[arg-type]
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
