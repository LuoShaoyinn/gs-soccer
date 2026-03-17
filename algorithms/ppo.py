from __future__ import annotations

import os
import torch

from skrl.agents.torch.ppo import PPO
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer

from .algorithm import Algorithm


class PPOAlgorithm(Algorithm):
    def __init__(
        self,
        *,
        cfg,
        env=None,
        agent_cfg: dict,
        models: dict,
        device: str,
        rollout_steps: int,
        trainer_cfg: dict,
    ):
        super().__init__(cfg, env=env)

        self.agent = PPO(
            models=models,
            memory=RandomMemory(
                memory_size=rollout_steps,
                num_envs=self.env.num_envs,
                device=device,
            ),
            cfg=agent_cfg,
            observation_space=self.env.observation_space,  # type: ignore[arg-type]
            action_space=self.env.action_space,  # type: ignore[arg-type]
            device=device,
        )
        self.trainer = SequentialTrainer(cfg=trainer_cfg, env=self.env, agents=[self.agent])  # type: ignore[arg-type]

    def execute(
        self,
        *,
        eval_mode: bool,
        resume_training: bool,
        checkpoint_path: str,
        eval_steps: int = 1000,
        compile_policy: bool = False,
    ) -> None:
        if compile_policy:
            self.agent.policy = torch.compile(self.agent.policy)  # type: ignore[assignment]

        if (eval_mode or resume_training) and os.path.exists(checkpoint_path):
            self.agent.load(checkpoint_path)
            print(f"Model loaded from {checkpoint_path}")

        if not eval_mode:
            self.trainer.train()
            return

        self.agent.policy.eval()  # type: ignore[union-attr]
        states, _ = self.env.reset()
        with torch.no_grad():
            for _ in range(eval_steps):
                actions, _, _ = self.agent.act(states, timestep=0, timesteps=0)
                states, _, _, _, _ = self.env.step(actions)
