from dataclasses import dataclass

import genesis as gs
import torch

from skrl.agents.torch.ppo import PPO, PPO_CFG
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer

from network import Policy, Value
from .algorithm import Algorithm, AlgorithmConfig


@dataclass(kw_only=True)
class PPOAlgorithmConfig(AlgorithmConfig):
    rollouts: int = 24
    learning_epochs: int = 5
    mini_batches: int = 4
    learning_rate: float = 1e-3
    random_timesteps: int = 0
    gae_lambda: float = 0.95
    ratio_clip: float = 0.2
    value_clip: float = 0.2
    grad_norm_clip: float = 1.0
    entropy_loss_scale: float = 0.005
    value_loss_scale: float = 1.0
    kl_threshold: float = 0.01
    initial_policy_log_std: float = 0.4


class PPOAlgorithm(Algorithm):
    def __init__(self, env, cfg: PPOAlgorithmConfig):
        super().__init__(env, cfg)
        self.models = {
            "policy": Policy(env.observation_space, env.action_space, cfg.device),
            "value": Value(env.observation_space, env.action_space, cfg.device),
        }
        for model in self.models.values():
            model.init_parameters(method_name=cfg.init_method_name, mean=cfg.init_mean, std=cfg.init_std)
        self.models["policy"].log_std_parameter.data.fill_(cfg.initial_policy_log_std)

        agent_cfg = PPO_CFG(
            rollouts=cfg.rollouts,
            learning_epochs=cfg.learning_epochs,
            mini_batches=cfg.mini_batches,
            discount_factor=cfg.discount_factor,
            gae_lambda=cfg.gae_lambda,
            learning_rate=cfg.learning_rate,
            random_timesteps=cfg.random_timesteps,
            ratio_clip=cfg.ratio_clip,
            value_clip=cfg.value_clip,
            grad_norm_clip=cfg.grad_norm_clip,
            entropy_loss_scale=cfg.entropy_loss_scale,
            value_loss_scale=cfg.value_loss_scale,
            kl_threshold=cfg.kl_threshold,
            mixed_precision=cfg.mixed_precision,
        )
        agent_cfg.experiment.directory = cfg.experiment_directory
        agent_cfg.experiment.experiment_name = cfg.experiment_name
        agent_cfg.experiment.write_interval = cfg.write_interval
        agent_cfg.experiment.checkpoint_interval = cfg.checkpoint_interval

        self.agent = PPO(
            models=self.models,
            memory=RandomMemory(memory_size=cfg.rollouts, num_envs=env.num_envs, device=cfg.device),
            cfg=agent_cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=cfg.device,
        )
        self.trainer = SequentialTrainer(
            cfg={
                "timesteps": cfg.timesteps,
                "headless": cfg.headless,
                "environment_info": cfg.environment_info,
                "disable_progressbar": False,
            },
            env=env,
            agents=[self.agent],
        )

        if cfg.resume and cfg.checkpoint_path:
            self.agent.load(cfg.checkpoint_path)
            print(f"Model loaded from {cfg.checkpoint_path}")

    def _maybe_compile_policy(self):
        if self.cfg.compile_policy:
            self.agent.policy = torch.compile(self.agent.policy)  # type: ignore[assignment]

    def train(self):
        self._maybe_compile_policy()
        self.trainer.train()

    def eval(self):
        self._maybe_compile_policy()
        self.agent.policy.eval()  # type: ignore[union-attr]
        states, _ = self.env.reset()
        with torch.no_grad():
            for t in range(self.cfg.eval_steps):
                actions, outputs = self.agent.act(states, None, timestep=t, timesteps=self.cfg.eval_steps)
                try:
                    states, _, _, _, _ = self.env.step(outputs.get("mean_actions", actions))
                except gs.GenesisException as exc:
                    if "Viewer closed" in str(exc):
                        break
                    raise
