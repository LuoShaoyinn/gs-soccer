from __future__ import annotations

from dataclasses import dataclass
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl import config
from skrl.agents.torch.sac import SAC
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer

from network import Policy, QNetwork

from .algorithm import Algorithm, AlgorithmConfig


@dataclass(kw_only=True)
class SACAlgorithmConfig(AlgorithmConfig):
    memory_size: int = 262144
    timesteps: int = 30000

    experiment_name: str = "mos9_gait_sac"
    experiment_directory: str = "runs"

    batch_size: int = 512
    random_timesteps: int = 256
    learning_starts: int = 1024
    gradient_steps: int = 2

    tau: float = 0.005
    lr: float = 3e-4
    alpha_lr: float = 3e-4
    initial_entropy_value: float = 0.2


class TrackedSAC(SAC):
    def update(self, *, timestep: int, timesteps: int) -> None:
        for _ in range(self.cfg.gradient_steps):
            (
                sampled_observations,
                sampled_states,
                sampled_actions,
                sampled_rewards,
                sampled_next_observations,
                sampled_next_states,
                sampled_terminated,
            ) = self.memory.sample(names=self._tensors_names, batch_size=self.cfg.batch_size)[0]

            with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                inputs = {
                    "observations": self._observation_preprocessor(sampled_observations, train=True),
                    "states": self._state_preprocessor(sampled_states, train=True),
                }
                next_inputs = {
                    "observations": self._observation_preprocessor(sampled_next_observations, train=True),
                    "states": self._state_preprocessor(sampled_next_states, train=True),
                }

                with torch.no_grad():
                    next_actions, outputs = self.policy.act(next_inputs, role="policy")
                    next_log_prob = outputs["log_prob"]

                    target_q1_values, _ = self.target_critic_1.act(
                        {**next_inputs, "taken_actions": next_actions}, role="target_critic_1"
                    )
                    target_q2_values, _ = self.target_critic_2.act(
                        {**next_inputs, "taken_actions": next_actions}, role="target_critic_2"
                    )
                    target_q_values = (
                        torch.min(target_q1_values, target_q2_values)
                        - self._entropy_coefficient * next_log_prob
                    )
                    target_values = (
                        sampled_rewards
                        + self.cfg.discount_factor
                        * sampled_terminated.logical_not()
                        * target_q_values
                    )

                critic_1_values, _ = self.critic_1.act(
                    {**inputs, "taken_actions": sampled_actions}, role="critic_1"
                )
                critic_2_values, _ = self.critic_2.act(
                    {**inputs, "taken_actions": sampled_actions}, role="critic_2"
                )
                critic_loss = (
                    F.mse_loss(critic_1_values, target_values)
                    + F.mse_loss(critic_2_values, target_values)
                ) / 2

            self.critic_optimizer.zero_grad()
            self.scaler.scale(critic_loss).backward()
            if config.torch.is_distributed:
                self.critic_1.reduce_parameters()
                self.critic_2.reduce_parameters()
            if self.cfg.grad_norm_clip > 0:
                self.scaler.unscale_(self.critic_optimizer)
                nn.utils.clip_grad_norm_(
                    itertools.chain(self.critic_1.parameters(), self.critic_2.parameters()),
                    self.cfg.grad_norm_clip,
                )
            self.scaler.step(self.critic_optimizer)

            with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                actions, outputs = self.policy.act(inputs, role="policy")
                log_prob = outputs["log_prob"]
                critic_1_values, _ = self.critic_1.act(
                    {**inputs, "taken_actions": actions}, role="critic_1"
                )
                critic_2_values, _ = self.critic_2.act(
                    {**inputs, "taken_actions": actions}, role="critic_2"
                )
                policy_loss = (
                    self._entropy_coefficient * log_prob
                    - torch.min(critic_1_values, critic_2_values)
                ).mean()

            self.policy_optimizer.zero_grad()
            self.scaler.scale(policy_loss).backward()
            if config.torch.is_distributed:
                self.policy.reduce_parameters()
            if self.cfg.grad_norm_clip > 0:
                self.scaler.unscale_(self.policy_optimizer)
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.grad_norm_clip)
            self.scaler.step(self.policy_optimizer)

            entropy_loss = torch.zeros((), device=self.device)
            if self.cfg.learn_entropy:
                with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                    entropy_loss = -(
                        self.log_entropy_coefficient
                        * (log_prob + self._target_entropy).detach()
                    ).mean()

                self.entropy_optimizer.zero_grad()
                self.scaler.scale(entropy_loss).backward()
                self.scaler.step(self.entropy_optimizer)
                self._entropy_coefficient = torch.exp(self.log_entropy_coefficient.detach())

            self.scaler.update()

            self.target_critic_1.update_parameters(self.critic_1, polyak=self.cfg.polyak)
            self.target_critic_2.update_parameters(self.critic_2, polyak=self.cfg.polyak)

            if self.policy_scheduler:
                self.policy_scheduler.step()
            if self.critic_scheduler:
                self.critic_scheduler.step()

            if self.write_interval > 0:
                total_loss = policy_loss.detach() + critic_loss.detach() + entropy_loss.detach()
                self.track_data("Loss / Total loss", total_loss.item())
                self.track_data("Loss / Policy loss", policy_loss.item())
                self.track_data("Loss / Critic loss", critic_loss.item())
                if self.cfg.learn_entropy:
                    self.track_data("Loss / Entropy loss", entropy_loss.item())

                self.track_data("Q-network / Q1 (max)", torch.max(critic_1_values).item())
                self.track_data("Q-network / Q1 (min)", torch.min(critic_1_values).item())
                self.track_data("Q-network / Q1 (mean)", torch.mean(critic_1_values).item())
                self.track_data("Q-network / Q2 (max)", torch.max(critic_2_values).item())
                self.track_data("Q-network / Q2 (min)", torch.min(critic_2_values).item())
                self.track_data("Q-network / Q2 (mean)", torch.mean(critic_2_values).item())
                self.track_data("Target / Target (max)", torch.max(target_values).item())
                self.track_data("Target / Target (min)", torch.min(target_values).item())
                self.track_data("Target / Target (mean)", torch.mean(target_values).item())
                self.track_data("Coefficient / Entropy coefficient", self._entropy_coefficient.item())

                if self.policy_scheduler:
                    self.track_data("Learning / Policy learning rate", self.policy_scheduler.get_last_lr()[0])
                if self.critic_scheduler:
                    self.track_data("Learning / Critic learning rate", self.critic_scheduler.get_last_lr()[0])


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

        agent_cfg = {
            "gradient_steps": cfg.gradient_steps,
            "batch_size": cfg.batch_size,
            "discount_factor": cfg.discount_factor,
            "polyak": cfg.tau,
            "learning_rate": (cfg.lr, cfg.lr, cfg.alpha_lr),
            "random_timesteps": cfg.random_timesteps,
            "learning_starts": cfg.learning_starts,
            "mixed_precision": cfg.mixed_precision,
            "initial_entropy_value": cfg.initial_entropy_value,
            "experiment": {
                "directory": cfg.experiment_directory,
                "write_interval": cfg.write_interval,
                "checkpoint_interval": cfg.checkpoint_interval,
                "experiment_name": cfg.experiment_name,
            },
        }

        self.agent = TrackedSAC(
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
            "disable_progressbar": False,
        }
        self.trainer = SequentialTrainer(cfg=trainer_cfg, env=self.env, agents=[self.agent])
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
            for timestep in range(self.cfg.eval_steps):
                actions, outputs = self.agent.act(
                    states,
                    None,
                    timestep=timestep,
                    timesteps=self.cfg.eval_steps,
                )
                states, _, _, _, _ = self.env.step(outputs.get("mean_actions", actions))
