import math

import torch
from torch.nn import functional as F

from .networks import Actor, Critic, WorldModel, frozen


def lambda_returns(rewards, deaths, values, gamma, lam):
    """Eq. 3, with values shaped [H+1, B] and all targets detached."""
    target = values[-1]
    result = []
    for i in reversed(range(len(rewards))):
        target = rewards[i] + gamma * (1 - deaths[i]) * (
            (1 - lam) * values[i + 1] + lam * target
        )
        result.append(target)
    return torch.stack(list(reversed(result))).detach()


def checked_step(loss, optimizer, parameters, element_clip=False):
    if not torch.isfinite(loss):
        raise FloatingPointError("nonfinite learning loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    parameters = list(parameters)
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
    if element_clip:
        for p in parameters:
            if p.grad is not None:
                p.grad.clamp_(-0.1, 0.1)
    optimizer.step()
    return float(norm)


class Learner:
    def __init__(self, task, library):
        self.task, self.library, self.cfg = task, library, task.cfg
        self.device = task.home.device
        self.model, self.actor, self.critic = (
            cls(task).to(self.device) for cls in (WorldModel, Actor, Critic)
        )
        # Task contains analytic buffers, not trainable parameters.
        self.optimizers = {
            "model": torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr),
            "actor": torch.optim.Adam(
                self.actor.parameters(), lr=self.cfg.lr, betas=(0.9, 0.95)
            ),
            "critic": torch.optim.Adam(self.critic.parameters(), lr=self.cfg.lr),
        }
        self.updates = 0

    def fit_model(self, batch):
        t = self.task
        state = batch["state"][:, 0]
        dynamics_loss, death_loss = 0.0, 0.0
        for k in range(batch["action"].shape[1]):
            state = self.model(state, batch["action"][:, k], batch["context"][:, k])
            target = batch["next_state"][:, k]
            dynamics_loss += (
                (
                    (state[..., : t.physical_dim] - target[..., : t.physical_dim])
                    / t.scale[: t.physical_dim]
                )
                .square()
                .mean()
            )
            ref = self.library.reference(
                batch["clip"][:, k],
                batch["frame"][:, k] + 1,
                context=batch["context"][:, k],
            )[:, 0]
            death_loss += F.binary_cross_entropy_with_logits(
                self.model.death_logit(state, ref), batch["terminal"][:, k].float()
            )
        length = batch["action"].shape[1]
        loss = (dynamics_loss + death_loss) / length
        norm = checked_step(loss, self.optimizers["model"], self.model.parameters())
        return {
            "model/state_loss": float(dynamics_loss.detach() / length),
            "model/termination_loss": float(death_loss.detach() / length),
            "model/gradient_norm": norm,
        }

    def improve(self, batch):
        t, cfg, lib = self.task, self.cfg, self.library
        state, context = batch["state"][:, 0], batch["context"][:, 0]
        history = batch["history"][:, 0]
        clips, frames = batch["clip"][:, 0], batch["frame"][:, 0]
        states, goals, rewards, deaths, values = [], [], [], [], []
        survival, objective = (
            torch.ones(len(state), device=self.device),
            torch.zeros(len(state), device=self.device),
        )
        self.optimizers["model"].zero_grad(set_to_none=True)
        self.optimizers["critic"].zero_grad(set_to_none=True)
        # No no_grad here: actor needs model Jacobians and the terminal-value derivative.
        with frozen(self.model, self.critic):
            for k in range(cfg.horizon):
                ref = lib.reference(clips, frames + k, cfg.future, context=context)
                goal = t.goal(state, ref)
                states.append(state.detach())
                goals.append(goal.detach())
                values.append(self.critic(state, goal, context))
                action = self.actor(history, goal)
                next_state = self.model(state, action, context)
                target = lib.reference(clips, frames + k + 1, context=context)[:, 0]
                reward, _ = t.reward(next_state, state[..., t.previous], action, target)
                death = torch.sigmoid(self.model.death_logit(next_state, target))
                # Clip end and time horizon are truncations, not physical failure labels.
                end = frames + k + 1 >= lib.lengths[clips] - 1
                death = torch.where(end, torch.ones_like(death), death)
                objective = objective + cfg.gamma**k * survival * reward
                survival = survival * (1 - death)
                rewards.append(reward)
                deaths.append(death)
                state = next_state
                history = torch.cat(
                    (
                        history[:, 1:],
                        t.observation(state, cfg.sensor_noise).unsqueeze(1),
                    ),
                    1,
                )
            goal = t.goal(
                state,
                lib.reference(clips, frames + cfg.horizon, cfg.future, context=context),
            )
            values.append(self.critic(state, goal, context))
            objective += cfg.gamma**cfg.horizon * survival * values[-1]
            actor_loss = -objective.mean()
            actor_norm = checked_step(
                actor_loss,
                self.optimizers["actor"],
                self.actor.parameters(),
                element_clip=True,
            )
        returns = lambda_returns(
            torch.stack(rewards).detach(),
            torch.stack(deaths).detach(),
            torch.stack(values).detach(),
            cfg.gamma,
            cfg.gae_lambda,
        ).flatten()
        states, goals = torch.cat(states), torch.cat(goals)
        contexts = context.repeat(cfg.horizon, 1)
        critic_loss = torch.zeros((), device=self.device)
        for _ in range(cfg.critic_epochs):
            order = torch.randperm(len(returns), device=self.device)
            for idx in order.tensor_split(min(cfg.critic_minibatches, len(order))):
                critic_loss = F.mse_loss(
                    self.critic(states[idx], goals[idx], contexts[idx]), returns[idx]
                )
                checked_step(
                    critic_loss, self.optimizers["critic"], self.critic.parameters()
                )
        self.updates += 1
        factor = 0.03 + 0.97 * 0.5 * (
            1 + math.cos(math.pi * min(self.updates, 250000) / 250000)
        )
        for optimizer in self.optimizers.values():
            for group in optimizer.param_groups:
                group["lr"] = cfg.lr * factor
        return {
            "policy/loss": float(actor_loss.detach()),
            "policy/gradient_norm": actor_norm,
            "critic/loss": float(critic_loss.detach()),
            "policy/imagined_return": float(objective.mean().detach()),
            "policy/survival": float(survival.mean().detach()),
        }

    def state_dict(self):
        return {
            "model": self.model.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizers": {k: o.state_dict() for k, o in self.optimizers.items()},
            "updates": self.updates,
        }

    def warm_start(self, state):
        """Copy a compatible tracker; zero newly added privileged input columns.

        Optimizers, replay and update counters deliberately start fresh because
        the dynamics distribution and vector batch size have changed.
        """
        expanded = []
        for name in ("model", "actor", "critic"):
            module = getattr(self, name)
            target = module.state_dict()
            source = state[name]
            if set(source) != set(target):
                raise ValueError(f"incompatible warm-start {name} keys")
            for key, value in source.items():
                if value.shape == target[key].shape:
                    target[key] = value
                elif (
                    (name, key)
                    in (("model", "dynamics.0.weight"), ("critic", "net.0.weight"))
                    and value.shape[0] == target[key].shape[0]
                    and value.shape[1] < target[key].shape[1]
                ):
                    target[key].zero_()
                    target[key][:, : value.shape[1]] = value
                    expanded.append(f"{name}.{key}")
                else:
                    raise ValueError(f"incompatible warm-start tensor {name}.{key}")
            module.load_state_dict(target)
        return expanded

    def load_state_dict(self, data):
        for key in ("model", "actor", "critic"):
            getattr(self, key).load_state_dict(data[key])
        for key, optimizer in self.optimizers.items():
            optimizer.load_state_dict(data["optimizers"][key])
        self.updates = data["updates"]
