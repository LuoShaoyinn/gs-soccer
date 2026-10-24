from contextlib import contextmanager

import torch
from torch import nn

from .geometry import from_rotation_vector, multiply, normalize


def mlp(inputs, widths, outputs, small_output=False):
    layers = []
    for width in widths:
        layers += [nn.Linear(inputs, width), nn.ELU()]
        inputs = width
    layers.append(nn.Linear(inputs, outputs))
    if small_output:
        nn.init.uniform_(layers[-1].weight, -0.001, 0.001)
        nn.init.zeros_(layers[-1].bias)
    return nn.Sequential(*layers)


@contextmanager
def frozen(*modules):
    # Freeze weights, preserving derivatives with respect to model/critic inputs.
    parameters = [(p, p.requires_grad) for m in modules for p in m.parameters()]
    try:
        for p, _ in parameters:
            p.requires_grad_(False)
        yield
    finally:
        for p, flag in parameters:
            p.requires_grad_(flag)


def model_features(task, state):
    # Absolute world xy cannot affect flat-ground dynamics or the critic's translation frame.
    x = state / task.scale
    return torch.cat(
        (
            x[..., : task.pos.start],
            torch.zeros_like(x[..., task.pos][..., :2]),
            x[..., task.pos.start + 2 :],
        ),
        -1,
    )


class WorldModel(nn.Module):
    def __init__(self, task):
        super().__init__()
        self.task = task
        self.dynamics = mlp(
            task.state_dim + task.n + task.model_context_dim,
            task.cfg.model_widths,
            task.physical_dim,
            small_output=True,
        )
        self.termination = mlp(task.state_dim + task.goal_frame_dim, (128, 128), 1)

    def forward(self, state, action, context):
        t = self.task
        delta = self.dynamics(
            torch.cat(
                (model_features(t, state), action, t.model_context(state, context)), -1
            )
        )
        # Kinematic integration is a stable initial prior. The network learns contact/actuation corrections.
        q = state[..., t.q] + state[..., t.dq] * t.cfg.dt
        p = state[..., t.pos] + state[..., t.vel] * t.cfg.dt
        r = multiply(
            from_rotation_vector(state[..., t.ang] * t.cfg.dt), state[..., t.quat]
        )
        prior = torch.cat(
            (q, state[..., t.dq], p, r, state[..., t.vel], state[..., t.ang]), -1
        )
        physical = prior + 0.1 * delta * t.scale[: t.physical_dim]
        return torch.cat(
            (
                physical[..., : t.quat.start],
                normalize(physical[..., t.quat]),
                physical[..., t.quat.stop :],
                action,
            ),
            -1,
        )

    def death_logit(self, state, reference):
        goal = self.task.goal(state, reference.unsqueeze(-2))
        return self.termination(
            torch.cat((model_features(self.task, state), goal), -1)
        ).squeeze(-1)


class Actor(nn.Module):
    def __init__(self, task):
        super().__init__()
        cfg = task.cfg
        self.encoder = mlp(
            task.goal_frame_dim * cfg.future, cfg.actor_widths, cfg.latent_dim
        )
        self.decoder = mlp(
            cfg.latent_dim + task.obs_dim * cfg.history,
            cfg.actor_widths,
            task.n,
            small_output=True,
        )

    def forward(self, history, goal):
        latent = torch.tanh(self.encoder(goal))
        return self.decoder(torch.cat((latent, history.flatten(-2)), -1))


class Critic(nn.Module):
    def __init__(self, task):
        super().__init__()
        self.task = task
        self.net = mlp(
            task.state_dim
            + task.goal_frame_dim * task.cfg.future
            + task.model_context_dim,
            task.cfg.critic_widths,
            1,
        )

    def forward(self, state, goal, context):
        return self.net(
            torch.cat(
                (
                    model_features(self.task, state),
                    goal,
                    self.task.model_context(state, context),
                ),
                -1,
            )
        ).squeeze(-1)
