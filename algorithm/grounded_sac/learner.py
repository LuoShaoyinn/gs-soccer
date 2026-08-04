from __future__ import annotations

import copy
from collections.abc import Mapping

import torch
import torch.nn.functional as F

from .config import GroundedSACConfig
from .networks import FrozenObservationNormalizer, VectorActor, VectorCritic
from .replay import ReplayBatch, VectorReplayBuffer


MONITOR_HORIZONS = (10, 20, 50, 100, 350)


def _rank_loss(value: torch.Tensor, step_penalty: float) -> torch.Tensor:
    """Horizon constraint with a positive *cost magnitude*.

    Runtime rewards use a negative step penalty.  The ranking relation is
    therefore `f_h >= f_(h-1) - abs(step_penalty)`, which permits all-one
    success vectors while constraining adjacent heads.
    """

    return torch.relu(value[:, :-1] - abs(step_penalty) - value[:, 1:]).square().mean()


def _td_loss(prediction: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    smooth = F.smooth_l1_loss(prediction, target)
    mse = F.mse_loss(prediction, target)
    return smooth + 0.5 * mse, smooth, mse


def _cat_batches(*batches: ReplayBatch) -> ReplayBatch:
    return ReplayBatch(**{
        name: torch.cat([getattr(batch, name) for batch in batches], dim=0)
        for name in ReplayBatch.__dataclass_fields__
    })


class GroundedSACLearner:
    """Moving human IQL reference plus vector SAC, without the fence.

    IQL is updated exclusively from successful human suffix rows.  SAC gets
    every real transition.  In this first experiment every state is considered
    familiar, so the controller and SAC continuation are always SAC and the
    outside-fence loss is deliberately zero.
    """

    def __init__(self, config: GroundedSACConfig, replay: VectorReplayBuffer) -> None:
        self.cfg = config
        self.replay = replay
        self.device = torch.device(config.device)
        self.normalizer = FrozenObservationNormalizer(config.observation_dim).to(self.device)

        actor_args = (config.observation_dim, config.action_dim, config.hidden_dim)
        critic_args = (*actor_args, config.horizons)
        self.iql_actor = VectorActor(*actor_args).to(self.device)
        self.iql_q1 = VectorCritic(*critic_args).to(self.device)
        self.iql_q2 = VectorCritic(*critic_args).to(self.device)
        self.iql_v = VectorCritic(config.observation_dim, 0, config.hidden_dim, config.horizons).to(self.device)
        # A value network takes only state; its zero-width action input is
        # explicit and keeps model construction uniform with Q networks.

        self.sac_actor = VectorActor(*actor_args).to(self.device)
        self.sac_q1 = VectorCritic(*critic_args).to(self.device)
        self.sac_q2 = VectorCritic(*critic_args).to(self.device)
        self.target_q1 = copy.deepcopy(self.sac_q1).eval().requires_grad_(False)
        self.target_q2 = copy.deepcopy(self.sac_q2).eval().requires_grad_(False)

        self.iql_q_optim = torch.optim.Adam(
            [*self.iql_q1.parameters(), *self.iql_q2.parameters()], lr=config.iql_learning_rate
        )
        self.iql_v_optim = torch.optim.Adam(self.iql_v.parameters(), lr=config.iql_learning_rate)
        self.iql_actor_optim = torch.optim.Adam(self.iql_actor.parameters(), lr=config.iql_learning_rate)
        self.sac_q_optim = torch.optim.Adam(
            [*self.sac_q1.parameters(), *self.sac_q2.parameters()], lr=config.learning_rate
        )
        self.sac_actor_optim = torch.optim.Adam(self.sac_actor.parameters(), lr=config.actor_learning_rate)
        self.update_count = 0

    @torch.no_grad()
    def fit_normalizer_once(self) -> None:
        self.normalizer.fit(self.replay.all_human_observations().to(self.device))

    def _obs(self, observation: torch.Tensor) -> torch.Tensor:
        # During initial fully-human collection, learner proposals are recorded
        # for audit only.  Identity normalization avoids a circular dependency;
        # no learner loss is allowed before `fit_normalizer_once()` freezes it.
        if not bool(self.normalizer.fitted):
            return observation.float()
        return self.normalizer(observation.float())

    def _value(self, observation: torch.Tensor) -> torch.Tensor:
        # VectorCritic with an empty action is a direct state-value MLP.
        return self.iql_v(self._obs(observation), observation.new_empty((len(observation), 0)))

    def _iql_q_min(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        obs = self._obs(observation)
        return torch.minimum(self.iql_q1(obs, action), self.iql_q2(obs, action))

    def _sac_q_min(self, observation: torch.Tensor, action: torch.Tensor, *, target: bool = False) -> torch.Tensor:
        obs = self._obs(observation)
        q1, q2 = (self.target_q1, self.target_q2) if target else (self.sac_q1, self.sac_q2)
        return torch.minimum(q1(obs, action), q2(obs, action))

    def iql_action(self, observation: torch.Tensor) -> torch.Tensor:
        return self.iql_actor(self._obs(observation))

    def sac_action(self, observation: torch.Tensor) -> torch.Tensor:
        return self.sac_actor(self._obs(observation))

    def controller_action(self, observation: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """No-fence runtime controller: SAC is always executable for now."""
        with torch.no_grad():
            sac = self.sac_action(observation)
            iql = self.iql_action(observation)
        flags = torch.ones(len(observation), dtype=torch.bool, device=observation.device)
        return sac, {
            "sac_proposal": sac,
            "iql_proposal": iql,
            "familiar": flags,
            "familiarity_score": torch.zeros(len(observation), device=observation.device),
            "familiarity_smoothed_score": torch.zeros(len(observation), device=observation.device),
            "controller_mode": torch.zeros(len(observation), dtype=torch.long, device=observation.device),
        }

    def _bellman(self, reward: torch.Tensor, success: torch.Tensor, terminal: torch.Tensor, continuation: torch.Tensor, *, mask_terminal: bool) -> torch.Tensor:
        target = reward[:, None].expand(-1, self.cfg.horizons).clone()
        if self.cfg.horizons > 1:
            can_continue = (~terminal if mask_terminal else torch.ones_like(terminal)).to(target.dtype)
            target[:, 1:] += self.cfg.gamma * continuation[:, :-1] * can_continue[:, None]
        return torch.where(success[:, None], torch.ones_like(target), target).detach()

    def _update_iql(self, batch: ReplayBatch) -> dict[str, torch.Tensor]:
        obs, next_obs, action = batch.observation, batch.next_observation, batch.action
        with torch.no_grad():
            # Human IQL never reads SAC transitions, physical timestep, or done.
            target = self._bellman(batch.reward, batch.success, batch.terminal, self._value(next_obs), mask_terminal=False)
        normalized = self._obs(obs)
        q1, q2 = self.iql_q1(normalized, action), self.iql_q2(normalized, action)
        q_loss_1, q_smooth_1, q_mse_1 = _td_loss(q1, target)
        q_loss_2, q_smooth_2, q_mse_2 = _td_loss(q2, target)
        q_rank = _rank_loss(q1, self.cfg.step_penalty) + _rank_loss(q2, self.cfg.step_penalty)
        q_loss = q_loss_1 + q_loss_2 + self.cfg.rank_weight * q_rank
        self.iql_q_optim.zero_grad(set_to_none=True)
        q_loss.backward()
        self.iql_q_optim.step()

        value = self._value(obs)
        with torch.no_grad():
            q_min = self._iql_q_min(obs, action)
        difference = q_min - value
        expectile_weight = torch.where(difference > 0, self.cfg.expectile, 1.0 - self.cfg.expectile)
        v_expectile = (expectile_weight * difference.square()).mean()
        v_rank = _rank_loss(value, self.cfg.step_penalty)
        v_loss = v_expectile + self.cfg.rank_weight * v_rank
        self.iql_v_optim.zero_grad(set_to_none=True)
        v_loss.backward()
        self.iql_v_optim.step()

        predicted = self.iql_action(obs)
        with torch.no_grad():
            advantage = (self._iql_q_min(obs, action) - self._value(obs)).mean(dim=1)
            weight = torch.exp(self.cfg.awr_beta * advantage).clamp_max(self.cfg.awr_max_weight)
        actor_loss = (weight[:, None] * (predicted - action).square()).mean()
        self.iql_actor_optim.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.iql_actor_optim.step()
        return {
            "iql/critic_td": (q_loss_1 + q_loss_2).detach(),
            "iql/critic_smooth_l1": (q_smooth_1 + q_smooth_2).detach(),
            "iql/critic_mse": (q_mse_1 + q_mse_2).detach(),
            "iql/expectile": v_expectile.detach(),
            "iql/actor_loss": actor_loss.detach(),
            "iql/horizon": (q_rank + v_rank).detach(),
            "iql/advantage": advantage.mean().detach(),
        }

    def _floor_loss(self, batch: ReplayBatch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        obs = batch.observation
        with torch.no_grad():
            reference_action = self.iql_action(obs)
            reference_value = self._iql_q_min(obs, reference_action).detach()
        normalized = self._obs(obs)
        q1 = self.sac_q1(normalized, reference_action)
        q2 = self.sac_q2(normalized, reference_action)
        violations = torch.cat((torch.relu(reference_value - q1), torch.relu(reference_value - q2)), dim=0)
        return violations.square().mean(), {
            "floor/loss": violations.square().mean().detach(),
            "floor/mean_violation": violations.mean().detach(),
            "floor/rms_violation": violations.square().mean().sqrt().detach(),
            "floor/p95_violation": torch.quantile(violations.detach().flatten(), 0.95),
            "floor/max_violation": violations.max().detach(),
            "floor/fraction_above_0.01": (violations > 0.01).float().mean().detach(),
        }

    def _update_sac(self, td_batch: ReplayBatch, floor_batch: ReplayBatch, human_count: int) -> dict[str, torch.Tensor]:
        obs, next_obs, action = td_batch.observation, td_batch.next_observation, td_batch.action
        with torch.no_grad():
            # With the fence disabled every next state is familiar.
            continuation = self._sac_q_min(next_obs, self.sac_action(next_obs), target=True)
            target = self._bellman(td_batch.reward, td_batch.success, td_batch.terminal, continuation, mask_terminal=True)
        normalized = self._obs(obs)
        q1, q2 = self.sac_q1(normalized, action), self.sac_q2(normalized, action)
        td1, smooth1, mse1 = _td_loss(q1, target)
        td2, smooth2, mse2 = _td_loss(q2, target)
        rank = _rank_loss(q1, self.cfg.step_penalty) + _rank_loss(q2, self.cfg.step_penalty)
        floor, floor_metrics = self._floor_loss(floor_batch)
        # No unfamiliar states exist in this revision; retain the named term
        # and metric to make fence enablement a local change later.
        outside = torch.zeros((), device=self.device)
        critic_loss = td1 + td2 + self.cfg.rank_weight * rank + self.cfg.floor_weight * floor + self.cfg.outside_weight * outside
        self.sac_q_optim.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.sac_q_optim.step()

        policy_action = self.sac_action(obs)
        actor_loss = -self._sac_q_min(obs, policy_action).mean()
        self.sac_actor_optim.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.sac_actor_optim.step()

        with torch.no_grad():
            for target_q, online_q in ((self.target_q1, self.sac_q1), (self.target_q2, self.sac_q2)):
                for target_parameter, parameter in zip(target_q.parameters(), online_q.parameters(), strict=True):
                    target_parameter.lerp_(parameter, self.cfg.target_tau)
            q_sac = self._sac_q_min(obs, self.sac_action(obs))
            q_iql = self._sac_q_min(obs, self.iql_action(obs))
        metrics = {
            "sac/td_smooth_l1": (smooth1 + smooth2).detach(),
            "sac/td_mse": (mse1 + mse2).detach(),
            "sac/actor_loss": actor_loss.detach(),
            "sac/horizon": rank.detach(),
            "sac/outside_loss": outside,
            "floor/human_suffix_samples": torch.tensor(float(human_count), device=self.device),
            "floor/familiar_samples": torch.tensor(float(len(floor_batch.reward) - human_count), device=self.device),
            "comparison/inside_sac_minus_iql": (q_sac - q_iql).mean().detach(),
            "comparison/outside_sac_minus_iql": torch.zeros((), device=self.device),
            "fence/familiarity_score": torch.zeros((), device=self.device),
            "fence/unfamiliar_fraction": torch.zeros((), device=self.device),
            "fence/recovery_burst_count": torch.zeros((), device=self.device),
            "fence/recovery_burst_duration": torch.zeros((), device=self.device),
            "fence/return_to_familiar_fraction": torch.ones((), device=self.device),
        }
        metrics.update(floor_metrics)
        for horizon in MONITOR_HORIZONS:
            metrics[f"sac/q_h{horizon}"] = q_sac[:, horizon - 1].mean().detach()
        return metrics

    def update(self) -> dict[str, float]:
        if len(self.replay) < self.cfg.batch_size:
            raise RuntimeError("not enough replay transitions")
        human_batch = self.replay.sample(self.cfg.batch_size, self.device, human_suffix=True)
        iql_metrics = self._update_iql(human_batch)
        td_batch = self.replay.sample(self.cfg.batch_size, self.device)
        familiar_batch = self.replay.sample(self.cfg.batch_size // 2, self.device)
        floor_human = self.replay.sample(self.cfg.batch_size // 2, self.device, human_suffix=True)
        sac_metrics = self._update_sac(td_batch, _cat_batches(floor_human, familiar_batch), len(floor_human.reward))
        with torch.no_grad():
            reference = self._iql_q_min(human_batch.observation, self.iql_action(human_batch.observation))
        for horizon in MONITOR_HORIZONS:
            iql_metrics[f"iql/q_ref_h{horizon}"] = reference[:, horizon - 1].mean().detach()
        self.update_count += 1
        return {key: float(value.detach().mean().cpu()) for key, value in {**iql_metrics, **sac_metrics}.items()}
