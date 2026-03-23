from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import genesis as gs
import torch

from network import Policy, Value
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer

from algorithms.algorithm import Algorithm, AlgorithmConfig


@dataclass(kw_only=True)
class LeagueMember:
    name: str
    checkpoint_path: str | None
    elo: float = 1200.0
    games: int = 0


@dataclass(kw_only=True)
class AlphaStarEvolutionConfig(AlgorithmConfig):
    rollout_steps: int = 64
    learning_epochs: int = 6
    mini_batches: int = 8

    generations: int = 8
    train_timesteps_per_generation: int = 4096
    eval_episodes: int = 16
    snapshot_interval: int = 2
    promote_win_rate: float = 0.55
    max_league_size: int = 8

    learner_idx: int = 0
    experiment_name: str = "soccer_1v1_alphastar"


class LeagueEnvAdapter:
    def __init__(self, game_env, learner_idx: int = 0):
        self.game_env = game_env
        self.learner_idx = learner_idx
        self.opponent_idx = 1 - learner_idx

        self.num_envs = game_env.num_envs
        self.num_agents = 1
        self.is_vector_env = True
        self.observation_space = game_env.observation_space[self.learner_idx]
        self.action_space = game_env.action_space[self.learner_idx]

        self._latest_obs: list[torch.Tensor] | None = None
        self._opponent_fn = self._default_opponent

    def set_opponent_sampler(self, sampler):
        self._opponent_fn = sampler

    def _default_opponent(self, obs: torch.Tensor) -> torch.Tensor:
        action = torch.zeros((obs.shape[0], 3), dtype=torch.float, device=gs.device)
        ball_rel = obs[:, 9:11]
        ball_to_goal = obs[:, 13:15]
        action[:, 0:2] = torch.tanh(1.5 * ball_rel + 0.3 * ball_to_goal)
        return action

    def reset(self):
        obs, info = self.game_env.reset()
        self._latest_obs = obs
        return obs[self.learner_idx], info

    def step(self, learner_action: torch.Tensor):
        if self._latest_obs is None:
            raise RuntimeError("Call reset() before step()")
        opponent_obs = self._latest_obs[self.opponent_idx]
        with torch.no_grad():
            opponent_action = self._opponent_fn(opponent_obs)

        actions = [None, None]
        actions[self.learner_idx] = learner_action
        actions[self.opponent_idx] = opponent_action
        obs, reward, terminated, truncated, info = self.game_env.step(actions)  # type: ignore[arg-type]
        if "extra" in info and isinstance(info["extra"], list):
            info["extra"] = info["extra"][self.learner_idx]
        self._latest_obs = obs
        return obs[self.learner_idx], reward[self.learner_idx], terminated[self.learner_idx], truncated[self.learner_idx], info

    def close(self):
        self.game_env.close()


class AlphaStarEvolutionAlgorithm(Algorithm):
    def __init__(self, env, cfg: AlphaStarEvolutionConfig):
        self.game_env = env
        self.env_adapter = LeagueEnvAdapter(env, learner_idx=cfg.learner_idx)
        super().__init__(self.env_adapter, cfg)
        self.cfg = cfg

        self.models = {
            "policy": Policy(self.env.observation_space, self.env.action_space, cfg.device),
            "value": Value(self.env.observation_space, self.env.action_space, cfg.device),
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
        agent_cfg["mini_batches"] = cfg.mini_batches
        agent_cfg["mixed_precision"] = cfg.mixed_precision

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
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            device=cfg.device,
        )

        self.run_dir = Path(cfg.experiment_directory) / cfg.experiment_name
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.league_dir = self.run_dir / "league"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.league_dir.mkdir(parents=True, exist_ok=True)

        self.league: list[LeagueMember] = [LeagueMember(name="scripted_bot", checkpoint_path=None, elo=1000.0)]
        self.best_eval_win_rate = -1.0

        if cfg.resume and cfg.checkpoint_path and Path(cfg.checkpoint_path).exists():
            self.agent.policy.load_state_dict(torch.load(cfg.checkpoint_path, map_location=gs.device))  # type: ignore[union-attr]
            print(f"Loaded learner from {cfg.checkpoint_path}")

        self._opponent_model = Policy(self.env.observation_space, self.env.action_space, cfg.device).to(gs.device)
        self._opponent_model.eval()

    def _trainer(self, timesteps: int) -> SequentialTrainer:
        trainer_cfg = {
            "timesteps": timesteps,
            "headless": self.cfg.headless,
            "environment_info": self.cfg.environment_info,
        }
        return SequentialTrainer(cfg=trainer_cfg, env=self.env, agents=[self.agent])  # type: ignore[arg-type]

    def _league_meta_path(self) -> Path:
        return self.league_dir / "league.json"

    def _save_league(self) -> None:
        payload = [asdict(member) for member in self.league]
        self._league_meta_path().write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _save_learner_policy(self, path: Path) -> None:
        torch.save(self.agent.policy.state_dict(), path)

    def _set_opponent(self, member: LeagueMember) -> None:
        if member.checkpoint_path is None:
            self.env_adapter.set_opponent_sampler(self.env_adapter._default_opponent)
            return

        state_dict = torch.load(member.checkpoint_path, map_location=gs.device)
        self._opponent_model.load_state_dict(state_dict)
        self._opponent_model.eval()

        def opponent_fn(obs: torch.Tensor) -> torch.Tensor:
            return self._opponent_model.net(obs)

        self.env_adapter.set_opponent_sampler(opponent_fn)

    def _sample_opponent(self) -> LeagueMember:
        if len(self.league) == 1:
            return self.league[0]

        learner_elo = 1200.0 + 15.0 * len(self.league)
        scores = []
        for member in self.league:
            scores.append(torch.exp(torch.tensor(-abs(member.elo - learner_elo) / 350.0)).item() + 0.05)
        probs = torch.tensor(scores, dtype=torch.float)
        probs = probs / probs.sum()
        idx = torch.multinomial(probs, num_samples=1).item()
        return self.league[idx]

    def _eval_vs_opponent(self, opponent: LeagueMember, episodes: int) -> tuple[float, float]:
        self._set_opponent(opponent)
        self.agent.policy.eval()  # type: ignore[union-attr]

        wins = 0.0
        games = 0.0
        rewards = 0.0
        obs, _ = self.env.reset()

        with torch.no_grad():
            while games < episodes:
                actions, _, _ = self.agent.act(obs, timestep=0, timesteps=0)
                obs, rew, terminated, truncated, info = self.env.step(actions)
                rewards += float(rew.mean().item())
                done = torch.logical_or(terminated, truncated).squeeze(1)
                if done.any():
                    mask = done
                    goal_for = info["goal_team0"][mask].float() if self.cfg.learner_idx == 0 else info["goal_team1"][mask].float()
                    goal_against = info["goal_team1"][mask].float() if self.cfg.learner_idx == 0 else info["goal_team0"][mask].float()
                    wins += float((goal_for > goal_against).float().sum().item())
                    games += float(mask.sum().item())

        win_rate = wins / max(games, 1.0)
        avg_reward = rewards / max(games, 1.0)
        return win_rate, avg_reward

    def _update_elo(self, opponent: LeagueMember, win_rate: float) -> None:
        k = 24.0
        learner_elo = 1200.0 + 15.0 * len(self.league)
        expected = 1.0 / (1.0 + 10.0 ** ((opponent.elo - learner_elo) / 400.0))
        opponent.elo += k * ((1.0 - win_rate) - (1.0 - expected))
        opponent.games += 1

    def _promote_snapshot(self, generation: int, eval_win_rate: float) -> LeagueMember:
        path = self.league_dir / f"gen_{generation:04d}.pt"
        self._save_learner_policy(path)
        member = LeagueMember(name=f"gen_{generation:04d}", checkpoint_path=str(path), elo=1200.0 + 8.0 * generation)
        self.league.append(member)

        if len(self.league) > self.cfg.max_league_size:
            scripted = [member for member in self.league if member.checkpoint_path is None]
            snapshots = [member for member in self.league if member.checkpoint_path is not None]
            snapshots.sort(key=lambda item: item.elo, reverse=True)
            self.league = scripted + snapshots[: self.cfg.max_league_size - len(scripted)]

        if eval_win_rate > self.best_eval_win_rate:
            self.best_eval_win_rate = eval_win_rate
            self._save_learner_policy(self.checkpoint_dir / "best_agent.pt")
        self._save_learner_policy(self.checkpoint_dir / "last_agent.pt")
        self._save_league()
        return member

    def train(self) -> None:
        if self.cfg.compile_policy:
            self.agent.policy = torch.compile(self.agent.policy)  # type: ignore[assignment]

        for generation in range(self.cfg.generations):
            opponent = self._sample_opponent()
            self._set_opponent(opponent)

            trainer = self._trainer(self.cfg.train_timesteps_per_generation)
            trainer.train()

            win_rate, avg_reward = self._eval_vs_opponent(opponent, self.cfg.eval_episodes)
            self._update_elo(opponent, win_rate)

            should_promote = (generation % self.cfg.snapshot_interval == 0) or (win_rate >= self.cfg.promote_win_rate)
            if should_promote:
                member = self._promote_snapshot(generation=generation, eval_win_rate=win_rate)
                print(
                    f"[Gen {generation}] promoted={member.name} opponent={opponent.name} "
                    f"win_rate={win_rate:.3f} avg_reward={avg_reward:.3f} league_size={len(self.league)}"
                )
            else:
                self._save_learner_policy(self.checkpoint_dir / "last_agent.pt")
                print(
                    f"[Gen {generation}] opponent={opponent.name} win_rate={win_rate:.3f} "
                    f"avg_reward={avg_reward:.3f}"
                )

    def eval(self) -> None:
        checkpoint = self.cfg.checkpoint_path or str(self.checkpoint_dir / "best_agent.pt")
        if Path(checkpoint).exists():
            self.agent.policy.load_state_dict(torch.load(checkpoint, map_location=gs.device))  # type: ignore[union-attr]
            print(f"Loaded eval policy from {checkpoint}")

        if self.cfg.compile_policy:
            self.agent.policy = torch.compile(self.agent.policy)  # type: ignore[assignment]

        opponent = self.league[0] if len(self.league) == 1 else self.league[-1]
        win_rate, avg_reward = self._eval_vs_opponent(opponent, self.cfg.eval_episodes)
        print(f"[Eval] opponent={opponent.name} win_rate={win_rate:.3f} avg_reward={avg_reward:.3f}")
