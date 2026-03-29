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
from policies import GoToBallPIDPolicy


@dataclass(kw_only=True)
class LeagueMember:
    name: str
    checkpoint_path: str | None
    elo: float = 1200.0
    games: int = 0


@dataclass(kw_only=True)
class AlphaStarEvolutionConfig(AlgorithmConfig):
    rollout_steps: int = 16
    learning_epochs: int = 4
    mini_batches: int = 16

    generations: int = 8
    train_timesteps_per_generation: int = 131072
    eval_episodes: int = 32
    snapshot_interval: int = 2
    promote_win_rate: float = 0.55
    max_league_size: int = 8

    learner_idx: int = 0
    symmetric_selfplay: bool = True
    scripted_opponent: str = "pid"
    experiment_name: str = "soccer_1v1_alphastar"


class LeagueEnvAdapter:
    def __init__(
        self,
        game_env,
        learner_idx: int = 0,
        symmetric_selfplay: bool = True,
        scripted_opponent: str = "pid",
    ):
        self.game_env = game_env
        self.learner_idx = learner_idx
        self.opponent_idx = 1 - learner_idx
        self.symmetric_selfplay = symmetric_selfplay
        self.team_names = ("red", "blue")

        self.num_envs = game_env.num_envs
        self.num_agents = 1
        self.is_vector_env = True
        self.observation_space = game_env.observation_space[self.team_names[self.learner_idx]]
        self.action_space = game_env.action_space[self.team_names[self.learner_idx]]
        obs_dim = int(self.observation_space.shape[0])
        action_dim = int(self.action_space.shape[0])
        self._obs_mirror_sign = torch.ones((obs_dim,), dtype=torch.float, device=gs.device)
        self._action_mirror_sign = torch.ones((action_dim,), dtype=torch.float, device=gs.device)
        if hasattr(game_env, "model"):
            model = game_env.model
            if hasattr(model, "obs_mirror_sign"):
                self._obs_mirror_sign = model.obs_mirror_sign.to(gs.device)
            if hasattr(model, "action_mirror_sign"):
                self._action_mirror_sign = model.action_mirror_sign.to(gs.device)

        self._latest_obs: dict[str, torch.Tensor] | None = None
        self._scripted_policy = GoToBallPIDPolicy(device=gs.device)
        self._scripted_opponent = scripted_opponent
        self._opponent_fn = self._default_opponent
        self._learner_side_idx = torch.full(
            (self.num_envs,),
            fill_value=self.learner_idx,
            dtype=torch.long,
            device=gs.device,
        )

    def set_opponent_sampler(self, sampler):
        self._opponent_fn = sampler

    def _sample_learner_sides(self, envs_idx: torch.Tensor | None = None) -> None:
        if envs_idx is None:
            envs_idx = torch.arange(self.num_envs, dtype=torch.long, device=gs.device)
        if self.symmetric_selfplay:
            self._learner_side_idx[envs_idx] = torch.randint(
                low=0,
                high=2,
                size=(envs_idx.shape[0],),
                device=gs.device,
            )
        else:
            self._learner_side_idx[envs_idx] = self.learner_idx

    def _mirror_obs(self, obs: torch.Tensor) -> torch.Tensor:
        return obs * self._obs_mirror_sign.unsqueeze(0)

    def _canonical_to_env_action(self, action: torch.Tensor, side_idx: torch.Tensor) -> torch.Tensor:
        env_action = action.clone()
        side1_mask = side_idx == 1
        if side1_mask.any():
            env_action[side1_mask] = env_action[side1_mask] * self._action_mirror_sign.unsqueeze(0)
        return env_action

    @staticmethod
    def _select_dict_by_side(values: dict[str, torch.Tensor], side_idx: torch.Tensor) -> torch.Tensor:
        selected = values["red"].clone()
        side1_mask = side_idx == 1
        if side1_mask.any():
            selected[side1_mask] = values["blue"][side1_mask]
        return selected

    def _canonical_obs_from_side(self, obs: dict[str, torch.Tensor], side_idx: torch.Tensor) -> torch.Tensor:
        canonical = self._select_dict_by_side(obs, side_idx)
        side1_mask = side_idx == 1
        if side1_mask.any():
            canonical[side1_mask] = self._mirror_obs(canonical[side1_mask])
        return canonical

    @staticmethod
    def _merge_extra_info(extra: dict[str, dict]) -> dict:
        if not isinstance(extra, dict) or "red" not in extra or "blue" not in extra:
            return {}
        keys = set(extra["red"].keys()) & set(extra["blue"].keys())
        merged = {}
        for key in keys:
            merged[key] = 0.5 * (extra["red"][key] + extra["blue"][key])
        return merged

    def _default_opponent(self, obs: torch.Tensor) -> torch.Tensor:
        if self._scripted_opponent == "legacy":
            action = torch.zeros((obs.shape[0], 3), dtype=torch.float, device=gs.device)
            vel_dim = obs.shape[1] - 16
            ball_rel_start = 2 + vel_dim + 1 + 2 + 2
            ball_to_goal_start = ball_rel_start + 2 + 2
            ball_rel = obs[:, ball_rel_start : ball_rel_start + 2]
            ball_to_goal = obs[:, ball_to_goal_start : ball_to_goal_start + 2]
            action[:, 0:2] = torch.tanh(1.5 * ball_rel + 0.3 * ball_to_goal)
            return action
        return self._scripted_policy.act(obs)

    def reset(self):
        obs, info = self.game_env.reset()
        self._latest_obs = obs
        self._sample_learner_sides()
        learner_obs = self._canonical_obs_from_side(obs, self._learner_side_idx)
        return learner_obs, info

    def step(self, learner_action: torch.Tensor):
        if self._latest_obs is None:
            raise RuntimeError("Call reset() before step()")
        learner_side_idx = self._learner_side_idx.clone()
        opponent_side_idx = 1 - learner_side_idx
        opponent_obs = self._canonical_obs_from_side(self._latest_obs, opponent_side_idx)
        with torch.no_grad():
            opponent_action = self._opponent_fn(opponent_obs)

        learner_action_env = self._canonical_to_env_action(learner_action, learner_side_idx)
        opponent_action_env = self._canonical_to_env_action(opponent_action, opponent_side_idx)
        learner_is_team_red = learner_side_idx == 0
        actions = {
            "red": torch.where(learner_is_team_red.unsqueeze(1), learner_action_env, opponent_action_env),
            "blue": torch.where(learner_is_team_red.unsqueeze(1), opponent_action_env, learner_action_env),
        }
        obs, reward, terminated, truncated, info = self.game_env.step(actions)
        learner_reward = self._select_dict_by_side(reward, learner_side_idx)
        learner_terminated = self._select_dict_by_side(terminated, learner_side_idx)
        learner_truncated = self._select_dict_by_side(truncated, learner_side_idx)
        learner_done = torch.logical_or(learner_terminated, learner_truncated).squeeze(1)

        if learner_done.any():
            done_envs = torch.nonzero(learner_done, as_tuple=False).squeeze(1)
            self._sample_learner_sides(done_envs)

        if "extra" in info and isinstance(info["extra"], dict):
            info["extra"] = self._merge_extra_info(info["extra"])
        info["learner_side_idx"] = learner_side_idx
        self._latest_obs = obs
        learner_obs = self._canonical_obs_from_side(obs, self._learner_side_idx)
        return learner_obs, learner_reward, learner_terminated, learner_truncated, info

    def close(self):
        self.game_env.close()


class AlphaStarEvolutionAlgorithm(Algorithm):
    def __init__(self, env, cfg: AlphaStarEvolutionConfig):
        self.game_env = env
        self.env_adapter = LeagueEnvAdapter(
            env,
            learner_idx=cfg.learner_idx,
            symmetric_selfplay=cfg.symmetric_selfplay,
            scripted_opponent=cfg.scripted_opponent,
        )
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
            self.agent.load(cfg.checkpoint_path)  # type: ignore[arg-type]
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
                    learner_side = info["learner_side_idx"]
                    goal_red = info["goal_team_red"]
                    goal_blue = info["goal_team_blue"]
                    goal_for = torch.where(
                        learner_side.unsqueeze(1) == 0,
                        goal_red,
                        goal_blue,
                    )[mask].float()
                    goal_against = torch.where(
                        learner_side.unsqueeze(1) == 0,
                        goal_blue,
                        goal_red,
                    )[mask].float()
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
            self.agent.load(checkpoint)  # type: ignore[arg-type]
            print(f"Loaded eval policy from {checkpoint}")

        if self.cfg.compile_policy:
            self.agent.policy = torch.compile(self.agent.policy)  # type: ignore[assignment]

        opponent = self.league[0] if len(self.league) == 1 else self.league[-1]
        win_rate, avg_reward = self._eval_vs_opponent(opponent, self.cfg.eval_episodes)
        print(f"[Eval] opponent={opponent.name} win_rate={win_rate:.3f} avg_reward={avg_reward:.3f}")
