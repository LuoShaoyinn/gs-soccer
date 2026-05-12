from __future__ import annotations
from dataclasses import dataclass
import os

import torch
import torch.nn.functional as F
import torch.distributions as D
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from network import Policy, QNetwork

from .algorithm import Algorithm, AlgorithmConfig


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, act_dim: int, device: str):
        self.obs = torch.zeros((capacity, obs_dim), device=device)
        self.act = torch.zeros((capacity, act_dim), device=device)
        self.rew = torch.zeros((capacity, 1), device=device)
        self.nobs = torch.zeros((capacity, obs_dim), device=device)
        self.done = torch.zeros((capacity, 1), device=device)
        self.ptr = 0
        self.cap = capacity
        self.full = False

    def add(self, obs, act, rew, nobs, done):
        n = obs.shape[0]
        if n >= self.cap:
            s = n - self.cap
            self.obs[:], self.act[:] = obs[s:], act[s:]
            self.rew[:], self.nobs[:], self.done[:] = rew[s:], nobs[s:], done[s:]
            self.ptr, self.full = 0, True
            return
        e = self.ptr + n
        if e <= self.cap:
            self.obs[self.ptr:e] = obs;  self.act[self.ptr:e] = act
            self.rew[self.ptr:e] = rew;  self.nobs[self.ptr:e] = nobs
            self.done[self.ptr:e] = done
        else:
            f = self.cap - self.ptr
            self.obs[self.ptr:] = obs[:f];  self.act[self.ptr:] = act[:f]
            self.rew[self.ptr:] = rew[:f];  self.nobs[self.ptr:] = nobs[:f]
            self.done[self.ptr:] = done[:f]
            r = n - f
            self.obs[:r] = obs[f:];  self.act[:r] = act[f:]
            self.rew[:r] = rew[f:];  self.nobs[:r] = nobs[f:]
            self.done[:r] = done[f:];  self.full = True
        self.ptr = e % self.cap
        if self.ptr == 0:
            self.full = True

    def size(self) -> int:
        return self.cap if self.full else self.ptr

    def sample(self, bs: int):
        idx = torch.randint(0, self.size(), (bs,), device=self.obs.device)
        return self.obs[idx], self.act[idx], self.rew[idx], self.nobs[idx], self.done[idx]


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

    expert_checkpoint_path: str = ""
    expert_collect_steps: int = 0
    expert_sample_ratio: float = 0.0


class SACAlgorithm(Algorithm):
    def __init__(self, env, cfg: SACAlgorithmConfig):
        super().__init__(env, cfg)
        device = cfg.device
        obs_space = env.observation_space
        act_space = env.action_space
        obs_dim = obs_space.shape[0]
        act_dim = act_space.shape[0]

        self.policy = Policy(obs_space, act_space, device)
        self.critic_1 = QNetwork(obs_space, act_space, device)
        self.critic_2 = QNetwork(obs_space, act_space, device)
        self.critic_tgt_1 = QNetwork(obs_space, act_space, device)
        self.critic_tgt_2 = QNetwork(obs_space, act_space, device)
        for m in (self.policy, self.critic_1, self.critic_2,
                  self.critic_tgt_1, self.critic_tgt_2):
            m.init_parameters(method_name=cfg.init_method_name,
                              mean=cfg.init_mean, std=cfg.init_std)
        self.critic_tgt_1.load_state_dict(self.critic_1.state_dict())
        self.critic_tgt_2.load_state_dict(self.critic_2.state_dict())

        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.target_entropy = -act_dim

        self.policy_opt = torch.optim.Adam(self.policy.parameters(), lr=cfg.lr)
        self.critic_opt = torch.optim.Adam(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()),
            lr=cfg.lr,
        )
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=cfg.alpha_lr)

        self.online_buf = ReplayBuffer(cfg.memory_size, obs_dim, act_dim, device)
        self.expert_buf = ReplayBuffer(cfg.memory_size, obs_dim, act_dim, device)

        if cfg.resume and cfg.checkpoint_path and os.path.exists(cfg.checkpoint_path):
            self._load_checkpoint(cfg.checkpoint_path)

    def _collect_expert_data(self) -> None:
        cfg: SACAlgorithmConfig = self.cfg
        if not cfg.expert_checkpoint_path or cfg.expert_collect_steps <= 0:
            return
        if not os.path.exists(cfg.expert_checkpoint_path):
            print(f"Expert not found: {cfg.expert_checkpoint_path}")
            return

        print(f"Collecting expert data: {cfg.expert_collect_steps} steps "
              f"from {cfg.expert_checkpoint_path}")
        expert = torch.jit.load(cfg.expert_checkpoint_path).to(cfg.device)

        obs, _ = self.env.reset()
        for i in range(cfg.expert_collect_steps):
            with torch.no_grad():
                action = expert(obs)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = (terminated | truncated).float()
            self.expert_buf.add(obs, action, reward, next_obs, done)
            obs = next_obs
            if (i + 1) % 200 == 0:
                print(f"  expert {i+1}/{cfg.expert_collect_steps}, "
                      f"buf={self.expert_buf.size()}")

        print(f"Expert buffer: {self.expert_buf.size()} transitions")
        self.env.reset()

    def _sac_update(self, obs, act, rew, nobs, done):
        alpha = self.log_alpha.exp()

        with torch.no_grad():
            nm = self.policy.net(nobs)
            ns = self.policy.log_std_parameter.exp().expand_as(nm)
            nd = D.Normal(nm, ns)
            na = nd.sample()
            nlp = nd.log_prob(na).sum(-1, keepdim=True)
            q1t = self.critic_tgt_1.net(torch.cat([nobs, na], -1))
            q2t = self.critic_tgt_2.net(torch.cat([nobs, na], -1))
            qt = torch.min(q1t, q2t) - alpha * nlp
            y = rew + (1.0 - done) * self.cfg.discount_factor * qt

        q1 = self.critic_1.net(torch.cat([obs, act], -1))
        q2 = self.critic_2.net(torch.cat([obs, act], -1))
        c_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.critic_opt.zero_grad()
        c_loss.backward()
        self.critic_opt.step()

        pm = self.policy.net(obs)
        ps = self.policy.log_std_parameter.exp().expand_as(pm)
        pd = D.Normal(pm, ps)
        pa = pd.rsample()
        plp = pd.log_prob(pa).sum(-1, keepdim=True)
        q1p = self.critic_1.net(torch.cat([obs, pa], -1))
        q2p = self.critic_2.net(torch.cat([obs, pa], -1))
        qp = torch.min(q1p, q2p)
        p_loss = (alpha.detach() * plp - qp).mean()
        self.policy_opt.zero_grad()
        p_loss.backward()
        self.policy_opt.step()

        a_loss = -(self.log_alpha * (plp + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        a_loss.backward()
        self.alpha_opt.step()

        tau = self.cfg.tau
        for p, tp in zip(self.critic_1.parameters(), self.critic_tgt_1.parameters()):
            tp.data.lerp_(p.data, tau)
        for p, tp in zip(self.critic_2.parameters(), self.critic_tgt_2.parameters()):
            tp.data.lerp_(p.data, tau)

    def train(self) -> None:
        self._collect_expert_data()

        run_dir = os.path.join(self.cfg.experiment_directory, self.cfg.experiment_name)
        writer = SummaryWriter(log_dir=run_dir)

        obs, _ = self.env.reset()
        for step in range(self.cfg.timesteps):
            if step < self.cfg.random_timesteps:
                action = (torch.rand((self.env.num_envs,
                                      self.env.action_space.shape[0]),
                                     device=self.cfg.device) * 2 - 1) * np.pi
            else:
                with torch.no_grad():
                    m = self.policy.net(obs)
                    s = self.policy.log_std_parameter.exp().expand_as(m)
                    action = D.Normal(m, s).sample().clamp(-np.pi, np.pi)

            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = (terminated | truncated).float()
            self.online_buf.add(obs, action, reward, next_obs, done)
            obs = next_obs

            if step >= self.cfg.learning_starts and self.online_buf.size() >= self.cfg.batch_size // 2:
                n_on = int(self.cfg.batch_size * (1 - self.cfg.expert_sample_ratio))
                n_ex = self.cfg.batch_size - n_on
                for _ in range(self.cfg.gradient_steps):
                    bo, ba, br, bn, bd = self.online_buf.sample(n_on)
                    if n_ex > 0 and self.expert_buf.size() >= n_ex:
                        eo, ea, er, en, ed = self.expert_buf.sample(n_ex)
                        bo = torch.cat([bo, eo])
                        ba = torch.cat([ba, ea])
                        br = torch.cat([br, er])
                        bn = torch.cat([bn, en])
                        bd = torch.cat([bd, ed])
                    self._sac_update(bo, ba, br, bn, bd)

            if step % self.cfg.write_interval == 0:
                if "extra" in info:
                    for k, v in info["extra"].items():
                        writer.add_scalar(k, float(v), step)
                writer.add_scalar("alpha", self.log_alpha.exp().item(), step)
                writer.add_scalar("online_buf", self.online_buf.size(), step)

            if step > 0 and step % self.cfg.checkpoint_interval == 0:
                ckpt_dir = os.path.join(run_dir, "checkpoints")
                os.makedirs(ckpt_dir, exist_ok=True)
                self._save_checkpoint(os.path.join(ckpt_dir, f"step_{step}.pt"))

        ckpt_dir = os.path.join(run_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        self._save_checkpoint(os.path.join(ckpt_dir, "best_agent.pt"))
        writer.close()

    def eval(self) -> None:
        obs, _ = self.env.reset()
        with torch.no_grad():
            for _ in range(self.cfg.eval_steps):
                mean = self.policy.net(obs)
                obs, _, _, _, _ = self.env.step(mean)

    def _save_checkpoint(self, path):
        torch.save({
            "policy": self.policy.state_dict(),
            "critic_1": self.critic_1.state_dict(),
            "critic_2": self.critic_2.state_dict(),
            "critic_tgt_1": self.critic_tgt_1.state_dict(),
            "critic_tgt_2": self.critic_tgt_2.state_dict(),
            "log_alpha": self.log_alpha.detach().clone(),
            "policy_opt": self.policy_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "alpha_opt": self.alpha_opt.state_dict(),
        }, path)
        print(f"Saved: {path}")

    def _load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.cfg.device, weights_only=False)
        self.policy.load_state_dict(ckpt["policy"])
        self.critic_1.load_state_dict(ckpt["critic_1"])
        self.critic_2.load_state_dict(ckpt["critic_2"])
        self.critic_tgt_1.load_state_dict(ckpt["critic_tgt_1"])
        self.critic_tgt_2.load_state_dict(ckpt["critic_tgt_2"])
        if "log_alpha" in ckpt:
            self.log_alpha.data.copy_(ckpt["log_alpha"])
        for k, opt in [("policy_opt", self.policy_opt),
                       ("critic_opt", self.critic_opt),
                       ("alpha_opt", self.alpha_opt)]:
            if k in ckpt:
                opt.load_state_dict(ckpt[k])
        print(f"Loaded: {path}")
