import argparse
import os
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch.distributions import Normal
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torchrl.data import LazyMemmapStorage

import genesis as gs
from algorithm.pi_visual_walk.sac import Policy, QNetwork
from algorithm.pi_visual_walk.teacher import WalkTeacher, action_clip_bounds, clip_walk_action
from env_factory import make_env


class HierarchicalReplay:
    """TorchRL memmap storage with explicit disk -> RAM -> VRAM sampling."""

    def __init__(self, capacity, obs_dim, act_dim, num_envs, device, scratch_dir):
        self.capacity = int(capacity)
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.scratch_dir = Path(scratch_dir)
        self.scratch_dir.mkdir(parents=True, exist_ok=True)
        self.storage = LazyMemmapStorage(
            self.capacity,
            scratch_dir=str(self.scratch_dir),
            device="cpu",
            existsok=True,
        )
        self.ptr = 0
        self.size = 0
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)

    def add(self, obs, action, reward, done):
        obs = obs.detach().to("cpu", dtype=torch.float16, non_blocking=True)
        action = action.detach().to("cpu", dtype=torch.float16, non_blocking=True)
        reward = reward.detach().reshape(-1, 1).to("cpu", dtype=torch.float16, non_blocking=True)
        done = done.detach().reshape(-1, 1).to("cpu", dtype=torch.bool, non_blocking=True)

        n = obs.shape[0]
        td = TensorDict({"obs": obs, "action": action, "reward": reward, "done": done}, batch_size=[n])
        end = self.ptr + n
        if end <= self.capacity:
            self.storage.set(torch.arange(self.ptr, end), td)
        else:
            first = self.capacity - self.ptr
            self.storage.set(torch.arange(self.ptr, self.capacity), td[:first])
            self.storage.set(torch.arange(0, n - first), td[first:])
        self.ptr = (self.ptr + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def _sample_ram(self, batch_size):
        if self.size < self.num_envs + 1:
            raise RuntimeError("Replay does not yet contain enough sequential samples")
        if self.size < self.capacity:
            high = self.size - self.num_envs
            idx = torch.randint(0, high, (batch_size,), device="cpu")
        else:
            latest = (torch.arange(self.num_envs, device="cpu") + self.ptr - self.num_envs) % self.capacity
            idx_chunks = []
            remaining = batch_size
            while remaining > 0:
                candidate = torch.randint(0, self.capacity, (remaining * 2,), device="cpu")
                valid = candidate[~torch.isin(candidate, latest)]
                idx_chunks.append(valid[:remaining])
                remaining -= idx_chunks[-1].numel()
            idx = torch.cat(idx_chunks, dim=0)
        next_idx = (idx + self.num_envs) % self.capacity

        td = self.storage.get(idx)
        next_td = self.storage.get(next_idx)
        obs = td["obs"].float()
        done = td["done"].bool()
        next_obs = torch.where(done, obs, next_td["obs"].float())
        return (
            obs,
            td["action"].float(),
            td["reward"].float(),
            next_obs,
            done.float(),
        )

    def sample(self, batch_size):
        return tuple(t.to(self.device, non_blocking=True) for t in self._sample_ram(batch_size))

    def __len__(self):
        return self.size


class ReplayBatchStager:
    def __init__(self, replay, batch_size):
        self.replay = replay
        self.batch_size = batch_size
        self.stream = None
        if replay.device.type == "cuda" and torch.cuda.is_available():
            self.stream = torch.cuda.Stream(device=replay.device)
        self.next_batch = None
        self._prefetch()

    def _stage(self, batch):
        out = []
        for tensor in batch:
            if self.stream is not None:
                try:
                    tensor = tensor.pin_memory()
                except RuntimeError:
                    pass
            out.append(tensor.to(self.replay.device, non_blocking=self.stream is not None))
        return tuple(out)

    def _prefetch(self):
        if self.stream is None:
            self.next_batch = self.replay.sample(self.batch_size)
            return
        with torch.cuda.stream(self.stream):
            self.next_batch = self._stage(self.replay._sample_ram(self.batch_size))

    def next(self, prefetch=True):
        if self.stream is not None:
            current = torch.cuda.current_stream(device=self.replay.device)
            current.wait_stream(self.stream)
        batch = self.next_batch
        if self.stream is not None:
            current = torch.cuda.current_stream(device=self.replay.device)
            for tensor in batch:
                tensor.record_stream(current)
        if prefetch:
            self._prefetch()
        return batch


def parse_args():
    p = argparse.ArgumentParser(description="Walk-only online RLPD")
    p.add_argument("--num-envs", type=int, default=512)
    p.add_argument("--teacher-model-dir", type=str, default="refs/piplus_soccer_sim2sim/models/exported")
    p.add_argument("--memory-size", type=int, default=8 * (2**20))
    p.add_argument("--replay-dir", type=str, default="/tmp/gs_soccer_walk_replay")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--alpha-init", type=float, default=0.001)
    p.add_argument("--target-entropy", type=float, default=-2.0)
    p.add_argument("--reward-scale", type=float, default=1.0)
    p.add_argument("--learning-starts", type=int, default=1000)
    p.add_argument("--utd", type=int, default=32)
    p.add_argument("--n-critics", type=int, default=32)
    p.add_argument("--n-min", type=int, default=2)
    p.add_argument("--grad-norm-clip", type=float, default=1.0)
    p.add_argument("--settle-steps", type=int, default=50)
    p.add_argument("--timesteps", type=int, default=500_000)
    p.add_argument("--log-interval", type=int, default=100)
    p.add_argument("--save-interval", type=int, default=10_000)
    p.add_argument("--save-dir", type=str, default=None)
    p.add_argument("--viewer", action="store_true", default=False)
    p.add_argument("--eval", action="store_true", default=False)
    p.add_argument("--eval-episodes", type=int, default=40)
    p.add_argument("--resume", type=str, default=None)
    return p.parse_args()


def _autocast():
    enabled = torch.cuda.is_available()
    return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=enabled)


def critic_update(critics, tgt_critics, policy, log_alpha, c_opt, scaler, batch, gamma, reward_scale, grad_clip, n_min, action_low, action_high):
    obs, act, rew, nobs, done = batch
    with _autocast():
        with torch.no_grad():
            n_mean, n_out = policy.compute({"observations": nobs})
            n_dist = Normal(n_mean, n_out["log_std"].exp())
            n_action = clip_walk_action(n_dist.rsample(), action_low, action_high)
            n_log_prob = n_dist.log_prob(n_action).sum(-1, keepdim=True)
            tgt_idx = torch.randperm(len(tgt_critics), device=obs.device)[:n_min]
            q_targets = torch.cat([tgt_critics[int(i)].net(torch.cat([nobs, n_action], -1)) for i in tgt_idx], dim=-1)
            target_q = reward_scale * rew + gamma * (1.0 - done) * (
                q_targets.min(dim=-1, keepdim=True).values - log_alpha.exp().detach() * n_log_prob
            )
        qs = [critic.net(torch.cat([obs, act], -1)) for critic in critics]
        critic_loss = sum(F.mse_loss(q, target_q) for q in qs)

    c_opt.zero_grad()
    scaler.scale(critic_loss).backward()
    if grad_clip > 0:
        scaler.unscale_(c_opt)
        params = []
        for critic in critics:
            params.extend(critic.parameters())
        torch.nn.utils.clip_grad_norm_(params, grad_clip)
    scaler.step(c_opt)
    scaler.update()
    return qs[0], critic_loss


def policy_update(policy, critics, log_alpha, p_opt, a_opt, scaler, batch, target_entropy, grad_clip, n_min, action_low, action_high):
    obs = batch[0]
    with _autocast():
        mean, out = policy.compute({"observations": obs})
        dist = Normal(mean, out["log_std"].exp())
        action = clip_walk_action(dist.rsample(), action_low, action_high)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        critic_idx = torch.randperm(len(critics), device=obs.device)[:n_min]
        q_policy = torch.cat([critics[int(i)].net(torch.cat([obs, action], -1)) for i in critic_idx], dim=-1)
        policy_loss = (log_alpha.exp().detach() * log_prob - q_policy.min(dim=-1, keepdim=True).values).mean()

    p_opt.zero_grad()
    scaler.scale(policy_loss).backward()
    if grad_clip > 0:
        scaler.unscale_(p_opt)
        torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
    scaler.step(p_opt)

    alpha_loss = -(log_alpha * (log_prob.detach() + target_entropy)).mean()
    a_opt.zero_grad()
    scaler.scale(alpha_loss).backward()
    scaler.step(a_opt)
    scaler.update()
    return policy_loss, alpha_loss, log_prob


def sac_step(policy, critics, tgt_critics, replay, p_opt, c_opt, a_opt, log_alpha, scaler, args, action_low, action_high):
    stager = ReplayBatchStager(replay, args.batch_size)
    q1 = critic_loss = None
    for _ in range(args.utd):
        q1, critic_loss = critic_update(
            critics, tgt_critics, policy, log_alpha, c_opt, scaler,
            stager.next(), args.gamma, args.reward_scale, args.grad_norm_clip, args.n_min,
            action_low, action_high,
        )
    policy_loss, alpha_loss, log_prob = policy_update(
        policy, critics, log_alpha, p_opt, a_opt, scaler,
        stager.next(prefetch=False), args.target_entropy, args.grad_norm_clip, args.n_min,
        action_low, action_high,
    )
    with torch.no_grad():
        for critic, target in zip(critics, tgt_critics):
            for p, tp in zip(critic.parameters(), target.parameters()):
                tp.mul_(1.0 - args.tau).add_(p, alpha=args.tau)
    return {
        "q1": q1.mean().detach().item(),
        "critic_loss": critic_loss.detach().item(),
        "policy_loss": policy_loss.detach().item(),
        "alpha_loss": alpha_loss.detach().item(),
        "alpha": log_alpha.exp().detach().item(),
        "entropy": -log_prob.float().mean().detach().item(),
    }


def main():
    args = parse_args()
    if args.viewer:
        os.environ.pop("PYOPENGL_PLATFORM", None)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    gs.init(backend=gs.gpu, performance_mode=True, logging_level="warning")

    dev = torch.device(gs.device)
    num_envs = args.num_envs
    half = num_envs // 2
    env = make_env(num_envs=num_envs, viewer=args.viewer or args.eval)
    teacher = WalkTeacher(args.teacher_model_dir, device=dev)

    obs_space = env.observation_space
    act_space = env.action_space
    policy = Policy(obs_space, act_space, dev).to(dev)
    critics = [QNetwork(obs_space, act_space, dev).to(dev) for _ in range(args.n_critics)]
    tgt_critics = [QNetwork(obs_space, act_space, dev).to(dev) for _ in range(args.n_critics)]
    for target, critic in zip(tgt_critics, critics):
        target.load_state_dict(critic.state_dict())

    log_alpha = torch.log(torch.tensor(args.alpha_init, device=dev)).requires_grad_(True)
    policy_opt = Adam(policy.parameters(), lr=args.learning_rate)
    critic_opt = Adam([p for critic in critics for p in critic.parameters()], lr=args.learning_rate)
    alpha_opt = Adam([log_alpha], lr=args.learning_rate)
    scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())
    action_low, action_high = action_clip_bounds(dev)

    replay = HierarchicalReplay(
        args.memory_size,
        obs_space.shape[0],
        act_space.shape[0],
        num_envs,
        dev,
        args.replay_dir,
    )

    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=dev, weights_only=True)
        policy.load_state_dict(ckpt["policy"])
        for i in range(args.n_critics):
            critics[i].load_state_dict(ckpt[f"critic_{i}"])
            tgt_critics[i].load_state_dict(ckpt[f"tgt_critic_{i}"])
        log_alpha.data = ckpt["log_alpha"]
        start_step = ckpt.get("step", 0)

    if args.eval:
        _eval(env, policy, args.eval_episodes, dev, action_low, action_high)
        return

    save_dir = args.save_dir or f"runs/rlpd_walk_{datetime.now().strftime('%m%d_%H%M')}"
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(save_dir)

    obs, info = env.reset()
    zeros = torch.zeros(num_envs, act_space.shape[0], dtype=torch.float32, device=dev)
    for _ in range(args.settle_steps):
        obs, _, _, _, _ = env.step(zeros)

    t0 = time.time()
    stats = {}
    print(f"=== Walk RLPD: {num_envs} envs ({half} teacher + {num_envs - half} student), {args.timesteps} steps ===")
    print(f"    replay: TorchRL LazyMemmapStorage at {args.replay_dir}, capacity={args.memory_size:,}")
    print(f"    obs={obs_space.shape[0]}, action={act_space.shape[0]}, critics={args.n_critics}, UTD={args.utd}")

    for step in range(start_step, args.timesteps):
        with torch.no_grad():
            teacher_actions = teacher.infer(obs[:half])
            mean, out = policy.compute({"observations": obs[half:]})
            student_dist = Normal(mean, out["log_std"].exp())
            student_actions = student_dist.rsample()
            actions = torch.cat([teacher_actions, student_actions], dim=0)
            actions = clip_walk_action(actions, action_low, action_high)

        next_obs, rewards, terminated, truncated, info = env.step(actions)
        done = terminated | truncated
        replay.add(obs, actions, rewards, done)
        obs = next_obs

        if step >= args.learning_starts and len(replay) >= max(args.batch_size, num_envs + 1):
            stats = sac_step(
                policy, critics, tgt_critics, replay,
                policy_opt, critic_opt, alpha_opt, log_alpha, scaler, args,
                action_low, action_high,
            )

        s = step + 1
        if s % args.log_interval == 0:
            elapsed = max(time.time() - t0, 1e-6)
            writer.add_scalar("train/fps", (s - start_step) / elapsed, s)
            writer.add_scalar("reward/total", rewards.float().mean().detach().item(), s)
            writer.add_scalar("replay/size", len(replay), s)
            for key, value in info.items():
                scalar = value.float().mean().detach().item() if torch.is_tensor(value) else float(value)
                if key.startswith("r_"):
                    writer.add_scalar(f"reward/{key[2:]}", scalar, s)
                else:
                    writer.add_scalar(f"env/{key}", scalar, s)
            for key, value in stats.items():
                writer.add_scalar(f"train/{key}", value, s)

        if s % args.save_interval == 0:
            _save(save_dir, s, policy, critics, tgt_critics, log_alpha)

    _save(save_dir, args.timesteps, policy, critics, tgt_critics, log_alpha)
    writer.close()


def _save(save_dir, step, policy, critics, tgt_critics, log_alpha):
    path = os.path.join(save_dir, f"checkpoint_{step}.pt")
    ckpt = {"step": step, "policy": policy.state_dict(), "log_alpha": log_alpha.data}
    for i, (critic, target) in enumerate(zip(critics, tgt_critics)):
        ckpt[f"critic_{i}"] = critic.state_dict()
        ckpt[f"tgt_critic_{i}"] = target.state_dict()
    torch.save(ckpt, path)
    print(f"  Saved {path}")


@torch.no_grad()
def _eval(env, policy, max_episodes, dev, action_low, action_high):
    obs, _ = env.reset()
    episodes = 0
    ep_rewards = torch.zeros(env.num_envs, device=dev)
    while episodes < max_episodes:
        mean, _ = policy.compute({"observations": obs})
        obs, reward, terminated, truncated, _ = env.step(clip_walk_action(mean, action_low, action_high))
        ep_rewards += reward.squeeze(-1)
        done = (terminated | truncated).squeeze(-1)
        if done.any():
            for idx in done.nonzero(as_tuple=True)[0]:
                episodes += 1
                print(f"ep={episodes} env={int(idx)} reward={ep_rewards[idx].item():+.3f}")
                ep_rewards[idx] = 0.0
                if episodes >= max_episodes:
                    break


if __name__ == "__main__":
    main()
