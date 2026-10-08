"""Branch-local MOS9 limit-action experiment with transition-based UTD."""
import argparse
from collections import defaultdict, deque
import time
from dataclasses import asdict
import json
from pathlib import Path
import signal
import torch
from algorithm.mos9_limit_action.tensorboard import TrainingSummaryWriter as SummaryWriter
from mos9_walk import parse_args as walk_args, build_walking_env
from algorithm.mos9_teacher.phase import PhaseWalkingTeacher
from algorithm.mos9_limit_action.config import GroundedSACConfig
from algorithm.mos9_limit_action.learner import GroundedSACLearner
from algorithm.mos9_limit_action.replay import ReplayBatch, VectorReplayBuffer


from algorithm.mos9_teacher.environment import WalkingEnv as TrainingEnv


class UpdateBudget:
    """UTD counts primary critic minibatch rows; auxiliary losses are separate."""
    def __init__(self, utd, batch_size):
        self.utd, self.batch_size = utd, batch_size
        self.credit = 0.0
    def add(self, transitions):
        self.credit += transitions * self.utd / self.batch_size
    def take(self):
        if self.credit + 1e-10 < 1:
            return False
        self.credit -= 1
        return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', type=Path, default=Path('runs/mos9_limit_action/run1'))
    p.add_argument('--num-envs', type=int, default=4)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--transitions', type=int, default=None, help='optional online transition limit; otherwise stop when replay fills')
    p.add_argument('--replay-capacity', type=int, default=100000000)
    p.add_argument('--teacher-checkpoint', type=Path, default=None)
    p.add_argument('--teacher-transitions', type=int, default=2048)
    p.add_argument('--pretrain-updates', type=int, default=500)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--utd', type=float, default=64.0)
    p.add_argument('--teacher-intervention-prob', type=float, default=0.01)
    p.add_argument('--replay-backend', choices=('memory','block'), default='block')
    p.add_argument('--replay-block-size', type=int, default=4096)
    p.add_argument('--replay-ram-blocks', type=int, default=4096)
    p.add_argument('--replay-gpu-cache-mib', type=int, default=16384)
    p.add_argument('--exploration-std', type=float, default=0.02)
    p.add_argument('--sim-freq', type=int, default=4000)
    p.add_argument('--action-slew', type=float, default=0.05)
    p.add_argument('--checkpoint-every', type=int, default=4096)
    p.add_argument('--flat-soles', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--terrain-height', type=float, default=0.001)
    p.add_argument('--teacher', choices=('planned', 'onnx'), default='planned')
    p.add_argument('--teacher-policy', type=Path, default=Path('/home/luoshaoyinn/workspace/URSoccerLab/py_example/models/policies/mos9_walk_v11_5500.onnx'))
    args = p.parse_args()
    if min(args.num_envs, args.replay_capacity, args.teacher_transitions, args.batch_size, args.checkpoint_every) <= 0 or (args.transitions is not None and args.transitions <= 0) or args.utd <= 0 or args.pretrain_updates < 0 or not 0 <= args.teacher_intervention_prob <= 1 or args.exploration_std < 0 or args.action_slew <= 0 or args.sim_freq <= 0 or args.sim_freq % 50:
        p.error('invalid positive counts, UTD, exploration, or teacher probability')
    if args.teacher_transitions < args.batch_size:
        p.error('teacher-transitions must be at least batch-size')
    if args.replay_capacity < args.teacher_transitions:
        p.error('replay-capacity must fit initial teacher transitions')
    if min(args.replay_block_size,args.replay_ram_blocks) < 1 or args.replay_gpu_cache_mib < 0:
        p.error('invalid replay cache budgets')
    args.run_dir.mkdir(parents=True, exist_ok=False)
    settings = walk_args(['--num-envs', str(args.num_envs), '--seed', str(args.seed), '--sim-freq', str(args.sim_freq), '--terrain-height', str(args.terrain_height)] + (['--flat-soles'] if args.flat_soles else []))
    env, plan, kin, _, _, _, gait = build_walking_env(settings, TrainingEnv)
    if args.teacher == 'onnx':
        from algorithm.mos9_teacher.onnx import OnnxWalkingTeacher
        teacher = OnnxWalkingTeacher(env, plan['names'], kin, settings, args.teacher_policy)
        teacher_metadata = teacher.metadata
    else:
        teacher = PhaseWalkingTeacher(plan, env.MDP.home.device, gait)
        teacher_metadata = {'kind':'planned','adaptive_phase':True,'gait':asdict(gait),'position_gain':teacher.position_gain.tolist(),'velocity_gain':teacher.velocity_gain.tolist(),'roll_gain':teacher.roll_gain,'pitch_gain':teacher.pitch_gain}
    lower, upper = torch.tensor([kin.bounds[n] for n in plan['names']], device=env.MDP.home.device).T
    cfg = GroundedSACConfig(observation_dim=env.observation_space.shape[0], action_dim=len(plan['names']), action_limit=3., action_slew=args.action_slew, horizons=500, hidden_dim=128, gamma=1., step_penalty=0., value_lower_bound=-1., action_limit_margin=1./500, batch_size=args.batch_size, exploration_std=args.exploration_std, iql_pretrain_updates=args.pretrain_updates, warmup_transitions=args.teacher_transitions, replay_capacity=args.replay_capacity, device=str(env.MDP.home.device))
    if args.replay_backend == 'block':
        if args.teacher_checkpoint is not None:
            p.error('block replay does not support importing/resuming an existing replay')
        from algorithm.mos9_limit_action.block_replay import DiskReplayBuffer
        replay = DiskReplayBuffer(cfg.replay_capacity,cfg.observation_dim,cfg.action_dim,directory=args.run_dir/'replay',device=cfg.device,batch_size=args.batch_size,block_size=args.replay_block_size,ram_blocks=args.replay_ram_blocks,gpu_cache_bytes=args.replay_gpu_cache_mib << 20,checkpoint_every=args.checkpoint_every,seed=args.seed)
    else:
        replay = VectorReplayBuffer(cfg.replay_capacity, cfg.observation_dim, cfg.action_dim, device=cfg.device)
    learner = GroundedSACLearner(cfg, replay)
    for actor in (learner.iql_actor, learner.sac_actor):
        actor.joint_lower.copy_(lower)
        actor.joint_upper.copy_(upper)
    obs = env.reset()[0]
    online_transitions = online_updates = seed_transitions = 0
    budget = UpdateBudget(args.utd, args.batch_size)
    if args.teacher_checkpoint is not None:
        initial = torch.load(args.teacher_checkpoint, map_location='cpu', weights_only=False)
        if initial['config']['environment'] != {'policy_hz':50,'max_steps':500,'integrator':'implicitfast','solver_iterations':50,'solver_tolerance':1e-5,'sim_freq':settings.sim_freq,'action_slew':args.action_slew,'terrain_height':settings.terrain_height,'flat_soles':args.flat_soles}:
            raise ValueError('teacher checkpoint task/model settings differ')
        if initial['online_transitions'] != 0:
            raise ValueError('seed import requires a teacher-only checkpoint')
        replay.load_state_dict(initial['replay'])
        seed_transitions = replay.size
        print(f'imported factual teacher transitions={seed_transitions}', flush=True)
    stopped = False
    def stop(*_):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    manifest = {'arguments': {k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}, 'learner':asdict(cfg), 'environment':{'policy_hz':50,'max_steps':500,'integrator':'implicitfast','solver_iterations':50,'solver_tolerance':1e-5,'sim_freq':settings.sim_freq,'action_slew':args.action_slew,'terrain_height':settings.terrain_height,'flat_soles':args.flat_soles}, 'teacher':teacher_metadata, 'physics_solver':'CG with full scene reset on episode boundaries', 'reference_data':'all factual teacher transitions, including terminal failures', 'replay_policy':'one physical buffer with teacher index view; stop at capacity without overwriting; final vector step truncated to remaining rows', 'controller':{'learner_first':True,'sticky_until_episode_end':True,'rescue_envs':list(range(args.num_envs//2,args.num_envs)),'per_step_probability':args.teacher_intervention_prob}, 'utd_definition':'primary SAC TD minibatch rows / new online transitions; pretraining excluded; each update also samples one IQL batch and one floor batch'}
    (args.run_dir/'config.json').write_text(json.dumps(manifest, indent=2)+'\n')
    writer = SummaryWriter(str(args.run_dir / "tensorboard"), flush_secs=10)
    writer.add_text("config", json.dumps(manifest, indent=2), 0)
    recent = defaultdict(lambda: deque(maxlen=100))
    episode_counts = defaultdict(int)
    started = time.monotonic()
    def save():
        checkpoint = {'learner':learner.state_dict(), 'replay':replay.state_dict(), 'config':manifest, 'online_transitions':online_transitions,'online_updates':online_updates, 'seed_transitions':seed_transitions, 'utd_credit':budget.credit, 'torch_rng':torch.get_rng_state(), 'cuda_rng':torch.cuda.get_rng_state_all()}
        temp = args.run_dir/'checkpoint.tmp'
        torch.save(checkpoint, temp)
        temp.replace(args.run_dir/'checkpoint.pt')
    def collect(action, teacher_mask):
        nonlocal obs
        with torch.no_grad():
            if not torch.isfinite(action).all() or not torch.isfinite(obs).all():
                raise FloatingPointError("non-finite policy action or observation")
            action = action.clamp(lower, upper)
            if args.teacher == 'onnx':
                teacher.observe_applied_action(action, teacher_mask)
            next_obs, reward, term, trunc, info = env.step(action)
            if not torch.isfinite(info['final_observation']).all() or not torch.isfinite(reward).all():
                raise FloatingPointError('non-finite physical transition')
            indices = replay.add_batch(ReplayBatch(obs.clone(),action.clone(),reward.flatten(),info['final_observation'],reward.flatten()>0,(term|trunc).flatten(),teacher_mask))
            replay.mark_human_suffix(indices[teacher_mask[:len(indices)]].tolist())
            obs = next_obs
            return len(indices)
    try:
        seed_cursor = len(env.MDP.completed)
        while seed_transitions < args.teacher_transitions and not replay.full and not stopped:
            state = env.get_state(env.all_envs_idx)
            action = teacher.act(state,env.MDP.steps,env.MDP.origins,env.MDP.yaws)
            seed_transitions += collect(action, torch.ones(args.num_envs, dtype=torch.bool, device=obs.device))
            writer.add_scalar("collection/teacher_transitions", seed_transitions, seed_transitions)
            for row in env.MDP.completed[seed_cursor:]:
                with (args.run_dir/'teacher_collection.jsonl').open('a') as log:
                    log.write(json.dumps(row)+'\n')
                for metric in ('steps', 'distance', 'success', 'fallen'):
                    writer.add_scalar('collection/teacher/'+metric, float(row[metric]), seed_transitions)
                print(f'teacher collection episode={row}', flush=True)
            seed_cursor = len(env.MDP.completed)
            if seed_transitions % 250 < args.num_envs:
                print(f'teacher transitions={seed_transitions} episodes={len(env.MDP.completed)}',flush=True)
        if stopped:
            return
        learner.fit_normalizer_once()
        for i in range(args.pretrain_updates):
            if stopped:
                break
            metrics = learner.pretrain_iql_update(replay.sample(args.batch_size,cfg.device,human_suffix=True))
            if (i+1)%10 == 0 or i+1 == args.pretrain_updates:
                for key, value in metrics.items():
                    writer.add_scalar("pretrain/"+key, float(value), i+1)
            if (i+1)%50 == 0:
                print(f'pretrain={i+1} actor_loss={float(metrics["iql/actor_loss"]):.6f}',flush=True)
        # Start the actor at the fitted reference rather than random joint targets.
        learner.sac_actor.load_state_dict(learner.iql_actor.state_dict())
        save()
        writer.flush()
        started = time.monotonic()
        obs = env.reset()[0]
        teacher_mode = torch.zeros(args.num_envs,dtype=torch.bool,device=obs.device)
        rescue_enabled = torch.arange(args.num_envs,device=obs.device) >= args.num_envs//2
        episode_teacher_steps = torch.zeros(args.num_envs,dtype=torch.long,device=obs.device)
        episode_takeover_step = torch.full((args.num_envs,),-1,dtype=torch.long,device=obs.device)
        teacher_transitions = 0
        cursor = len(env.MDP.completed)
        metrics = {}
        metrics_transition = 0
        with (args.run_dir/'metrics.jsonl').open('a') as log:
            while not replay.full and (args.transitions is None or online_transitions < args.transitions) and not stopped:
                state = env.get_state(env.all_envs_idx)
                with torch.no_grad():
                    proposed = learner.sac_action(obs) + args.exploration_std*torch.randn_like(env.MDP.last_action)
                    proposed = torch.maximum(env.MDP.last_action-args.action_slew, torch.minimum(env.MDP.last_action+args.action_slew, proposed))
                    trigger = (torch.rand(args.num_envs,device=obs.device) < args.teacher_intervention_prob) & rescue_enabled & ~teacher_mode & (env.MDP.steps > 0)
                    episode_takeover_step[trigger] = env.MDP.steps[trigger]
                    teacher_mode |= trigger
                    guided = teacher.act(state,env.MDP.steps,env.MDP.origins,env.MDP.yaws)
                    action = torch.where(teacher_mode[:,None],guided,proposed)
                episode_teacher_steps += teacher_mode.long()
                collected = collect(action,teacher_mode)
                teacher_transitions += int(teacher_mode[:collected].sum())
                online_transitions += collected
                budget.add(collected)
                while budget.take():
                    metrics = learner.update()
                    online_updates += 1
                    metrics_transition = online_transitions
                episodes = env.MDP.completed[cursor:]
                for row in episodes:
                    env_id = row['env_id']
                    row = {**row,'teacher':bool(teacher_mode[env_id]),'teacher_steps':int(episode_teacher_steps[env_id]),'teacher_fraction':float(episode_teacher_steps[env_id])/row['steps'],'takeover_step':int(episode_takeover_step[env_id]),'rescue_enabled':bool(rescue_enabled[env_id])}
                    log.write(json.dumps({'episode':row,'online_transitions':online_transitions})+'\n')
                    group = "teacher" if row['teacher'] else "learner"
                    recent[group].append(row)
                    episode_counts[group] += 1
                    for metric in ('steps', 'distance', 'success', 'fallen'):
                        writer.add_scalar(f"episode/{group}/{metric}", float(row[metric]), online_transitions)
                        writer.add_scalar(f"rolling/{group}/{metric}",
                                          sum(float(e[metric]) for e in recent[group])/len(recent[group]), online_transitions)
                    writer.add_scalar(f"episode/{group}/count", episode_counts[group], online_transitions)
                    print(f'episode={row}',flush=True)
                    teacher_mode[env_id] = False
                    episode_teacher_steps[env_id] = 0
                    episode_takeover_step[env_id] = -1
                cursor = len(env.MDP.completed)
                if online_transitions % 100 < args.num_envs:
                    record = {'online_transitions':online_transitions,'online_updates':online_updates,'effective_utd':online_updates*args.batch_size/online_transitions,'update_credit':budget.credit,'metrics_transition':metrics_transition,**metrics}
                    for key, value in metrics.items():
                        writer.add_scalar(key, value, online_transitions)
                    writer.add_scalar('training/target_utd', args.utd, online_transitions)
                    writer.add_scalar('training/effective_utd', record['effective_utd'], online_transitions)
                    writer.add_scalar('training/online_updates', online_updates, online_transitions)
                    writer.add_scalar('training/update_credit', budget.credit, online_transitions)
                    writer.add_scalar('training/transitions_per_second', online_transitions/max(time.monotonic()-started, 1e-6), online_transitions)
                    writer.add_scalar('training/teacher_fraction', teacher_transitions/online_transitions, online_transitions)
                    writer.add_scalar('training/current_teacher_env_fraction', float(teacher_mode.float().mean()), online_transitions)
                    writer.add_scalar('training/num_envs', args.num_envs, online_transitions)
                    writer.add_scalar('replay/size', len(replay), online_transitions)
                    writer.add_scalar('replay/capacity', replay.capacity, online_transitions)
                    log.write(json.dumps(record)+'\n');log.flush()
                    print(f'transitions={online_transitions} updates={online_updates} UTD={record["effective_utd"]:.3f}',flush=True)
                if online_transitions % args.checkpoint_every < args.num_envs:
                    save()
        if replay.full:
            print(f'replay buffer full rows={len(replay)}/{replay.capacity}; stopping training', flush=True)
    except Exception as error:
        (args.run_dir/'failure.json').write_text(json.dumps({'error':str(error),'online_transitions':online_transitions,'online_updates':online_updates},indent=2)+'\n')
        raise
    finally:
        save()
        (args.run_dir/'status.json').write_text(json.dumps({'reason':'buffer_full' if replay.full else ('signal' if stopped else 'transition_limit_or_error'),'replay_size':len(replay),'replay_capacity':replay.capacity,'online_transitions':online_transitions,'online_updates':online_updates},indent=2)+'\n')
        (args.run_dir/'episodes.json').write_text(json.dumps(env.MDP.completed,indent=2)+'\n')
        writer.close()
        if args.replay_backend == 'block':
            replay.close()
        env.scene.destroy()


if __name__ == '__main__':
    main()
