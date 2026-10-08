"""Check completed preflight data against episode provenance and phase snapshots."""
import argparse
import json
import math
from pathlib import Path

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from algorithm.mos9_limit_action.tensorboard import SCALAR_TAGS


def equal_state(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal_state(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(equal_state(x, y) for x, y in zip(a, b))
    return a == b


def verify(run):
    load = lambda name: torch.load(run/name, map_location='cpu', weights_only=False)
    status = json.loads((run/'status.json').read_text())
    checkpoint = load('checkpoint.pt')
    before = load('audit_before_warmup.pt')
    after = load('audit_after_warmup.pt')
    cfg = checkpoint['config']
    args = cfg['arguments']
    n = args['num_envs']
    seed = checkpoint['seed_transitions']
    size = checkpoint['replay']['size']
    online = checkpoint['online_transitions']
    updates = checkpoint['online_updates']
    assert cfg['algorithm_contract'] == 'old_branch_fidelity_v2'
    assert status['reason'] == 'buffer_full' and size == args['replay_capacity']
    assert not (run/'failure.json').exists()
    assert checkpoint['warmup_transitions'] == args['warmup_transitions']
    assert seed + checkpoint['warmup_transitions'] + online == size
    assert before['warmup_transitions'] == before['online_updates'] == 0
    assert after['warmup_transitions'] == args['warmup_transitions']
    assert after['online_transitions'] == after['online_updates'] == 0
    assert equal_state(before['learner'], after['learner']), 'warmup changed learner state'
    assert not equal_state(before['learner']['iql_actor'], before['learner']['sac_actor'])
    assert not before['learner']['sac_actor_optim']['state']
    assert not before['learner']['sac_q_optim']['state']
    assert cfg['learner']['gamma'] == .99
    assert cfg['learner']['hidden_dim'] == 512 and cfg['learner']['batch_size'] == 4096
    assert cfg['environment']['action_slew'] == 0
    last_batch = size % n or n
    expected_updates = math.floor((online-last_batch)*args['utd']/args['batch_size'])
    assert updates == expected_updates
    assert math.isclose(checkpoint['utd_credit'], online*args['utd']/args['batch_size']-updates)
    assert checkpoint['learner']['update_count'] == args['pretrain_updates'] + updates
    for key in ('iql_actor','iql_q1','iql_q2','iql_v','sac_actor','sac_q1','sac_q2','action_likeness'):
        assert all(torch.isfinite(v).all() for v in checkpoint['learner'][key].values())

    blocks = [torch.load(p,map_location='cpu',weights_only=True)
              for p in sorted((run/'replay').glob('block_*.pt'))]
    data = {key: torch.cat([b[key] for b in blocks]) for key in blocks[0]}
    assert torch.equal(data['row_id'], torch.arange(size)), 'missing or overwritten rows'
    assert not data['human_suffix'].any(), 'physical flags must not override the suffix index'
    assert torch.equal(data['success'], data['reward'] > 0)
    assert torch.isin(data['reward'], torch.tensor([-1.,0.,1.])).all()
    assert data['terminal'][data['success']].all()
    assert torch.equal(data['action'], data['next_observation'][:,36:54])
    assert (data['action'].abs() <= 3).all()
    assert torch.allclose(data['next_observation'][:,-1]-data['observation'][:,-1],
                          torch.full((size,),1/500),atol=1e-7,rtol=0)
    assert (data['next_observation'][data['terminal'],-1] > 0).all()
    assert torch.allclose(data['next_observation'][data['success'],-1],
                          torch.ones(int(data['success'].sum())))

    expected = torch.zeros(size,dtype=torch.bool)
    expected[:seed] = True
    elapsed = [0]*n
    successful_rescues = failed_rescues = autonomous_successes = 0
    for line in (run/'metrics.jsonl').read_text().splitlines():
        record = json.loads(line)
        if 'episode' not in record:
            continue
        row = record['episode']
        env = row['env_id']
        first = elapsed[env]
        end = first + row['steps']
        elapsed[env] = end
        if row['teacher']:
            takeover = row['takeover_step']
            assert env >= n//2 and takeover >= 1
            assert row['teacher_steps'] == row['steps'] - takeover
            if row['success']:
                successful_rescues += 1
                ids = seed + torch.arange(first+takeover,end)*n + env
                expected[ids[ids < size]] = True
            else:
                failed_rescues += 1
        else:
            assert row['teacher_steps'] == 0 and row['takeover_step'] == -1
            autonomous_successes += int(row['success'])
    actual = torch.load(checkpoint['replay']['suffix_index'],weights_only=True)
    assert torch.equal(actual, expected), 'suffix bitmap differs from real episode provenance'
    assert successful_rescues > 0, 'preflight did not exercise successful rescue confirmation'
    assert int(actual.sum()) == checkpoint['replay']['suffix_rows']
    events = EventAccumulator(str(run/'tensorboard')).Reload()
    tags = set(events.Tags()['scalars'])
    assert tags <= SCALAR_TAGS and events.Tags()['tensors'] == []
    report = dict(passed=True,replay_rows=size,seed_rows=seed,
                  warmup_rows=checkpoint['warmup_transitions'],online_rows=online,
                  online_updates=updates,effective_utd=updates*args['batch_size']/online,
                  pending_update_credit=checkpoint['utd_credit'],
                  successful_teacher_suffix_rows=int(actual.sum()),
                  successful_rescues=successful_rescues,failed_rescues=failed_rescues,
                  autonomous_successes=autonomous_successes,
                  unchanged_learner_during_warmup=True,independent_sac_initialization=True,
                  physical_terminal_observations=True,scalar_tags=sorted(tags))
    (run/'algorithm_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run',type=Path)
    verify(parser.parse_args().run.resolve())
