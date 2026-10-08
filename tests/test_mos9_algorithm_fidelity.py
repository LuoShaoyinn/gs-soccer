"""Executable checks of the inherited learner and collection protocol."""
import ast
from pathlib import Path
import subprocess
import unittest
from dataclasses import asdict
import torch
from algorithm.mos9_limit_action.config import GroundedSACConfig
from algorithm.mos9_limit_action.learner import GroundedSACLearner
from algorithm.mos9_limit_action.replay import ReplayBatch, VectorReplayBuffer
from train_mos9_limit_action import UpdateBudget


class StripDocs(ast.NodeTransformer):
    def visit_FunctionDef(self,node):
        self.generic_visit(node)
        if node.body and isinstance(node.body[0],ast.Expr) and isinstance(node.body[0].value,ast.Constant) and isinstance(node.body[0].value.value,str):node.body.pop(0)
        return node
    visit_ClassDef=visit_FunctionDef


def normalized(source):
    tree=StripDocs().visit(ast.parse(source))
    tree.body=[n for n in tree.body if not (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='MONITOR_HORIZONS' for t in n.targets))]
    return ast.dump(tree,include_attributes=False)


class FidelityTest(unittest.TestCase):
    def test_only_declared_task_and_resource_config_differences(self):
        namespace = {}
        source = subprocess.check_output(['git','show','9778c66:algorithm/grounded_sac/config.py'],text=True)
        exec(source, namespace)
        old = asdict(namespace['GroundedSACConfig']())
        new = asdict(GroundedSACConfig())
        allowed = {'observation_dim', 'action_dim', 'action_limit', 'horizons',
                   'step_penalty', 'value_lower_bound', 'action_limit_margin',
                   'replay_capacity', 'updates_per_env_step', 'action_slew'}
        changed = {k for k in old.keys() | new.keys() if old.get(k) != new.get(k)}
        self.assertEqual(changed, allowed)
        self.assertEqual(new['action_slew'], 0)

    def test_learner_and_network_computations_match_baseline(self):
        for module in ('learner','networks'):
            old=subprocess.check_output(['git','show',f'9778c66:algorithm/grounded_sac/{module}.py'],text=True)
            new=Path(f'algorithm/mos9_limit_action/{module}.py').read_text()
            self.assertEqual(normalized(old),normalized(new),module)

    def test_reference_updates_do_not_initialize_sac_actor(self):
        cfg=GroundedSACConfig(observation_dim=3,action_dim=2,hidden_dim=8,horizons=4,batch_size=4,device='cpu')
        replay=VectorReplayBuffer(8,3,2)
        batch=ReplayBatch(torch.randn(4,3),torch.randn(4,2),torch.tensor([0.,0.,0.,1.]),torch.randn(4,3),torch.tensor([False,False,False,True]),torch.tensor([False,False,False,True]),torch.zeros(4,dtype=torch.bool))
        ids=replay.add_batch(batch); replay.mark_human_suffix(ids.tolist())
        learner=GroundedSACLearner(cfg,replay); learner.fit_normalizer_once()
        before={k:v.clone() for k,v in learner.sac_actor.state_dict().items()}
        learner.pretrain_iql_update(replay.sample(4,'cpu',human_suffix=True))
        self.assertTrue(all(torch.equal(v,learner.sac_actor.state_dict()[k]) for k,v in before.items()))
        self.assertFalse(torch.equal(learner.sac_actor.output.weight,learner.iql_actor.output.weight))

    def test_fractional_utd_matches_old_row_definition(self):
        for envs,batch in ((4,256),(4,4096),(256,4096)):
            b=UpdateBudget(64,batch); updates=0
            for _ in range(64):
                b.add(envs)
                while b.take():updates+=1
            self.assertEqual(updates*batch/(64*envs),64)

if __name__=='__main__':unittest.main()
