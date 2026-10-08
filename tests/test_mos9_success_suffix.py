"""Regression checks for the algorithm's success-only reference dataset."""
import tempfile
import unittest
import torch
from algorithm.mos9_limit_action.replay import ReplayBatch, VectorReplayBuffer
from algorithm.mos9_limit_action.block_replay import DiskReplayBuffer
from algorithm.mos9_limit_action.suffix import SuccessfulSuffixTracker


class SuccessSuffixTest(unittest.TestCase):
    def check_backend(self, replay, device):
        tracker = SuccessfulSuffixTracker(2, replay)
        def step(start, teachers, success=(False,False), done=(False,False)):
            ids = torch.arange(start,start+2,device=device,dtype=torch.float32)
            flags = torch.tensor(teachers,device=device)
            batch = ReplayBatch(ids[:,None].repeat(1,2),ids[:,None],torch.tensor(success,device=device,dtype=torch.float32),
                                (ids+100)[:,None].repeat(1,2),torch.tensor(success,device=device),
                                torch.tensor(done,device=device),torch.zeros(2,device=device,dtype=torch.bool))
            inserted = replay.add_batch(batch)
            tracker.record(inserted, flags, batch.success, batch.terminal)
        step(0,(False,False))
        step(2,(True,True))
        if hasattr(replay,'flush'): replay.flush()
        # Teacher actions remain ineligible while the outcome is unknown.
        with self.assertRaises(RuntimeError): replay.sample(8,device,human_suffix=True)
        # Warm the uniform GPU pool before confirming any published row.
        replay.sample(8,device)
        step(4,(True,True),success=(False,True),done=(True,True))
        if hasattr(replay,'flush'): replay.flush()
        for _ in range(10):
            batch = replay.sample(8,device,human_suffix=True)
            self.assertTrue(batch.human_suffix.all())
            self.assertTrue(torch.isin(batch.action.flatten(),torch.tensor([3.,5.],device=device)).all())
            self.assertTrue(torch.equal(batch.next_observation[:,0],batch.action[:,0]+100))
        # Autonomous success has no teacher suffix; unfinished rescue is ineligible.
        step(6,(False,False),success=(True,True),done=(True,True))
        step(8,(True,True))
        if hasattr(replay,'flush'): replay.flush()
        self.assertEqual(tracker.confirmed_suffixes,1)
        if isinstance(replay,DiskReplayBuffer):
            self.assertEqual(replay.suffix_index[:10].nonzero().flatten().tolist(),[3,5])
            saved=replay.state_dict()
            self.assertEqual(torch.load(saved['suffix_index'],weights_only=True).nonzero().flatten().tolist(),[3,5])
        else:
            self.assertEqual(replay.human_suffix[:10].nonzero().flatten().tolist(),[3,5])

    def test_memory(self):
        self.check_backend(VectorReplayBuffer(10,2,1),'cpu')

    def test_capacity_does_not_confirm_an_unstored_terminal(self):
        replay = VectorReplayBuffer(5, 2, 1)
        tracker = SuccessfulSuffixTracker(2, replay)
        for step in range(3):
            done = torch.tensor([step == 2, step == 2])
            success = torch.tensor([False, step == 2])
            batch = ReplayBatch(torch.zeros(2,2), torch.zeros(2,1), success.float(),
                                torch.ones(2,2), success, done, torch.zeros(2,dtype=torch.bool))
            ids = replay.add_batch(batch)
            tracker.record(ids, torch.ones(2,dtype=torch.bool), success, done)
        self.assertTrue(replay.full)
        self.assertEqual(len(replay), 5)
        self.assertFalse(replay.human_suffix.any())
        self.assertEqual(tracker.confirmed_suffixes, 0)
        with self.assertRaises(BufferError):
            replay.add_batch(batch)

    def test_checkpoint_suffix_snapshots_are_immutable(self):
        with tempfile.TemporaryDirectory() as path:
            replay = DiskReplayBuffer(8,2,1,directory=path,device='cpu',batch_size=2,
                                      block_size=2,ram_blocks=2,gpu_cache_bytes=0)
            try:
                batch = ReplayBatch(torch.zeros(2,2),torch.zeros(2,1),torch.zeros(2),
                                    torch.ones(2,2),torch.zeros(2,dtype=torch.bool),
                                    torch.zeros(2,dtype=torch.bool),torch.zeros(2,dtype=torch.bool))
                replay.add_batch(batch)
                first = replay.state_dict()
                replay.mark_human_suffix([1])
                second = replay.state_dict()
                self.assertNotEqual(first['suffix_index'], second['suffix_index'])
                self.assertFalse(torch.load(first['suffix_index'],weights_only=True).any())
                self.assertEqual(torch.load(second['suffix_index'],weights_only=True).tolist(),[False,True])
            finally:
                replay.close()

    def test_disk_cpu(self):
        with tempfile.TemporaryDirectory() as path:
            b=DiskReplayBuffer(10,2,1,directory=path,device='cpu',batch_size=8,block_size=4,ram_blocks=1,gpu_cache_bytes=0)
            try:self.check_backend(b,'cpu')
            finally:b.close()

    @unittest.skipUnless(torch.cuda.is_available(),'requires GPU')
    def test_disk_gpu(self):
        with tempfile.TemporaryDirectory() as path:
            b=DiskReplayBuffer(10,2,1,directory=path,device='cuda:0',batch_size=8,block_size=4,ram_blocks=2,gpu_cache_bytes=1<<20)
            try:self.check_backend(b,'cuda:0')
            finally:b.close()

if __name__=='__main__': unittest.main()
