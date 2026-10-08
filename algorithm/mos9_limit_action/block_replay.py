"""Disk replay adapter with a teacher index view over one physical buffer.

Pinned to torch-block-replay f276e76. Teacher sampling uses its immutable RAM
blocks under the package metadata lock; no second transition store is created.
"""
import math
from pathlib import Path
import torch
from block_replay import BlockReplayBuffer
from .replay import ReplayBatch


class TaggedBlockBuffer(BlockReplayBuffer):
    def __init__(self, *args, suffix_index, **kwargs):
        self.suffix_index = suffix_index
        self.teacher_blocks = set()
        self.row_ranges = []
        super().__init__(*args, **kwargs)

    def _publish(self):
        index = self._next
        start = int(self._building['row_id'][0])
        stop = int(self._building['row_id'][self._used-1])+1
        super()._publish()
        with self._lock:
            self.row_ranges.append((start, stop))
            if bool(self.suffix_index[start:stop].any()):
                self.teacher_blocks.add(index)


class DiskReplayBuffer:
    def __init__(self, capacity, observation_dim, action_dim, *, directory,
                 device, batch_size, block_size=32768, ram_blocks=32,
                 gpu_cache_bytes=256 << 20, checkpoint_every=1000, seed=0):
        self.capacity = capacity
        self.device = torch.device(device)
        self.size = self.total_transitions = 0
        self.suffix_index = torch.zeros(capacity, dtype=torch.bool, device="cpu")
        self._tag_version = 0
        # A checkpoint/warmup flush publishes partial blocks. Reserve slots for
        # these too, so FIFO pruning cannot occur before the row quota fills.
        slots = math.ceil(capacity/block_size) + math.ceil(capacity/checkpoint_every) + 4
        self.store = TaggedBlockBuffer(directory, capacity=slots*block_size,
            block_size=block_size, ram_blocks=ram_blocks, batch_size=batch_size, suffix_index=self.suffix_index,
            device=device, prefetch=4, refresh_every=4,
            gpu_cache_bytes=gpu_cache_bytes, device_chunk_size=4096,
            device_refresh_every=8, device_refresh_chunks=8, seed=seed)
        self._teacher_indices = {}

    def __len__(self):
        return self.size

    @property
    def full(self):
        return self.size == self.capacity

    def add_batch(self, batch):
        if self.full:
            raise BufferError('replay full; stop collection')
        n = min(len(batch.reward), self.capacity-self.size)
        fields = {k:getattr(batch,k)[:n].to(self.device) for k in ReplayBatch.__dataclass_fields__}
        fields["human_suffix"] = torch.zeros(n, dtype=torch.bool, device=self.device)
        fields["row_id"] = torch.arange(self.size, self.size+n, device=self.device)
        self.store.append_async(fields)
        ids = torch.arange(self.size, self.size+n, device=self.device)
        self.size += n
        self.total_transitions += n
        return ids

    def mark_human_suffix(self, indices):
        if not indices:
            return
        import bisect
        ids = torch.tensor(indices, dtype=torch.long, device='cpu')
        if int(ids.min()) < 0 or int(ids.max()) >= self.size:
            raise IndexError('suffix index outside accepted replay rows')
        with self.store._lock:
            self.suffix_index[ids] = True
            self._tag_version += 1
            starts = [start for start,stop in self.store.row_ranges]
            for row_id in indices:
                block = bisect.bisect_right(starts, row_id)-1
                if block >= 0 and row_id < self.store.row_ranges[block][1]:
                    self.store.teacher_blocks.add(block)

    def flush(self):
        self.store.flush()

    def _all_teacher(self, field):
        # Called only once after initial collection, before online training.
        self.flush()
        values = []
        for path in sorted(self.store.directory.glob('block_*.pt')):
            data = torch.load(path, map_location='cpu', weights_only=True)
            values.append(data[field][self.suffix_index[data['row_id']]])
        if not values or not sum(len(v) for v in values):
            raise RuntimeError('no teacher rows')
        return torch.cat(values).to(self.device)

    def all_human_observations(self):
        return self._all_teacher('observation')

    def all_human_actions(self):
        return self._all_teacher('action')

    def sample(self, count, target_device, *, human_suffix=False):
        if not human_suffix:
            if count != self.store.batch_size:
                raise ValueError('block sampler batch size must match learner batch size')
            data = self.store.sample_async().wait()
            result = {k:data[k].to(target_device) for k in ReplayBatch.__dataclass_fields__}
            result['human_suffix'] = self.suffix_index[data['row_id'].cpu()].to(target_device)
            return ReplayBatch(**result)
        with self.store._lock:
            entries = list(self.store._cache.items())
        valid_ids = {i for i,_ in entries}
        self._teacher_indices = {i:v for i,v in self._teacher_indices.items() if i in valid_ids}
        views = []
        for index, data in entries:
            previous = self._teacher_indices.get(index)
            if previous is None or previous[0] is not data or previous[2] != self._tag_version:
                previous = (data, torch.nonzero(self.suffix_index[data['row_id']]).flatten(), self._tag_version)
                self._teacher_indices[index] = previous
            if len(previous[1]):
                views.append(previous[:2])
        if not views:
            # Rescue sampling must remain available if random RAM admission has
            # temporarily omitted all teacher blocks. Load one known tagged block.
            with self.store._lock:
                index = next(iter(self.store.teacher_blocks), None)
            if index is None:
                raise RuntimeError('no published teacher block')
            self.store._load(index)
            return self.sample(count, target_device, human_suffix=True)
        lengths = torch.tensor([len(ids) for _,ids in views], device='cpu')
        cumulative = lengths.cumsum(0)
        draws = torch.randint(int(cumulative[-1]), (count,), device='cpu')
        block_ids = torch.searchsorted(cumulative, draws, right=True)
        result = {}
        for k in ReplayBatch.__dataclass_fields__:
            example = views[0][0][k]
            out = torch.empty((count,*example.shape[1:]), dtype=example.dtype, device='cpu')
            for i,(data,ids) in enumerate(views):
                selected = block_ids == i
                if selected.any():
                    offset = int(cumulative[i-1]) if i else 0
                    out[selected] = data[k][ids[draws[selected]-offset]]
            result[k] = out.to(target_device)
        result["human_suffix"] = torch.ones(count, dtype=torch.bool, device=target_device)
        return ReplayBatch(**result)

    def state_dict(self):
        self.flush()
        path = self.store.directory/'suffix_index.pt'
        temporary = path.with_suffix('.tmp')
        torch.save(self.suffix_index[:self.size].clone(), temporary)
        temporary.replace(path)
        return dict(suffix_index=str(path),suffix_rows=int(self.suffix_index[:self.size].sum()),backend='torch-block-replay',directory=str(self.store.directory),
                    capacity=self.capacity,size=self.size,total_transitions=self.total_transitions,
                    stats=self.store.stats(),resumable=False)

    def close(self):
        self.store.close()
