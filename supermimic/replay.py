"""CPU ring buffer of vector steps with explicit episode boundaries and physical next states."""

import torch


class Replay:
    def __init__(self, capacity):
        self.capacity, self.cursor, self.size = capacity, 0, 0
        self.data = {}

    def append(self, **row):
        row = {k: v.detach().cpu() for k, v in row.items()}
        if not self.data:
            self.data = {
                k: torch.empty((self.capacity, *v.shape), dtype=v.dtype)
                for k, v in row.items()
            }
        if set(row) != set(self.data):
            raise ValueError("replay schema changed")
        for k, v in row.items():
            self.data[k][self.cursor].copy_(v)
        self.cursor = (self.cursor + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, window, device):
        if self.size < window:
            raise ValueError("not enough replay vector steps")
        envs = self.data["action"].shape[1]
        accepted, attempts, remaining = [], 0, batch_size
        while remaining:
            attempts += 1
            if attempts > 100:
                raise ValueError(
                    "no sufficient within-episode replay windows; collect more data"
                )
            count = max(remaining * 2, 64)
            # Logical indices are chronological even after ring wraparound.
            logical = torch.randint(self.size - window + 1, (count,))
            start = (self.cursor - self.size + logical) % self.capacity
            e = torch.randint(envs, (count,))
            times = (start[:, None] + torch.arange(window)) % self.capacity
            done = self.data["done"][times, e[:, None]]
            good = ~done[:, :-1].any(-1)
            chosen = good.nonzero().flatten()[:remaining]
            if len(chosen):
                accepted.append((times[chosen], e[chosen]))
                remaining -= len(chosen)
        times = torch.cat([x[0] for x in accepted])
        e = torch.cat([x[1] for x in accepted])
        # Gathering on CPU also keeps GPU reservation independent of replay capacity.
        return {k: v[times, e[:, None]].to(device) for k, v in self.data.items()}

    def state_dict(self):
        return {
            "capacity": self.capacity,
            "cursor": self.cursor,
            "size": self.size,
            "data": {k: v[: self.size].clone() for k, v in self.data.items()},
        }

    def load_state_dict(self, state):
        if state["capacity"] != self.capacity:
            raise ValueError("replay capacity mismatch")
        self.cursor, self.size = state["cursor"], state["size"]
        self.data = {
            k: torch.empty((self.capacity, *v.shape[1:]), dtype=v.dtype)
            for k, v in state["data"].items()
        }
        for k, v in state["data"].items():
            self.data[k][: len(v)].copy_(v)

    def cut_episodes(self):
        # A resumed physics scene cannot continue a recorded episode.
        if self.size:
            self.data["done"][(self.cursor - 1) % self.capacity] = True
