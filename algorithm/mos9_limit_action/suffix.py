"""Confirm a teacher suffix only after its episode reaches real success."""

class SuccessfulSuffixTracker:
    def __init__(self, num_envs, replay):
        self.replay = replay
        self.episodes = [[] for _ in range(num_envs)]
        self.confirmed_suffixes = 0

    def record(self, indices, teacher_mask, success, done):
        indices = indices.tolist()
        teacher_mask, success, done = teacher_mask.tolist(), success.tolist(), done.tolist()
        for env_id, index in enumerate(indices):
            episode = self.episodes[env_id]
            episode.append((index, teacher_mask[env_id]))
            if not done[env_id]:
                continue
            if success[env_id]:
                suffix = []
                for row_id, teacher in reversed(episode):
                    if not teacher:
                        break
                    suffix.append(row_id)
                if suffix:
                    self.replay.mark_human_suffix(suffix)
                    self.confirmed_suffixes += 1
            self.episodes[env_id] = []
