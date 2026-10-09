"""The sixteen scalar observations used by the MOS9 training dashboard."""
from collections import deque
from torch.utils.tensorboard import SummaryWriter

SCALAR_TAGS = frozenset({
    'rolling/learner/success', 'rolling/learner/steps', 'rolling/learner/distance',
    'rolling/teacher/success', 'rolling/teacher/steps', 'rolling/teacher/distance',
    'training/teacher_fraction',
    'sac/td_mse', 'sac/actor_loss', 'iql/critic_td',
    'action_likeness/loss', 'floor/rms_violation', 'action_limit/loss',
    'training/effective_utd', 'replay/size', 'training/transitions_per_second',
})
assert len(SCALAR_TAGS) == 16


class TrainingSummaryWriter(SummaryWriter):
    def add_scalar(self, tag, *args, **kwargs):
        if tag in SCALAR_TAGS:
            return super().add_scalar(tag, *args, **kwargs)

    def add_text(self, *args, **kwargs):
        # Full configuration is already persisted in config.json.
        pass


class CollectionSummaryWriter(TrainingSummaryWriter):
    """Use existing tags in a separate run whose x-axis is attempted episodes."""
    def __init__(self, log_dir, *, row_limit, **kwargs):
        super().__init__(log_dir, **kwargs)
        self.recent = deque(maxlen=100)
        self.episodes = self.accepted_rows = 0
        self.row_limit = row_limit

    def add_episode(self, row, *, replay_size=None):
        self.episodes += 1
        self.recent.append(row)
        if row['success']:
            self.accepted_rows = min(self.row_limit, self.accepted_rows + row['steps'])
        for metric in ('success', 'steps', 'distance'):
            self.add_scalar('rolling/teacher/'+metric,
                            sum(float(r[metric]) for r in self.recent)/len(self.recent),
                            self.episodes)
        self.add_scalar('replay/size', self.accepted_rows if replay_size is None else replay_size,
                        self.episodes)
