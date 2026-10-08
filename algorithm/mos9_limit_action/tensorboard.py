"""The sixteen scalar observations used by the MOS9 training dashboard."""
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
