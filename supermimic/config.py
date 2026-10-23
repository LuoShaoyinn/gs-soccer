import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT / "assets/MOS9/MOS9_walk.urdf"
JOINT_NAMES = tuple(
    j.attrib["name"]
    for j in ET.parse(URDF).getroot().findall("joint")
    if j.attrib["type"] == "revolute"
)
TRACKED_LINKS = (
    "base_link",
    "Rshoulder",
    "Rarm",
    "Rhand",
    "Lshoulder",
    "Larm",
    "Lhand",
    "Rhip",
    "Rleg1",
    "Rleg2",
    "Rfoot",
    "Lhip",
    "Lleg1",
    "Lleg2",
    "Lfoot",
)


@dataclass
class Config:
    seed: int = 0
    num_envs: int = 128
    dt: float = 0.02
    substeps: int = 10
    history: int = 10
    future: int = 10
    model_window: int = 8
    horizon: int = 32
    replay_steps: int = 1000
    collect_steps: int = 5
    update_ratio: float = 0.4  # learner updates per collected vector step
    model_batches: int = 10
    batch_size: int = 128
    imagination_batch: int = 128
    critic_epochs: int = 5
    critic_minibatches: int = 4
    model_widths: tuple = (512, 512, 512)
    actor_widths: tuple = (512, 256, 128)
    critic_widths: tuple = (256, 128)
    latent_dim: int = 32
    lr: float = 2e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    action_noise: float = 0.1
    action_scale: float = 0.5
    kp: float = 40.0
    kd: float = 2.0
    torque_limit: float = 20.0
    height_threshold: float = 0.15  # MOS9 is smaller than the paper's G1
    tilt_threshold: float = 0.8
    randomize: bool = True
    sensor_noise: bool = True
    episode_steps: int = 1000

    def validate(self):
        for name in (
            "num_envs",
            "substeps",
            "history",
            "future",
            "model_window",
            "horizon",
            "replay_steps",
            "collect_steps",
            "model_batches",
            "batch_size",
            "imagination_batch",
            "critic_epochs",
            "critic_minibatches",
            "episode_steps",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.replay_steps < self.model_window:
            raise ValueError("replay_steps must cover model_window")
        if not (0 < self.gamma < 1 and 0 <= self.gae_lambda <= 1):
            raise ValueError("invalid discount or GAE lambda")
        for name in (
            "dt",
            "lr",
            "action_scale",
            "kp",
            "kd",
            "torque_limit",
            "update_ratio",
            "height_threshold",
            "tilt_threshold",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")

    def paper_networks(self):
        self.model_widths = (2048, 2048, 2048)
        self.actor_widths = (2048, 1024, 512, 256)
