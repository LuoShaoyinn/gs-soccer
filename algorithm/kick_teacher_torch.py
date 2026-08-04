"""Native Torch form of the exported 646-D / 22-action kick teacher."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class KickTeacherTorch(nn.Module):
    """Exact Torch reconstruction of the small exported ONNX actor graph."""

    def __init__(self) -> None:
        super().__init__()
        self.ball_encoder = nn.Sequential(
            nn.Linear(20, 128), nn.ELU(), nn.Linear(128, 64), nn.ELU(), nn.Linear(64, 10),
        )
        self.actor = nn.Sequential(
            nn.Linear(636, 512), nn.ELU(), nn.Linear(512, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 22),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.ndim != 2 or observation.shape[1] != 646:
            raise ValueError("KickTeacherTorch expects [batch, 646] observations")
        ball = self.ball_encoder(observation[:, 626:646])
        feature = torch.cat((observation[:, :24], observation[:, 24:48], observation[:, 48:98], observation[:, 98:274], observation[:, 274:450], observation[:, 450:626], ball), dim=1)
        return self.actor(feature)


def convert_onnx_kick_actor(onnx_path: str | Path, checkpoint_path: str | Path) -> Path:
    """Convert the known simple Gemm/ELU kick actor to a portable `.pt` file."""

    import onnx
    from onnx import numpy_helper

    source, destination = Path(onnx_path), Path(checkpoint_path)
    graph = onnx.load(source)
    weights = {item.name: torch.from_numpy(numpy_helper.to_array(item)).clone() for item in graph.graph.initializer}
    model = KickTeacherTorch()
    named = dict(model.named_parameters())
    mapping = {
        "ball_encoder.0.weight": "ac.ball_encoder.model.0.weight",
        "ball_encoder.0.bias": "ac.ball_encoder.model.0.bias",
        "ball_encoder.2.weight": "ac.ball_encoder.model.2.weight",
        "ball_encoder.2.bias": "ac.ball_encoder.model.2.bias",
        "ball_encoder.4.weight": "ac.ball_encoder.model.4.weight",
        "ball_encoder.4.bias": "ac.ball_encoder.model.4.bias",
        "actor.0.weight": "ac.actor.0.weight", "actor.0.bias": "ac.actor.0.bias",
        "actor.2.weight": "ac.actor.2.weight", "actor.2.bias": "ac.actor.2.bias",
        "actor.4.weight": "ac.actor.4.weight", "actor.4.bias": "ac.actor.4.bias",
        "actor.6.weight": "ac.actor.6.weight", "actor.6.bias": "ac.actor.6.bias",
    }
    for target, source_name in mapping.items():
        if target not in named or source_name not in weights or named[target].shape != weights[source_name].shape:
            raise ValueError(f"unexpected kick actor tensor: {target} <- {source_name}")
        named[target].data.copy_(weights[source_name])
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format": "kick-teacher-torch-v1", "state_dict": model.state_dict()}, destination)
    return destination


def load_kick_teacher(checkpoint_path: str | Path, device: torch.device | str) -> KickTeacherTorch:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if payload.get("format") != "kick-teacher-torch-v1":
        raise ValueError(f"not a kick-teacher checkpoint: {checkpoint_path}")
    model = KickTeacherTorch()
    model.load_state_dict(payload["state_dict"])
    return model.to(device).eval()
