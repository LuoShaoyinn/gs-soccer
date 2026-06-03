from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(kw_only=True)
class CameraLocationRNNConfig:
    image_size: tuple[int, int] = (640, 480)
    proprio_dim: int = 4
    hidden_dim: int = 128
    lstm_layers: int = 1
    dropout: float = 0.0


class CameraLocationRNN(nn.Module):
    """CNN image encoder plus LSTM sequence head for camera xy localization."""

    def __init__(self, cfg: CameraLocationRNNConfig):
        super().__init__()
        self.cfg = cfg
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.feature = nn.Sequential(
            nn.Linear(64 + cfg.proprio_dim, cfg.hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.rnn = nn.LSTM(
            input_size=cfg.hidden_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.lstm_layers,
            dropout=cfg.dropout if cfg.lstm_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.hidden_dim, 2),
        )

    def forward(self, image_depth: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        """Predict normalized xy.

        image_depth: [B, T, 4, H, W] RGB in [0, 1] plus normalized depth.
        proprio: [B, T, 4] projected gravity xyz plus yaw.
        """
        batch, steps = image_depth.shape[:2]
        visual = self.cnn(image_depth.reshape(batch * steps, *image_depth.shape[2:]))
        fused = torch.cat((visual, proprio.reshape(batch * steps, -1)), dim=-1)
        fused = self.feature(fused).reshape(batch, steps, -1)
        out, _ = self.rnn(fused)
        return self.head(out)


def camera_observation_to_tensors(
    image: torch.Tensor,
    depth: torch.Tensor,
    projected_gravity: torch.Tensor,
    yaw: torch.Tensor,
    image_size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert raw Genesis camera state to model tensors for one or more frames."""
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if depth.ndim == 2:
        depth = depth.unsqueeze(0)
    rgb = image.to(torch.float32).permute(0, 3, 1, 2) / 255.0
    depth = depth.to(torch.float32).unsqueeze(1)
    depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    depth = torch.clamp(depth / 10.0, 0.0, 1.0)
    target_hw = (image_size[1], image_size[0])
    rgb = F.interpolate(rgb, size=target_hw, mode="bilinear", align_corners=False)
    depth = F.interpolate(depth, size=target_hw, mode="nearest")
    visual = torch.cat((rgb, depth), dim=1)
    yaw = yaw.reshape(-1, 1).to(torch.float32)
    proprio = torch.cat((projected_gravity.to(torch.float32), yaw), dim=-1)
    return visual, proprio
