import torch
import numpy as np
import genesis as gs
from typing import Any
from dataclasses import dataclass, field


@dataclass(kw_only=True)
class FloatingCameraConfig:
    res: tuple[int, int] = (320, 240)
    pos: np.ndarray = field(default_factory=\
            lambda: np.array([1.2, -1.8, 0.9], dtype=np.float32))
    target_offset: np.ndarray = field(default_factory=\
            lambda: np.array([0.0, 0.0, 0.25], dtype=np.float32))
    up: np.ndarray = field(default_factory=\
            lambda: np.array([0.0, 0.0, 1.0], dtype=np.float32))
    fov: float = 70.0
    near: float = 0.01
    far: float = 20.0
    GUI: bool = False
    env_idx: int | None = 0


class FloatingCameraRobot:
    def __init__(self, cfg: FloatingCameraConfig, scene: gs.Scene):
        self.cfg = cfg
        self.scene = scene
        self.camera = None
        self.target = None

    def look_at(self, target: Any) -> None:
        self.target = target

    def build(self) -> None:
        lookat = self.cfg.pos + np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.camera = self.scene.add_camera(
            res=self.cfg.res,
            pos=tuple(self.cfg.pos),
            lookat=tuple(lookat),
            up=tuple(self.cfg.up),
            fov=self.cfg.fov,
            GUI=self.cfg.GUI,
            near=self.cfg.near,
            far=self.cfg.far,
            env_idx=self.cfg.env_idx,
        )

    def config(self) -> None:
        pass

    def reset(self, envs_idx: torch.Tensor | None = None, **kwargs) -> None:
        pass

    def step(self) -> None:
        self.update_pose()

    def update_pose(self) -> None:
        if self.camera is None:
            raise RuntimeError("Floating camera has not been built. Call build() before scene.build().")
        if self.target is None:
            return

        target_pos = self.target.get_pos(envs_idx=self.cfg.env_idx)
        if target_pos.ndim > 1:
            target_pos = target_pos[0]
        offset = torch.as_tensor(self.cfg.target_offset, dtype=gs.tc_float, device=gs.device)
        up = torch.as_tensor(self.cfg.up, dtype=gs.tc_float, device=gs.device)
        self.camera.set_pose(
            pos=torch.as_tensor(self.cfg.pos, dtype=gs.tc_float, device=gs.device),
            lookat=target_pos + offset,
            up=up,
        )

    def render(self, **kwargs) -> Any:
        self.update_pose()
        return self.camera.render(**kwargs)
