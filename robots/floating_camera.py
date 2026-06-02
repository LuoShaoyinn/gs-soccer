import torch
import numpy as np
import genesis as gs
from typing import Any
from dataclasses import dataclass, field

from .robot import Robot


@dataclass(kw_only=True)
class FloatingCameraConfig:
    res: tuple[int, int] = (320, 240)
    pos: np.ndarray = field(default_factory=\
            lambda: np.array([2.2, -3.0, 1.4], dtype=np.float32))
    target_offset: np.ndarray = field(default_factory=\
            lambda: np.array([0.0, 0.0, 0.45], dtype=np.float32))
    quat: np.ndarray = field(default_factory=\
            lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    up: np.ndarray = field(default_factory=\
            lambda: np.array([0.0, 0.0, 1.0], dtype=np.float32))
    fov: float = 45.0
    near: float = 0.01
    far: float = 20.0
    env_idx: int | None = 0
    use_madrona: bool = False


class FloatingCameraRobot(Robot):
    cfg: FloatingCameraConfig
    def __init__(self, cfg: FloatingCameraConfig, scene: gs.Scene):
        self.cfg = cfg
        self.scene = scene
        self.camera = None
        self.pos = None
        self.quat = None

    def __apply_pose(self, pos: torch.Tensor, quat: torch.Tensor) -> None:
        if self.camera is None:
            raise RuntimeError("Floating camera has not been built. Call build() before scene.build().")
        if not hasattr(self.camera, "_apply_camera_transform"):
            raise RuntimeError("Genesis camera sensor does not expose _apply_camera_transform().")
        camera_T = gs.trans_quat_to_T(pos, quat)
        self.camera._apply_camera_transform(camera_T)
        self.camera._stale = True
        self.pos = pos
        self.quat = quat

    def __action_to_pose(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(action, torch.Tensor):
            raise TypeError(f"FloatingCameraRobot.step() expects torch.Tensor, got {type(action).__name__}.")
        camera_action = action.to(device=gs.device)
        if camera_action.ndim == 2 and camera_action.shape[0] == 1:
            camera_action = camera_action[0]
        if camera_action.numel() != 7:
            raise ValueError(f"FloatingCameraRobot.step() expects a 7D action, got shape {tuple(camera_action.shape)}.")
        camera_action = camera_action.reshape(7)
        pos = camera_action[:3]
        # Genesis quaternions use [qw, qx, qy, qz], matching gs.T_to_quat and gs.trans_quat_to_T.
        quat = camera_action[3:7]
        quat = quat / torch.clamp(torch.linalg.norm(quat), min=1e-6)
        return pos, quat

    def __normalize_image(self, image: torch.Tensor, envs_idx: torch.Tensor) -> torch.Tensor:
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.shape[0] == envs_idx.shape[0]:
            return image
        return image[envs_idx]

    def __normalize_depth(self, depth: torch.Tensor, envs_idx: torch.Tensor) -> torch.Tensor:
        if depth.ndim == 2:
            depth = depth.unsqueeze(0)
        if depth.ndim == 4 and depth.shape[1] == 1:
            depth = depth[:, 0]
        if depth.ndim == 4 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        if depth.shape[0] == envs_idx.shape[0]:
            return depth
        return depth[envs_idx]

    def __fallback_depth(self, envs_idx: torch.Tensor) -> torch.Tensor:
        width, height = self.cfg.res
        return torch.full((envs_idx.shape[0], height, width), torch.nan, dtype=torch.float32, device=gs.device)

    def __read_depth(self, envs_idx: torch.Tensor) -> torch.Tensor:
        if self.camera is None:
            raise RuntimeError("Floating camera has not been built. Call build() before scene.build().")
        try:
            if hasattr(self.camera, "_ensure_camera_registered"):
                self.camera._ensure_camera_registered()
            if hasattr(self.camera, "_camera_wrapper"):
                _, depth, _, _ = self.camera._shared_metadata.renderer.render_camera(
                    self.camera._camera_wrapper,
                    rgb=False,
                    depth=True,
                    segmentation=False,
                    normal=False,
                )
            elif hasattr(self.camera, "_shared_metadata") and hasattr(self.camera, "_camera_obj"):
                _, depth, *_ = self.camera._shared_metadata.renderer.render(
                    rgb=False,
                    depth=True,
                    segmentation=False,
                    normal=False,
                    antialiasing=False,
                    force_render=True,
                )
            else:
                return self.__fallback_depth(envs_idx)
        except Exception:
            return self.__fallback_depth(envs_idx)

        depth_tensor = torch.as_tensor(depth, dtype=torch.float32, device=gs.device)
        return self.__normalize_depth(depth_tensor, envs_idx)

    def build(self) -> None:
        lookat = self.cfg.pos + np.array([1.0, 0.0, 0.0], dtype=np.float32)
        camera_options = (
            gs.sensors.BatchRendererCameraOptions
            if self.cfg.use_madrona
            else gs.sensors.RasterizerCameraOptions
        )
        kwargs = {"use_rasterizer": True} if self.cfg.use_madrona else {}
        self.camera = self.scene.add_sensor(
            camera_options(
                res=self.cfg.res,
                pos=tuple(self.cfg.pos),
                lookat=tuple(lookat),
                up=tuple(self.cfg.up),
                fov=self.cfg.fov,
                near=self.cfg.near,
                far=self.cfg.far,
                **kwargs,
            )
        )

    def config(self) -> None:
        self.n_envs = self.scene.n_envs
        self.n_dofs = 7
        self.pos = torch.from_numpy(self.cfg.pos).to(gs.device)
        self.quat = torch.from_numpy(self.cfg.quat).to(gs.device)

    def reset(self, envs_idx: torch.Tensor | None = None, **kwargs) -> None:
        self.pos = torch.from_numpy(self.cfg.pos).to(gs.device)
        self.quat = torch.from_numpy(self.cfg.quat).to(gs.device)
        self.__apply_pose(pos=self.pos, quat=self.quat)

    def step(self, action: torch.Tensor) -> None:
        pos, quat = self.__action_to_pose(action)
        self.__apply_pose(pos=pos, quat=quat)

    def render(self, **kwargs) -> Any:
        return self.camera.read(**kwargs).rgb

    def get_state(self, envs_idx: torch.Tensor) -> dict[str, torch.Tensor]:
        image = self.camera.read(envs_idx=envs_idx).rgb
        pos = self.pos.broadcast_to((envs_idx.shape[0], 3))
        quat = self.quat.broadcast_to((envs_idx.shape[0], 4))
        return {
            "body_pos": pos,
            "body_quat": quat,
            "image": self.__normalize_image(image, envs_idx),
            "depth_image": self.__read_depth(envs_idx),
        }
