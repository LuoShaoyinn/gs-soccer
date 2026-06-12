import os
import numpy as np
import onnxruntime as ort


class OnnxPolicy:
    def __init__(self, model_dir: str):
        actor_path = os.path.join(model_dir, "actor.onnx")
        if not os.path.exists(actor_path):
            raise FileNotFoundError(f"Actor not found: {actor_path}")
        available = ort.get_available_providers()
        providers = [p for p in ["ROCMExecutionProvider", "MIGraphXExecutionProvider",
                                  "CUDAExecutionProvider", "CPUExecutionProvider"]
                     if p in available]
        try:
            self.session = ort.InferenceSession(actor_path, providers=providers)
        except RuntimeError:
            print("[WARN] GPU EPs failed, falling back to CPUExecutionProvider")
            self.session = ort.InferenceSession(actor_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        info_in = self.session.get_inputs()[0]
        info_out = self.session.get_outputs()[0]
        self.proprio_dim = int(info_in.shape[1])
        self.action_dim = int(info_out.shape[1])
        print(f"OnnxPolicy: {info_in.shape} -> {info_out.shape}")

    def infer(self, proprio: np.ndarray) -> np.ndarray:
        if proprio.shape[1] != self.proprio_dim:
            raise ValueError(f"Expected dim {self.proprio_dim}, got {proprio.shape[1]}")
        return self.session.run(None, {self.input_name: proprio.astype(np.float32)})[0]
