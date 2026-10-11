"""Runtime emotion predictor backed by ONNX Runtime (no PyTorch needed at inference time)."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np

from moodtune.preprocess import prepare, softmax


class OnnxPredictor:
    def __init__(self, model_path: str | Path, providers: list[str] | None = None):
        import onnxruntime as ort

        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"model not found: {path} (run `moodtune export` first)")
        available = ort.get_available_providers()
        if providers is None:
            providers = [
                p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available
            ]
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(path), sess_options=opts, providers=providers)
        self.meta = {
            k: json.loads(v) for k, v in self.session.get_modelmeta().custom_metadata_map.items()
        }
        self.labels: list[str] = self.meta["labels"]
        self.image_size: int = int(self.meta["image_size"])
        self.face_size: int = int(self.meta["face_size"])
        self.temperature: float = float(self.meta.get("temperature", 1.0))
        self._lock = threading.Lock()

    def predict(self, faces: list[np.ndarray]) -> np.ndarray:
        """Face crops (gray or BGR) -> calibrated probabilities, shape (N, len(labels))."""
        if not faces:
            return np.zeros((0, len(self.labels)), dtype=np.float32)
        batch = np.stack([prepare(f, self.face_size, self.image_size) for f in faces])
        with self._lock:
            (logits,) = self.session.run(None, {"image": batch})
        return softmax(logits, self.temperature)
