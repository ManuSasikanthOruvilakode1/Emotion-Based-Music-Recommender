"""Frame -> faces -> emotions. Works entirely in memory: nothing is written to disk."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from moodtune.config import DetectorConfig
from moodtune.face import FaceDetector, crop_face


class Predictor(Protocol):
    """Anything with ``labels`` and ``predict(list of face crops) -> (N, C) probabilities``."""

    labels: list[str]
    meta: dict

    def predict(self, faces: list[np.ndarray]) -> np.ndarray: ...


@dataclass
class FaceResult:
    box: tuple[int, int, int, int]
    detector_score: float
    probs: np.ndarray
    labels: list[str] = field(repr=False)

    @property
    def emotion(self) -> str:
        return self.labels[int(self.probs.argmax())]

    @property
    def confidence(self) -> float:
        return float(self.probs.max())

    def to_dict(self) -> dict:
        return {
            "box": list(self.box),
            "detector_score": round(self.detector_score, 3),
            "emotion": self.emotion,
            "confidence": round(self.confidence, 4),
            "probs": {k: round(float(v), 4) for k, v in zip(self.labels, self.probs, strict=True)},
        }


class Analyzer:
    def __init__(self, predictor: Predictor, detector: FaceDetector, cfg: DetectorConfig):
        self.predictor = predictor
        self.detector = detector
        self.cfg = cfg

    def analyze(self, bgr: np.ndarray, max_faces: int = 4) -> list[FaceResult]:
        """Detect faces (largest first) and classify each one."""
        faces = self.detector.detect(bgr)[:max_faces]
        if not faces:
            return []
        crops = [crop_face(bgr, f, self.cfg.margin, self.cfg.align) for f in faces]
        probs = self.predictor.predict(crops)
        return [
            FaceResult(f.box, f.score, p, list(self.predictor.labels))
            for f, p in zip(faces, probs, strict=True)
        ]
