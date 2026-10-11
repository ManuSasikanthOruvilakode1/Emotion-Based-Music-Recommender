"""Face detection and cropping.

OpenCV's YuNet (a small CNN detector, 2023) is used when its ONNX file is present: it finds
turned, tilted and partly lit faces that the 2001 Haar cascade misses, and it returns eye
landmarks so the crop can be rotated upright like the dataset faces. The YuNet file ships in
``models/``. If it is missing, the Haar cascade is used, but only with OpenCV 4.x: OpenCV 5
moved ``CascadeClassifier`` out of the main package.
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass

import cv2
import numpy as np

from moodtune.config import DetectorConfig

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Face:
    box: tuple[int, int, int, int]  # x, y, w, h in image pixels
    score: float
    # (right eye, left eye) centres in image pixels, if the detector provides landmarks
    eyes: tuple[tuple[float, float], tuple[float, float]] | None = None

    @property
    def area(self) -> int:
        return self.box[2] * self.box[3]


class FaceDetector:
    def __init__(self, cfg: DetectorConfig):
        self.cfg = cfg
        self._lock = threading.Lock()
        self._yunet = None
        if cfg.yunet_path.exists():
            self._yunet = cv2.FaceDetectorYN.create(
                str(cfg.yunet_path), "", (320, 320), cfg.score_threshold, 0.3, 5000
            )
            self.kind = "yunet"
        elif hasattr(cv2, "CascadeClassifier"):
            log.warning("%s not found; falling back to the Haar cascade", cfg.yunet_path)
            self._haar = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            self.kind = "haar"
        else:
            raise FileNotFoundError(
                f"{cfg.yunet_path} not found, and OpenCV {cv2.__version__} has no Haar cascade "
                "fallback. Download face_detection_yunet_2023mar.onnx from the OpenCV model zoo."
            )

    def detect(self, bgr: np.ndarray) -> list[Face]:
        """All faces, largest first."""
        faces = self._detect_yunet(bgr) if self._yunet is not None else self._detect_haar(bgr)
        faces = [f for f in faces if min(f.box[2], f.box[3]) >= self.cfg.min_face_px]
        return sorted(faces, key=lambda f: f.area, reverse=True)

    def _detect_yunet(self, bgr: np.ndarray) -> list[Face]:
        h, w = bgr.shape[:2]
        with self._lock:  # the detector object holds the input size, so serialise calls
            self._yunet.setInputSize((w, h))
            _, rows = self._yunet.detect(bgr)
        out = []
        for row in rows if rows is not None else []:
            x, y, bw, bh = row[:4].round().astype(int).tolist()
            out.append(
                Face(
                    box=_clip_box(x, y, bw, bh, w, h),
                    score=float(row[14]),
                    eyes=((float(row[4]), float(row[5])), (float(row[6]), float(row[7]))),
                )
            )
        return out

    def _detect_haar(self, bgr: np.ndarray) -> list[Face]:
        gray = cv2.equalizeHist(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
        h, w = gray.shape
        boxes = self._haar.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(self.cfg.min_face_px,) * 2
        )
        return [Face(_clip_box(*map(int, b), w, h), 1.0) for b in boxes]


def _clip_box(x: int, y: int, w: int, h: int, img_w: int, img_h: int) -> tuple[int, int, int, int]:
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(img_w, x + w), min(img_h, y + h)
    return x0, y0, max(0, x1 - x0), max(0, y1 - y0)


def crop_face(
    bgr: np.ndarray, face: Face, margin: float, align: bool, out_size: int = 96
) -> np.ndarray:
    """Square grayscale crop centred on the face, optionally rotated so the eyes are level.

    Parts of the crop that fall outside the frame are filled by edge replication rather than
    black, so the model does not see artificial borders it never saw in training.
    """
    x, y, w, h = face.box
    cx, cy = x + w / 2, y + h / 2
    side = max(w, h) * (1 + margin)
    angle = 0.0
    if align and face.eyes is not None:
        (rx, ry), (lx, ly) = face.eyes
        angle = math.degrees(math.atan2(ly - ry, lx - rx))
    scale = out_size / side
    m = cv2.getRotationMatrix2D((cx, cy), angle, scale)
    m[0, 2] += out_size / 2 - cx
    m[1, 2] += out_size / 2 - cy
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    return cv2.warpAffine(
        gray, m, (out_size, out_size), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE
    )
