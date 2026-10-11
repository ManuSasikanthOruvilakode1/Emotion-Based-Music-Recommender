"""Deterministic preprocessing shared by the image cache, evaluation, ONNX export and the app.

Depends only on NumPy and OpenCV so the deployed app does not need PyTorch. Every face, be it
a 48 px dataset JPEG, a 224 px dataset PNG or a webcam crop, goes through ``canonical`` and
then ``prepare``, so offline metrics measure the deployed code path.
"""

from __future__ import annotations

import cv2
import numpy as np

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def to_gray(img: np.ndarray) -> np.ndarray:
    """uint8 grayscale from a gray, BGR or BGRA image."""
    if img.ndim == 2:
        return img
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canonical(gray: np.ndarray, face_size: int) -> np.ndarray:
    """Square uint8 face of ``face_size`` px: the common format for every image source."""
    h, w = gray.shape[:2]
    if h != w:  # centre-crop to square
        s = min(h, w)
        y0, x0 = (h - s) // 2, (w - s) // 2
        gray = gray[y0 : y0 + s, x0 : x0 + s]
    if gray.shape[0] == face_size:
        return np.ascontiguousarray(gray)
    interp = cv2.INTER_AREA if gray.shape[0] > face_size else cv2.INTER_LINEAR
    return cv2.resize(gray, (face_size, face_size), interpolation=interp)


def prepare(face: np.ndarray, face_size: int, image_size: int) -> np.ndarray:
    """Face crop (gray or BGR, any size) -> normalised float32 3 x S x S network input."""
    x = canonical(to_gray(face), face_size)
    x = cv2.resize(x, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    x = x.astype(np.float32) / 255.0
    x = (x[None, :, :] - MEAN[:, None, None]) / STD[:, None, None]
    return np.ascontiguousarray(x, dtype=np.float32)


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits / temperature
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)
