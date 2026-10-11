"""PyTorch dataset over the cached canonical faces."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

from moodtune.data.manifest import Record, load_face
from moodtune.labels import EMOTIONS, SOURCES
from moodtune.preprocess import MEAN, STD, prepare


class _GrayToRGB(torch.nn.Module):
    """1-channel -> 3-channel (a module, not a lambda, so DataLoader workers can pickle it)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.expand(3, -1, -1)


def train_transform(image_size: int) -> v2.Compose:
    return v2.Compose(
        [
            v2.ToImage(),  # 1 x 48 x 48 uint8
            v2.RandomResizedCrop(
                (image_size, image_size), scale=(0.75, 1.0), ratio=(0.9, 1.1), antialias=True
            ),
            v2.RandomHorizontalFlip(),
            v2.RandomApply([v2.RandomAffine(degrees=12, translate=(0.06, 0.06))], p=0.5),
            v2.ColorJitter(brightness=0.35, contrast=0.35),
            # Re-compressing every source blurs the JPEG-vs-PNG difference between the sources.
            v2.RandomApply([v2.JPEG(quality=(30, 95))], p=0.4),
            v2.RandomApply([v2.GaussianBlur(kernel_size=5, sigma=(0.1, 1.2))], p=0.15),
            v2.ToDtype(torch.float32, scale=True),
            _GrayToRGB(),
            v2.Normalize(mean=MEAN.tolist(), std=STD.tolist()),
            v2.RandomErasing(p=0.25, scale=(0.02, 0.12)),
        ]
    )


class FaceDataset(Dataset):
    """Yields (image, label index, source index). Eval mode uses the deployed ``prepare``."""

    def __init__(self, records: Sequence[Record], face_size: int, image_size: int, train: bool):
        self.records = list(records)
        self.face_size = face_size
        self.image_size = image_size
        self.transform = train_transform(image_size) if train else None

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        r = self.records[idx]
        face = load_face(r)
        if self.transform is not None:
            x = self.transform(torch.from_numpy(np.ascontiguousarray(face))[None])
        else:
            x = torch.from_numpy(prepare(face, self.face_size, self.image_size))
        return x, EMOTIONS.index(r.label), SOURCES.index(r.source)


def class_weights(records: Sequence[Record], mode: str) -> torch.Tensor:
    """Per-class loss weights, normalised so the average weight over training samples is 1."""
    freq = Counter(r.label for r in records)
    n = torch.tensor([freq.get(c, 0) for c in EMOTIONS], dtype=torch.float64)
    if mode == "none":
        w = torch.ones_like(n)
    elif mode == "inverse":
        w = 1.0 / n.clamp_min(1)
    elif mode == "inverse_sqrt":
        w = n.clamp_min(1).rsqrt()
    else:
        raise ValueError(f"unknown class weighting {mode!r}")
    w = torch.where(n > 0, w, torch.zeros_like(w))
    w = w * n.sum() / (w * n).sum()
    return w.float()
