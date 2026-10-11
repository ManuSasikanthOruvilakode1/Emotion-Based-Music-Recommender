from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from moodtune.labels import EMOTIONS


def random_face(rng: np.random.Generator, size: int) -> np.ndarray:
    """A smooth random grayscale texture: distinct images have low pixel correlation."""
    small = rng.integers(0, 255, (6, 6), dtype=np.uint8)
    img = cv2.resize(small, (size, size), interpolation=cv2.INTER_CUBIC)
    noise = rng.normal(0, 6, (size, size))
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def write(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    assert ok
    buf.tofile(str(path))


@pytest.fixture(scope="session")
def dataset(tmp_path_factory) -> Path:
    """A tiny dataset laid out like the real one, with planted duplicates.

    Per class: 12 unique 48 px JPEGs, 3 unique 224 px PNGs (contempt: PNGs only).
    Planted: each class's first JPEG copied under a new number, a mirrored copy, a
    re-encoded copy at lower quality, one cross-class conflict, and one blank image.
    """
    root = tmp_path_factory.mktemp("data")
    rng = np.random.default_rng(0)
    n = 0
    for c in EMOTIONS:
        folder = {"angry": "angry", "surprise": "surprise"}.get(c, c)
        stem = {"angry": "img-anger", "surprise": "img-suprise"}.get(c, f"img-{c}")
        jpgs = [] if c == "contempt" else [random_face(rng, 48) for _ in range(12)]
        pngs = [random_face(rng, 224) for _ in range(6 if c == "contempt" else 3)]
        for img in jpgs:
            n += 1
            write(root / folder / f"{stem} ({n}).jpg", img)
        for img in pngs:
            n += 1
            write(root / folder / f"{stem} ({n}).png", img)
        if jpgs:
            write(root / folder / f"{stem} ({n + 1000}).jpg", jpgs[0])  # exact copy
            write(root / folder / f"{stem} ({n + 2000}).jpg", jpgs[1][:, ::-1].copy())  # mirror
            _, buf = cv2.imencode(".jpg", jpgs[2], [cv2.IMWRITE_JPEG_QUALITY, 40])
            buf.tofile(str(root / folder / f"{stem} ({n + 3000}).jpg"))  # re-encoded
    # the same image filed under two emotions
    conflict = random_face(rng, 48)
    write(root / "happy" / "img-happy (9001).jpg", conflict)
    write(root / "sad" / "img-sad (9002).jpg", conflict)
    write(root / "neutral" / "img-neutral (9003).jpg", np.full((48, 48), 128, np.uint8))
    return root
