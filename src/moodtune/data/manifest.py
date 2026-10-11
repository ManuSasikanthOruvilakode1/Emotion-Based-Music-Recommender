"""Scan the class folders, canonicalise every image, and remove duplicates.

The folders hold 77k files but only ~39k distinct images: most 48 px images are stored twice
under different numbers (``img-anger (1).jpg`` == ``img-anger (5961).jpg``). Duplicates must be
removed *before* splitting, otherwise copies of a test image sit in the training set.

Each file becomes a :class:`Record` with a ``status``:

* ``kept``            - used for train / val / test
* ``duplicate``       - pixel-identical to a kept image (``dup_of`` names it)
* ``near_duplicate``  - re-encoded, mirrored or slightly altered copy (pixel correlation
                        >= ``near_dup_corr``) of a kept image
* ``label_conflict``  - (near-)identical images filed under different emotions; all copies dropped
* ``blank``           - near-uniform image
"""

from __future__ import annotations

import csv
import hashlib
from collections import defaultdict
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from moodtune.labels import parse_folder
from moodtune.preprocess import canonical

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Record:
    path: str
    label: str
    source: str
    width: int
    height: int
    sha: str  # hash of the decoded pixels (format-independent exact identity)
    dhash: str  # 64-bit difference hash of the canonical face
    std: float
    cache_path: str = ""
    status: str = ""
    dup_of: str = ""
    split: str = ""


def source_of(width: int, height: int) -> str:
    """The dataset mixes FER-style 48 px JPEGs with 224 px PNGs from a second collection."""
    return "fer48" if max(width, height) <= 64 else "hr224"


def read_image(path: Path) -> np.ndarray:
    """Decode with OpenCV in a way that also works for non-ASCII paths on Windows."""
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"could not decode {path}")
    return img


def dhash_bits(face: np.ndarray) -> int:
    small = cv2.resize(face, (9, 8), interpolation=cv2.INTER_AREA).astype(np.int16)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int(np.packbits(bits).view(">u8")[0])


def _process(args: tuple[Path, Path, int]) -> tuple[Record, bytes]:
    src, root, face_size = args
    with Image.open(src) as im:
        gray = np.asarray(im.convert("L"))
    h, w = gray.shape
    face = canonical(gray, face_size)
    rec = Record(
        path=src.relative_to(root).as_posix(),
        label=parse_folder(src.parent.name),
        source=source_of(w, h),
        width=w,
        height=h,
        sha=hashlib.sha1(gray.tobytes() + f"{h}x{w}".encode()).hexdigest(),
        dhash=f"{dhash_bits(face):016x}",
        std=round(float(gray.std()), 3),
    )
    return rec, face.tobytes()


def scan(root: Path) -> list[Path]:
    files = [p for p in sorted(root.rglob("*")) if p.suffix.lower() in IMAGE_SUFFIXES]
    if not files:
        raise FileNotFoundError(f"no images found under {root}")
    return files


def build_records(root: Path, face_size: int, workers: int = 8) -> tuple[list[Record], np.ndarray]:
    """Decode every image once. Returns the records and their canonical faces (N, S, S) uint8."""
    jobs = [(p, root, face_size) for p in scan(root)]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            out = list(pool.map(_process, jobs, chunksize=256))
    else:
        out = [_process(j) for j in jobs]
    records = [r for r, _ in out]
    faces = np.stack([np.frombuffer(b, np.uint8).reshape(face_size, face_size) for _, b in out])
    return records, faces


# -- duplicate detection ----------------------------------------------------------------------
def _standardise(faces: np.ndarray) -> np.ndarray:
    x = faces.reshape(len(faces), -1).astype(np.float32)
    x -= x.mean(1, keepdims=True)
    x /= np.linalg.norm(x, axis=1, keepdims=True).clip(1e-6)
    return x


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def near_duplicate_pairs(
    faces: np.ndarray, min_corr: float, chunk: int = 1024
) -> list[tuple[int, int, float]]:
    """All pairs (i < j) whose faces, or one face and the other's mirror image, nearly match.

    Exhaustive: Pearson correlation of the canonical pixels for every pair, computed as one
    chunked matrix product, so no near copy is missed by a hashing shortcut.
    """
    n = len(faces)
    if n < 2:
        return []
    z = _standardise(faces)
    z_flip = _standardise(np.ascontiguousarray(faces[:, :, ::-1]))
    pairs: dict[tuple[int, int], float] = {}
    for start in range(0, n, chunk):
        rows = slice(start, min(start + chunk, n))
        for other in (z, z_flip):
            corr = z[rows] @ other.T
            ii, jj = np.nonzero(corr >= min_corr)
            vals = corr[ii, jj]
            ii = ii + start
            for a, b, c in zip(ii.tolist(), jj.tolist(), vals.tolist(), strict=True):
                if a != b:
                    key = (min(a, b), max(a, b))
                    pairs[key] = max(c, pairs.get(key, -1.0))
    return [(a, b, c) for (a, b), c in sorted(pairs.items())]


def deduplicate(
    records: list[Record],
    faces: np.ndarray,
    min_std: float,
    min_corr: float,
) -> dict:
    """Set ``status`` and ``dup_of`` on every record in place. Returns summary counts."""
    for r in records:
        r.status, r.dup_of = "", ""
    order = sorted(range(len(records)), key=lambda i: records[i].path)

    for i in order:
        if records[i].std < min_std:
            records[i].status = "blank"

    # 1. exact copies (same decoded pixels)
    by_sha: dict[str, list[int]] = defaultdict(list)
    for i in order:
        if not records[i].status:
            by_sha[records[i].sha].append(i)
    candidates = []
    for group in by_sha.values():
        if len({records[i].label for i in group}) > 1:
            for i in group:
                records[i].status = "label_conflict"
            continue
        head = group[0]
        candidates.append(head)
        for i in group[1:]:
            records[i].status, records[i].dup_of = "duplicate", records[head].path

    # 2. near copies among the survivors (re-encoded, mirrored, lightly edited)
    candidates.sort(key=lambda i: records[i].path)
    pairs = near_duplicate_pairs(faces[candidates], min_corr)
    uf = _UnionFind(len(candidates))
    for a, b, _ in pairs:
        uf.union(a, b)
    groups: dict[int, list[int]] = defaultdict(list)
    for local in range(len(candidates)):
        groups[uf.find(local)].append(candidates[local])
    for members in groups.values():
        members.sort(key=lambda i: records[i].path)
        if len({records[i].label for i in members}) > 1:
            for i in members:
                records[i].status = "label_conflict"
            # exact copies of a conflicting image go too
            paths = {records[i].path for i in members}
            for r in records:
                if r.status == "duplicate" and r.dup_of in paths:
                    r.status = "label_conflict"
            continue
        records[members[0]].status = "kept"
        for i in members[1:]:
            records[i].status, records[i].dup_of = "near_duplicate", records[members[0]].path

    counts: dict[str, int] = defaultdict(int)
    for r in records:
        counts[r.status] += 1
    return {"near_duplicate_pairs": len(pairs), "status": dict(sorted(counts.items()))}


def write_cache(records: Sequence[Record], faces: np.ndarray, cache_root: Path) -> None:
    """Write the canonical face of every kept image as a lossless PNG named by its pixel hash."""
    for r, face in zip(records, faces, strict=True):
        if r.status != "kept":
            r.cache_path = ""
            continue
        path = cache_root / r.sha[:2] / f"{r.sha}.png"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            ok, buf = cv2.imencode(".png", face)
            if not ok:
                raise RuntimeError(f"could not encode {r.path}")
            buf.tofile(str(path))
        r.cache_path = path.as_posix()


def load_face(record: Record) -> np.ndarray:
    return read_image(Path(record.cache_path))


def write_manifest(records: Iterable[Record], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = [f.name for f in fields(Record)]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=names)
        w.writeheader()
        for r in records:
            w.writerow(asdict(r))


def read_manifest(path: Path) -> list[Record]:
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return [
        Record(
            **{
                **row,
                "width": int(row["width"]),
                "height": int(row["height"]),
                "std": float(row["std"]),
            }
        )
        for row in rows
    ]
