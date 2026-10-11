"""Stratified train / val / test split over the de-duplicated images, plus a leakage check.

Stratification is by (emotion, source) so that the small 224 px collection, the only source of
``contempt``, is represented in every split in the same proportion.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence

import numpy as np
from sklearn.model_selection import train_test_split

from moodtune.data.manifest import Record, _standardise


def _strata(records: Sequence[Record], idx: Sequence[int]) -> list[str]:
    return [f"{records[i].label}/{records[i].source}" for i in idx]


def assign_splits(
    records: list[Record], val_fraction: float, test_fraction: float, seed: int
) -> None:
    """Set ``split`` to train / val / test for kept records (empty for dropped ones)."""
    kept = [i for i, r in enumerate(records) if r.status == "kept"]
    rest, test = train_test_split(
        kept, test_size=test_fraction, stratify=_strata(records, kept), random_state=seed
    )
    train, val = train_test_split(
        rest,
        test_size=val_fraction / (1 - test_fraction),
        stratify=_strata(records, rest),
        random_state=seed + 1,
    )
    for r in records:
        r.split = ""
    for name, idx in (("train", train), ("val", val), ("test", test)):
        for i in idx:
            records[i].split = name


def nearest_correlation(
    queries: np.ndarray, reference: np.ndarray, chunk: int = 1024
) -> np.ndarray:
    """For each query face, the highest pixel correlation with any reference face (or its mirror)."""
    q = _standardise(queries)
    q_flip = _standardise(np.ascontiguousarray(queries[:, :, ::-1]))
    best = np.full(len(queries), -1.0, dtype=np.float32)
    for start in range(0, len(reference), chunk * 8):
        block = _standardise(reference[start : start + chunk * 8])
        best = np.maximum(best, (q @ block.T).max(1))
        best = np.maximum(best, (q_flip @ block.T).max(1))
    return best


def leakage_report(
    records: Sequence[Record], faces: np.ndarray, seed: int = 0, sample: int = 2000
) -> dict:
    """Split sizes, and how close held-out test faces are to their nearest training face.

    The same statistic is computed for a naive random split of the raw files (what the old
    ``validation_split`` did) to show how much of that test set had copies in training.
    """
    rng = np.random.default_rng(seed)
    counts: dict[str, Counter] = defaultdict(Counter)
    sources: dict[str, Counter] = defaultdict(Counter)
    for r in records:
        if r.split:
            counts[r.split][r.label] += 1
            sources[r.split][r.source] += 1

    def describe(corr: np.ndarray, exact: np.ndarray) -> dict:
        return {
            "n_sampled": len(corr),
            "share_exact_copy_in_train": float(exact.mean()),
            "share_corr_ge_0.95": float((corr >= 0.95).mean()),
            "share_corr_ge_0.90": float((corr >= 0.90).mean()),
            "median_nearest_corr": float(np.median(corr)),
        }

    train = [i for i, r in enumerate(records) if r.split == "train"]
    test = [i for i, r in enumerate(records) if r.split == "test"]
    q = rng.choice(test, size=min(sample, len(test)), replace=False)
    train_sha = {records[i].sha for i in train}
    ours = describe(
        nearest_correlation(faces[q], faces[train]),
        np.array([records[i].sha in train_sha for i in q]),
    )

    usable = [i for i, r in enumerate(records) if r.status != "blank"]
    perm = rng.permutation(usable)
    n_test = round(len(usable) * 0.2)
    naive_test, naive_train = perm[:n_test], perm[n_test:]
    naive_sha = {records[i].sha for i in naive_train}
    nq = rng.choice(naive_test, size=min(sample, len(naive_test)), replace=False)
    naive = describe(
        nearest_correlation(faces[nq], faces[naive_train]),
        np.array([records[i].sha in naive_sha for i in nq]),
    )

    return {
        "counts": {s: dict(sorted(c.items())) for s, c in sorted(counts.items())},
        "sources": {s: dict(sorted(c.items())) for s, c in sorted(sources.items())},
        "test_to_nearest_train_face": {"dedup_stratified": ours, "naive_random_80_20": naive},
    }
