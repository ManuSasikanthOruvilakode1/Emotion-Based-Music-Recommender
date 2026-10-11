"""Classification and calibration metrics, overall and per image source."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch import nn
from torch.utils.data import DataLoader

from moodtune.labels import EMOTIONS, SOURCES
from moodtune.preprocess import softmax


@dataclass
class Predictions:
    logits: np.ndarray
    labels: np.ndarray
    sources: np.ndarray
    embeddings: np.ndarray | None = None


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool = True,
    embeddings: bool = False,
) -> Predictions:
    model.eval()
    logits, emb, ys, ss = [], [], [], []
    for x, y, s in loader:
        with torch.autocast(
            device.type, dtype=torch.bfloat16, enabled=amp and device.type == "cuda"
        ):
            fmap = model.features(x.to(device, non_blocking=True))
            z = model.embed(fmap)
            lg = model.head(z)
        logits.append(lg.float().cpu())
        if embeddings:
            emb.append(z.float().cpu())
        ys.append(y)
        ss.append(s)
    return Predictions(
        torch.cat(logits).numpy(),
        torch.cat(ys).numpy(),
        torch.cat(ss).numpy(),
        torch.cat(emb).numpy() if embeddings else None,
    )


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    conf = probs.max(axis=1)
    correct = probs.argmax(axis=1) == labels
    ece = 0.0
    for lo, hi in itertools.pairwise(np.linspace(0, 1, n_bins + 1)):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def fit_temperature(
    logits: np.ndarray, labels: np.ndarray, bounds: tuple[float, float] = (0.5, 5.0)
) -> float:
    """Single-parameter temperature scaling minimising validation NLL."""
    lg = torch.tensor(logits, dtype=torch.float64)
    y = torch.tensor(labels, dtype=torch.long)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = nn.functional.cross_entropy(lg / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(np.clip(log_t.exp().item(), *bounds))


def _nll(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(-np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1)).mean())


def classification_report(probs: np.ndarray, labels: np.ndarray) -> dict:
    """Metrics over the classes present in ``labels`` (macro-F1 ignores absent classes)."""
    pred = probs.argmax(1)
    present = sorted(set(labels.tolist()))
    k = len(EMOTIONS)
    prec, rec, f1, support = precision_recall_fscore_support(
        labels, pred, labels=range(k), zero_division=0
    )
    top2 = np.argsort(-probs, axis=1)[:, :2]
    no_contempt = [c for c in present if EMOTIONS[c] != "contempt"]
    return {
        "n": len(labels),
        "accuracy": float((pred == labels).mean()),
        "top2_accuracy": float((top2 == labels[:, None]).any(1).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(labels, pred)),
        "macro_f1": float(f1_score(labels, pred, labels=present, average="macro", zero_division=0)),
        "macro_f1_without_contempt": float(
            f1_score(labels, pred, labels=no_contempt, average="macro", zero_division=0)
        )
        if no_contempt
        else None,
        "weighted_f1": float(f1_score(labels, pred, average="weighted", zero_division=0)),
        "per_class": {
            EMOTIONS[i]: {
                "precision": float(prec[i]),
                "recall": float(rec[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(k)
            if support[i] > 0 or (pred == i).any()
        },
        "confusion_matrix": confusion_matrix(labels, pred, labels=range(k)).tolist(),
        "ece": expected_calibration_error(probs, labels),
        "nll": _nll(probs, labels),
    }


def summarize(p: Predictions, temperature: float = 1.0) -> dict:
    probs = softmax(p.logits, temperature)
    out = classification_report(probs, p.labels)
    out["temperature"] = temperature
    out["by_source"] = {
        name: classification_report(probs[p.sources == i], p.labels[p.sources == i])
        for i, name in enumerate(SOURCES)
        if (p.sources == i).any()
    }
    return out


def bootstrap_ci(probs: np.ndarray, labels: np.ndarray, n_boot: int = 1000, seed: int = 0) -> dict:
    """95% percentile intervals (resampling test images) for accuracy, macro-F1, per-class F1.

    Per-class intervals matter here: the test set has ~30 contempt faces, so its F1 can move a
    lot with a handful of images.
    """
    rng = np.random.default_rng(seed)
    pred = probs.argmax(1)
    k = len(EMOTIONS)
    present = sorted(set(labels.tolist()))
    acc, macro, per = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(labels), len(labels))
        y, p = labels[idx], pred[idx]
        acc.append((y == p).mean())
        f1 = f1_score(y, p, labels=range(k), average=None, zero_division=0)
        macro.append(f1[present].mean())
        per.append(f1)
    per = np.array(per)

    def ci(v) -> list[float]:
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

    return {
        "n_boot": n_boot,
        "accuracy": ci(acc),
        "macro_f1": ci(macro),
        "per_class_f1": {EMOTIONS[c]: ci(per[:, c]) for c in present},
    }


def source_probe(embeddings: np.ndarray, sources: np.ndarray, seed: int = 0) -> dict:
    """How well a linear probe recovers the image source from the model's embedding.

    The two sources were normalised to the same 48 px grayscale format; if the source is still
    trivially recoverable, the model could use it as a shortcut (e.g. for ``contempt``, which
    only exists in one source).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
    )
    scores = cross_val_score(clf, embeddings, sources, cv=5, scoring="balanced_accuracy")
    return {"balanced_accuracy_mean": float(scores.mean()), "std": float(scores.std())}
