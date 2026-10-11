"""Figures for the evaluation report: data, curves, confusion, calibration, Grad-CAM."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import itertools

import matplotlib.pyplot as plt
import numpy as np
import torch

from moodtune.labels import EMOTIONS
from moodtune.preprocess import MEAN, STD, softmax

BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#b9b8b2"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SEQ = ["#ffffff", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281"]

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.bbox": "tight",
        "font.size": 9,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": GRID,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 2,
        "legend.frameon": False,
    }
)


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_dataset(records, path: Path) -> Path:
    """Files per class before and after de-duplication, split by source."""
    raw = Counter(r.label for r in records)
    kept = Counter((r.label, r.source) for r in records if r.status == "kept")
    y = np.arange(len(EMOTIONS))
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    ax.barh(y, [raw[c] for c in EMOTIONS], color=GRID, height=0.75, label="files on disk")
    k48 = np.array([kept[(c, "fer48")] for c in EMOTIONS])
    k224 = np.array([kept[(c, "hr224")] for c in EMOTIONS])
    ax.barh(y, k48, color=BLUE, height=0.45, label="kept, 48 px source")
    ax.barh(y, k224, left=k48, color=ORANGE, height=0.45, label="kept, 224 px source")
    for i, c in enumerate(EMOTIONS):
        ax.text(raw[c] + 150, i, f"{k48[i] + k224[i]:,} / {raw[c]:,}", va="center", fontsize=8)
    ax.set_yticks(y, EMOTIONS)
    ax.invert_yaxis()
    ax.set_xlabel("images")
    ax.set_xlim(0, max(raw.values()) * 1.18)
    ax.grid(axis="y", visible=False)
    ax.set_title("Unique images kept after removing duplicates", loc="left")
    ax.legend(loc="lower right")
    return _save(fig, path)


def plot_history(history: list[dict], path: Path) -> Path:
    ep = [h["epoch"] for h in history]
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.2))
    a.plot(ep, [h["train_loss"] for h in history], color=BLUE, label="train loss (weighted)")
    a.plot(ep, [h["val_nll"] for h in history], color=ORANGE, label="val NLL")
    a.set_xlabel("epoch")
    a.set_title("Loss", loc="left")
    a.legend()
    b.plot(ep, [h["val_macro_f1"] for h in history], color=BLUE, label="val macro-F1")
    b.plot(ep, [h["val_accuracy"] for h in history], color=ORANGE, label="val accuracy")
    b.plot(ep, [h["val_contempt_f1"] for h in history], color=AQUA, label="val contempt F1")
    b.set_xlabel("epoch")
    b.set_ylim(0, 1)
    b.set_title("Validation", loc="left")
    b.legend(loc="lower right")
    return _save(fig, path)


def plot_confusion(
    cm: Sequence[Sequence[int]], labels: Sequence[str], title: str, path: Path
) -> Path:
    from matplotlib.colors import LinearSegmentedColormap

    m = np.asarray(cm, dtype=float)
    norm = m / np.clip(m.sum(1, keepdims=True), 1, None)
    n = len(labels)
    fig, ax = plt.subplots(figsize=(0.8 * n + 1.6, 0.7 * n + 1.0))
    ax.imshow(norm, cmap=LinearSegmentedColormap.from_list("seq", SEQ), vmin=0, vmax=1)
    ax.grid(False)
    for i in range(n):
        for j in range(n):
            if m[i, j] == 0 and m[i].sum() == 0:
                continue
            ax.text(
                j,
                i,
                f"{norm[i, j]:.0%}\n{int(m[i, j])}",
                ha="center",
                va="center",
                color="white" if norm[i, j] > 0.6 else INK,
                fontsize=7,
            )
    ax.set_xticks(range(n), labels, rotation=35, ha="right")
    ax.set_yticks(range(n), [f"{lbl} ({int(m[i].sum())})" for i, lbl in enumerate(labels)])
    ax.set_xlabel("predicted")
    ax.set_ylabel("true (support)")
    ax.set_title(title, loc="left")
    return _save(fig, path)


def plot_reliability(
    logits: np.ndarray, labels: np.ndarray, temperature: float, path: Path, bins: int = 10
) -> Path:
    fig, ax = plt.subplots(figsize=(4.2, 4))
    ax.plot([0, 1], [0, 1], color=GRID, linestyle="--", linewidth=1)
    for t, colour, name in [
        (1.0, ORANGE, "uncalibrated"),
        (temperature, BLUE, f"temperature {temperature:.2f}"),
    ]:
        probs = softmax(logits, t)
        conf, correct = probs.max(1), probs.argmax(1) == labels
        xs, ys = [], []
        for lo, hi in itertools.pairwise(np.linspace(0, 1, bins + 1)):
            mask = (conf > lo) & (conf <= hi)
            if mask.sum() >= 10:
                xs.append(conf[mask].mean())
                ys.append(correct[mask].mean())
        ax.plot(xs, ys, marker="o", markersize=5, color=colour, label=name)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("confidence (top class)")
    ax.set_ylabel("accuracy")
    ax.set_title("Calibration (test)", loc="left")
    ax.legend(loc="lower right")
    return _save(fig, path)


def plot_f1_by_source(metrics: dict, path: Path) -> Path:
    """Per-class F1 on each image source: shows where a class score comes from."""
    sources = list(metrics["by_source"])
    colours = [BLUE, ORANGE]
    x = np.arange(len(EMOTIONS))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8, 3.2))
    for k, src in enumerate(sources):
        per = metrics["by_source"][src]["per_class"]
        vals = [
            per.get(c, {}).get("f1", np.nan) if per.get(c, {}).get("support") else np.nan
            for c in EMOTIONS
        ]
        ax.bar(x + (k - 0.5) * width, vals, width, color=colours[k], label=src)
    ax.set_xticks(x, EMOTIONS)
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1")
    ax.grid(axis="x", visible=False)
    ax.set_title(
        "Per-class F1 by image source (test; gaps = class absent from that source)", loc="left"
    )
    ax.legend(ncols=2, loc="upper right")
    return _save(fig, path)


# -- Grad-CAM -------------------------------------------------------------------------------
def gradcam(model, x: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Grad-CAM over the last feature map for the predicted class. Returns (cams, probs)."""
    model.eval()
    fmap = model.features(x)
    fmap.retain_grad()
    logits = model.logits_from(fmap)
    idx = logits.argmax(1)
    logits.gather(1, idx.view(-1, 1)).sum().backward()
    weights = fmap.grad.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * fmap).sum(1, keepdim=True))
    cam = torch.nn.functional.interpolate(
        cam, size=x.shape[-2:], mode="bilinear", align_corners=False
    ).squeeze(1)
    cam = cam / cam.flatten(1).amax(1).clamp_min(1e-8).view(-1, 1, 1)
    return cam.detach().numpy(), torch.softmax(logits.detach(), 1).numpy()


def plot_gradcam_grid(
    model, images: torch.Tensor, titles: Sequence[str], path: Path, cols: int = 8
) -> Path:
    cams, probs = gradcam(model, images.clone())
    gray = (images[:, 0].numpy() * STD[0] + MEAN[0]).clip(0, 1)
    rows = int(np.ceil(len(images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.45, rows * 1.85))
    for ax in np.atleast_1d(axes).flat:
        ax.axis("off")
    for i, ax in enumerate(np.atleast_1d(axes).flat[: len(images)]):
        ax.imshow(gray[i], cmap="gray", vmin=0, vmax=1)
        ax.imshow(cams[i], cmap="inferno", alpha=0.45)
        pred = EMOTIONS[int(probs[i].argmax())]
        ax.set_title(f"{titles[i]}\n→ {pred} {probs[i].max():.2f}", fontsize=7)
    fig.suptitle("Grad-CAM on held-out test faces (true label, then prediction)", x=0.01, ha="left")
    return _save(fig, path)
