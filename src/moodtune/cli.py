"""Command-line interface: ``moodtune <command> --help``."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from moodtune.config import Config

app = typer.Typer(
    add_completion=False, no_args_is_help=True, help="Facial-emotion music recommender."
)
console = Console()

ConfigOpt = Annotated[Path, typer.Option("--config", "-c", help="YAML config file.")]
OverrideOpt = Annotated[
    list[str] | None, typer.Option("--set", "-s", help="Override, e.g. train.epochs=5")
]
DEFAULT_CONFIG = Path("configs/default.yaml")


@app.callback()
def _setup(verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        datefmt="%H:%M:%S",
        handlers=[RichHandler(console=console, show_path=False)],
    )


def _load(config: Path, overrides: list[str] | None) -> Config:
    return Config.load(config if config.exists() else None, overrides)


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


@app.command()
def prepare(
    config: ConfigOpt = DEFAULT_CONFIG, overrides: OverrideOpt = None, workers: int = 6
) -> None:
    """Scan images, remove duplicates, cache canonical faces, write a stratified split."""
    from collections import Counter

    from moodtune.data.manifest import build_records, deduplicate, write_cache, write_manifest
    from moodtune.data.split import assign_splits, leakage_report
    from moodtune.labels import EMOTIONS
    from moodtune.report import plot_dataset

    cfg = _load(config, overrides)
    d = cfg.data
    t0 = time.perf_counter()
    records, faces = build_records(d.root, d.face_size, workers)
    console.print(f"decoded {len(records):,} files in {time.perf_counter() - t0:.0f}s")
    dedup = deduplicate(records, faces, d.min_std, d.near_dup_corr)
    write_cache(records, faces, d.cache_dir)
    assign_splits(records, d.val_fraction, d.test_fraction, d.seed)
    write_manifest(records, d.manifest)
    report = {"dedup": dedup, **leakage_report(records, faces, seed=d.seed)}
    _write_json(d.manifest.with_name("split_report.json"), report)
    plot_dataset(records, d.manifest.with_name("dataset.png"))

    table = Table(
        title=f"{len(records):,} files -> {dedup['status'].get('kept', 0):,} unique images"
    )
    for col in (
        "emotion",
        "files",
        "duplicates",
        "conflicts",
        "kept",
        "train",
        "val",
        "test",
        "224px kept",
    ):
        table.add_column(col, justify="left" if col == "emotion" else "right")
    for c in EMOTIONS:
        rs = [r for r in records if r.label == c]
        st = Counter(r.status for r in rs)
        sp = Counter(r.split for r in rs)
        table.add_row(
            c,
            f"{len(rs):,}",
            f"{st['duplicate'] + st['near_duplicate']:,}",
            f"{st['label_conflict']:,}",
            f"{st['kept']:,}",
            f"{sp['train']:,}",
            f"{sp['val']:,}",
            f"{sp['test']:,}",
            f"{sum(1 for r in rs if r.status == 'kept' and r.source == 'hr224'):,}",
        )
    console.print(table)
    console.print(
        f"status counts: {dedup['status']}  near-duplicate pairs: {dedup['near_duplicate_pairs']:,}"
    )
    for name, g in report["test_to_nearest_train_face"].items():
        console.print(
            f"{name:>20}: test images with an exact copy in train {g['share_exact_copy_in_train']:.1%}, "
            f"with a >=0.95-correlated (or mirrored) twin {g['share_corr_ge_0.95']:.1%}"
        )


def _records(cfg: Config):
    from moodtune.data.manifest import read_manifest

    if not cfg.data.manifest.exists():
        raise typer.BadParameter(f"{cfg.data.manifest} not found - run `moodtune prepare` first")
    return read_manifest(cfg.data.manifest)


@app.command()
def train(
    config: ConfigOpt = DEFAULT_CONFIG,
    overrides: OverrideOpt = None,
    name: Annotated[str, typer.Option(help="Run name under train.output_dir.")] = "main",
) -> None:
    """Train on the train split; select and calibrate on val."""
    from moodtune.report import plot_history
    from moodtune.train import train_model

    cfg = _load(config, overrides)
    records = _records(cfg)
    out = cfg.train.output_dir / name
    result = train_model(
        cfg,
        [r for r in records if r.split == "train"],
        [r for r in records if r.split == "val"],
        out,
    )
    plot_history(result["history"], out / "report" / "history.png")
    v = result["val"]
    console.print(
        f"[bold]best epoch {result['best_epoch']}[/]  val macro-F1 {v['macro_f1']:.4f}  "
        f"acc {v['accuracy']:.4f}  ECE {v['ece']:.4f}  T={result['temperature']:.2f}  -> {out / 'best.pt'}"
    )


@app.command()
def evaluate(
    run: Annotated[Path, typer.Argument(help="Run directory containing best.pt")] = Path(
        "runs/main"
    ),
    config: ConfigOpt = DEFAULT_CONFIG,
    overrides: OverrideOpt = None,
) -> None:
    """Score the held-out test split; write metrics, confusion, calibration and Grad-CAM figures."""
    import numpy as np
    import torch

    from moodtune.data.dataset import FaceDataset
    from moodtune.labels import EMOTIONS
    from moodtune.metrics import bootstrap_ci, predict, source_probe, summarize
    from moodtune.model import load_checkpoint
    from moodtune.preprocess import softmax
    from moodtune.report import (
        plot_confusion,
        plot_f1_by_source,
        plot_gradcam_grid,
        plot_reliability,
    )
    from moodtune.train import get_device, make_loader

    cfg = _load(config, overrides)
    model, _ = load_checkpoint(run / "best.pt")
    cfg.model = model.cfg
    test = [r for r in _records(cfg) if r.split == "test"]
    device = get_device()
    model.to(device)
    preds = predict(model, make_loader(test, cfg, train=False), device, embeddings=True)
    t = float(model.temperature)
    metrics = summarize(preds, t)
    metrics["uncalibrated_ece"] = summarize(preds, 1.0)["ece"]
    metrics["source_probe_on_embeddings"] = source_probe(preds.embeddings, preds.sources)
    metrics["bootstrap_95ci"] = bootstrap_ci(softmax(preds.logits, t), preds.labels)
    _write_json(run / "test_metrics.json", metrics)

    rep = run / "report"
    plot_confusion(
        metrics["confusion_matrix"],
        EMOTIONS,
        "Test confusion (row-normalised)",
        rep / "confusion.png",
    )
    for src, m in metrics["by_source"].items():
        plot_confusion(
            m["confusion_matrix"],
            EMOTIONS,
            f"Test confusion, {src} source",
            rep / f"confusion_{src}.png",
        )
    plot_reliability(preds.logits, preds.labels, t, rep / "calibration.png")
    plot_f1_by_source(metrics, rep / "f1_by_source.png")

    # Grad-CAM: two correct faces per class, then a row of mistakes.
    model.cpu().float()
    pred = preds.logits.argmax(1)
    rng = np.random.default_rng(0)
    picks: list[int] = []
    for c in range(len(EMOTIONS)):
        ok = np.flatnonzero((preds.labels == c) & (pred == c))
        picks += rng.choice(ok, size=min(2, len(ok)), replace=False).tolist()
    wrong = np.flatnonzero(pred != preds.labels)
    picks += rng.choice(wrong, size=min(8, len(wrong)), replace=False).tolist()
    ds = FaceDataset(
        [test[i] for i in picks], cfg.data.face_size, cfg.model.image_size, train=False
    )
    imgs = torch.stack([ds[i][0] for i in range(len(ds))])
    plot_gradcam_grid(model, imgs, [test[i].label for i in picks], rep / "gradcam.png")

    table = Table(
        title=f"Test n={metrics['n']:,}  ({model.cfg.backbone}, pretrained={model.cfg.pretrained})"
    )
    for col in ("", "accuracy", "macro-F1", "macro-F1 w/o contempt", "ECE"):
        table.add_column(col, justify="right")
    rows = [("all", metrics), *metrics["by_source"].items()]
    for name, m in rows:
        mfc = m["macro_f1_without_contempt"]
        table.add_row(
            name,
            f"{m['accuracy']:.4f}",
            f"{m['macro_f1']:.4f}",
            f"{mfc:.4f}" if mfc is not None else "-",
            f"{m['ece']:.4f}",
        )
    console.print(table)
    per = Table(title="Per class (test)")
    for col in ("emotion", "precision", "recall", "F1", "F1 95% CI", "support"):
        per.add_column(col, justify="right")
    cis = metrics["bootstrap_95ci"]["per_class_f1"]
    for c, v in metrics["per_class"].items():
        lo, hi = cis.get(c, [float("nan")] * 2)
        per.add_row(
            c,
            f"{v['precision']:.3f}",
            f"{v['recall']:.3f}",
            f"{v['f1']:.3f}",
            f"{lo:.2f}-{hi:.2f}",
            str(v["support"]),
        )
    console.print(per)
    probe = metrics["source_probe_on_embeddings"]
    console.print(
        f"uncalibrated ECE {metrics['uncalibrated_ece']:.4f} -> calibrated {metrics['ece']:.4f} (T={t:.2f}); "
        f"source recoverable from embeddings: balanced acc {probe['balanced_accuracy_mean']:.3f}"
    )


@app.command()
def export(
    run: Annotated[Path, typer.Argument()] = Path("runs/main"),
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("models/moodtune.onnx"),
    config: ConfigOpt = DEFAULT_CONFIG,
) -> None:
    """Export best.pt to ONNX (with metadata) and verify numerical parity on test faces."""
    import numpy as np
    import torch

    from moodtune.data.dataset import FaceDataset
    from moodtune.export import export_onnx, parity_check

    cfg = _load(config, None)
    meta = export_onnx(run / "best.pt", output, cfg.data.face_size)
    test = [r for r in _records(cfg) if r.split == "test"][:64]
    ds = FaceDataset(test, meta["face_size"], meta["image_size"], train=False)
    batch = torch.stack([ds[i][0] for i in range(len(ds))]).numpy().astype(np.float32)
    parity = parity_check(run / "best.pt", output, batch)
    console.print(
        f"exported {output} ({output.stat().st_size / 1e6:.1f} MB)  "
        f"max |torch - onnx| = {parity['max_abs_diff']:.2e}  top-1 agreement {parity['top1_agreement']:.0%}"
    )
    _write_json(run / "export_parity.json", {**parity, "onnx": str(output), "meta": meta})
    if parity["max_abs_diff"] > 1e-3 or parity["top1_agreement"] < 1.0:
        raise typer.Exit(code=1)


@app.command()
def predict(
    images: Annotated[list[Path], typer.Argument(help="Image files")],
    config: ConfigOpt = DEFAULT_CONFIG,
    detect: Annotated[
        bool, typer.Option(help="Detect faces; --no-detect treats each image as a face crop.")
    ] = True,
) -> None:
    """Classify images with the exported model, through the same pipeline as the app."""
    import cv2

    from moodtune.analyzer import Analyzer
    from moodtune.data.manifest import read_image
    from moodtune.face import FaceDetector
    from moodtune.predictor import OnnxPredictor

    cfg = _load(config, None)
    predictor = OnnxPredictor(cfg.app.model_path)
    analyzer = Analyzer(predictor, FaceDetector(cfg.app.detector), cfg.app.detector)
    table = Table()
    for col in ("image", "faces", "emotion", "confidence", "runner-up"):
        table.add_column(col)
    for path in images:
        img = read_image(path)
        if detect:
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            results = analyzer.analyze(img[:, :, :3])
            probs = results[0].probs if results else None
            n = str(len(results))
        else:
            probs, n = predictor.predict([img])[0], "-"
        if probs is None:
            table.add_row(path.name, n, "no face", "", "")
            continue
        order = probs.argsort()[::-1]
        lbl = predictor.labels
        table.add_row(
            path.name,
            n,
            lbl[order[0]],
            f"{probs[order[0]]:.3f}",
            f"{lbl[order[1]]} {probs[order[1]]:.3f}",
        )
    console.print(table)


@app.command()
def bench(config: ConfigOpt = DEFAULT_CONFIG, iters: int = 200) -> None:
    """Latency of face detection (640x480 frame) and emotion inference (one face crop)."""
    import numpy as np

    from moodtune.face import FaceDetector
    from moodtune.predictor import OnnxPredictor

    cfg = _load(config, None)
    predictor = OnnxPredictor(cfg.app.model_path)
    detector = FaceDetector(cfg.app.detector)
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
    crop = rng.integers(0, 255, (96, 96), dtype=np.uint8)

    def timeit(fn) -> tuple[float, float]:
        for _ in range(10):
            fn()
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter()
            fn()
            ts.append((time.perf_counter() - t0) * 1000)
        ts.sort()
        return ts[len(ts) // 2], ts[int(len(ts) * 0.95)]

    d50, d95 = timeit(lambda: detector.detect(frame))
    p50, p95 = timeit(lambda: predictor.predict([crop]))
    console.print(
        f"detector={detector.kind}: p50 {d50:.2f} ms  p95 {d95:.2f} ms | "
        f"emotion model ({predictor.session.get_providers()[0]}): p50 {p50:.2f} ms  p95 {p95:.2f} ms"
    )


@app.command()
def serve(
    config: ConfigOpt = DEFAULT_CONFIG,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    """Start the web app (http://HOST:PORT). Browsers allow the camera on localhost or HTTPS."""
    import uvicorn

    from moodtune.api import create_app

    uvicorn.run(create_app(_load(config, None)), host=host, port=port, log_level="info")


if __name__ == "__main__":
    app()
