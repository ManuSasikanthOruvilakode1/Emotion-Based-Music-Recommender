"""Training loop."""

from __future__ import annotations

import json
import logging
import math
import random
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from timm.utils import ModelEmaV3
from torch import nn
from torch.utils.data import DataLoader

from moodtune.config import Config
from moodtune.data.dataset import FaceDataset, class_weights
from moodtune.data.manifest import Record
from moodtune.labels import EMOTIONS
from moodtune.metrics import fit_temperature, predict, summarize
from moodtune.model import EmotionNet, save_checkpoint

log = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_loader(records: Sequence[Record], cfg: Config, train: bool, seed: int = 0) -> DataLoader:
    ds = FaceDataset(records, cfg.data.face_size, cfg.model.image_size, train=train)
    workers = cfg.train.num_workers
    return DataLoader(
        ds,
        batch_size=cfg.train.batch_size,
        shuffle=train,
        generator=torch.Generator().manual_seed(seed) if train else None,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        drop_last=train and len(ds) > cfg.train.batch_size,
    )


def _lr_lambda(warmup_steps: int, total_steps: int):
    def fn(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    return fn


def _score(metrics: dict) -> tuple[float, float]:
    """Model selection: macro-F1 (every emotion counts equally), then lower NLL."""
    return metrics["macro_f1"], -metrics["nll"]


def train_model(
    cfg: Config, train_records: Sequence[Record], val_records: Sequence[Record], out_dir: Path
) -> dict:
    tc = cfg.train
    seed_everything(tc.seed)
    device = get_device()
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.dump(out_dir / "config.yaml")

    train_loader = make_loader(train_records, cfg, train=True, seed=tc.seed)
    val_loader = make_loader(val_records, cfg, train=False)

    model = EmotionNet(cfg.model).to(device).to(memory_format=torch.channels_last)
    ema = ModelEmaV3(model, decay=tc.ema_decay, use_warmup=True)
    opt = torch.optim.AdamW(model.param_groups(tc.lr, tc.backbone_lr_mult, tc.weight_decay))
    steps_per_epoch = len(train_loader)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, _lr_lambda(tc.warmup_epochs * steps_per_epoch, tc.epochs * steps_per_epoch)
    )
    weights = class_weights(train_records, tc.class_weighting)
    log.info("class weights: %s", dict(zip(EMOTIONS, weights.tolist(), strict=True)))
    criterion = nn.CrossEntropyLoss(weight=weights.to(device), label_smoothing=tc.label_smoothing)
    use_amp = tc.amp and device.type == "cuda"

    history: list[dict] = []
    best_score, best_epoch, bad_epochs = (-1.0, -math.inf), -1, 0
    ckpt_path = out_dir / "best.pt"
    step = 0
    for epoch in range(1, tc.epochs + 1):
        model.train()
        t0, total, n = time.perf_counter(), 0.0, 0
        for x, y, _source in train_loader:
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                loss = criterion(model(x).float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            sched.step()
            step += 1
            ema.update(model, step=step)
            total += loss.item() * len(x)
            n += len(x)

        val = summarize(predict(ema.module, val_loader, device, use_amp))
        row = {
            "epoch": epoch,
            "train_loss": total / max(n, 1),
            "val_macro_f1": val["macro_f1"],
            "val_accuracy": val["accuracy"],
            "val_nll": val["nll"],
            "val_contempt_f1": val["per_class"].get("contempt", {}).get("f1", float("nan")),
            "lr": opt.param_groups[-1]["lr"],
            "seconds": time.perf_counter() - t0,
        }
        history.append(row)
        log.info(
            "epoch %02d  loss %.4f  val macro-F1 %.4f  acc %.4f  nll %.4f  contempt-F1 %.3f  (%.1fs)",
            epoch,
            row["train_loss"],
            row["val_macro_f1"],
            row["val_accuracy"],
            row["val_nll"],
            row["val_contempt_f1"],
            row["seconds"],
        )
        score = _score(val)
        if score > best_score:
            best_score, best_epoch, bad_epochs = score, epoch, 0
            save_checkpoint(ema.module, ckpt_path, epoch=epoch)
        else:
            bad_epochs += 1
            if bad_epochs >= tc.patience:
                log.info("early stopping at epoch %d (best %d)", epoch, best_epoch)
                break

    (out_dir / "history.json").write_text(json.dumps(history, indent=2))

    # Reload the best EMA weights, then calibrate on validation.
    best = EmotionNet(cfg.model.model_copy(update={"pretrained": False})).to(device)
    best.load_state_dict(
        torch.load(ckpt_path, map_location=device, weights_only=False)["state_dict"]
    )
    best.cfg = cfg.model
    val_pred = predict(best, val_loader, device, use_amp)
    temperature = fit_temperature(val_pred.logits, val_pred.labels)
    best.temperature.fill_(temperature)
    val_metrics = summarize(val_pred, temperature)
    save_checkpoint(best, ckpt_path, epoch=best_epoch, val_metrics=val_metrics)
    (out_dir / "val_metrics.json").write_text(json.dumps(val_metrics, indent=2))
    return {
        "best_epoch": best_epoch,
        "temperature": temperature,
        "val": val_metrics,
        "history": history,
    }
