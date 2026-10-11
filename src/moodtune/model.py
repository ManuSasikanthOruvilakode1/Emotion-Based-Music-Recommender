"""Emotion classifier: a timm backbone, or a port of the original Keras CNN as a baseline."""

from __future__ import annotations

from pathlib import Path

import timm
import torch
from torch import nn

from moodtune.config import ModelConfig
from moodtune.labels import EMOTIONS


class LegacyCNN(nn.Module):
    """The network from the original ``train_test.py`` (Keras), on a 48 x 48 grayscale input."""

    def __init__(self, drop_rate: float = 0.5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout(0.25),
            nn.Conv2d(64, 128, 3), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3), nn.ReLU(inplace=True),
        )  # fmt: skip
        self.pool = nn.Sequential(nn.MaxPool2d(2), nn.Dropout(0.25), nn.Flatten())
        self.fc = nn.Sequential(nn.LazyLinear(1024), nn.ReLU(inplace=True), nn.Dropout(drop_rate))
        self.num_features = 1024

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # The preprocessed input is gray replicated to 3 normalised channels; undo to 1 channel.
        return self.conv(x.mean(1, keepdim=True))

    def forward_head(self, fmap: torch.Tensor, pre_logits: bool = True) -> torch.Tensor:
        return self.fc(self.pool(fmap))


class EmotionNet(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.backbone == "legacy_cnn":
            self.backbone = LegacyCNN(cfg.drop_rate)
            self.backbone.forward_head(
                self.backbone.forward_features(torch.zeros(1, 3, cfg.image_size, cfg.image_size))
            )  # materialise the lazy layer
            dim, drop = 1024, 0.0
        else:
            self.backbone = timm.create_model(
                cfg.backbone,
                pretrained=cfg.pretrained,
                num_classes=0,
                drop_path_rate=cfg.drop_path_rate,
            )
            dim = getattr(self.backbone, "head_hidden_size", None) or self.backbone.num_features
            drop = cfg.drop_rate
        self.dropout = nn.Dropout(drop)
        self.head = nn.Linear(dim, len(EMOTIONS))
        # Post-hoc calibration (temperature scaling), fitted on the validation split.
        self.register_buffer("temperature", torch.ones(()))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Last spatial feature map (B, C, H, W): the Grad-CAM target."""
        return self.backbone.forward_features(x)

    def embed(self, fmap: torch.Tensor) -> torch.Tensor:
        return self.backbone.forward_head(fmap, pre_logits=True)

    def logits_from(self, fmap: torch.Tensor) -> torch.Tensor:
        return self.head(self.dropout(self.embed(fmap)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.logits_from(self.features(x))

    def param_groups(self, lr: float, backbone_lr_mult: float, weight_decay: float) -> list[dict]:
        """Lower LR for a pretrained backbone; no weight decay on norms and biases."""
        groups: dict[tuple[bool, bool], list[nn.Parameter]] = {}
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            groups.setdefault((name.startswith("backbone."), p.ndim <= 1), []).append(p)
        mult = (
            backbone_lr_mult if self.cfg.pretrained and self.cfg.backbone != "legacy_cnn" else 1.0
        )
        return [
            {
                "params": params,
                "lr": lr * (mult if is_backbone else 1.0),
                "weight_decay": 0.0 if no_decay else weight_decay,
            }
            for (is_backbone, no_decay), params in groups.items()
        ]


def save_checkpoint(model: EmotionNet, path: Path, **extra) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_cfg": model.cfg.model_dump(mode="json"),
            "labels": list(EMOTIONS),
            "state_dict": model.state_dict(),
            **extra,
        },
        path,
    )


def load_checkpoint(
    path: Path, map_location: str | torch.device = "cpu"
) -> tuple[EmotionNet, dict]:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    if ckpt.get("labels", list(EMOTIONS)) != list(EMOTIONS):
        raise ValueError(f"{path} was trained on labels {ckpt['labels']}, expected {EMOTIONS}")
    cfg = ModelConfig.model_validate({**ckpt["model_cfg"], "pretrained": False})
    model = EmotionNet(cfg)
    model.load_state_dict(ckpt["state_dict"])
    # Keep the original flag for reporting (weights are already loaded).
    model.cfg = ModelConfig.model_validate(ckpt["model_cfg"])
    model.eval()
    return model, ckpt
