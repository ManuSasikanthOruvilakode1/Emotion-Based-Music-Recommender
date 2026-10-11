"""Typed configuration loaded from YAML.

Every pipeline stage reads one ``Config`` object so that a run is fully described by
a single file that is also copied next to its artifacts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from moodtune.labels import EMOTIONS

Strategy = Literal["match", "lift"]


class DataConfig(BaseModel):
    root: Path = Path("data")
    cache_dir: Path = Path(".cache/faces")
    manifest: Path = Path("artifacts/manifest.csv")
    # Every image, whatever its source, is reduced to this canonical grayscale size before
    # training, evaluation and live inference, so image size and format cannot identify the source.
    face_size: int = 48
    # Two canonical faces (or one and the other's mirror image) with pixel correlation at or
    # above this are treated as copies of one image.
    near_dup_corr: float = 0.95
    # Images with pixel standard deviation below this are blank or corrupt and are dropped.
    min_std: float = 3.0
    val_fraction: float = 0.1
    test_fraction: float = 0.15
    seed: int = 42


class ModelConfig(BaseModel):
    # A timm model name, or "legacy_cnn" for a PyTorch port of the original Keras network.
    backbone: str = "convnext_nano.in12k_ft_in1k"
    pretrained: bool = True
    drop_rate: float = 0.3
    drop_path_rate: float = 0.1
    # Square network input; the canonical face is upsampled to this size.
    image_size: int = 160


class TrainConfig(BaseModel):
    epochs: int = 40
    batch_size: int = 128
    lr: float = 1e-3
    backbone_lr_mult: float = 0.1
    weight_decay: float = 0.05
    warmup_epochs: int = 2
    label_smoothing: float = 0.1
    # Per-class loss weights against the 90:1 happy:contempt imbalance.
    class_weighting: Literal["none", "inverse", "inverse_sqrt"] = "inverse_sqrt"
    ema_decay: float = 0.999
    patience: int = 8
    amp: bool = True
    num_workers: int = 4
    seed: int = 42
    output_dir: Path = Path("runs")


class DetectorConfig(BaseModel):
    # OpenCV YuNet ONNX file; if missing, the bundled Haar cascade is used instead.
    yunet_path: Path = Path("models/face_detection_yunet_2023mar.onnx")
    score_threshold: float = 0.7
    min_face_px: int = 40
    # Square crop around the detected box, enlarged by this fraction to match the dataset framing.
    margin: float = 0.1
    # Rotate so the eyes are level (YuNet only; Haar gives no landmarks).
    align: bool = True


class SmoothingConfig(BaseModel):
    """Turns noisy per-frame probabilities into a stable emotion for the webcam stream."""

    ema_alpha: float = 0.3
    # A new emotion must lead the current one by this much (smoothed probability) ...
    switch_margin: float = 0.1
    # ... for this long before it replaces the current one.
    min_dwell_s: float = 1.0
    # Forget the state when no face has been seen for this long.
    reset_after_s: float = 2.0


class MusicConfig(BaseModel):
    """YouTube search queries per emotion.

    ``match`` plays to the detected mood; ``lift`` steers negative moods towards calmer or
    happier music (and falls back to ``match`` for emotions without a ``lift`` entry).
    """

    default_strategy: Strategy = "match"
    search_url: str = "https://www.youtube.com/results?search_query={query}"
    # Below this (calibrated) confidence the app says it is unsure and suggests neutral music.
    min_confidence: float = 0.4
    queries: dict[str, dict[str, list[str]]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> MusicConfig:
        unknown = set(self.queries) - set(EMOTIONS)
        if unknown:
            raise ValueError(f"music.queries has unknown emotions: {sorted(unknown)}")
        for emotion, by_strategy in self.queries.items():
            bad = set(by_strategy) - {"match", "lift"}
            if bad:
                raise ValueError(f"music.queries.{emotion}: unknown strategies {sorted(bad)}")
            if not by_strategy.get("match"):
                raise ValueError(f"music.queries.{emotion} needs a non-empty 'match' list")
        return self


class AppConfig(BaseModel):
    model_path: Path = Path("models/moodtune.onnx")
    max_upload_mb: float = 10.0
    detector: DetectorConfig = Field(default_factory=DetectorConfig)
    smoothing: SmoothingConfig = Field(default_factory=SmoothingConfig)


class Config(BaseModel):
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    app: AppConfig = Field(default_factory=AppConfig)
    music: MusicConfig = Field(default_factory=MusicConfig)

    @classmethod
    def load(cls, path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
        raw: dict = {}
        if path is not None:
            with open(path, encoding="utf-8") as fh:
                raw = yaml.safe_load(fh) or {}
        for item in overrides or []:
            _apply_override(raw, item)
        return cls.model_validate(raw)

    def dump(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self.model_dump(mode="json"), fh, sort_keys=False)


def _apply_override(raw: dict, item: str) -> None:
    """Apply a ``section.key=value`` override, parsing the value as YAML."""
    if "=" not in item:
        raise ValueError(f"override must look like section.key=value, got {item!r}")
    dotted, value = item.split("=", 1)
    node = raw
    *parents, leaf = dotted.split(".")
    for key in parents:
        node = node.setdefault(key, {})
    node[leaf] = yaml.safe_load(value)
