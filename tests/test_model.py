from __future__ import annotations

import numpy as np
import pytest
import torch

from moodtune.config import Config, ModelConfig
from moodtune.data.manifest import build_records, deduplicate, write_cache
from moodtune.data.split import assign_splits
from moodtune.export import export_onnx, parity_check
from moodtune.labels import EMOTIONS
from moodtune.model import EmotionNet, load_checkpoint, save_checkpoint
from moodtune.predictor import OnnxPredictor

LEGACY = ModelConfig(backbone="legacy_cnn", pretrained=False, image_size=48)
TINY = ModelConfig(backbone="mobilenetv4_conv_small", pretrained=False, image_size=64)


@pytest.mark.parametrize("cfg", [LEGACY, TINY], ids=["legacy", "timm"])
def test_forward_and_gradcam_hooks(cfg):
    model = EmotionNet(cfg).eval()
    x = torch.randn(2, 3, cfg.image_size, cfg.image_size)
    fmap = model.features(x)
    assert fmap.ndim == 4
    assert model(x).shape == (2, len(EMOTIONS))
    torch.testing.assert_close(model.logits_from(fmap), model(x))


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    torch.manual_seed(0)
    model = EmotionNet(LEGACY).eval()
    model.temperature.fill_(1.7)
    path = tmp_path_factory.mktemp("ckpt") / "best.pt"
    save_checkpoint(model, path, epoch=3)
    return path


def test_checkpoint_roundtrip(checkpoint):
    model, ckpt = load_checkpoint(checkpoint)
    assert ckpt["epoch"] == 3 and ckpt["labels"] == list(EMOTIONS)
    assert float(model.temperature) == pytest.approx(1.7)


def test_onnx_export_parity_and_predictor(checkpoint, tmp_path):
    out = tmp_path / "m.onnx"
    meta = export_onnx(checkpoint, out, face_size=48)
    assert meta["labels"] == list(EMOTIONS) and meta["image_size"] == 48
    batch = np.random.default_rng(0).normal(size=(4, 3, 48, 48)).astype(np.float32)
    parity = parity_check(checkpoint, out, batch)
    assert parity["max_abs_diff"] < 1e-4 and parity["top1_agreement"] == 1.0

    pred = OnnxPredictor(out, providers=["CPUExecutionProvider"])
    assert pred.temperature == pytest.approx(1.7) and pred.labels == list(EMOTIONS)
    rng = np.random.default_rng(1)
    crops = [
        rng.integers(0, 255, (96, 96), dtype=np.uint8),
        rng.integers(0, 255, (120, 90, 3), dtype=np.uint8),
    ]
    probs = pred.predict(crops)
    assert probs.shape == (2, len(EMOTIONS))
    np.testing.assert_allclose(probs.sum(1), 1.0, rtol=1e-5)
    assert pred.predict([]).shape == (0, len(EMOTIONS))


def test_training_smoke(dataset, tmp_path):
    """One epoch of the full loop (legacy CNN, CPU): checkpoint, calibration, metrics."""
    from moodtune.train import train_model

    records, faces = build_records(dataset, face_size=48, workers=1)
    deduplicate(records, faces, 3.0, 0.95)
    write_cache(records, faces, tmp_path / "cache")
    assign_splits(records, 0.2, 0.25, seed=0)
    cfg = Config.load(
        None,
        [
            "model.backbone=legacy_cnn",
            "model.pretrained=false",
            "model.image_size=48",
            "train.epochs=1",
            "train.batch_size=16",
            "train.num_workers=0",
            "train.amp=false",
        ],
    )
    result = train_model(
        cfg,
        [r for r in records if r.split == "train"],
        [r for r in records if r.split == "val"],
        tmp_path / "run",
    )
    assert (tmp_path / "run" / "best.pt").exists()
    assert 0.5 <= result["temperature"] <= 5.0
    v = result["val"]
    assert 0.0 <= v["macro_f1"] <= 1.0 and set(v["by_source"]) <= {"fer48", "hr224"}
    assert len(v["confusion_matrix"]) == len(EMOTIONS)
