from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
import torch

from moodtune.config import Config, MusicConfig
from moodtune.data.dataset import FaceDataset, class_weights
from moodtune.data.manifest import (
    build_records,
    deduplicate,
    load_face,
    read_manifest,
    write_cache,
    write_manifest,
)
from moodtune.data.split import assign_splits, leakage_report
from moodtune.labels import EMOTIONS, parse_folder
from moodtune.preprocess import canonical, prepare


@pytest.fixture(scope="module")
def prepared(dataset, tmp_path_factory):
    records, faces = build_records(dataset, face_size=48, workers=1)
    summary = deduplicate(records, faces, min_std=3.0, min_corr=0.95)
    write_cache(records, faces, tmp_path_factory.mktemp("cache"))
    return records, faces, summary


@pytest.mark.parametrize(
    ("folder", "expected"),
    [("angry", "angry"), ("anger", "angry"), ("suprise", "surprise"), ("Contempt", "contempt")],
)
def test_parse_folder(folder, expected):
    assert parse_folder(folder) == expected


def test_parse_folder_rejects_unknown():
    with pytest.raises(ValueError):
        parse_folder("bored")


def test_sources_and_canonical_size(prepared):
    records, faces, _ = prepared
    assert faces.shape[1:] == (48, 48)
    assert {r.source for r in records if r.label == "contempt"} == {"hr224"}
    assert {r.source for r in records if r.width == 48} == {"fer48"}


def test_dedup_removes_planted_copies(prepared):
    records, _, summary = prepared
    status = Counter(r.status for r in records)
    non_contempt = len(EMOTIONS) - 1
    assert status["duplicate"] == non_contempt
    assert status["near_duplicate"] == 2 * non_contempt  # mirrored + re-encoded
    assert status["label_conflict"] == 2
    assert status["blank"] == 1
    assert status["kept"] == non_contempt * 15 + 6
    kept = [r for r in records if r.status == "kept"]
    assert len({r.sha for r in kept}) == len(kept)
    paths = {r.path for r in kept}
    assert all(r.dup_of in paths for r in records if r.status in {"duplicate", "near_duplicate"})
    assert summary["status"]["kept"] == len(kept)


def test_split_is_stratified_deterministic_and_disjoint(prepared):
    records, faces, _ = prepared
    assign_splits(records, val_fraction=0.2, test_fraction=0.25, seed=0)
    first = [r.split for r in records]
    assign_splits(records, val_fraction=0.2, test_fraction=0.25, seed=0)
    assert [r.split for r in records] == first
    assert {r.split for r in records if r.status == "kept"} == {"train", "val", "test"}
    assert all(r.split == "" for r in records if r.status != "kept")
    for split in ("train", "test"):
        assert {r.label for r in records if r.split == split} == set(EMOTIONS)

    report = leakage_report(records, faces, seed=0)
    ours = report["test_to_nearest_train_face"]["dedup_stratified"]
    assert ours["share_exact_copy_in_train"] == 0.0
    assert ours["share_corr_ge_0.95"] == 0.0


def test_manifest_roundtrip_and_cache(prepared, tmp_path):
    records, _, _ = prepared
    write_manifest(records, tmp_path / "m.csv")
    back = read_manifest(tmp_path / "m.csv")
    assert back == records
    kept = next(r for r in back if r.status == "kept")
    assert load_face(kept).shape == (48, 48)


def test_dataset_shapes(prepared):
    records, _, _ = prepared
    kept = [r for r in records if r.status == "kept"][:4]
    for train in (True, False):
        x, y, s = FaceDataset(kept, 48, 64, train=train)[0]
        assert x.shape == (3, 64, 64) and x.dtype == torch.float32
        assert 0 <= y < len(EMOTIONS) and s in (0, 1)


def test_eval_preprocessing_matches_for_any_input_size():
    rng = np.random.default_rng(0)
    big = rng.integers(0, 255, (224, 224), dtype=np.uint8)
    small = canonical(big, 48)
    # a 224 px face and its canonical 48 px version give the same network input
    np.testing.assert_allclose(prepare(big, 48, 96), prepare(small, 48, 96))
    bgr = np.repeat(big[:, :, None], 3, axis=2)
    np.testing.assert_allclose(prepare(bgr, 48, 96), prepare(big, 48, 96), atol=1e-6)


def test_class_weights():
    from moodtune.data.manifest import Record

    recs = [Record("", "happy", "fer48", 48, 48, "", "", 10.0)] * 90 + [
        Record("", "contempt", "hr224", 224, 224, "", "", 10.0)
    ] * 10
    n = torch.tensor([Counter(r.label for r in recs)[c] for c in EMOTIONS], dtype=torch.float)
    for mode in ("none", "inverse", "inverse_sqrt"):
        w = class_weights(recs, mode)
        assert float((w * n).sum()) == pytest.approx(len(recs))
    w = class_weights(recs, "inverse_sqrt")
    c, h = EMOTIONS.index("contempt"), EMOTIONS.index("happy")
    assert float(w[c] / w[h]) == pytest.approx(3.0)
    assert float(w[EMOTIONS.index("sad")]) == 0.0  # absent classes get no weight


def test_config_loads_and_validates(tmp_path):
    cfg = Config.load("configs/default.yaml", ["train.epochs=3", "model.backbone=legacy_cnn"])
    assert cfg.train.epochs == 3 and cfg.model.backbone == "legacy_cnn"
    assert set(cfg.music.queries) == set(EMOTIONS)
    cfg.dump(tmp_path / "c.yaml")
    assert Config.load(tmp_path / "c.yaml") == cfg
    with pytest.raises(ValueError):
        MusicConfig(queries={"bored": {"match": ["x"]}})
    with pytest.raises(ValueError):
        MusicConfig(queries={"sad": {"lift": ["x"]}})  # needs a match list
