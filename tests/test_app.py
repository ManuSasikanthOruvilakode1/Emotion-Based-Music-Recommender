from __future__ import annotations

import math
from pathlib import Path
from typing import ClassVar

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from moodtune.analyzer import FaceResult
from moodtune.api import create_app
from moodtune.config import Config, DetectorConfig, SmoothingConfig
from moodtune.face import Face, FaceDetector, crop_face
from moodtune.labels import EMOTIONS
from moodtune.music import Recommender
from moodtune.smoothing import EmotionSmoother

LABELS = list(EMOTIONS)


def onehot(label: str, p: float = 0.9) -> np.ndarray:
    v = np.full(len(LABELS), (1 - p) / (len(LABELS) - 1))
    v[LABELS.index(label)] = p
    return v


# -- face detection and cropping ---------------------------------------------------------------
@pytest.mark.parametrize("yunet", [True, False], ids=["yunet", "haar"])
def test_detector_on_blank_frame(yunet, tmp_path):
    path = Path("models/face_detection_yunet_2023mar.onnx") if yunet else tmp_path / "missing.onnx"
    if yunet and not path.exists():
        pytest.skip("YuNet model not downloaded")
    if not yunet and not hasattr(cv2, "CascadeClassifier"):
        with pytest.raises(FileNotFoundError, match="model zoo"):
            FaceDetector(DetectorConfig(yunet_path=path))
        return
    det = FaceDetector(DetectorConfig(yunet_path=path))
    assert det.kind == ("yunet" if yunet else "haar")
    assert det.detect(np.zeros((240, 320, 3), np.uint8)) == []


def test_crop_aligns_eyes():
    img = np.zeros((200, 200), np.uint8)
    right, left = (80.0, 90.0), (120.0, 110.0)  # tilted ~26.6 degrees
    for x, y in (right, left):
        cv2.circle(img, (int(x), int(y)), 4, 255, -1)
    face = Face(box=(60, 60, 80, 80), score=1.0, eyes=(right, left))
    crop = crop_face(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), face, margin=0.2, align=True)
    assert crop.shape == (96, 96)
    _, _, _, centroids = cv2.connectedComponentsWithStats((crop > 100).astype(np.uint8))
    (x1, y1), (x2, y2) = sorted(centroids[1:].tolist())
    assert abs(y1 - y2) < 1.5 and x2 - x1 > 20  # eyes now level

    unaligned = crop_face(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), face, margin=0.2, align=False)
    _, _, _, c2 = cv2.connectedComponentsWithStats((unaligned > 100).astype(np.uint8))
    (_, ya), (_, yb) = sorted(c2[1:].tolist())
    assert abs(ya - yb) > 10


# -- smoothing ---------------------------------------------------------------------------------
def test_smoother_hysteresis_dwell_and_reset():
    cfg = SmoothingConfig(ema_alpha=0.5, switch_margin=0.1, min_dwell_s=1.0, reset_after_s=2.0)
    s = EmotionSmoother(cfg, LABELS)
    assert s.update(None, 0.0)["emotion"] is None
    assert s.update(onehot("happy"), 0.0)["emotion"] == "happy"
    # one sad frame: the EMA moves, but the switch needs a lead held for min_dwell_s
    assert s.update(onehot("sad"), 0.2)["emotion"] == "happy"
    assert s.update(onehot("sad"), 0.4)["emotion"] == "happy"
    st = s.update(onehot("sad"), 1.0)
    assert st["emotion"] == "happy" and st["pending"] == "sad"
    st = s.update(onehot("sad"), 1.5)
    assert st["emotion"] == "sad"
    assert math.isclose(sum(st["probs"].values()), 1.0, rel_tol=1e-3)
    # brief face loss keeps the state; long loss resets it
    assert s.update(None, 2.5)["emotion"] == "sad"
    assert s.update(None, 4.0)["emotion"] is None


def test_smoother_ignores_small_leads():
    s = EmotionSmoother(SmoothingConfig(ema_alpha=1.0, switch_margin=0.2, min_dwell_s=0.0), LABELS)
    s.update(onehot("neutral", 0.5), 0.0)
    near_tie = onehot("neutral", 0.4)
    near_tie[LABELS.index("happy")] = 0.5
    assert s.update(near_tie, 1.0)["emotion"] == "neutral"


# -- music -------------------------------------------------------------------------------------
def test_recommender_strategies():
    r = Recommender(Config.load("configs/default.yaml").music)
    match = r.recommend("sad", 0.9, "match")
    lift = r.recommend("sad", 0.9, "lift")
    assert match["query"] != lift["query"] and lift["applied"] == "lift"
    assert match["url"].startswith("https://www.youtube.com/results?search_query=")
    assert " " not in match["url"]
    happy = r.recommend("happy", 0.9, "lift")
    assert happy["applied"] == "match" and happy["strategy"] == "lift"
    unsure = r.recommend("angry", 0.2, "match")
    assert unsure["uncertain"] and unsure["query"] == r.recommend("neutral", 0.9)["query"]
    assert r.recommend(None, None)["uncertain"]
    assert len(match["alternatives"]) >= 1


# -- API ---------------------------------------------------------------------------------------
class StubPredictor:
    labels = LABELS
    meta: ClassVar[dict] = {"backbone": "stub", "labels": LABELS}


class StubDetector:
    kind = "stub"


class StubAnalyzer:
    """Bright frame -> one happy face; dark frame -> no face."""

    predictor = StubPredictor()
    detector = StubDetector()

    def analyze(self, bgr, max_faces=4):
        if bgr.mean() < 100:
            return []
        return [FaceResult((10, 10, 40, 40), 0.99, onehot("happy"), LABELS)]


def jpeg(value: int) -> bytes:
    ok, buf = cv2.imencode(".jpg", np.full((120, 160, 3), value, np.uint8))
    assert ok
    return buf.tobytes()


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # anything written to the working directory would show up here
    cfg = Config.load(Path(__file__).parents[1] / "configs" / "default.yaml")
    with TestClient(create_app(cfg, analyzer=StubAnalyzer())) as c:
        yield c
    assert list(tmp_path.iterdir()) == []


def test_health_meta_and_page(client):
    assert client.get("/api/health").json()["detector"] == "stub"
    meta = client.get("/api/meta").json()
    assert meta["labels"] == LABELS and meta["default_strategy"] == "match"
    page = client.get("/")
    assert page.status_code == 200 and "MoodTune" in page.text
    assert "cdn" not in page.text.lower()
    assert client.get("/static/app.js").status_code == 200


def test_analyze_upload(client):
    r = client.post(
        "/api/analyze?strategy=lift", files={"file": ("x.jpg", jpeg(255), "image/jpeg")}
    )
    body = r.json()
    assert r.status_code == 200 and body["faces"][0]["emotion"] == "happy"
    assert body["frame"] == {"width": 160, "height": 120}
    assert body["recommendation"]["strategy"] == "lift"

    none = client.post("/api/analyze", files={"file": ("x.jpg", jpeg(0), "image/jpeg")}).json()
    assert none["faces"] == [] and none["recommendation"]["uncertain"]

    bad = client.post("/api/analyze", files={"file": ("x.jpg", b"not an image", "image/jpeg")})
    assert bad.status_code == 400


def test_recommend_endpoint(client):
    r = client.get(
        "/api/recommend", params={"emotion": "sad", "confidence": 0.8, "strategy": "lift"}
    )
    assert r.status_code == 200 and r.json()["applied"] == "lift"
    assert client.get("/api/recommend", params={"emotion": "bored"}).status_code == 422
    assert client.get("/api/recommend", params={"strategy": "shuffle"}).status_code == 422


def test_websocket_stream_smooths_and_switches_strategy(client):
    with client.websocket_connect("/api/stream") as ws:
        ws.send_bytes(jpeg(0))
        first = ws.receive_json()
        assert first["faces"] == [] and first["smoothed"]["emotion"] is None
        ws.send_bytes(jpeg(255))
        msg = ws.receive_json()
        assert msg["smoothed"]["emotion"] == "happy"
        assert msg["recommendation"]["strategy"] == "match"
        ws.send_text('{"strategy": "lift"}')
        ws.send_bytes(jpeg(255))
        assert ws.receive_json()["recommendation"]["strategy"] == "lift"
        ws.send_bytes(b"garbage")
        assert "error" in ws.receive_json()
