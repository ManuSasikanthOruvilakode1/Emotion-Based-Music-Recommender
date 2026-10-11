"""HTTP + WebSocket API and the browser app.

The camera runs in the user's browser (the old app opened the webcam of the machine running
the server). Frames arrive over a WebSocket, are analysed in memory and discarded; nothing is
written to disk, and each connection keeps its own smoothing state.
"""

from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from moodtune import __version__
from moodtune.analyzer import Analyzer, FaceResult
from moodtune.config import Config, Strategy
from moodtune.music import Recommender
from moodtune.smoothing import EmotionSmoother

STATIC = Path(__file__).parent / "static"
MAX_FRAME_BYTES = 2 * 1024 * 1024
MAX_FRAME_SIDE = 1280


def decode_image(data: bytes) -> np.ndarray:
    bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("could not decode image")
    h, w = bgr.shape[:2]
    if max(h, w) > MAX_FRAME_SIDE:  # big phone photos: detection does not need the detail
        s = MAX_FRAME_SIDE / max(h, w)
        bgr = cv2.resize(bgr, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    return bgr


def build_analyzer(cfg: Config) -> Analyzer:
    from moodtune.face import FaceDetector
    from moodtune.predictor import OnnxPredictor

    return Analyzer(
        OnnxPredictor(cfg.app.model_path), FaceDetector(cfg.app.detector), cfg.app.detector
    )


def create_app(cfg: Config, analyzer: Analyzer | None = None) -> FastAPI:
    """Build the app. ``analyzer`` can be injected (tests); otherwise it is built on startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.analyzer = analyzer or build_analyzer(cfg)
        yield

    app = FastAPI(title="MoodTune", version=__version__, lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    recommender = Recommender(cfg.music)
    max_upload = int(cfg.app.max_upload_mb * 1024 * 1024)

    def _analyzer() -> Analyzer:
        return app.state.analyzer

    def _payload(faces: list[FaceResult], frame_shape: tuple[int, ...]) -> dict:
        return {
            "frame": {"width": frame_shape[1], "height": frame_shape[0]},
            "faces": [f.to_dict() for f in faces],
        }

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})

    @app.get("/api/health")
    def health() -> dict:
        a = _analyzer()
        return {
            "status": "ok",
            "version": __version__,
            "model": a.predictor.meta,
            "detector": a.detector.kind,
        }

    @app.get("/api/meta")
    def meta() -> dict:
        return {
            "labels": list(_analyzer().predictor.labels),
            "strategies": ["match", "lift"],
            "default_strategy": cfg.music.default_strategy,
            "min_confidence": cfg.music.min_confidence,
        }

    @app.get("/api/recommend")
    def recommend(
        emotion: str | None = None,
        confidence: float | None = Query(None, ge=0, le=1),
        strategy: Strategy | None = None,
    ) -> dict:
        if emotion is not None and emotion not in _analyzer().predictor.labels:
            raise HTTPException(422, f"unknown emotion {emotion!r}")
        return recommender.recommend(emotion, confidence, strategy)

    @app.post("/api/analyze")
    async def analyze(file: UploadFile = File(...), strategy: Strategy | None = None) -> dict:
        """Analyse one uploaded photo. The image is decoded in memory and then discarded."""
        data = await file.read(max_upload + 1)
        if len(data) > max_upload:
            raise HTTPException(413, "image too large")
        try:
            bgr = await run_in_threadpool(decode_image, data)
        except ValueError as e:
            raise HTTPException(400, str(e)) from None
        faces = await run_in_threadpool(_analyzer().analyze, bgr)
        main = faces[0] if faces else None
        return {
            **_payload(faces, bgr.shape),
            "recommendation": recommender.recommend(
                main.emotion if main else None, main.confidence if main else None, strategy
            ),
        }

    @app.websocket("/api/stream")
    async def stream(ws: WebSocket) -> None:
        """Binary messages: JPEG frames. Text messages: ``{"strategy": "match" | "lift"}``.

        Each frame gets one JSON reply, so the client sends its next frame only after the
        previous one is answered and the server never queues work.
        """
        await ws.accept()
        smoother = EmotionSmoother(cfg.app.smoothing, list(_analyzer().predictor.labels))
        strategy: Strategy = cfg.music.default_strategy
        try:
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if msg.get("text") is not None:
                    try:
                        s = json.loads(msg["text"]).get("strategy")
                    except (json.JSONDecodeError, AttributeError):
                        s = None
                    if s in ("match", "lift"):
                        strategy = s
                    continue
                data = msg.get("bytes") or b""
                if len(data) > MAX_FRAME_BYTES:
                    await ws.send_json({"error": "frame too large"})
                    continue
                t0 = time.perf_counter()
                try:
                    bgr = await run_in_threadpool(decode_image, data)
                except ValueError as e:
                    await ws.send_json({"error": str(e)})
                    continue
                faces = await run_in_threadpool(_analyzer().analyze, bgr)
                state = smoother.update(faces[0].probs if faces else None, time.monotonic())
                await ws.send_json(
                    {
                        **_payload(faces, bgr.shape),
                        "smoothed": state,
                        "recommendation": recommender.recommend(
                            state["emotion"], state["confidence"], strategy
                        ),
                        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                    }
                )
        except WebSocketDisconnect:
            pass

    return app
