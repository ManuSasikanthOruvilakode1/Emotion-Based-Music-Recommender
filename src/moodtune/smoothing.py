"""Temporal smoothing of per-frame emotion probabilities for the webcam stream.

A single-frame classifier flickers between emotions as the face moves, blinks or talks. Each
stream gets a small state machine:

* exponential moving average of the probability vector,
* hysteresis: a challenger must lead the current emotion by ``switch_margin``,
* a minimum dwell time before the challenger is committed,
* a reset when no face has been seen for ``reset_after_s``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from moodtune.config import SmoothingConfig


@dataclass
class EmotionSmoother:
    cfg: SmoothingConfig
    labels: list[str]
    probs: np.ndarray | None = None
    emotion: str | None = None
    since: float | None = None
    last_face_t: float | None = None
    _candidate: str | None = field(default=None, repr=False)
    _candidate_since: float | None = field(default=None, repr=False)

    def reset(self) -> None:
        self.probs = self.emotion = self.since = None
        self._candidate = self._candidate_since = None

    def update(self, probs: np.ndarray | None, t: float) -> dict:
        """Feed one frame's probabilities (``None`` = no face). Returns the current state."""
        if probs is None:
            if self.last_face_t is not None and t - self.last_face_t > self.cfg.reset_after_s:
                self.reset()
            return self.state(t)

        self.last_face_t = t
        a = self.cfg.ema_alpha
        p = np.asarray(probs, dtype=np.float64)
        self.probs = p.copy() if self.probs is None else a * p + (1 - a) * self.probs

        top = self.labels[int(self.probs.argmax())]
        if self.emotion is None:
            self.emotion, self.since = top, t
        elif top == self.emotion:
            self._candidate = self._candidate_since = None
        else:
            lead = self.probs.max() - self.probs[self.labels.index(self.emotion)]
            if lead < self.cfg.switch_margin:
                self._candidate = self._candidate_since = None
            else:
                if top != self._candidate:
                    self._candidate, self._candidate_since = top, t
                if t - self._candidate_since >= self.cfg.min_dwell_s:
                    self.emotion, self.since = top, self._candidate_since
                    self._candidate = self._candidate_since = None
        return self.state(t)

    def state(self, t: float) -> dict:
        if self.probs is None:
            return {"emotion": None, "confidence": None, "probs": None, "for_s": None}
        return {
            "emotion": self.emotion,
            "confidence": round(float(self.probs[self.labels.index(self.emotion)]), 4),
            "probs": {k: round(float(v), 4) for k, v in zip(self.labels, self.probs, strict=True)},
            "for_s": round(t - self.since, 2) if self.since is not None else None,
            "pending": self._candidate,
        }
