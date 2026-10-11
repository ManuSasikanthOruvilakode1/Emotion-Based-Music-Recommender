"""Emotion -> music suggestion as YouTube search links.

The original app opened ``"<emotion> song"`` in a browser on the *server*. Here the query
comes from the config, depends on a strategy (``match`` the mood or ``lift`` it), and is
returned as a link for the user's own browser.
"""

from __future__ import annotations

from urllib.parse import quote_plus

from moodtune.config import MusicConfig, Strategy

_FALLBACK = {"match": ["{emotion} songs"]}


class Recommender:
    def __init__(self, cfg: MusicConfig):
        self.cfg = cfg

    def link(self, query: str) -> str:
        return self.cfg.search_url.format(query=quote_plus(query))

    def recommend(
        self, emotion: str | None, confidence: float | None, strategy: Strategy | None = None
    ) -> dict:
        strategy = strategy or self.cfg.default_strategy
        uncertain = emotion is None or confidence is None or confidence < self.cfg.min_confidence
        target = "neutral" if uncertain else emotion
        by_strategy = self.cfg.queries.get(target, _FALLBACK)
        used = strategy if strategy in by_strategy else "match"
        queries = [q.format(emotion=target) for q in by_strategy[used]]
        return {
            "emotion": emotion,
            "uncertain": uncertain,
            "strategy": strategy,
            "applied": used,  # "match" when no lift list exists for this emotion
            "query": queries[0],
            "url": self.link(queries[0]),
            "alternatives": [{"query": q, "url": self.link(q)} for q in queries[1:]],
        }
